from __future__ import annotations

import os
import re
import signal
import statistics
import time
from collections import deque
from dataclasses import dataclass
from functools import cached_property

import requests

from bot_template import BaseBot, OrderBook, OrderRequest, Side, Trade


@dataclass
class ProductState:
    tick_size: float
    fair_value: float | None = None
    last_mid: float | None = None
    mid_changes: deque[float] | None = None
    paused_until: float = 0.0


@dataclass
class QuoteDecision:
    bid: float | None
    ask: float | None
    fair_value: float
    volatility: float
    half_spread: float
    reason: str = ""


class MarketMakerBot(BaseBot):
    def __init__(
        self,
        cmi_url: str,
        username: str,
        password: str,
        alpha: float = 0.2,
        order_size: int = 6,
        loop_sleep_s: float = 1.2,
    ):
        super().__init__(cmi_url, username, password)
        # self.username = username
        # self._password = password

        self.alpha = max(0.1, min(0.3, alpha))
        self.order_size = max(1, order_size)
        self.loop_sleep_s = max(0.8, loop_sleep_s)

        self.max_abs_position = 100
        self.vol_window = 180
        self.base_half_spread_ticks = 2.0
        self.vol_spread_multiplier = 2.5
        self.max_inventory_skew_ticks = 3.0
        self.max_book_spread_ticks = 30
        self.min_top_volume = 1
        self.vol_spike_frac = 0.02
        self.pause_seconds = 8.0

        self.target_market_ids = {1, 3, 5, 7}
        self.target_products: list[str] = []
        self.product_states: dict[str, ProductState] = {}

        self._running = False
        self._product_idx = 0
        self._last_log_ts = 0.0

    @cached_property
    def auth_token(self) -> str:
        response = requests.post(
            f"{self._cmi_url}/api/user/authenticate",
            headers={"Content-Type": "application/json; charset=utf-8"},
            json={"username": self.username, "password": self._password},
            timeout=10,
        )
        response.raise_for_status()
        return response.headers["Authorization"]

    def on_orderbook(self, orderbook: OrderBook) -> None:
        return

    def on_trades(self, trade: Trade) -> None:
        return

    def initialize_products(self) -> None:
        products = self.get_products()
        product_by_symbol = {p.symbol.upper(): p for p in products}
        chosen: list[str] = []

        preferred_symbols = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT", "LON_ETF"]
        for symbol in preferred_symbols:
            product = product_by_symbol.get(symbol)
            if product:
                chosen.append(product.symbol)

        if len(chosen) < 4:
            numeric_candidates: list[str] = []
            for product in products:
                market_id = self._extract_market_id(product.symbol)
                if market_id in self.target_market_ids:
                    numeric_candidates.append(product.symbol)

            merged = []
            seen = set()
            for symbol in [*chosen, *numeric_candidates]:
                if symbol not in seen:
                    seen.add(symbol)
                    merged.append(symbol)
            chosen = merged

        if len(chosen) < 4 and len(products) >= 7:
            fallback_index_symbols = [products[i].symbol for i in (0, 2, 4, 6)]
            print(
                "Warning: using fallback index mapping for markets 1,3,5,7 -> "
                f"{fallback_index_symbols}"
            )
            chosen = fallback_index_symbols

        if len(chosen) < 4:
            available = [p.symbol for p in products]
            raise RuntimeError(
                "Could not detect market symbols for markets 1, 3, 5, 7. "
                f"Available symbols: {available}"
            )

        self.product_states.clear()
        for symbol in chosen:
            product = next((p for p in products if p.symbol == symbol), None)
            if not product:
                continue
            self.product_states[symbol] = ProductState(
                tick_size=product.tickSize,
                mid_changes=deque(maxlen=self.vol_window),
            )

        self.target_products = chosen
        print(f"Target products: {self.target_products}")

    def run_loop(self) -> None:
        self.initialize_products()
        self._running = True
        while self._running:
            if not self.target_products:
                time.sleep(self.loop_sleep_s)
                continue

            product = self.target_products[self._product_idx]
            self._product_idx = (self._product_idx + 1) % len(self.target_products)

            try:
                self._process_product(product)
            except Exception as exc:
                print(f"Error on {product}: {exc}")

            time.sleep(self.loop_sleep_s)

    def stop_loop(self) -> None:
        self._running = False

    def _process_product(self, product: str) -> None:
        state = self.product_states[product]
        positions = self.get_positions()
        position = positions.get(product, 0)
        orderbook = self.get_orderbook(product)
        decision = self._compute_quote_decision(orderbook, state, position)

        if decision.bid is None or decision.ask is None:
            self._cancel_existing(product)
            self._periodic_log(product, position, decision)
            return

        self._replace_quotes(product, decision.bid, decision.ask, position)
        self._periodic_log(product, position, decision)

    def _compute_quote_decision(
        self,
        orderbook: OrderBook,
        state: ProductState,
        position: int,
    ) -> QuoteDecision:
        now = time.time()

        if now < state.paused_until:
            return QuoteDecision(
                bid=None,
                ask=None,
                fair_value=state.fair_value or 0.0,
                volatility=0.0,
                half_spread=0.0,
                reason="paused",
            )

        if not orderbook.buy_orders or not orderbook.sell_orders:
            state.paused_until = now + self.pause_seconds
            return QuoteDecision(
                bid=None,
                ask=None,
                fair_value=state.fair_value or 0.0,
                volatility=0.0,
                half_spread=0.0,
                reason="book missing one side",
            )

        best_bid = orderbook.buy_orders[0]
        best_ask = orderbook.sell_orders[0]

        if best_bid.volume < self.min_top_volume or best_ask.volume < self.min_top_volume:
            state.paused_until = now + self.pause_seconds
            return QuoteDecision(
                bid=None,
                ask=None,
                fair_value=state.fair_value or 0.0,
                volatility=0.0,
                half_spread=0.0,
                reason="book too thin",
            )

        mid = 0.5 * (best_bid.price + best_ask.price)
        if state.fair_value is None:
            state.fair_value = mid
        else:
            state.fair_value = self.alpha * mid + (1.0 - self.alpha) * state.fair_value

        if state.last_mid is not None and state.mid_changes is not None:
            state.mid_changes.append(mid - state.last_mid)
        state.last_mid = mid

        volatility = 0.0
        if state.mid_changes and len(state.mid_changes) > 1:
            volatility = statistics.stdev(state.mid_changes)

        if state.fair_value > 0 and volatility > self.vol_spike_frac * state.fair_value:
            state.paused_until = now + self.pause_seconds
            return QuoteDecision(
                bid=None,
                ask=None,
                fair_value=state.fair_value,
                volatility=volatility,
                half_spread=0.0,
                reason="volatility spike",
            )

        tick = state.tick_size
        book_spread = max(tick, best_ask.price - best_bid.price)
        book_spread_ticks = book_spread / tick
        if book_spread_ticks > self.max_book_spread_ticks:
            state.paused_until = now + self.pause_seconds
            return QuoteDecision(
                bid=None,
                ask=None,
                fair_value=state.fair_value,
                volatility=volatility,
                half_spread=0.0,
                reason="book spread too wide",
            )

        half_spread = (
            self.base_half_spread_ticks * tick
            + 0.5 * book_spread
            + self.vol_spread_multiplier * volatility
        )

        inv_ratio = max(-1.0, min(1.0, position / self.max_abs_position))
        inv_adjust = inv_ratio * self.max_inventory_skew_ticks * tick

        bid_raw = state.fair_value - half_spread - inv_adjust
        ask_raw = state.fair_value + half_spread + inv_adjust

        bid = self._round_down(bid_raw, tick)
        ask = self._round_up(ask_raw, tick)
        if ask - bid < tick:
            ask = bid + tick

        if bid <= 0:
            return QuoteDecision(
                bid=None,
                ask=None,
                fair_value=state.fair_value,
                volatility=volatility,
                half_spread=half_spread,
                reason="invalid bid",
            )

        return QuoteDecision(
            bid=bid,
            ask=ask,
            fair_value=state.fair_value,
            volatility=volatility,
            half_spread=half_spread,
        )

    def _replace_quotes(self, product: str, bid: float, ask: float, position: int) -> None:
        self._cancel_existing(product)

        max_buy = self.max_abs_position - position
        max_sell = self.max_abs_position + position

        orders: list[OrderRequest] = []
        if max_buy > 0:
            buy_size = min(self.order_size, max_buy)
            if buy_size > 0:
                orders.append(OrderRequest(product=product, price=bid, side=Side.BUY, volume=buy_size))

        if max_sell > 0:
            sell_size = min(self.order_size, max_sell)
            if sell_size > 0:
                orders.append(OrderRequest(product=product, price=ask, side=Side.SELL, volume=sell_size))

        for order in orders:
            self.send_order(order)

    def _cancel_existing(self, product: str) -> None:
        open_orders = self.get_orders(product)
        for order in open_orders:
            order_id = order.get("id")
            if order_id:
                self.cancel_order(order_id)

    def _periodic_log(self, product: str, position: int, decision: QuoteDecision) -> None:
        now = time.time()
        if now - self._last_log_ts < 2.5:
            return
        self._last_log_ts = now

        spread = 2.0 * decision.half_spread
        print(
            f"[{product}] pos={position:>4} fv={decision.fair_value:>8.2f} "
            f"vol={decision.volatility:>7.4f} spread={spread:>7.3f} "
            f"bid={decision.bid} ask={decision.ask} {decision.reason}"
        )

    @staticmethod
    def _round_down(price: float, tick: float) -> float:
        return tick * int(price / tick)

    @staticmethod
    def _round_up(price: float, tick: float) -> float:
        rounded_down = tick * int(price / tick)
        if abs(price - rounded_down) < 1e-12:
            return rounded_down
        return rounded_down + tick

    @staticmethod
    def _extract_market_id(symbol: str) -> int | None:
        m = re.search(r"(\d+)$", symbol)
        if m:
            return int(m.group(1))

        all_digits = re.findall(r"\d+", symbol)
        if len(all_digits) == 1:
            return int(all_digits[0])
        return None

TEST_USER = "WeAreIshaanX"
TEST_PASS = "1234"

def main() -> None:
    # exchange_url = os.getenv(
    #     "CMI_URL",
    #     "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com",
    # )
    exchange_url = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
 
    # username = os.getenv("CMI_USERNAME", "")
    # password = os.getenv("CMI_PASSWORD", "")

    # if not username or not password:
    #     raise RuntimeError("Set CMI_USERNAME and CMI_PASSWORD environment variables")

    username = TEST_USER
    password = TEST_PASS

    bot = MarketMakerBot(
        cmi_url=exchange_url,
        username=username,
        password=password,
        alpha=float(os.getenv("MM_ALPHA", "0.2")),
        order_size=int(os.getenv("MM_ORDER_SIZE", "6")),
        loop_sleep_s=float(os.getenv("MM_LOOP_SLEEP", "1.2")),
    )

    shutting_down = False

    def _shutdown_handler(signum, frame):
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        bot.stop_loop()
        bot.stop()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    bot.start()
    try:
        bot.run_loop()
    finally:
        bot.stop_loop()
        bot.stop()


if __name__ == "__main__":
    main()
