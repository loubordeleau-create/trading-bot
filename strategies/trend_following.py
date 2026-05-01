"""Stratégie C : Trend Following EMA crossover — anti-lookahead."""
import numpy as np
from typing import Optional
from core.market_data import OHLCV
from core.indicators import ema, atr, adx
from strategies.base import Strategy, Signal


class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    preferred_regimes = ["trending", "accumulation"]

    def __init__(self, fast_period=20, slow_period=50, atr_period=14,
                 adx_min=20, atr_stop_mult=2.0, atr_target_mult=3.0):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.adx_min = adx_min
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def analyze(self, symbol, ohlcv: OHLCV, context) -> Optional[Signal]:
        min_len = self.slow_period + 12
        if len(ohlcv.close) < min_len:
            return None
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low

        fast = ema(close, self.fast_period)
        slow = ema(close, self.slow_period)
        atr_vals = atr(high, low, close, self.atr_period)
        adx_vals = adx(high, low, close, 14)
        price = close[-1]
        curr_atr = atr_vals[-1]
        curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0

        if np.isnan(curr_atr) or curr_atr <= 0:
            return None
        if curr_adx < self.adx_min:
            return None

        prev_diff = fast[-2] - slow[-2]
        curr_diff = fast[-1] - slow[-1]

        if prev_diff <= 0 and curr_diff > 0 and price > slow[-1]:
            strength = min(0.4 + 0.3 * min(curr_adx / 50, 1.0) + 0.3 * min(abs(curr_diff) / curr_atr, 1.0), 1.0)
            return Signal(
                action="buy", strength=strength, entry_price=price,
                stop_loss=price - self.atr_stop_mult * curr_atr,
                take_profit=price + self.atr_target_mult * curr_atr,
                reasoning=f"Trend long: EMA{self.fast_period}>EMA{self.slow_period}, ADX {curr_adx:.1f}",
                strategy_name=self.name,
            )
        if prev_diff >= 0 and curr_diff < 0 and price < slow[-1]:
            strength = min(0.4 + 0.3 * min(curr_adx / 50, 1.0) + 0.3 * min(abs(curr_diff) / curr_atr, 1.0), 1.0)
            return Signal(
                action="sell", strength=strength, entry_price=price,
                stop_loss=price + self.atr_stop_mult * curr_atr,
                take_profit=price - self.atr_target_mult * curr_atr,
                reasoning=f"Trend short: EMA{self.fast_period}<EMA{self.slow_period}, ADX {curr_adx:.1f}",
                strategy_name=self.name,
            )
        return None
