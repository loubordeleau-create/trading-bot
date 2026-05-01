"""Stratégie B : Momentum Breakout — anti-lookahead."""
import numpy as np
from typing import Optional
from core.market_data import OHLCV
from core.indicators import atr, adx, ema, volume_profile_imbalance
from strategies.base import Strategy, Signal


class MomentumBreakoutStrategy(Strategy):
    name = "momentum_breakout"
    preferred_regimes = ["trending"]

    def __init__(self, lookback_range=20, atr_period=14, adx_min=25,
                 volume_confirm_mult=1.5, atr_stop_mult=1.5, atr_target_mult=3.0):
        self.lookback_range = lookback_range
        self.atr_period = atr_period
        self.adx_min = adx_min
        self.volume_confirm_mult = volume_confirm_mult
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def analyze(self, symbol, ohlcv: OHLCV, context) -> Optional[Signal]:
        min_len = max(52, self.lookback_range + 5)
        if len(ohlcv.close) < min_len:
            return None
        # Tronque à la dernière bougie fermée
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low
        volume = ohlcv.volume

        atr_vals = atr(high, low, close, self.atr_period)
        adx_vals = adx(high, low, close, 14)
        ema50 = ema(close, 50)
        price = close[-1]
        curr_atr = atr_vals[-1]
        curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0

        if np.isnan(curr_atr) or curr_atr <= 0:
            return None
        if curr_adx < self.adx_min:
            return None

        # Range des N bougies AVANT celle en cours (déjà fermées)
        range_high = high[-self.lookback_range - 1:-1].max()
        range_low = low[-self.lookback_range - 1:-1].min()
        avg_volume = volume[-self.lookback_range - 1:-1].mean()
        curr_vol = volume[-1]
        vol_spike = curr_vol / avg_volume if avg_volume > 0 else 0

        vol_imb = volume_profile_imbalance(volume, close, 10)

        if price > range_high and vol_spike >= self.volume_confirm_mult and vol_imb > 0.1:
            if price <= ema50[-1]:
                return None
            strength = self._strength(
                (price - range_high) / range_high,
                min(vol_spike / 3, 1.0),
                min((curr_adx - self.adx_min) / 30, 1.0),
                max(vol_imb, 0),
            )
            return Signal(
                action="buy", strength=strength, entry_price=price,
                stop_loss=max(range_high - self.atr_stop_mult * curr_atr, range_low),
                take_profit=price + self.atr_target_mult * curr_atr,
                reasoning=f"Breakout up: price>range_high {range_high:.2f}, vol spike {vol_spike:.1f}x, ADX {curr_adx:.1f}",
                strategy_name=self.name,
            )

        if price < range_low and vol_spike >= self.volume_confirm_mult and vol_imb < -0.1:
            if price >= ema50[-1]:
                return None
            strength = self._strength(
                (range_low - price) / range_low,
                min(vol_spike / 3, 1.0),
                min((curr_adx - self.adx_min) / 30, 1.0),
                max(-vol_imb, 0),
            )
            return Signal(
                action="sell", strength=strength, entry_price=price,
                stop_loss=min(range_low + self.atr_stop_mult * curr_atr, range_high),
                take_profit=price - self.atr_target_mult * curr_atr,
                reasoning=f"Breakout down: price<range_low {range_low:.2f}, vol spike {vol_spike:.1f}x, ADX {curr_adx:.1f}",
                strategy_name=self.name,
            )
        return None

    def _strength(self, a, b, c, d):
        return max(0.0, min(1.0, 0.30 * min(a * 100, 1.0) + 0.30 * b + 0.25 * c + 0.15 * d))
