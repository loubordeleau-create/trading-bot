"""Stratégie A : Mean Reversion — anti-lookahead (signal sur bougie FERMÉE)."""
import numpy as np
from typing import Optional
from core.market_data import OHLCV
from core.indicators import rsi, bollinger_bands, atr, adx, volume_profile_imbalance
from strategies.base import Strategy, Signal


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    preferred_regimes = ["ranging", "accumulation"]

    def __init__(self, rsi_oversold=28, rsi_overbought=72, bb_period=20, bb_std=2.0,
                 atr_period=14, adx_max=22, atr_stop_mult=1.5, atr_target_mult=2.0):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.adx_max = adx_max
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def analyze(self, symbol, ohlcv: OHLCV, context) -> Optional[Signal]:
        if len(ohlcv.close) < 52:
            return None
        # Tronque à la dernière bougie fermée pour éviter tout lookahead
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low
        volume = ohlcv.volume

        rsi_vals = rsi(close, 14)
        lower, middle, upper = bollinger_bands(close, self.bb_period, self.bb_std)
        atr_vals = atr(high, low, close, self.atr_period)
        adx_vals = adx(high, low, close, 14)
        vol_imb = volume_profile_imbalance(volume, close, 20)

        price = close[-1]
        curr_rsi = rsi_vals[-1]
        curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 50
        curr_atr = atr_vals[-1]

        if np.isnan(curr_atr) or curr_atr <= 0 or np.isnan(curr_rsi):
            return None
        if curr_adx > self.adx_max:
            return None

        if curr_rsi < self.rsi_oversold and price <= lower[-1] and vol_imb > -0.3:
            strength = self._strength(
                (self.rsi_oversold - curr_rsi) / self.rsi_oversold,
                (lower[-1] - price) / lower[-1] if lower[-1] > 0 else 0,
                max(0, vol_imb),
                1 - curr_adx / self.adx_max,
            )
            return Signal(
                action="buy", strength=strength, entry_price=price,
                stop_loss=price - self.atr_stop_mult * curr_atr,
                take_profit=middle[-1],
                reasoning=f"MR long: RSI {curr_rsi:.1f}, price<BBL, ADX {curr_adx:.1f}, vol_imb {vol_imb:+.2f}",
                strategy_name=self.name,
            )

        if curr_rsi > self.rsi_overbought and price >= upper[-1] and vol_imb < 0.3:
            strength = self._strength(
                (curr_rsi - self.rsi_overbought) / (100 - self.rsi_overbought),
                (price - upper[-1]) / upper[-1] if upper[-1] > 0 else 0,
                max(0, -vol_imb),
                1 - curr_adx / self.adx_max,
            )
            return Signal(
                action="sell", strength=strength, entry_price=price,
                stop_loss=price + self.atr_stop_mult * curr_atr,
                take_profit=middle[-1],
                reasoning=f"MR short: RSI {curr_rsi:.1f}, price>BBU, ADX {curr_adx:.1f}, vol_imb {vol_imb:+.2f}",
                strategy_name=self.name,
            )
        return None

    def _strength(self, a, b, c, d):
        return max(0.0, min(1.0, 0.35 * min(a, 1.0) + 0.25 * min(b * 50, 1.0) + 0.20 * c + 0.20 * d))
