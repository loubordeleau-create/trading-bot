"""Stratégie D : Volatility Harvesting — anti-lookahead."""
import numpy as np
from typing import Optional
from core.market_data import OHLCV
from core.indicators import atr, rsi, bollinger_bands, adx
from strategies.base import Strategy, Signal


class VolatilityHarvestStrategy(Strategy):
    name = "vol_harvest"
    preferred_regimes = ["chaotic"]

    def __init__(self, atr_period=14, atr_spike_mult=1.5, bb_period=20,
                 atr_stop_mult=1.0, atr_target_mult=1.3):
        self.atr_period = atr_period
        self.atr_spike_mult = atr_spike_mult
        self.bb_period = bb_period
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult

    def analyze(self, symbol, ohlcv: OHLCV, context) -> Optional[Signal]:
        if len(ohlcv.close) < 62:
            return None
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low

        atr_vals = atr(high, low, close, self.atr_period)
        rsi_vals = rsi(close, 7)
        lower, middle, upper = bollinger_bands(close, self.bb_period, 2.5)
        adx_vals = adx(high, low, close, 14)

        price = close[-1]
        curr_atr = atr_vals[-1]
        atr_avg_30 = np.nanmean(atr_vals[-30:])
        curr_rsi = rsi_vals[-1]
        curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 30

        if np.isnan(curr_atr) or curr_atr <= 0:
            return None

        vol_spike = curr_atr / atr_avg_30 if atr_avg_30 > 0 else 0
        if vol_spike < self.atr_spike_mult:
            return None
        if curr_adx > 30:
            return None

        if curr_rsi < 25 and price < lower[-1] * 1.002:
            return Signal(
                action="buy",
                strength=0.5 + 0.2 * min(vol_spike - self.atr_spike_mult, 1.0),
                entry_price=price,
                stop_loss=price - self.atr_stop_mult * curr_atr,
                take_profit=price + self.atr_target_mult * curr_atr,
                reasoning=f"VolHarvest long: vol x{vol_spike:.2f}, RSI7 {curr_rsi:.1f}, ADX {curr_adx:.1f}",
                strategy_name=self.name,
            )
        if curr_rsi > 75 and price > upper[-1] * 0.998:
            return Signal(
                action="sell",
                strength=0.5 + 0.2 * min(vol_spike - self.atr_spike_mult, 1.0),
                entry_price=price,
                stop_loss=price + self.atr_stop_mult * curr_atr,
                take_profit=price - self.atr_target_mult * curr_atr,
                reasoning=f"VolHarvest short: vol x{vol_spike:.2f}, RSI7 {curr_rsi:.1f}, ADX {curr_adx:.1f}",
                strategy_name=self.name,
            )
        return None
