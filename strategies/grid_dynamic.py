"""Stratégie E : Grid dynamique — anti-lookahead."""
import numpy as np
from typing import Optional, Dict, List
from core.market_data import OHLCV
from core.indicators import atr, adx
from strategies.base import Strategy, Signal


class GridDynamicStrategy(Strategy):
    name = "grid_dynamic"
    preferred_regimes = ["accumulation", "ranging"]

    def __init__(self, levels=5, range_lookback=50, adx_max=20, grid_spacing_atr=0.5):
        self.levels = levels
        self.range_lookback = range_lookback
        self.adx_max = adx_max
        self.grid_spacing_atr = grid_spacing_atr
        self._grids: Dict[str, List[float]] = {}
        self._last_signal_level: Dict[str, float] = {}

    def analyze(self, symbol, ohlcv: OHLCV, context) -> Optional[Signal]:
        if len(ohlcv.close) < self.range_lookback + 2:
            return None
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low

        adx_vals = adx(high, low, close, 14)
        atr_vals = atr(high, low, close, 14)
        curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 50
        curr_atr = atr_vals[-1]
        price = close[-1]

        if np.isnan(curr_atr) or curr_atr <= 0:
            return None
        if curr_adx > self.adx_max:
            self._grids.pop(symbol, None)
            return None

        range_high = high[-self.range_lookback:].max()
        range_low = low[-self.range_lookback:].min()
        range_mid = (range_high + range_low) / 2
        range_size = range_high - range_low

        if range_size < curr_atr * 3 or range_size > curr_atr * 15:
            return None

        if symbol not in self._grids:
            step = range_size / (self.levels + 1)
            self._grids[symbol] = [range_low + step * (i + 1) for i in range(self.levels)]

        grid = self._grids[symbol]
        last_level = self._last_signal_level.get(symbol, range_mid)

        below_levels = [l for l in grid if l < price]
        above_levels = [l for l in grid if l > price]

        if below_levels and price < range_mid:
            nearest_below = max(below_levels)
            recently_touched = bool(np.any(low[-3:] <= nearest_below * 1.002))
            if recently_touched and abs(nearest_below - last_level) > curr_atr * 0.5:
                self._last_signal_level[symbol] = nearest_below
                return Signal(
                    action="buy",
                    strength=0.45 + 0.3 * ((range_mid - price) / (range_size / 2)),
                    entry_price=price,
                    stop_loss=range_low - curr_atr,
                    take_profit=range_mid,
                    reasoning=f"Grid long lvl {nearest_below:.2f}, range [{range_low:.2f}, {range_high:.2f}], ADX {curr_adx:.1f}",
                    strategy_name=self.name,
                )
        if above_levels and price > range_mid:
            nearest_above = min(above_levels)
            recently_touched = bool(np.any(high[-3:] >= nearest_above * 0.998))
            if recently_touched and abs(nearest_above - last_level) > curr_atr * 0.5:
                self._last_signal_level[symbol] = nearest_above
                return Signal(
                    action="sell",
                    strength=0.45 + 0.3 * ((price - range_mid) / (range_size / 2)),
                    entry_price=price,
                    stop_loss=range_high + curr_atr,
                    take_profit=range_mid,
                    reasoning=f"Grid short lvl {nearest_above:.2f}, range [{range_low:.2f}, {range_high:.2f}], ADX {curr_adx:.1f}",
                    strategy_name=self.name,
                )
        return None
