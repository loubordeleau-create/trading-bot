"""
Stop manager v2 - Trailing stop progressif 3 niveaux.
Philosophie: Letting winners run.
"""
import time
from typing import Optional


class StopManager:
    def __init__(self, config):
        self.level1_atr_profit = 1.0
        self.level2_atr_profit = 2.0
        self.level2_atr_lock = 0.5
        self.level3_atr_profit = 3.0
        self.level3_atr_trail = 2.0
        self.time_stop_hours = 48
        self.time_stop_progress_required_atr = 0.5

    def check_stop(self, position, current_price: float, current_atr: float) -> Optional[str]:
        if not hasattr(position, 'best_price') or position.best_price is None:
            position.best_price = current_price
            position.trailing_level = 0

        if position.side == "long":
            position.best_price = max(position.best_price, current_price)
        else:
            position.best_price = min(position.best_price, current_price)

        if position.side == "long":
            if current_price <= position.stop_loss:
                return "sl"
            if current_price >= position.take_profit:
                return "tp"
        else:
            if current_price >= position.stop_loss:
                return "sl"
            if current_price <= position.take_profit:
                return "tp"

        if current_atr > 0:
            self._update_trailing_stop(position, current_atr)

        hours_open = (time.time() - position.entry_time) / 3600
        if hours_open >= self.time_stop_hours and current_atr > 0:
            profit_atr = self._profit_in_atr(position, current_price, current_atr)
            if profit_atr < self.time_stop_progress_required_atr:
                return "time_stop"

        return None

    def _update_trailing_stop(self, position, current_atr: float):
        best = position.best_price
        entry = position.entry_price
        current_level = getattr(position, 'trailing_level', 0)

        if position.side == "long":
            profit_atr = (best - entry) / current_atr if current_atr > 0 else 0

            if profit_atr >= self.level3_atr_profit:
                new_stop = best - self.level3_atr_trail * current_atr
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    if current_level < 3:
                        position.trailing_level = 3
                return

            if profit_atr >= self.level2_atr_profit:
                new_stop = entry + self.level2_atr_lock * current_atr
                if new_stop > position.stop_loss:
                    position.stop_loss = new_stop
                    if current_level < 2:
                        position.trailing_level = 2
                return

            if profit_atr >= self.level1_atr_profit:
                if entry > position.stop_loss:
                    position.stop_loss = entry
                    if current_level < 1:
                        position.trailing_level = 1

        else:  # short
            profit_atr = (entry - best) / current_atr if current_atr > 0 else 0

            if profit_atr >= self.level3_atr_profit:
                new_stop = best + self.level3_atr_trail * current_atr
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    if current_level < 3:
                        position.trailing_level = 3
                return

            if profit_atr >= self.level2_atr_profit:
                new_stop = entry - self.level2_atr_lock * current_atr
                if new_stop < position.stop_loss:
                    position.stop_loss = new_stop
                    if current_level < 2:
                        position.trailing_level = 2
                return

            if profit_atr >= self.level1_atr_profit:
                if entry < position.stop_loss:
                    position.stop_loss = entry
                    if current_level < 1:
                        position.trailing_level = 1

    def _profit_in_atr(self, position, current_price: float, atr: float) -> float:
        if atr <= 0:
            return 0
        if position.side == "long":
            return (current_price - position.entry_price) / atr
        else:
            return (position.entry_price - current_price) / atr
