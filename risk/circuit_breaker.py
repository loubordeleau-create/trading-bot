"""
Circuit breaker : stoppe le trading si conditions de risque extrêmes.
- Perte journalière trop importante
- Drawdown hebdomadaire
- Séries de pertes consécutives
- Kill-switch manuel ou par IA
"""
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class CircuitBreakerStatus:
    tripped: bool
    reason: Optional[str] = None
    cooldown_until: Optional[float] = None


class CircuitBreaker:
    def __init__(self, config):
        self.config = config
        self.risk = config.risk
        self._cooldown_until: Optional[float] = None
        self._tripped_reason: Optional[str] = None

    def check(self, state, capital: float) -> CircuitBreakerStatus:
        """Appelé avant chaque décision de trade. Retourne le statut courant."""
        now = time.time()

        # Si on est en cooldown, on attend
        if self._cooldown_until and now < self._cooldown_until:
            return CircuitBreakerStatus(
                tripped=True,
                reason=self._tripped_reason,
                cooldown_until=self._cooldown_until,
            )
        if self._cooldown_until and now >= self._cooldown_until:
            self._cooldown_until = None
            self._tripped_reason = None

        # Kill-switch manuel / IA
        if state.kill_switch:
            return CircuitBreakerStatus(tripped=True, reason="Kill-switch actif (IA ou manuel)")

        # Perte journalière
        daily_pnl = state.daily_pnl()
        if daily_pnl < 0 and abs(daily_pnl) / capital > self.risk.max_daily_loss_pct:
            self._trip(f"Perte journalière {daily_pnl:.2f} USDT "
                      f"({abs(daily_pnl)/capital*100:.1f}%) > seuil "
                      f"{self.risk.max_daily_loss_pct*100:.1f}%",
                      cooldown_hours=24)
            return CircuitBreakerStatus(True, self._tripped_reason, self._cooldown_until)

        # Drawdown hebdomadaire
        weekly_pnl = state.weekly_pnl()
        if weekly_pnl < 0 and abs(weekly_pnl) / capital > self.risk.max_weekly_drawdown_pct:
            self._trip(f"Drawdown hebdo {weekly_pnl:.2f} USDT "
                      f"({abs(weekly_pnl)/capital*100:.1f}%) > seuil "
                      f"{self.risk.max_weekly_drawdown_pct*100:.1f}%",
                      cooldown_hours=48)
            return CircuitBreakerStatus(True, self._tripped_reason, self._cooldown_until)

        # Pertes consécutives
        if state.consecutive_losses >= self.risk.max_consecutive_losses:
            self._trip(f"{state.consecutive_losses} pertes consécutives "
                      f">= seuil {self.risk.max_consecutive_losses}",
                      cooldown_hours=12)
            return CircuitBreakerStatus(True, self._tripped_reason, self._cooldown_until)

        return CircuitBreakerStatus(tripped=False)

    def _trip(self, reason: str, cooldown_hours: float):
        self._tripped_reason = reason
        self._cooldown_until = time.time() + cooldown_hours * 3600

    def reset(self):
        self._cooldown_until = None
        self._tripped_reason = None
