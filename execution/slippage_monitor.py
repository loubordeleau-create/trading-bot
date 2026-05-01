"""
Slippage Monitor. Track le slippage moyen par symbol et alerte si dégradation.
Signal : un slippage qui grimpe = liquidité qui baisse = danger imminent.
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Deque

logger = logging.getLogger(__name__)


@dataclass
class SlippageStats:
    recent_bps: Deque[float] = field(default_factory=lambda: deque(maxlen=50))
    total_samples: int = 0
    cumulative_bps: float = 0.0

    @property
    def avg_bps(self) -> float:
        if not self.recent_bps:
            return 0
        return sum(self.recent_bps) / len(self.recent_bps)

    @property
    def p95_bps(self) -> float:
        if len(self.recent_bps) < 5:
            return 0
        sorted_vals = sorted(self.recent_bps)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]


class SlippageMonitor:
    def __init__(self, alert_threshold_bps: float = 30, degradation_ratio: float = 2.0):
        self.alert_threshold_bps = alert_threshold_bps
        self.degradation_ratio = degradation_ratio
        self._stats: Dict[str, SlippageStats] = {}
        self._baseline: Dict[str, float] = {}

    def record(self, symbol: str, slippage_bps: float) -> dict:
        """
        Enregistre un slippage et retourne un dict d'alertes.
        {
          'alert': bool,
          'reason': str,
          'avg_bps': float,
          'p95_bps': float
        }
        """
        stats = self._stats.setdefault(symbol, SlippageStats())
        stats.recent_bps.append(slippage_bps)
        stats.total_samples += 1
        stats.cumulative_bps += slippage_bps

        # Baseline : moyenne des 20 premiers échantillons
        if symbol not in self._baseline and len(stats.recent_bps) >= 20:
            self._baseline[symbol] = stats.avg_bps

        alert = False
        reason = "OK"

        if slippage_bps > self.alert_threshold_bps:
            alert = True
            reason = f"Slippage {slippage_bps:.1f}bps > seuil {self.alert_threshold_bps}bps"

        baseline = self._baseline.get(symbol)
        if baseline and baseline > 0 and stats.avg_bps > baseline * self.degradation_ratio:
            alert = True
            reason = (
                f"Slippage moyen {stats.avg_bps:.1f}bps > "
                f"{self.degradation_ratio}x baseline {baseline:.1f}bps — liquidité dégradée"
            )

        if alert:
            logger.warning(f"[{symbol}] Slippage alert: {reason}")

        return {
            "alert": alert,
            "reason": reason,
            "avg_bps": stats.avg_bps,
            "p95_bps": stats.p95_bps,
        }

    def get_stats(self, symbol: str) -> dict:
        stats = self._stats.get(symbol)
        if not stats:
            return {"samples": 0, "avg_bps": 0, "p95_bps": 0}
        return {
            "samples": stats.total_samples,
            "avg_bps": stats.avg_bps,
            "p95_bps": stats.p95_bps,
            "baseline": self._baseline.get(symbol),
        }
