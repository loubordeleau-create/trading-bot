"""
Funding rate data fetcher.

Les funding rates sont publics sur Bybit/Binance et ne nécessitent PAS de clé API.
On les utilise comme indicateur externe pour trader le spot sur Kraken.

Usage:
    fetcher = FundingFetcher(exchange="bybit")
    current = fetcher.get_current_funding("BTC/USDT")  # Dernier funding connu
    history = fetcher.get_funding_history("BTC/USDT", days=30)  # Historique
"""
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FundingRate:
    timestamp_ms: int
    rate: float          # Ex: 0.0001 = 0.01% par 8h
    rate_annualized: float  # Ex: 0.01% × 3 × 365 = ~11%

    @classmethod
    def from_raw(cls, ts: int, rate: float) -> "FundingRate":
        # La plupart des exchanges paient toutes les 8h = 3x par jour
        annualized = rate * 3 * 365
        return cls(timestamp_ms=int(ts), rate=float(rate), rate_annualized=float(annualized))


class FundingFetcher:
    """
    Fetch funding rates depuis un exchange perp (Bybit ou Binance).
    Pas besoin de clé API - les endpoints funding sont publics.
    """

    # Conversion symbol spot → symbol perp
    # Ex: "BTC/USDT" (spot Kraken) -> "BTC/USDT:USDT" (perp Bybit)
    PERP_SUFFIX = ":USDT"

    def __init__(self, exchange: str = "bybit"):
        try:
            import ccxt
        except ImportError:
            raise ImportError("ccxt required: pip install ccxt")

        exchange_class = getattr(ccxt, exchange)
        self.client = exchange_class({"enableRateLimit": True})
        self.exchange_name = exchange

    def _to_perp_symbol(self, spot_symbol: str) -> str:
        """Convertit 'BTC/USDT' -> 'BTC/USDT:USDT' (format perp ccxt)."""
        if ":" in spot_symbol:
            return spot_symbol  # Déjà un perp
        return spot_symbol + self.PERP_SUFFIX

    def get_current_funding(self, symbol: str) -> Optional[FundingRate]:
        """
        Retourne le funding rate le plus récent.
        Returns None si échec.
        """
        perp_symbol = self._to_perp_symbol(symbol)
        try:
            data = self.client.fetch_funding_rate(perp_symbol)
            if not data:
                return None
            rate = data.get("fundingRate")
            ts = data.get("timestamp") or int(time.time() * 1000)
            if rate is None:
                return None
            return FundingRate.from_raw(ts, rate)
        except Exception as e:
            logger.warning(f"Funding fetch failed for {perp_symbol}: {e}")
            return None

    def get_funding_history(self, symbol: str, days: int = 30,
                            limit_per_call: int = 200) -> List[FundingRate]:
        """
        Fetch l'historique des funding rates sur N jours.
        Funding est publié toutes les 8h = ~3 entries/jour.
        """
        perp_symbol = self._to_perp_symbol(symbol)
        since_ms = int((time.time() - days * 86400) * 1000)

        all_rates: List[FundingRate] = []
        current_since = since_ms
        max_iter = 50  # safety

        for _ in range(max_iter):
            try:
                chunk = self.client.fetch_funding_rate_history(
                    perp_symbol, since=current_since, limit=limit_per_call
                )
            except Exception as e:
                logger.warning(f"Funding history fetch failed: {e}")
                break

            if not chunk:
                break

            for entry in chunk:
                rate = entry.get("fundingRate")
                ts = entry.get("timestamp")
                if rate is not None and ts is not None:
                    all_rates.append(FundingRate.from_raw(ts, rate))

            last_ts = chunk[-1].get("timestamp")
            if not last_ts or last_ts <= current_since:
                break
            current_since = last_ts + 1

            if len(chunk) < limit_per_call:
                break
            time.sleep(max(self.client.rateLimit / 1000, 0.2))

        # Dedup et sort
        seen = set()
        deduped = []
        for r in all_rates:
            if r.timestamp_ms not in seen:
                seen.add(r.timestamp_ms)
                deduped.append(r)
        deduped.sort(key=lambda r: r.timestamp_ms)

        return deduped


def compute_funding_zscore(history: List[FundingRate], current: FundingRate,
                           lookback_days: int = 14) -> float:
    """
    Z-score du funding actuel vs la moyenne des N derniers jours.

    - Z > +2 = funding anormalement élevé (longs surpayés) → signal SHORT
    - Z < -2 = funding anormalement négatif (shorts surpayés) → signal LONG
    - |Z| < 1 = normal, pas de signal

    Pourquoi z-score et pas seuil absolu :
    Le funding "normal" varie selon la paire et la période. BTC tourne autour
    de 0.01%, mais en bull market ça peut être 0.04% normal. On veut détecter
    les EXTRÊMES relatifs, pas des seuils fixes.
    """
    if not history or current is None:
        return 0.0

    cutoff_ms = current.timestamp_ms - lookback_days * 86400 * 1000
    recent = [r.rate for r in history if r.timestamp_ms >= cutoff_ms]

    if len(recent) < 10:
        return 0.0

    arr = np.array(recent)
    mean = arr.mean()
    std = arr.std()
    if std < 1e-9:
        return 0.0

    z = (current.rate - mean) / std
    return float(z)
