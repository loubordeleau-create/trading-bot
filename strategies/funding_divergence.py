"""
Funding Rate Divergence Strategy.

THÈSE :
Quand le funding rate sur les perpetual futures devient anormalement extrême
(z-score > +2 ou < -2 vs historique 14 jours), le positionnement est déséquilibré
et un reversal mean-reverting est probable dans les heures/jours qui suivent.

- Funding très POSITIF = longs surpayés = probable CORRECTION → signal SHORT
- Funding très NÉGATIF = shorts surpayés = probable BOUNCE → signal LONG

IMPORTANT :
- On utilise les funding perps (Bybit/Binance) comme indicateur externe
- On trade en SPOT (Kraken) donc pas de short direct → on ignore les signaux short pour l'instant
  OU on les implémente comme "sortie de long" si on a une position long ouverte
- Version simple d'abord : ON NE TRADE QUE LES LONGS (funding très négatif = achat)

Paramètres :
- z_threshold: 2.0 par défaut (signal à ±2 sigma)
- lookback_days: 14 jours pour calcul de la baseline
- cooldown_hours: 24h entre signaux (évite overtrading sur extrême persistant)
"""
import logging
import time
from typing import Optional

import numpy as np

from core.funding_data import FundingFetcher, compute_funding_zscore
from core.indicators import atr
from core.market_data import OHLCV
from strategies.base import Signal, Strategy

logger = logging.getLogger(__name__)


class FundingDivergenceStrategy(Strategy):
    name = "funding_divergence"
    preferred_regimes = ["ranging", "trending", "chaotic"]  # Marche dans tous les régimes

    def __init__(
        self,
        z_threshold: float = 2.0,
        lookback_days: int = 14,
        cooldown_hours: int = 24,
        atr_stop_mult: float = 2.0,
        atr_target_mult: float = 3.0,
        funding_source: str = "bybit",
        enable_shorts: bool = False,  # False pour Kraken spot
    ):
        self.z_threshold = z_threshold
        self.lookback_days = lookback_days
        self.cooldown_sec = cooldown_hours * 3600
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.enable_shorts = enable_shorts

        self._fetcher: Optional[FundingFetcher] = None
        self._funding_source = funding_source

        # Cache pour éviter de re-fetch le funding à chaque bougie
        # (le funding change toutes les 8h seulement)
        self._cache_symbol: Optional[str] = None
        self._cache_history: list = []
        self._cache_fetched_at: float = 0
        self._cache_ttl_sec: int = 3600  # refetch chaque heure

        # Anti-spam: timestamp du dernier signal par symbol
        self._last_signal_ts: dict = {}

    def _get_fetcher(self) -> FundingFetcher:
        if self._fetcher is None:
            self._fetcher = FundingFetcher(self._funding_source)
        return self._fetcher

    def _refresh_funding_data(self, symbol: str) -> tuple:
        """
        Retourne (current_funding, history) avec cache.
        """
        now = time.time()
        cache_valid = (
            self._cache_symbol == symbol
            and (now - self._cache_fetched_at) < self._cache_ttl_sec
            and len(self._cache_history) > 0
        )

        fetcher = self._get_fetcher()

        if not cache_valid:
            try:
                history = fetcher.get_funding_history(symbol, days=self.lookback_days + 2)
                if history:
                    self._cache_symbol = symbol
                    self._cache_history = history
                    self._cache_fetched_at = now
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to fetch funding history: {e}")
                return None, []

        current = fetcher.get_current_funding(symbol)
        return current, self._cache_history

    def analyze(self, symbol: str, ohlcv: OHLCV, context: dict) -> Optional[Signal]:
        if len(ohlcv.close) < 30:
            return None

        # 1. Fetch funding data
        current_funding, history = self._refresh_funding_data(symbol)
        if current_funding is None or len(history) < 10:
            return None

        # 2. Compute z-score
        z = compute_funding_zscore(history, current_funding, self.lookback_days)

        # 3. Anti-spam cooldown
        now = time.time()
        last_ts = self._last_signal_ts.get(symbol, 0)
        if (now - last_ts) < self.cooldown_sec:
            return None

        # 4. Décision
        action = None
        reasoning = None

        if z <= -self.z_threshold:
            # Funding très négatif = shorts surpayés = probable bounce
            action = "buy"
            reasoning = (
                f"Funding z={z:.2f} (rate={current_funding.rate*100:.4f}% per 8h, "
                f"annualized={current_funding.rate_annualized*100:.1f}%). "
                f"Shorts surpayés → probable bounce mean-reverting."
            )

        elif z >= self.z_threshold and self.enable_shorts:
            # Funding très positif = longs surpayés = probable correction
            action = "sell"
            reasoning = (
                f"Funding z={z:.2f} (rate={current_funding.rate*100:.4f}% per 8h, "
                f"annualized={current_funding.rate_annualized*100:.1f}%). "
                f"Longs surpayés → probable correction."
            )

        if action is None:
            return None

        # 5. Calcul stops/targets via ATR
        curr_atr = float(atr(ohlcv.high, ohlcv.low, ohlcv.close, 14)[-1])
        if np.isnan(curr_atr) or curr_atr <= 0:
            return None

        entry = float(ohlcv.close[-1])

        if action == "buy":
            stop_loss = entry - self.atr_stop_mult * curr_atr
            take_profit = entry + self.atr_target_mult * curr_atr
        else:
            stop_loss = entry + self.atr_stop_mult * curr_atr
            take_profit = entry - self.atr_target_mult * curr_atr

        # 6. Strength basée sur l'amplitude du z-score
        # z=2 → strength 0.6, z=3 → 0.8, z=4+ → 0.95
        strength = min(0.95, 0.4 + 0.2 * (abs(z) - self.z_threshold + 1))

        # Enregistre le signal pour cooldown
        self._last_signal_ts[symbol] = now

        return Signal(
            action=action,
            strength=strength,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            strategy_name=self.name,
        )
