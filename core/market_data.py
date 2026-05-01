"""
Fetch + cache des données de marché. Évite les appels API redondants.
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class OHLCV:
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    @classmethod
    def from_raw(cls, raw: list) -> "OHLCV":
        arr = np.array(raw, dtype=float)
        return cls(
            timestamps=arr[:, 0],
            open=arr[:, 1],
            high=arr[:, 2],
            low=arr[:, 3],
            close=arr[:, 4],
            volume=arr[:, 5],
        )


class MarketData:
    """Fetch OHLCV avec cache court terme pour limiter les appels API."""

    def __init__(self, exchange, cache_ttl_seconds: int = 30):
        self.exchange = exchange
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[Tuple[str, str], Tuple[float, OHLCV]] = {}

    def get_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> OHLCV:
        key = (symbol, timeframe)
        now = time.time()
        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts < self.cache_ttl:
                return data
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit)
        data = OHLCV.from_raw(raw)
        self._cache[key] = (now, data)
        return data

    def get_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker.get("last") or ticker.get("close"))

    def get_spread_bps(self, symbol: str) -> float:
        """Spread bid-ask en basis points (1bp = 0.01%)."""
        ob = self.exchange.fetch_orderbook(symbol, limit=5)
        if not ob["bids"] or not ob["asks"]:
            return 9999
        bid, ask = ob["bids"][0][0], ob["asks"][0][0]
        mid = (bid + ask) / 2
        return (ask - bid) / mid * 10000

    def correlation_matrix(self, symbols: list, timeframe: str = "1h",
                          lookback: int = 100) -> Dict[Tuple[str, str], float]:
        """Matrice de corrélation des returns entre paires."""
        returns = {}
        for s in symbols:
            data = self.get_ohlcv(s, timeframe, lookback + 1)
            if len(data.close) < 2:
                continue
            returns[s] = np.diff(np.log(data.close))

        correlations = {}
        symbols_list = list(returns.keys())
        for i, s1 in enumerate(symbols_list):
            for s2 in symbols_list[i + 1:]:
                r1, r2 = returns[s1], returns[s2]
                n = min(len(r1), len(r2))
                if n < 10:
                    correlations[(s1, s2)] = 0.0
                    continue
                c = np.corrcoef(r1[-n:], r2[-n:])[0, 1]
                correlations[(s1, s2)] = float(c) if not np.isnan(c) else 0.0
        return correlations
