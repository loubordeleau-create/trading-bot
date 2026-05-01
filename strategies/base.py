"""
Base class pour toutes les stratégies.

RÈGLE ANTI-LOOKAHEAD:
Toutes les stratégies évaluent sur la DERNIÈRE BOUGIE du tableau OHLCV qui leur
est présenté. Le contrat est le suivant:

- En LIVE: main.py ne doit appeler `analyze()` qu'APRÈS la clôture d'une bougie,
  et l'OHLCV passé doit se terminer sur cette bougie fermée (pas la courante).
- En BACKTEST: le backtester présente les données jusqu'à la bougie i-1 fermée,
  puis exécute à l'open de la bougie i. Donc ohlcv[-1] = dernière bougie fermée.

Les stratégies utilisent donc directement `close[-1]`, `high[-1]`, etc.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from core.market_data import OHLCV


@dataclass
class Signal:
    action: str
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    strategy_name: str


class Strategy(ABC):
    name: str = "base"
    preferred_regimes: list = []

    @abstractmethod
    def analyze(self, symbol: str, ohlcv: OHLCV, context: dict) -> Optional[Signal]:
        raise NotImplementedError
