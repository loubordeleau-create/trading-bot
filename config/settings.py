"""
Configuration centrale du bot - Kraken (Quebec legal).
NEXT LEVEL: Claude a plus de pouvoir decisionnel. Sizing agressif.
Les cles API se mettent dans le fichier .env, pas ici.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ExchangeConfig:
    name: str = "kraken"
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False


@dataclass
class TradingConfig:
    symbols: List[str] = field(default_factory=lambda: [
        "BTC/USDT",
        "ETH/USDT",
        "SOL/USDT",
        "XRP/USDT",
        "LINK/USDT",
        "ADA/USDT",
        "DOGE/USDT",
    ])

    timeframes: List[str] = field(default_factory=lambda: ["15m", "1h", "4h"])
    primary_tf: str = "1h"

    # Capital bumped pour voir du vrai mouvement en paper
    capital_usdt: float = 10000.0

    # Sizing plus agressif
    max_position_pct: float = 0.25
    max_total_exposure: float = 0.80
    max_leverage: float = 1.0

    # Risque 0.25% par trade
    kelly_fraction: float = 0.25

    atr_stop_mult: float = 1.5
    atr_target_mult: float = 2.0

    stop_mode: str = "bot_managed"


@dataclass
class AIConfig:
    anthropic_api_key: str = ""
    model: str = "claude-haiku-4-5"

    # Claude evalue plus souvent
    regime_check_minutes: int = 30
    event_scan_minutes: int = 60
    post_mortem_day: str = "Sunday"

    max_daily_api_cost_usd: float = 5.0

    # NEXT LEVEL: pouvoir de Claude
    ai_sizing_mult_min: float = 0.3
    ai_sizing_mult_max: float = 2.5
    ai_can_kill_strategy: bool = True
    ai_can_pause_bot: bool = True
    ai_max_pause_minutes: int = 120


@dataclass
class RiskConfig:
    # Circuit breakers HARDCODED (Claude ne peut pas les bypasser)
    max_daily_loss_pct: float = 0.05
    max_weekly_drawdown_pct: float = 0.08
    max_consecutive_losses: int = 5

    max_portfolio_correlation: float = 0.75

    max_slippage_bps: float = 20


@dataclass
class BotConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    mode: str = "paper"
    log_level: str = "INFO"


CONFIG = BotConfig()
