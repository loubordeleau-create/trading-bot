"""
Regime Detector - NEXT LEVEL.
Claude ne classifie plus seulement: il DECIDE quelles strategies activer,
peut pauser le bot, et module le sizing sur un range elargi.

Output enrichi (backward compatible avec ancien code):
  - enabled_strategies : list[str] - lesquelles activer maintenant
  - pause_minutes : int - combien de minutes pauser le bot (0 = pas de pause)
  - conviction_tier : 'high' | 'medium' | 'low'
"""
import json
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from core.indicators import atr, adx, rsi, ema

logger = logging.getLogger(__name__)


# Liste officielle des strategies (doit matcher les .name des classes Strategy)
# Philosophy LOCKED: pure trend-following.
# Les anciennes strategies (timing reversal) sont conservees dans le code main.py mais
# ne sont plus listees ici - donc Claude NE PEUT PAS les activer (whitelist strict).
ALL_STRATEGIES = [
    "trend_surfer",        # SEULE strategie autorisee - pure trend following with trailing stop
]

# Fallback mapping: si Claude rate son output, on fallback vers trend_surfer si trending
DEFAULT_STRATEGIES_BY_REGIME = {
    "trending":     ["trend_surfer"],
    "ranging":      [],
    "accumulation": [],
    "chaotic":      [],
}


@dataclass
class RegimeAssessment:
    regime: str
    regime_score: float
    confidence_mult: float
    kill_switch: bool
    kill_reason: Optional[str]
    reasoning: str
    raw: dict
    # NEXT LEVEL fields (defaults pour backward compat)
    enabled_strategies: List[str] = field(default_factory=list)
    pause_minutes: int = 0
    conviction_tier: str = "medium"


SYSTEM_PROMPT_NEXT_LEVEL = """Tu es un portfolio manager senior qui pilote un bot de trading crypto.

PHILOSOPHIE: PURE TREND-FOLLOWING

Le bot trade UNE seule facon: il surfe les trends confirmes avec trailing stop.
Pas de timing de reversal. Pas de mean reversion. Pas de grid. Pas de scalping.

La seule strategie active est trend_surfer qui:
1. Attend un trend confirme (ADX > 25 sur 4h)
2. Entre sur breakout du high/low recent avec volume
3. Trail le stop progressivement (breakeven -> +0.5 ATR -> trail du high)
4. Laisse courir jusqu'a ce que le trend casse

TON JOB:
- Identifier si on est en vrai trending (go) ou non (wait)
- Donner une conviction (multiplier sizing)
- Activer trend_surfer si conditions ok, ne pas l'activer sinon

HIERARCHIE DES TIMEFRAMES:
- 4h = LA DECISION (le seul qui compte vraiment)
- 1h = CONFIRMATION (conviction)
- 15m = IGNORE (on timing pas les reversals)

REGIMES:

TRENDING (activer trend_surfer):
- 4h ADX > 22
- 4h EMA20 > EMA50 (bullish) OU EMA20 < EMA50 (bearish)
- 4h price dans le bon cote de EMA50
- enabled_strategies: ["trend_surfer"]
- conviction_mult:
  * 4h ADX 22-25: 0.7 (trend modere, prudence)
  * 4h ADX 25-30: 1.0
  * 4h ADX 30-40: 1.3
  * 4h ADX 40+: 1.6
  * Bonus +0.2 si 1h confirme (aligned)
  * Max 2.0

RANGING / ACCUMULATION / CHAOTIC (ne rien activer):
- enabled_strategies: []
- conviction_mult: 0.3 (ne sera pas utilise car rien ne trade)
- PAS de pause_minutes (pas besoin, rien ne trade)

CAS EXTREMES (pause):
Pause UNIQUEMENT si:
- Event macro imminent (FOMC dans <30min)
- ATR% > 10% sur 4h (flash crash)
- News majeure (hack, SEC action en cours)

Ne PAS pauser pour:
- "Divergence multi-TF" (normal)
- "Vol faible" (normal en trending mature)
- "RSI extreme" (on s'en fout, on suit le trend)
- "Configuration mixte" (on trade pas, on attend simplement)

Kill-switch: UNIQUEMENT danger systemique (hack majeur, fork, depegging stable).

REPONSE JSON STRICT:

{
  "regime": "trending|ranging|chaotic|accumulation",
  "regime_score": 0-10,
  "confidence_mult": 0.3-2.0,
  "enabled_strategies": ["trend_surfer"] ou [],
  "pause_minutes": 0,
  "conviction_tier": "high|medium|low",
  "kill_switch": false,
  "kill_reason": null,
  "reasoning": "2-3 phrases focalisees sur le 4h. Trending ou pas? Action: trend_surfer ou wait."
}

REGLE D'OR: Preferer NE RIEN ACTIVER que d'activer dans du non-trending.
Un bon trend-follower fait 60-70% du temps en cash. C'est normal."""


class RegimeDetector:
    def __init__(self, claude_client):
        self.claude = claude_client
        self._last_assessment: Optional[RegimeAssessment] = None

    def assess(self, symbol: str, ohlcv_map: dict,
               external_context: Optional[dict] = None) -> RegimeAssessment:
        tech_snapshot = self._build_tech_snapshot(symbol, ohlcv_map)
        context_str = self._format_context(external_context or {})

        user_prompt = f"""Symbol: {symbol}

SNAPSHOT TECHNIQUE:
{tech_snapshot}

CONTEXTE EXTERNE:
{context_str}

STRATEGIES DISPONIBLES: {", ".join(ALL_STRATEGIES)}

Analyse le contexte et retourne le JSON avec regime, enabled_strategies, confidence_mult, pause_minutes, reasoning."""

        response = self.claude.ask_json(SYSTEM_PROMPT_NEXT_LEVEL, user_prompt, max_tokens=800)

        if response is None:
            logger.warning("Regime detector: reponse Claude invalide, fallback neutre.")
            return self._fallback_assessment()

        try:
            regime = response["regime"]
            regime_score = float(response["regime_score"])

            # confidence_mult: range elargi 0.3 - 2.5
            conf_mult = max(0.3, min(2.5, float(response["confidence_mult"])))

            # enabled_strategies: valide que ce sont des strategies connues
            raw_enabled = response.get("enabled_strategies", [])
            if not isinstance(raw_enabled, list):
                raw_enabled = []
            enabled = [s for s in raw_enabled if s in ALL_STRATEGIES]

            # Si Claude n'a rien renvoye ou que des trucs invalides, fallback par regime
            if not enabled and regime in DEFAULT_STRATEGIES_BY_REGIME:
                enabled = DEFAULT_STRATEGIES_BY_REGIME[regime]
                logger.info(f"enabled_strategies vide/invalide, fallback par regime '{regime}': {enabled}")

            # pause_minutes: cap a 120 pour securite
            pause_min = int(response.get("pause_minutes", 0) or 0)
            pause_min = max(0, min(120, pause_min))

            conviction_tier = response.get("conviction_tier", "medium")
            if conviction_tier not in ("high", "medium", "low"):
                conviction_tier = "medium"

            assessment = RegimeAssessment(
                regime=regime,
                regime_score=regime_score,
                confidence_mult=conf_mult,
                kill_switch=bool(response.get("kill_switch", False)),
                kill_reason=response.get("kill_reason"),
                reasoning=response.get("reasoning", ""),
                raw=response,
                enabled_strategies=enabled,
                pause_minutes=pause_min,
                conviction_tier=conviction_tier,
            )
            self._last_assessment = assessment
            return assessment
        except (KeyError, ValueError, TypeError) as e:
            logger.exception(f"Regime assessment parse error: {e}")
            return self._fallback_assessment()

    def _build_tech_snapshot(self, symbol: str, ohlcv_map: dict) -> str:
        lines = []
        for tf, data in ohlcv_map.items():
            if len(data.close) < 50:
                continue
            close = data.close
            price = close[-1]
            price_24_ago = close[-24] if len(close) >= 24 else close[0]
            change_pct = (price - price_24_ago) / price_24_ago * 100

            atr_vals = atr(data.high, data.low, close, 14)
            adx_vals = adx(data.high, data.low, close, 14)
            rsi_vals = rsi(close, 14)
            ema20 = ema(close, 20)
            ema50 = ema(close, 50)

            curr_atr = atr_vals[-1]
            atr_pct = curr_atr / price * 100
            curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0
            curr_rsi = rsi_vals[-1]

            vol_last_5 = data.volume[-5:].mean()
            vol_prev_20 = data.volume[-25:-5].mean() if len(data.volume) >= 25 else vol_last_5
            vol_ratio = vol_last_5 / vol_prev_20 if vol_prev_20 > 0 else 1.0

            trend_pos = "above" if price > ema50[-1] else "below"
            ema_cross = "bullish" if ema20[-1] > ema50[-1] else "bearish"

            lines.append(
                f"  [{tf}] price={price:.2f} (24p change {change_pct:+.2f}%), "
                f"ATR%={atr_pct:.2f}, ADX={curr_adx:.1f}, RSI={curr_rsi:.1f}, "
                f"EMA20/50={ema_cross}, price {trend_pos} EMA50, "
                f"vol_ratio={vol_ratio:.2f}x"
            )
        return "\n".join(lines)

    def _format_context(self, ctx: dict) -> str:
        if not ctx:
            return "  (aucun contexte externe fourni)"
        return "\n".join(f"  {k}: {v}" for k, v in ctx.items())

    def _fallback_assessment(self) -> RegimeAssessment:
        """Fallback safe: ranging, sizing reduit, toutes strategies OK pour pas tout bloquer."""
        return RegimeAssessment(
            regime="ranging",
            regime_score=5.0,
            confidence_mult=0.5,
            kill_switch=False,
            kill_reason=None,
            reasoning="Fallback: Claude indisponible, regime presume neutre, sizing reduit.",
            raw={},
            enabled_strategies=ALL_STRATEGIES.copy(),
            pause_minutes=0,
            conviction_tier="low",
        )
