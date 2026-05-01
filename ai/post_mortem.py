"""
Post-Mortem Analyzer. Une fois par semaine, Claude analyse les trades récents
et propose des ajustements de paramètres. L'humain valide.
"""
import logging
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PostMortemReport:
    summary: str
    win_rate_by_strategy: Dict[str, float]
    win_rate_by_regime: Dict[str, float]
    observations: List[str]
    suggested_adjustments: List[dict]    # Liste de {param, current, suggested, reason}
    raw: dict


SYSTEM_PROMPT = """Tu es un trader/quant senior qui fait un post-mortem hebdomadaire
sur un bot de trading crypto multi-stratégies.

Tu reçois les statistiques de performance par stratégie et par régime, ainsi qu'un
échantillon de trades. Ton job :

1. Identifier ce qui a marché et ce qui a mal marché
2. Détecter les patterns (ex: "mean reversion perd systématiquement en régime chaotic")
3. Proposer des ajustements CONCRETS de paramètres, justifiés

Tu réponds TOUJOURS en JSON valide, rien d'autre :

{
  "summary": "2-3 phrases résumant la semaine",
  "observations": [
    "observation 1 concrète",
    "observation 2 concrète"
  ],
  "suggested_adjustments": [
    {
      "param": "mean_reversion.rsi_oversold",
      "current": 28,
      "suggested": 25,
      "reason": "Les entrées à RSI 28 se font stop-out 60% du temps en régime chaotic"
    }
  ],
  "confidence": 0-10
}

Règles :
- Sois honnête. Si une stratégie performe mal, dis-le.
- Ne propose PAS d'ajustements si l'échantillon est trop petit (<20 trades par stratégie).
- Sois conservateur sur les ajustements (max ±20% des valeurs courantes).
- Priorise la réduction des pertes avant l'augmentation des gains."""


class PostMortemAnalyzer:
    def __init__(self, claude_client):
        self.claude = claude_client

    def analyze(self, state, lookback_days: int = 7) -> PostMortemReport:
        import time
        cutoff = time.time() - lookback_days * 86400
        recent = [t for t in state.closed_trades if t.exit_time >= cutoff]

        if len(recent) < 5:
            return PostMortemReport(
                summary=f"Échantillon insuffisant ({len(recent)} trades), analyse skippée.",
                win_rate_by_strategy={}, win_rate_by_regime={},
                observations=[], suggested_adjustments=[], raw={},
            )

        # Stats par stratégie
        by_strategy = defaultdict(list)
        for t in recent:
            by_strategy[t.strategy].append(t)

        wr_strat = {
            s: sum(1 for t in trades if t.pnl_usdt > 0) / len(trades)
            for s, trades in by_strategy.items()
        }

        # Stats par régime (regime stocké dans ai_context)
        by_regime = defaultdict(list)
        for t in recent:
            regime = t.ai_context.get("regime", "unknown")
            by_regime[regime].append(t)

        wr_regime = {
            r: sum(1 for t in trades if t.pnl_usdt > 0) / len(trades)
            for r, trades in by_regime.items()
        }

        # Build prompt
        trades_summary = self._summarize_trades(recent)
        strat_breakdown = "\n".join(
            f"  {s}: {len(trades)} trades, wr={wr_strat[s]:.1%}, "
            f"total_pnl={sum(t.pnl_usdt for t in trades):.2f} USDT"
            for s, trades in by_strategy.items()
        )
        regime_breakdown = "\n".join(
            f"  {r}: {len(trades)} trades, wr={wr_regime[r]:.1%}, "
            f"total_pnl={sum(t.pnl_usdt for t in trades):.2f} USDT"
            for r, trades in by_regime.items()
        )

        user_prompt = f"""Période: {lookback_days} derniers jours
Total trades: {len(recent)}
Win rate global: {sum(1 for t in recent if t.pnl_usdt > 0) / len(recent):.1%}
PnL total: {sum(t.pnl_usdt for t in recent):.2f} USDT
Pertes consécutives actuelles: {state.consecutive_losses}

PAR STRATÉGIE:
{strat_breakdown}

PAR RÉGIME:
{regime_breakdown}

ÉCHANTILLON DE TRADES (10 derniers):
{trades_summary}

Fais le post-mortem et retourne le JSON attendu."""

        response = self.claude.ask_json(SYSTEM_PROMPT, user_prompt, max_tokens=2000)

        if response is None:
            return PostMortemReport(
                summary="Analyse Claude indisponible.",
                win_rate_by_strategy=wr_strat,
                win_rate_by_regime=wr_regime,
                observations=[], suggested_adjustments=[], raw={},
            )

        return PostMortemReport(
            summary=response.get("summary", ""),
            win_rate_by_strategy=wr_strat,
            win_rate_by_regime=wr_regime,
            observations=list(response.get("observations", [])),
            suggested_adjustments=list(response.get("suggested_adjustments", [])),
            raw=response,
        )

    def _summarize_trades(self, trades: list, n: int = 10) -> str:
        recent = trades[-n:]
        lines = []
        for t in recent:
            duration_h = (t.exit_time - t.entry_time) / 3600
            lines.append(
                f"  {t.symbol} {t.side} [{t.strategy}] "
                f"entry={t.entry_price:.2f} exit={t.exit_price:.2f} "
                f"pnl={t.pnl_pct*100:+.2f}% ({t.pnl_usdt:+.2f} USDT) "
                f"duration={duration_h:.1f}h reason={t.exit_reason}"
            )
        return "\n".join(lines)
