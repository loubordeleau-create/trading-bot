"""
Event Scanner. Claude scrute le contexte news/social/macro et flag les risques événementiels.
Idéalement combiné avec feeds externes (CryptoPanic, Twitter API) qu'on passe en contexte.
"""
import logging
from dataclasses import dataclass
from typing import Optional, List


logger = logging.getLogger(__name__)


@dataclass
class EventAssessment:
    risk_level: int                # 0-10
    events: List[str]              # Liste d'événements identifiés
    recommend_pause: bool
    time_horizon_hours: float      # Dans combien d'heures ces events jouent
    reasoning: str


SYSTEM_PROMPT = """Tu es un event risk analyst pour un bot de trading crypto.

Ton job : lire le contexte news/social/macro fourni et identifier les événements
susceptibles d'impacter violemment les prix crypto dans les 24-48h.

Tu dois TOUJOURS répondre en JSON valide :

{
  "risk_level": 0-10,
  "events": ["description courte d'event 1", "description courte event 2"],
  "recommend_pause": true|false,
  "time_horizon_hours": float,
  "reasoning": "2-3 phrases"
}

Événements majeurs à flagger :
- FOMC, CPI US, NFP, discours Fed
- Décisions SEC (ETF, enforcement)
- Hacks/exploits majeurs
- Liquidations en cascade
- Hard forks, mises à jour majeures
- Annonces régulatoires (US, UE, Chine)
- Géopolitique à impact crypto
- Expiration d'options massives
- Unlocks de tokens > $50M

risk_level :
- 0-3 : calme, environnement normal
- 4-6 : vigilance, event modéré dans le pipeline
- 7-9 : risque élevé, réduire exposition
- 10 : danger imminent, pause conseillée

recommend_pause = true seulement si un event MAJEUR est dans les 6h et pourrait
provoquer un mouvement > 5% de BTC."""


class EventScanner:
    def __init__(self, claude_client):
        self.claude = claude_client

    def scan(self, context: dict) -> EventAssessment:
        """
        context : dict libre avec les infos qu'on a pu rassembler.
        Exemples de clés : 'news_headlines', 'upcoming_macro', 'twitter_sentiment',
        'funding_rates', 'liquidations_24h', 'btc_change_24h'.
        """
        context_str = self._format_context(context)

        user_prompt = f"""CONTEXTE ACTUEL:
{context_str}

Identifie les risques événementiels et retourne le JSON attendu."""

        response = self.claude.ask_json(SYSTEM_PROMPT, user_prompt, max_tokens=500)

        if response is None:
            return EventAssessment(
                risk_level=5, events=[], recommend_pause=False,
                time_horizon_hours=24,
                reasoning="Fallback: analyse indisponible, risk neutre.",
            )

        try:
            return EventAssessment(
                risk_level=int(response["risk_level"]),
                events=list(response.get("events", [])),
                recommend_pause=bool(response.get("recommend_pause", False)),
                time_horizon_hours=float(response.get("time_horizon_hours", 24)),
                reasoning=response.get("reasoning", ""),
            )
        except (KeyError, ValueError) as e:
            logger.exception(f"EventScanner parse error: {e}")
            return EventAssessment(5, [], False, 24, "Parse error fallback.")

    def _format_context(self, ctx: dict) -> str:
        if not ctx:
            return "(aucune info externe fournie — tu te bases sur ta connaissance générale)"
        parts = []
        for k, v in ctx.items():
            if isinstance(v, list):
                parts.append(f"{k}:")
                for item in v[:10]:  # cap
                    parts.append(f"  - {item}")
            else:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
