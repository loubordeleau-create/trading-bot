"""
Client Claude API. Wrapper fin avec retry, JSON parsing, cost tracking.
"""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)


@dataclass
class AICost:
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls_today: int = 0
    cost_today: float = 0.0
    _day_start: float = field(default_factory=time.time)


class ClaudeClient:
    # Pricing Opus 4.7 (approximatif, per 1M tokens)
    # À ajuster selon le modèle effectivement utilisé
    PRICING = {
        "claude-opus-4-7":    {"input": 15.0, "output": 75.0},
        "claude-opus-4-6":    {"input": 15.0, "output": 75.0},
        "claude-sonnet-4-6":  {"input": 3.0,  "output": 15.0},
        "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    }

    def __init__(self, config):
        self.config = config
        self.ai_config = config.ai

        if Anthropic is None:
            raise RuntimeError("Package anthropic non installé. pip install anthropic")

        self.client = Anthropic(api_key=config.ai.anthropic_api_key)
        self.cost = AICost()

    def ask_json(self, system_prompt: str, user_prompt: str,
                 max_tokens: int = 1024, retries: int = 2) -> Optional[dict]:
        """
        Appel LLM attendant du JSON en retour.
        Retourne le dict parsé, ou None si échec.
        """
        # Check daily cost cap
        self._reset_daily_if_needed()
        if self.cost.cost_today >= self.ai_config.max_daily_api_cost_usd:
            logger.warning("Daily AI cost cap atteint, skip appel.")
            return None

        for attempt in range(retries + 1):
            try:
                msg = self.client.messages.create(
                    model=self.ai_config.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = msg.content[0].text if msg.content else ""
                self._track_cost(msg)

                # Extraire JSON (résistant aux markdown fences)
                cleaned = text.strip()
                if "```json" in cleaned:
                    cleaned = cleaned.split("```json")[1].split("```")[0]
                elif "```" in cleaned:
                    cleaned = cleaned.split("```")[1].split("```")[0]

                return json.loads(cleaned.strip())

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse failed (attempt {attempt+1}): {e}. Raw: {text[:200]}")
                if attempt == retries:
                    return None
            except Exception as e:
                logger.exception(f"Claude API error (attempt {attempt+1}): {e}")
                if attempt == retries:
                    return None
                time.sleep(1 + attempt)

        return None

    def _track_cost(self, msg):
        usage = msg.usage
        self.cost.total_calls += 1
        self.cost.calls_today += 1
        self.cost.total_input_tokens += usage.input_tokens
        self.cost.total_output_tokens += usage.output_tokens

        pricing = self.PRICING.get(self.ai_config.model, {"input": 3.0, "output": 15.0})
        call_cost = (
            usage.input_tokens / 1_000_000 * pricing["input"] +
            usage.output_tokens / 1_000_000 * pricing["output"]
        )
        self.cost.total_cost_usd += call_cost
        self.cost.cost_today += call_cost

    def _reset_daily_if_needed(self):
        if time.time() - self.cost._day_start > 86400:
            self.cost.calls_today = 0
            self.cost.cost_today = 0
            self.cost._day_start = time.time()
