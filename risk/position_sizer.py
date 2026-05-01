"""
Position Sizer. Calcul de la taille optimale d'une position.
Utilise Kelly fractionnaire ajusté par conviction IA et volatilité.
"""
import math
from dataclasses import dataclass


@dataclass
class SizingResult:
    qty: float
    position_value_usdt: float
    capital_used_pct: float
    kelly_raw: float
    kelly_adjusted: float
    rationale: str


class PositionSizer:
    def __init__(self, config):
        self.config = config
        self.trading = config.trading

    def size(self, capital_usdt: float, entry_price: float, stop_price: float,
             win_rate: float, avg_win_loss_ratio: float,
             signal_strength: float, ai_confidence_mult: float,
             current_exposure_usdt: float) -> SizingResult:
        """
        Calcule la taille optimale d'une position.

        Kelly formula: f* = W - (1-W)/R
            W = win rate, R = avg_win / avg_loss

        On applique:
        - fraction kelly (0.25 typiquement) pour réduire la variance
        - multiplicateur signal strength [0-1]
        - multiplicateur IA confidence
        - caps durs (max_position_pct, max_total_exposure)
        """
        # Sanity
        if entry_price <= 0 or stop_price <= 0 or entry_price == stop_price:
            return SizingResult(0, 0, 0, 0, 0, "Prix invalides")

        # Kelly brut
        # Prior: quand on a peu d'historique, on assume un léger edge positif
        # (sinon Kelly = 0 au démarrage -> le bot ne trade jamais = chicken-and-egg)
        W = max(0.1, min(0.9, win_rate))
        R = max(0.3, avg_win_loss_ratio)

        # Si on est encore avec des priors naïfs (0.5, 1.0), utiliser prior optimiste modéré
        if abs(W - 0.5) < 0.01 and abs(R - 1.0) < 0.01:
            W = 0.52
            R = 1.2

        kelly_raw = W - (1 - W) / R

        # Floor Kelly: même si edge négatif, on garde un minimum pour continuer à trader
        # mais à taille fortement réduite. Ça évite que le bot s'auto-bloque totalement
        # après une mauvaise série. Le signal_strength et les autres filtres (circuit
        # breaker, correlation) restent actifs comme safety nets.
        MIN_KELLY = 0.02  # 2% = très petit mais non-zéro
        kelly_raw = max(MIN_KELLY, kelly_raw)

        # Kelly ajusté
        kelly_adj = (
            kelly_raw
            * self.trading.kelly_fraction
            * signal_strength
            * ai_confidence_mult
        )

        # Risque par trade (fraction du capital perdue si stop touché)
        risk_per_unit = abs(entry_price - stop_price) / entry_price

        if risk_per_unit < 0.001:
            return SizingResult(0, 0, 0, kelly_raw, kelly_adj,
                               "Stop trop proche du prix (<0.1%)")

        # Fraction du capital à risquer sur ce trade
        capital_to_risk = kelly_adj * capital_usdt

        # Position sizing : value_at_risk = position_value * risk_per_unit
        # Donc position_value = capital_to_risk / risk_per_unit
        position_value = capital_to_risk / risk_per_unit

        # Caps durs
        max_by_pos = capital_usdt * self.trading.max_position_pct
        max_by_total = capital_usdt * self.trading.max_total_exposure - current_exposure_usdt
        max_by_total = max(0, max_by_total)

        position_value = min(position_value, max_by_pos, max_by_total)

        if position_value <= 10:  # minimum $10 de position
            return SizingResult(0, 0, 0, kelly_raw, kelly_adj,
                               f"Position trop petite (${position_value:.2f})")

        qty = position_value / entry_price
        # Arrondi raisonnable
        qty = math.floor(qty * 1e6) / 1e6

        rationale = (
            f"Kelly brut {kelly_raw:.3f} (W={W:.2f}, R={R:.2f}) | "
            f"ajusté {kelly_adj:.4f} (frac={self.trading.kelly_fraction}, "
            f"sig={signal_strength:.2f}, ai={ai_confidence_mult:.2f}) | "
            f"risk={risk_per_unit*100:.2f}% | "
            f"size=${position_value:.2f} ({position_value/capital_usdt*100:.1f}% capital)"
        )

        return SizingResult(
            qty=qty,
            position_value_usdt=position_value,
            capital_used_pct=position_value / capital_usdt,
            kelly_raw=kelly_raw,
            kelly_adjusted=kelly_adj,
            rationale=rationale,
        )