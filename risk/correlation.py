"""
Portfolio correlation. Refuse les trades redondants : si BTC/ETH/SOL sont
corrélés à 0.95, ouvrir 3 positions = 1 position avec 3x le risque.
"""
from typing import Dict, Tuple


class CorrelationChecker:
    def __init__(self, config):
        self.max_corr = config.risk.max_portfolio_correlation

    def allow_new_position(self, candidate_symbol: str, candidate_side: str,
                          open_positions: dict,
                          correlations: Dict[Tuple[str, str], float]) -> tuple[bool, str]:
        """
        Retourne (allow, reason).
        open_positions: symbol -> Position (side 'long'/'short')
        correlations: {(sym1, sym2): corr} pour tous les couples.
        """
        if not open_positions:
            return True, "Aucune position ouverte, corrélation non-contrainte."

        for sym, pos in open_positions.items():
            if sym == candidate_symbol:
                return False, f"Position déjà ouverte sur {sym}."

            # Récupérer la corrélation (key peut être dans un sens ou l'autre)
            corr = correlations.get((sym, candidate_symbol),
                    correlations.get((candidate_symbol, sym), 0.0))

            # Si positions dans le MÊME sens et corrélées positivement: cumul de risque
            # Si positions dans des sens OPPOSÉS et corrélées positivement: auto-hedge
            # Donc on vérifie le "directional exposure"
            same_direction = (pos.side == ("long" if candidate_side == "buy" else "short"))

            if same_direction and corr > self.max_corr:
                return False, (
                    f"Corrélation {corr:.2f} avec {sym} (même direction) "
                    f"> seuil {self.max_corr:.2f}. Risque cumulatif trop élevé."
                )
            if not same_direction and corr < -self.max_corr:
                return False, (
                    f"Corrélation {corr:.2f} avec {sym} (sens opposés mais corrélés négativement) "
                    f"signifie positions redondantes."
                )

        return True, "OK corrélation."
