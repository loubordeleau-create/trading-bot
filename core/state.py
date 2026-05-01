"""
État persistant du bot. Positions ouvertes, trades fermés, métriques de perf.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional, Dict


@dataclass
class Position:
    symbol: str
    side: str                      # 'long' ou 'short'
    qty: float
    entry_price: float
    entry_time: float
    stop_loss: float
    take_profit: float
    strategy: str
    ai_context: dict = field(default_factory=dict)  # Snapshot du contexte IA à l'entrée
    order_id: Optional[str] = None

    def unrealized_pnl(self, current_price: float) -> float:
        mult = 1 if self.side == "long" else -1
        return (current_price - self.entry_price) * self.qty * mult

    def unrealized_pnl_pct(self, current_price: float) -> float:
        mult = 1 if self.side == "long" else -1
        return (current_price - self.entry_price) / self.entry_price * mult


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: float
    exit_time: float
    pnl_usdt: float
    pnl_pct: float
    fees: float
    strategy: str
    exit_reason: str                # 'tp', 'sl', 'ai_signal', 'manual', 'circuit_breaker'
    ai_context: dict = field(default_factory=dict)


class BotState:
    """Source de vérité pour tout l'état du bot."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.positions: Dict[str, Position] = {}   # symbol -> Position
        self.closed_trades: List[ClosedTrade] = []
        self.regime_score: float = 5.0             # Default neutre
        self.confidence_mult: float = 1.0
        self.kill_switch: bool = False
        self.last_ai_update: float = 0
        self.session_start: float = time.time()
        self.consecutive_losses: int = 0

    # ---------- Sync with exchange ----------

    def sync_with_exchange(self, exchange, default_atr_mult: float = 1.5) -> dict:
        """
        Au démarrage, récupère les positions ouvertes sur l'exchange et reconstruit
        l'état interne. Si l'exchange expose des conditional orders (SL/TP attachés),
        on les récupère pour reconstituer les vrais stops au lieu de placeholders.

        Retourne un dict de diagnostic {added, removed, updated, warnings}.
        """
        import logging
        log = logging.getLogger(__name__)
        diag = {"added": [], "removed": [], "updated": [], "warnings": []}

        # 1. Fetch live positions
        try:
            live_positions = exchange.get_positions()
        except Exception as e:
            log.warning(f"sync: cannot fetch positions: {e}")
            diag["warnings"].append(f"fetch_positions_failed: {e}")
            return diag

        # 2. Fetch open orders (pour extraire les SL/TP conditionnels)
        # Organisé par symbol -> list of orders
        open_orders_by_symbol: Dict[str, List[dict]] = {}
        try:
            if hasattr(exchange, "client") and exchange.client is not None:
                all_orders = exchange.client.fetch_open_orders()
                for o in all_orders:
                    sym = o.get("symbol")
                    if sym:
                        open_orders_by_symbol.setdefault(sym, []).append(o)
        except Exception as e:
            log.warning(f"sync: cannot fetch open orders: {e}")
            diag["warnings"].append(f"fetch_open_orders_failed: {e}")

        # 3. Pour chaque position live, reconstruire ou mettre à jour
        live_symbols = set()
        for symbol, data in live_positions.items():
            live_symbols.add(symbol)
            # Normaliser le format selon exchange
            qty, side, entry_price, leverage, entry_time_ms = self._parse_position_data(data)
            if qty == 0 or entry_price == 0:
                continue

            # Chercher SL/TP dans les open orders de ce symbol
            stop_loss, take_profit = self._extract_sl_tp_from_orders(
                open_orders_by_symbol.get(symbol, []), side, entry_price
            )

            # Fallback si pas trouvé: placeholders larges avec warning
            used_fallback = False
            if stop_loss is None or take_profit is None:
                used_fallback = True
                stop_dist = entry_price * 0.03
                if side == "long":
                    stop_loss = stop_loss or (entry_price - stop_dist)
                    take_profit = take_profit or (entry_price + stop_dist * 2)
                else:
                    stop_loss = stop_loss or (entry_price + stop_dist)
                    take_profit = take_profit or (entry_price - stop_dist * 2)
                diag["warnings"].append(
                    f"{symbol}: SL/TP non trouvés sur exchange, fallback placeholders 3%/6%"
                )

            entry_time = entry_time_ms / 1000.0 if entry_time_ms else time.time()

            if symbol in self.positions:
                # Update d'une position existante (cas: qty changée suite à fill partiel)
                existing = self.positions[symbol]
                if abs(existing.qty - qty) > 1e-9:
                    diag["updated"].append(f"{symbol}: qty {existing.qty} -> {qty}")
                    existing.qty = qty
            else:
                recovered = Position(
                    symbol=symbol, side=side, qty=qty, entry_price=entry_price,
                    entry_time=entry_time,
                    stop_loss=stop_loss, take_profit=take_profit,
                    strategy="recovered",
                    ai_context={
                        "note": "Position découverte au sync" + (
                            " (stops placeholder)" if used_fallback else " (stops récupérés de l'exchange)"
                        ),
                        "leverage": leverage,
                    },
                )
                self.positions[symbol] = recovered
                diag["added"].append(symbol)
                self._log_event("position_recovered", asdict(recovered))

        # 4. Fermer les positions "fantômes" (présentes dans state mais plus sur l'exchange)
        #    Ça arrive si l'exchange a liquidé ou si un stop a été hit pendant qu'on était offline
        for symbol in list(self.positions.keys()):
            if symbol not in live_symbols:
                pos = self.positions.pop(symbol)
                diag["removed"].append(symbol)
                self._log_event("position_ghost_removed", {
                    "symbol": symbol,
                    "note": "Position absente de l'exchange au sync, supposée fermée hors du bot",
                    "last_known": asdict(pos),
                })

        return diag

    def _parse_position_data(self, data: dict):
        """Normalise les différents formats ccxt selon l'exchange."""
        qty = 0.0
        side = "long"
        entry_price = 0.0
        leverage = 1.0
        entry_time_ms = None

        # Format standard ccxt unified
        if "contracts" in data and data.get("contracts") is not None:
            qty = abs(float(data["contracts"]))
            side_raw = data.get("side", "long")
            side = "long" if side_raw in ("long", "buy") else "short"
            entry_price = float(data.get("entryPrice") or data.get("average") or 0)
            leverage = float(data.get("leverage") or 1.0)
            # Timestamp si dispo
            entry_time_ms = data.get("timestamp") or data.get("datetime")
            if isinstance(entry_time_ms, str):
                entry_time_ms = None  # on ignore les strings, on laisse fallback now()
        # Format alternatif (signed qty)
        elif "qty" in data:
            raw_qty = float(data["qty"])
            qty = abs(raw_qty)
            side = "long" if raw_qty > 0 else "short"
            entry_price = float(data.get("avg_price") or data.get("entryPrice") or 0)
            leverage = float(data.get("leverage") or 1.0)

        return qty, side, entry_price, leverage, entry_time_ms

    def _extract_sl_tp_from_orders(self, orders: List[dict], side: str, entry_price: float):
        """
        Parse les conditional orders pour extraire stop-loss et take-profit.
        ccxt normalise dans order['info'] selon l'exchange. On check plusieurs variantes.
        Retourne (stop_loss, take_profit) ou (None, None) si pas trouvé.
        """
        stop_loss = None
        take_profit = None

        for o in orders:
            order_type = (o.get("type") or "").lower()
            stop_price = o.get("stopPrice") or o.get("triggerPrice")
            if not stop_price:
                info = o.get("info", {})
                stop_price = info.get("stopPrice") or info.get("triggerPrice") or info.get("stopLossPrice") or info.get("takeProfitPrice")
            if not stop_price:
                continue
            try:
                stop_price = float(stop_price)
            except (TypeError, ValueError):
                continue

            # Classification: un stop pour long est SL si sous entry, TP si au-dessus
            is_sl = False
            is_tp = False
            if "stop_loss" in order_type or "stoploss" in order_type:
                is_sl = True
            elif "take_profit" in order_type or "takeprofit" in order_type:
                is_tp = True
            else:
                # Heuristique basée sur la position du prix
                if side == "long":
                    if stop_price < entry_price:
                        is_sl = True
                    else:
                        is_tp = True
                else:
                    if stop_price > entry_price:
                        is_sl = True
                    else:
                        is_tp = True

            if is_sl and (stop_loss is None or
                         (side == "long" and stop_price > stop_loss) or
                         (side == "short" and stop_price < stop_loss)):
                # Garde le stop le plus serré (le plus protecteur)
                stop_loss = stop_price
            elif is_tp and take_profit is None:
                take_profit = stop_price

        return stop_loss, take_profit

    def reconcile_with_exchange(self, exchange) -> dict:
        """
        Réconciliation périodique (à appeler toutes les 5-10 min en live).
        Même logique que sync_with_exchange mais avec logs différents pour
        distinguer startup-sync vs periodic-reconcile.
        """
        diag = self.sync_with_exchange(exchange)
        if diag["added"] or diag["removed"] or diag["updated"]:
            import logging
            log = logging.getLogger(__name__)
            log.warning(
                f"RECONCILE: added={diag['added']} removed={diag['removed']} "
                f"updated={diag['updated']} warnings={len(diag['warnings'])}"
            )
        return diag


    # ---------- Position management ----------

    def open_position(self, pos: Position):
        self.positions[pos.symbol] = pos
        self._log_event("position_opened", asdict(pos))

    def close_position(self, symbol: str, exit_price: float, fees: float,
                       exit_reason: str) -> Optional[ClosedTrade]:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        mult = 1 if pos.side == "long" else -1
        pnl = (exit_price - pos.entry_price) * pos.qty * mult - fees
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * mult

        trade = ClosedTrade(
            symbol=pos.symbol, side=pos.side, qty=pos.qty,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time, exit_time=time.time(),
            pnl_usdt=pnl, pnl_pct=pnl_pct, fees=fees,
            strategy=pos.strategy, exit_reason=exit_reason,
            ai_context=pos.ai_context,
        )
        self.closed_trades.append(trade)
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        self._log_event("trade_closed", asdict(trade))
        return trade

    # ---------- Metrics ----------

    def total_exposure_usdt(self, prices: Dict[str, float]) -> float:
        return sum(
            abs(p.qty) * prices.get(s, p.entry_price)
            for s, p in self.positions.items()
        )

    def session_pnl(self) -> float:
        return sum(t.pnl_usdt for t in self.closed_trades)

    def daily_pnl(self) -> float:
        cutoff = time.time() - 86400
        return sum(t.pnl_usdt for t in self.closed_trades if t.exit_time >= cutoff)

    def weekly_pnl(self) -> float:
        cutoff = time.time() - 86400 * 7
        return sum(t.pnl_usdt for t in self.closed_trades if t.exit_time >= cutoff)

    def win_rate(self, lookback: int = 50) -> float:
        recent = self.closed_trades[-lookback:]
        if not recent:
            return 0.5  # Prior neutre
        wins = sum(1 for t in recent if t.pnl_usdt > 0)
        return wins / len(recent)

    def avg_win_loss_ratio(self, lookback: int = 50) -> float:
        recent = self.closed_trades[-lookback:]
        wins = [t.pnl_usdt for t in recent if t.pnl_usdt > 0]
        losses = [-t.pnl_usdt for t in recent if t.pnl_usdt < 0]
        if not wins or not losses:
            return 1.0
        return (sum(wins) / len(wins)) / (sum(losses) / len(losses))

    # ---------- Logging ----------

    def _log_event(self, event_type: str, data: dict):
        path = os.path.join(self.log_dir, "events.jsonl")
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": data,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
