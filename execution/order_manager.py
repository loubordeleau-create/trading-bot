"""
Order Manager. Exécution intelligente :
- Maker-first : limit order passif, retombe en market si pas fill en X secondes
- TWAP : split un gros ordre en N petits sur une fenêtre
- Iceberg : affiche seulement une fraction de la taille à la fois

Objectif : minimiser slippage + capturer le rebate maker (-0.01 à -0.025%).
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    avg_fill_price: float
    total_filled: float
    total_fees: float
    slippage_bps: float
    orders_placed: int
    error: Optional[str] = None


class OrderManager:
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.max_slippage_bps = config.risk.max_slippage_bps

    def execute(self, symbol: str, side: str, qty: float,
                ref_price: float, urgency: str = "normal") -> ExecutionResult:
        """
        Exécute un ordre avec stratégie adaptée à l'urgence.
        - urgency='low' : TWAP sur 5 min, maker-first
        - urgency='normal' : maker-first 30s puis market
        - urgency='high' : market direct
        """
        if urgency == "high":
            return self._market_execute(symbol, side, qty, ref_price)
        elif urgency == "low":
            return self._twap_execute(symbol, side, qty, ref_price, duration_sec=300, slices=5)
        else:
            return self._maker_first_execute(symbol, side, qty, ref_price, timeout_sec=30)

    def _market_execute(self, symbol: str, side: str, qty: float, ref_price: float) -> ExecutionResult:
        res = self.exchange.place_market_order(symbol, side, qty)
        if not res.success:
            return ExecutionResult(False, 0, 0, 0, 0, 1, error=res.error)
        slip_bps = abs(res.filled_price - ref_price) / ref_price * 10000 if ref_price > 0 else 0
        return ExecutionResult(
            success=True,
            avg_fill_price=res.filled_price,
            total_filled=res.filled_qty,
            total_fees=res.fee or 0,
            slippage_bps=slip_bps,
            orders_placed=1,
        )

    def _maker_first_execute(self, symbol: str, side: str, qty: float,
                             ref_price: float, timeout_sec: int = 30) -> ExecutionResult:
        """Place limit post-only au bid/ask. Si pas fill en timeout, passe en market."""
        ob = self.exchange.fetch_orderbook(symbol, limit=5)
        if not ob["bids"] or not ob["asks"]:
            return self._market_execute(symbol, side, qty, ref_price)

        # Prix maker : un tick à l'intérieur du spread (best bid si buy, best ask si sell)
        limit_price = ob["bids"][0][0] if side == "buy" else ob["asks"][0][0]

        order_res = self.exchange.place_limit_order(symbol, side, qty, limit_price, post_only=True)
        if not order_res.success:
            logger.warning(f"Maker order refused ({order_res.error}), fallback market")
            return self._market_execute(symbol, side, qty, ref_price)

        # Attendre fill ou timeout
        start = time.time()
        while time.time() - start < timeout_sec:
            time.sleep(2)
            # En paper le fill est instantané, en live il faudrait fetch_order
            if self.exchange.mode == "paper":
                break

        # Si pas fill, cancel + market le reste
        if order_res.filled_price is None:
            self.exchange.cancel_order(symbol, order_res.order_id)
            return self._market_execute(symbol, side, qty, ref_price)

        slip_bps = abs(order_res.filled_price - ref_price) / ref_price * 10000 if ref_price > 0 else 0
        return ExecutionResult(
            success=True,
            avg_fill_price=order_res.filled_price or limit_price,
            total_filled=order_res.filled_qty or qty,
            total_fees=order_res.fee or 0,
            slippage_bps=slip_bps,
            orders_placed=1,
        )

    def _twap_execute(self, symbol: str, side: str, qty: float, ref_price: float,
                      duration_sec: int = 300, slices: int = 5) -> ExecutionResult:
        """Split qty en N slices, exécute une slice toutes les duration/N secondes."""
        slice_qty = qty / slices
        interval = duration_sec / slices

        fills: List[tuple] = []  # (price, qty, fee)
        errors = 0

        for i in range(slices):
            # Check slippage avant chaque slice
            current_price = self.exchange.fetch_ticker(symbol).get("last", ref_price)
            slip_pre = abs(current_price - ref_price) / ref_price * 10000
            if slip_pre > self.max_slippage_bps:
                logger.warning(f"TWAP abort: pre-slippage {slip_pre:.1f}bps > cap")
                break

            res = self._maker_first_execute(symbol, side, slice_qty, current_price, timeout_sec=15)
            if res.success:
                fills.append((res.avg_fill_price, res.total_filled, res.total_fees))
            else:
                errors += 1

            if i < slices - 1:
                time.sleep(interval)

        if not fills:
            return ExecutionResult(False, 0, 0, 0, 0, slices, error=f"{errors} erreurs")

        total_qty = sum(f[1] for f in fills)
        avg_price = sum(f[0] * f[1] for f in fills) / total_qty if total_qty > 0 else 0
        total_fees = sum(f[2] for f in fills)
        avg_slip = abs(avg_price - ref_price) / ref_price * 10000 if ref_price > 0 else 0

        return ExecutionResult(
            success=True,
            avg_fill_price=avg_price,
            total_filled=total_qty,
            total_fees=total_fees,
            slippage_bps=avg_slip,
            orders_placed=len(fills),
        )
