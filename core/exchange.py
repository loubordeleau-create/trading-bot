"""
Wrapper exchange unifié. ccxt sous le capot, mais interface propre pour le bot.
"""
import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

try:
    import ccxt
except ImportError:
    ccxt = None  # Permet au reste du code de se charger sans ccxt installé

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_qty: Optional[float] = None
    fee: Optional[float] = None
    error: Optional[str] = None


class Exchange:
    """
    Wrapper fin autour de ccxt. Unifie l'interface entre Bybit/Binance/Kraken.
    Gère aussi le mode paper (simulation).
    """

    def __init__(self, config, mode: str = "paper"):
        self.config = config
        self.mode = mode
        self._paper_balance = config.trading.capital_usdt if mode == "paper" else 0
        self._paper_positions: Dict[str, dict] = {}

        if mode != "paper":
            if ccxt is None:
                raise RuntimeError("ccxt non installé. pip install ccxt")
            exchange_class = getattr(ccxt, config.exchange.name)
            self.client = exchange_class({
                "apiKey": config.exchange.api_key,
                "secret": config.exchange.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},  # perps
            })
            if config.exchange.testnet:
                self.client.set_sandbox_mode(True)
        else:
            self.client = None
            logger.info("Exchange en mode PAPER — aucun ordre réel.")

    # ---------- Market data ----------

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> List[List[float]]:
        """Retourne [[timestamp, open, high, low, close, volume], ...]"""
        if self.mode == "paper" and self.client is None:
            # En paper pur, on a quand même besoin de vraies données de marché
            # Fallback: on initialise un client public read-only
            if ccxt is None:
                raise RuntimeError("ccxt requis même en paper pour les données de marché")
            public_client = getattr(ccxt, self.config.exchange.name)({"enableRateLimit": True})
            return public_client.fetch_ohlcv(symbol, timeframe, limit=limit)
        return self.client.fetch_ohlcv(symbol, timeframe, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        if self.mode == "paper" and self.client is None:
            public_client = getattr(ccxt, self.config.exchange.name)({"enableRateLimit": True})
            return public_client.fetch_ticker(symbol)
        return self.client.fetch_ticker(symbol)

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> dict:
        if self.mode == "paper" and self.client is None:
            public_client = getattr(ccxt, self.config.exchange.name)({"enableRateLimit": True})
            return public_client.fetch_order_book(symbol, limit=limit)
        return self.client.fetch_order_book(symbol, limit=limit)

    # ---------- Trading ----------

    def place_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        """side: 'buy' ou 'sell'"""
        if self.mode == "paper":
            return self._paper_market_order(symbol, side, qty)
        try:
            order = self.client.create_market_order(symbol, side, qty)
            return OrderResult(
                success=True,
                order_id=order.get("id"),
                filled_price=order.get("average") or order.get("price"),
                filled_qty=order.get("filled"),
                fee=order.get("fee", {}).get("cost"),
            )
        except Exception as e:
            logger.exception(f"Market order error: {e}")
            return OrderResult(success=False, error=str(e))

    def place_limit_order(self, symbol: str, side: str, qty: float, price: float,
                          post_only: bool = True) -> OrderResult:
        """post_only=True = maker only, rejeté si taker."""
        if self.mode == "paper":
            return self._paper_limit_order(symbol, side, qty, price)
        try:
            params = {"postOnly": post_only} if post_only else {}
            order = self.client.create_limit_order(symbol, side, qty, price, params)
            return OrderResult(success=True, order_id=order.get("id"))
        except Exception as e:
            logger.exception(f"Limit order error: {e}")
            return OrderResult(success=False, error=str(e))

    def cancel_order(self, symbol: str, order_id: str) -> bool:
        if self.mode == "paper":
            return True
        try:
            self.client.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.exception(f"Cancel error: {e}")
            return False

    def get_balance(self) -> float:
        if self.mode == "paper":
            return self._paper_balance
        bal = self.client.fetch_balance()
        return bal.get("USDT", {}).get("free", 0)

    def get_positions(self) -> Dict[str, dict]:
        if self.mode == "paper":
            return dict(self._paper_positions)
        positions = self.client.fetch_positions()
        return {p["symbol"]: p for p in positions if p.get("contracts", 0) != 0}

    # ---------- Paper trading simulation ----------

    def _paper_market_order(self, symbol: str, side: str, qty: float) -> OrderResult:
        ticker = self.fetch_ticker(symbol)
        # Simule un petit slippage
        price = ticker["ask"] if side == "buy" else ticker["bid"]
        slippage = 0.0005  # 5bps
        fill_price = price * (1 + slippage if side == "buy" else 1 - slippage)
        cost = fill_price * qty
        fee = cost * 0.0006  # 6bps taker

        if side == "buy":
            if self._paper_balance < cost + fee:
                return OrderResult(success=False, error="Insufficient paper balance")
            self._paper_balance -= (cost + fee)
            pos = self._paper_positions.get(symbol, {"qty": 0, "avg_price": 0})
            new_qty = pos["qty"] + qty
            pos["avg_price"] = (pos["avg_price"] * pos["qty"] + fill_price * qty) / new_qty if new_qty else 0
            pos["qty"] = new_qty
            self._paper_positions[symbol] = pos
        else:
            pos = self._paper_positions.get(symbol, {"qty": 0, "avg_price": 0})
            self._paper_balance += (cost - fee)
            pos["qty"] -= qty
            if abs(pos["qty"]) < 1e-9:
                self._paper_positions.pop(symbol, None)
            else:
                self._paper_positions[symbol] = pos

        return OrderResult(
            success=True,
            order_id=f"paper_{int(time.time()*1000)}",
            filled_price=fill_price,
            filled_qty=qty,
            fee=fee,
        )

    def _paper_limit_order(self, symbol: str, side: str, qty: float, price: float) -> OrderResult:
        # Simplification : en paper on simule un fill immédiat si le prix est touchable
        ticker = self.fetch_ticker(symbol)
        if side == "buy" and price >= ticker["ask"]:
            return self._paper_market_order(symbol, side, qty)
        if side == "sell" and price <= ticker["bid"]:
            return self._paper_market_order(symbol, side, qty)
        return OrderResult(
            success=True,
            order_id=f"paper_limit_{int(time.time()*1000)}",
            filled_price=None,  # Pending
            filled_qty=0,
        )
