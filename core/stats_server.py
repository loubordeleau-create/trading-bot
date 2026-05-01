"""
Serveur HTTP minimal pour exposer les stats du bot en temps réel.

Utilise le module http.server de Python (stdlib, zéro dépendance).
Le serveur tourne sur un thread séparé pour ne pas bloquer le bot.

Endpoints:
    GET /          → HTML simple avec lien vers /stats
    GET /stats     → JSON avec toutes les stats
    GET /health    → "ok" (pour healthcheck Railway)

Security:
    - Lecture seule: aucun endpoint ne modifie l'état du bot
    - Token optionnel via variable d'env DASHBOARD_TOKEN:
        si défini, les requêtes doivent inclure ?token=XXX
    - Aucun secret exposé (pas de clés API, pas de raw logs)
"""
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class StatsServer:
    """
    Serveur HTTP qui expose les stats du bot.

    Usage:
        server = StatsServer(port=8080, stats_provider=lambda: {...})
        server.start()  # lance sur thread séparé
    """

    def __init__(self, port: int, stats_provider: Callable[[], dict],
                 token: str = ""):
        self.port = port
        self.stats_provider = stats_provider
        self.token = token
        self._server = None
        self._thread = None

    def start(self):
        """Lance le serveur sur un thread background."""
        handler_class = self._make_handler_class()
        self._server = ThreadingHTTPServer(("0.0.0.0", self.port), handler_class)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Stats server démarré sur port {self.port}")

    def stop(self):
        if self._server:
            self._server.shutdown()
            logger.info("Stats server arrêté")

    def _make_handler_class(self):
        provider = self.stats_provider
        token_required = self.token

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                # Silence les logs HTTP par défaut (sinon ça spam)
                pass

            def _send_json(self, data: dict, status: int = 200):
                body = json.dumps(data, default=str).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_text(self, text: str, status: int = 200, content_type: str = "text/plain"):
                body = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _check_auth(self, query: dict) -> bool:
                if not token_required:
                    return True  # pas de token requis
                provided = query.get("token", [""])[0]
                return provided == token_required

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                query = parse_qs(parsed.query)

                if path == "/health":
                    self._send_text("ok")
                    return

                if path == "/":
                    html = """<!DOCTYPE html>
<html><head><title>Trading Bot</title></head>
<body style="font-family: sans-serif; padding: 2em;">
<h1>🤖 Trading Bot — Stats Endpoint</h1>
<p>JSON stats: <a href="/stats">/stats</a></p>
<p>Health check: <a href="/health">/health</a></p>
</body></html>"""
                    self._send_text(html, content_type="text/html")
                    return

                if path == "/stats":
                    if not self._check_auth(query):
                        self._send_json({"error": "Invalid token"}, status=401)
                        return

                    try:
                        stats = provider()
                        self._send_json(stats)
                    except Exception as e:
                        logger.error(f"Stats provider error: {e}")
                        self._send_json({"error": str(e)}, status=500)
                    return

                # 404
                self._send_json({"error": "Not found"}, status=404)

        return Handler


def build_stats_payload(state, config, start_ts: float,
                        ai_insights: list = None,
                        ai_state: dict = None) -> dict:
    """
    Construit le payload de stats à partir de l'état du bot.
    Ne retourne AUCUN secret (pas de clés API, pas de chemins serveur).

    ai_insights: liste des derniers commentaires Claude
                 [{ts, symbol, regime, regime_score, confidence_mult, reasoning, kill_switch}]
    ai_state: état adaptatif courant de l'IA
              {pause_until, pause_reason, enabled_strategies, regime, sizing_mult, now}
    """
    now = time.time()
    uptime_sec = int(now - start_ts)

    # Trades fermés (liste de ClosedTrade dataclass)
    closed = state.closed_trades or []

    def _t(obj, attr, default=None):
        """Accès sécurisé à un attribut sur dataclass ou dict."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(attr, default)
        return getattr(obj, attr, default)

    # Stats globales
    total_trades = len(closed)
    wins = [t for t in closed if _t(t, "pnl_usdt", 0) > 0]
    losses = [t for t in closed if _t(t, "pnl_usdt", 0) < 0]
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0.0

    total_pnl = sum(_t(t, "pnl_usdt", 0) for t in closed)
    total_fees = sum(_t(t, "fees", 0) for t in closed)

    # Profit factor
    gross_wins = sum(_t(t, "pnl_usdt", 0) for t in wins)
    gross_losses = abs(sum(_t(t, "pnl_usdt", 0) for t in losses))
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else 0.0

    # Breakdown par stratégie
    by_strategy = {}
    for t in closed:
        strat = _t(t, "strategy", "unknown") or "unknown"
        if strat not in by_strategy:
            by_strategy[strat] = {"n": 0, "wins": 0, "pnl": 0.0}
        by_strategy[strat]["n"] += 1
        if _t(t, "pnl_usdt", 0) > 0:
            by_strategy[strat]["wins"] += 1
        by_strategy[strat]["pnl"] += _t(t, "pnl_usdt", 0)

    strategy_stats = []
    for name, s in by_strategy.items():
        wr = (s["wins"] / s["n"] * 100) if s["n"] > 0 else 0
        strategy_stats.append({
            "name": name,
            "n_trades": s["n"],
            "win_rate": round(wr, 1),
            "pnl_usdt": round(s["pnl"], 2),
        })
    strategy_stats.sort(key=lambda x: x["pnl_usdt"], reverse=True)

    # Positions ouvertes (dict {symbol: Position})
    open_positions = []
    for symbol, pos in (state.positions or {}).items():
        ai_ctx = _t(pos, "ai_context") or {}
        open_positions.append({
            "symbol": symbol,
            "side": _t(pos, "side"),
            "qty": round(_t(pos, "qty", 0), 6),
            "entry_price": round(_t(pos, "entry_price", 0), 4),
            "stop_loss": round(_t(pos, "stop_loss", 0), 4),
            "take_profit": round(_t(pos, "take_profit", 0), 4),
            "strategy": _t(pos, "strategy"),
            "entry_time": _t(pos, "entry_time"),
            "ai_regime": ai_ctx.get("regime"),
            "ai_score": ai_ctx.get("regime_score"),
            "ai_reasoning": ai_ctx.get("reasoning"),
            "ai_confidence_mult": ai_ctx.get("confidence_mult"),
            "signal_reasoning": ai_ctx.get("signal_reasoning") or ai_ctx.get("reasoning"),
        })

    # Derniers trades (10)
    recent_trades = []
    for t in (closed[-10:] if closed else [])[::-1]:
        t_ai_ctx = _t(t, "ai_context") or {}
        recent_trades.append({
            "symbol": _t(t, "symbol"),
            "side": _t(t, "side"),
            "strategy": _t(t, "strategy"),
            "entry_price": round(_t(t, "entry_price", 0), 4),
            "exit_price": round(_t(t, "exit_price", 0), 4),
            "pnl_usdt": round(_t(t, "pnl_usdt", 0), 2),
            "pnl_pct": round(_t(t, "pnl_pct", 0), 2),
            "exit_reason": _t(t, "exit_reason"),
            "entry_time": _t(t, "entry_time"),
            "exit_time": _t(t, "exit_time"),
            "ai_regime": t_ai_ctx.get("regime"),
            "ai_reasoning": t_ai_ctx.get("reasoning"),
        })

    # Equity curve (liste des PnL cumulatifs)
    equity_curve = []
    running = config.trading.capital_usdt
    equity_curve.append({
        "ts": start_ts,
        "equity": round(running, 2),
    })
    for t in closed:
        running += _t(t, "pnl_usdt", 0)
        equity_curve.append({
            "ts": _t(t, "exit_time", now),
            "equity": round(running, 2),
        })

    # PnL (méthodes ou attributs selon l'API)
    def _maybe_call(obj, name, default=0):
        v = getattr(obj, name, None)
        if v is None:
            return default
        if callable(v):
            try:
                return v()
            except Exception:
                return default
        return v

    daily_pnl = _maybe_call(state, "daily_pnl", 0)
    session_pnl = _maybe_call(state, "session_pnl", 0)
    consec_losses = getattr(state, "consecutive_losses", 0) or 0

    return {
        "status": {
            "mode": config.mode,
            "uptime_sec": uptime_sec,
            "timestamp": now,
            "symbols": config.trading.symbols,
            "ai_enabled": bool(config.ai.anthropic_api_key),
            "ai_model": config.ai.model if config.ai.anthropic_api_key else None,
        },
        "capital": {
            "initial_usdt": config.trading.capital_usdt,
            "current_usdt": round(config.trading.capital_usdt + total_pnl, 2),
            "total_pnl_usdt": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / config.trading.capital_usdt * 100, 2) if config.trading.capital_usdt > 0 else 0,
            "total_fees_usdt": round(total_fees, 2),
            "daily_pnl_usdt": round(daily_pnl, 2),
            "session_pnl_usdt": round(session_pnl, 2),
        },
        "performance": {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "consecutive_losses": consec_losses,
        },
        "ai_stats": {
            "cost_today_usd": round(getattr(state, "ai_cost_today_usd", 0) or 0, 4),
            "calls_today": getattr(state, "ai_calls_today", 0) or 0,
        },
        "open_positions": open_positions,
        "recent_trades": recent_trades,
        "by_strategy": strategy_stats,
        "equity_curve": equity_curve[-200:],  # derniers 200 points max
        "ai_insights": ai_insights or [],     # dernières analyses Claude (avec reasoning)
        "ai_state": _build_ai_state_block(ai_state or {}),  # état adaptatif courant
    }


def _build_ai_state_block(ai_state: dict) -> dict:
    """Construit le bloc d'état adaptatif IA pour le dashboard."""
    now = ai_state.get("now") or time.time()

    # NEXT LEVEL FIX: pause par symbole au lieu de globale
    pause_by_symbol = ai_state.get("pause_until_by_symbol", {}) or {}
    pause_reason_by_symbol = ai_state.get("pause_reason_by_symbol", {}) or {}

    # Construit la liste des symboles pausés avec leurs infos
    paused_symbols = []
    for symbol, pause_until in pause_by_symbol.items():
        remaining_sec = max(0, int(pause_until - now))
        if remaining_sec > 0:
            paused_symbols.append({
                "symbol": symbol,
                "pause_remaining_sec": remaining_sec,
                "pause_reason": pause_reason_by_symbol.get(symbol, ""),
            })

    # Global "is_paused" = vrai seulement si TOUS les symboles sont pausés
    # (Ce n'est plus "le bot entier est pausé" mais "aucun symbole ne peut trader")
    all_paused = len(paused_symbols) > 0 and all(
        pause_by_symbol.get(s, 0) > now
        for s in ai_state.get("all_symbols", [])
    )

    # Pause remaining globale = max des pauses actives (si all_paused)
    global_pause_remaining = max([p["pause_remaining_sec"] for p in paused_symbols], default=0)
    global_pause_reason = paused_symbols[0]["pause_reason"] if paused_symbols else ""

    return {
        "is_paused": all_paused,
        "pause_remaining_sec": global_pause_remaining if all_paused else 0,
        "pause_reason": global_pause_reason if all_paused else "",
        "paused_symbols": paused_symbols,  # NEW: liste des symboles pausés individuellement
        "enabled_strategies": ai_state.get("enabled_strategies", []) or [],
        "current_regime": ai_state.get("regime", "") or "",
        "current_sizing_mult": round(float(ai_state.get("sizing_mult", 1.0) or 1.0), 2),
    }
