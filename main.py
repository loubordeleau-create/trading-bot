"""
Main Orchestrator. Le chef d'orchestre du bot.

Boucle principale :
1. Check circuit breaker
2. Update IA regime toutes les 2h (cached sinon)
3. Pour chaque symbol :
   a. Fetch OHLCV multi-TF
   b. Gérer les positions existantes (stops, trailing, time-stop)
   c. Si pas de position : faire tourner les stratégies, choisir la meilleure selon régime
   d. Valider avec correlation + sizing
   e. Exécuter
4. Dormir X secondes
5. Une fois par semaine : post-mortem
"""
import argparse
import logging
import sys
import time
from typing import Dict, Optional

from config.settings import CONFIG
from core.exchange import Exchange
from core.market_data import MarketData
from core.state import BotState, Position
from core.indicators import atr

from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum_breakout import MomentumBreakoutStrategy
from strategies.trend_following import TrendFollowingStrategy
from strategies.trend_pullback import TrendPullbackStrategy
from strategies.trend_surfer import TrendSurferStrategy
from strategies.vol_harvest import VolatilityHarvestStrategy
from strategies.grid_dynamic import GridDynamicStrategy

from risk.position_sizer import PositionSizer
from risk.circuit_breaker import CircuitBreaker
from risk.correlation import CorrelationChecker
from risk.stop_manager import StopManager

from ai.claude_client import ClaudeClient
from ai.regime_detector import RegimeDetector
from ai.event_scanner import EventScanner
from ai.post_mortem import PostMortemAnalyzer

from execution.order_manager import OrderManager
from execution.slippage_monitor import SlippageMonitor

from core.stats_server import StatsServer, build_stats_payload


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bot")


class TradingBot:
    def __init__(self, config, mode: str = "paper"):
        self.config = config
        config.mode = mode

        logger.info(f"Init bot en mode {mode.upper()}")

        # Infra
        self.exchange = Exchange(config, mode=mode)
        self.market_data = MarketData(self.exchange)
        self.state = BotState(log_dir="logs")

        # Sync positions existantes sur l'exchange (anti-doublon après crash/reboot)
        if mode == "live":
            self.state.sync_with_exchange(self.exchange)
            if self.state.positions:
                logger.warning(f"⚠ {len(self.state.positions)} position(s) existante(s) récupérée(s) "
                              f"de l'exchange avec stops par défaut — review manuellement!")

        # IA (optionnelle — si pas de clé API, les modules fallback en mode dégradé)
        self.claude = None
        self.regime_detector = None
        self.event_scanner = None
        self.post_mortem = None
        if config.ai.anthropic_api_key:
            try:
                self.claude = ClaudeClient(config)
                self.regime_detector = RegimeDetector(self.claude)
                self.event_scanner = EventScanner(self.claude)
                self.post_mortem = PostMortemAnalyzer(self.claude)
                logger.info("Claude AI layer activée")
            except Exception as e:
                logger.warning(f"Claude layer indisponible: {e}. Bot tourne en mode technique pur.")
        else:
            logger.warning("Pas de clé Anthropic. Bot tourne sans couche IA (regime_score=5, mult=1).")

        # Stratégies
        self.strategies = [
            # PRIMARY STRATEGY - pure trend following with trailing stop
            TrendSurferStrategy(atr_stop_mult=2.0, atr_target_mult=5.0),
            # Legacy strategies - kept in code but Claude won't activate them
            MeanReversionStrategy(atr_stop_mult=config.trading.atr_stop_mult,
                                   atr_target_mult=config.trading.atr_target_mult),
            MomentumBreakoutStrategy(atr_stop_mult=config.trading.atr_stop_mult,
                                      atr_target_mult=config.trading.atr_target_mult * 1.5),
            TrendFollowingStrategy(atr_stop_mult=config.trading.atr_stop_mult * 1.3,
                                    atr_target_mult=config.trading.atr_target_mult * 1.5),
            TrendPullbackStrategy(atr_stop_mult=config.trading.atr_stop_mult,
                                   atr_target_mult=config.trading.atr_target_mult * 1.25),
            VolatilityHarvestStrategy(),
            GridDynamicStrategy(),
        ]

        # Risk
        self.sizer = PositionSizer(config)
        self.circuit_breaker = CircuitBreaker(config)
        self.corr_checker = CorrelationChecker(config)
        self.stop_manager = StopManager(config)

        # Execution
        self.order_manager = OrderManager(self.exchange, config)
        self.slippage_monitor = SlippageMonitor()

        # AI cache
        self._regime_cache: Dict[str, tuple] = {}   # symbol -> (ts, assessment)
        self._last_post_mortem: float = 0

        # AI insights buffer (30 dernières analyses Claude pour le dashboard)
        from collections import deque
        self._ai_insights: deque = deque(maxlen=30)

        # NEXT LEVEL: AI adaptive state (exposé dans le dashboard)
        # Pause PAR SYMBOLE, pas globale (fix bug où pause ETH bloque BTC)
        self._ai_pause_until_by_symbol: Dict[str, float] = {}   # symbol -> timestamp
        self._ai_pause_reason_by_symbol: Dict[str, str] = {}    # symbol -> reason
        self._ai_current_enabled: list = []  # strategies actuellement enabled par Claude
        self._ai_current_regime: str = ""    # dernier régime global
        self._ai_current_mult: float = 1.0   # multiplicateur de sizing courant

        # Réconciliation périodique (live seulement)
        self._last_reconcile: float = time.time()
        self._reconcile_interval_sec: int = 600  # Toutes les 10 min

        # Start time (pour uptime)
        self._start_ts = time.time()

        # Serveur HTTP pour dashboard (optionnel, activé si PORT env var)
        self.stats_server = None
        import os as _os
        port_str = _os.environ.get("PORT") or _os.environ.get("STATS_PORT")
        if port_str:
            try:
                port = int(port_str)
                token = _os.environ.get("DASHBOARD_TOKEN", "")
                self.stats_server = StatsServer(
                    port=port,
                    stats_provider=lambda: build_stats_payload(
                        self.state, self.config, self._start_ts,
                        ai_insights=list(self._ai_insights),
                        ai_state={
                            "pause_until_by_symbol": dict(self._ai_pause_until_by_symbol),
                            "pause_reason_by_symbol": dict(self._ai_pause_reason_by_symbol),
                            "all_symbols": list(self.config.trading.symbols),
                            "enabled_strategies": list(self._ai_current_enabled),
                            "regime": self._ai_current_regime,
                            "sizing_mult": self._ai_current_mult,
                            "now": time.time(),
                        },
                    ),
                    token=token,
                )
                self.stats_server.start()
                logger.info(f"Dashboard endpoint actif sur port {port}"
                            + (" (token required)" if token else " (no token)"))
            except Exception as e:
                logger.warning(f"Stats server failed to start: {e}")

    # ============================================================
    # MAIN LOOP
    # ============================================================

    def run(self):
        logger.info(f"Bot démarré. Symbols: {self.config.trading.symbols}")
        while True:
            try:
                self._iteration()
            except KeyboardInterrupt:
                logger.info("Arrêt manuel.")
                break
            except Exception as e:
                logger.exception(f"Erreur dans l'itération: {e}")
            time.sleep(30)  # Boucle toutes les 30s

    def _iteration(self):
        # 0. Réconciliation périodique (live uniquement)
        self._maybe_reconcile()

        # 1. Circuit breaker
        capital = self.exchange.get_balance()
        if capital <= 0:
            capital = self.config.trading.capital_usdt  # fallback paper

        cb_status = self.circuit_breaker.check(self.state, capital)
        if cb_status.tripped:
            logger.warning(f"Circuit breaker actif: {cb_status.reason}")
            # On gère quand même les positions ouvertes (stops)
            self._manage_open_positions()
            return

        # 2. Pour chaque symbol
        for symbol in self.config.trading.symbols:
            self._process_symbol(symbol, capital)

        # 3. Post-mortem hebdo
        self._maybe_run_post_mortem()

        # 4. Log périodique d'état
        self._log_status(capital)

    def _maybe_reconcile(self):
        """Toutes les N minutes en live, resync l'état avec l'exchange."""
        if self.config.mode != "live":
            return
        now = time.time()
        if now - self._last_reconcile < self._reconcile_interval_sec:
            return
        self._last_reconcile = now
        try:
            diag = self.state.reconcile_with_exchange(self.exchange)
            if diag["warnings"]:
                for w in diag["warnings"]:
                    logger.warning(f"RECONCILE warning: {w}")
        except Exception as e:
            logger.exception(f"Reconcile error: {e}")

    def _process_symbol(self, symbol: str, capital: float):
        # Fetch multi-TF
        ohlcv_map = {}
        for tf in self.config.trading.timeframes:
            try:
                ohlcv_map[tf] = self.market_data.get_ohlcv(symbol, tf, limit=200)
            except Exception as e:
                logger.warning(f"Fetch OHLCV failed {symbol} {tf}: {e}")
                return

        primary = ohlcv_map.get(self.config.trading.primary_tf)
        if primary is None or len(primary.close) < 50:
            return

        current_price = float(primary.close[-1])
        current_atr = float(atr(primary.high, primary.low, primary.close, 14)[-1])

        # 1. Gérer position existante (JAMAIS bloqué par une pause Claude)
        # Les positions déjà ouvertes continuent leur cycle TP/SL/trailing normalement
        if symbol in self.state.positions:
            self._manage_position(symbol, current_price, current_atr)
            return

        # NEXT LEVEL FIX: Check si Claude a pausé CE symbole spécifiquement
        # (avant: pause globale qui bloquait BTC quand ETH était pausé = bug)
        symbol_pause_until = self._ai_pause_until_by_symbol.get(symbol, 0.0)
        if symbol_pause_until > time.time():
            remaining = int((symbol_pause_until - time.time()) / 60)
            pause_reason = self._ai_pause_reason_by_symbol.get(symbol, "")
            logger.debug(f"[{symbol}] Pausé par IA ({remaining}m restantes): {pause_reason}")
            return

        # 2. Obtenir évaluation IA (cached)
        assessment = self._get_regime_assessment(symbol, ohlcv_map)

        # Kill-switch IA
        if assessment.kill_switch:
            self.state.kill_switch = True
            logger.warning(f"AI kill-switch: {assessment.kill_reason}")
            return

        # NEXT LEVEL FIX: Pause PAR SYMBOLE (pas globale)
        if assessment.pause_minutes > 0 and self.config.ai.ai_can_pause_bot:
            self._ai_pause_until_by_symbol[symbol] = time.time() + (assessment.pause_minutes * 60)
            self._ai_pause_reason_by_symbol[symbol] = assessment.reasoning[:200]
            logger.warning(f"[{symbol}] Symbole pausé {assessment.pause_minutes}m par Claude: {assessment.reasoning[:150]}")
            return

        # Clear pause pour ce symbole si Claude dit plus de pause
        if symbol in self._ai_pause_until_by_symbol and assessment.pause_minutes == 0:
            if self._ai_pause_until_by_symbol[symbol] > time.time():
                pass  # pause encore active, on la laisse expirer naturellement
            else:
                # Pause expirée et Claude ne re-pause pas, on clean
                self._ai_pause_until_by_symbol.pop(symbol, None)
                self._ai_pause_reason_by_symbol.pop(symbol, None)

        # Update état global pour dashboard
        self._ai_current_regime = assessment.regime
        self._ai_current_mult = assessment.confidence_mult
        self._ai_current_enabled = list(assessment.enabled_strategies)

        # 3. Faire tourner les stratégies et filtrer par enabled_strategies de Claude
        context = {
            "regime_score": assessment.regime_score,
            "regime": assessment.regime,
            "confidence_mult": assessment.confidence_mult,
            "current_positions": self.state.positions,
        }

        signals = []

        # NEXT LEVEL CRITICAL FIX:
        # Si Claude n'active aucune strategie (enabled=[]), on ne scan RIEN.
        # Avant: le fallback laissait tourner les strategies si regime_score <= 6 → BUG
        # Maintenant: enabled=[] = vraiment rien, on est cash, point.
        if self.config.ai.ai_can_kill_strategy and not assessment.enabled_strategies:
            logger.debug(f"[{symbol}] Aucune strategie activee par Claude, skip total")
            return

        for strat in self.strategies:
            # Filtre strict par Claude's enabled_strategies
            if self.config.ai.ai_can_kill_strategy and assessment.enabled_strategies:
                if strat.name not in assessment.enabled_strategies:
                    logger.debug(f"[{symbol}] Skip {strat.name}: Claude l'a desactivee pour ce regime")
                    continue

            sig = strat.analyze(symbol, primary, context)
            if sig and sig.action in ("buy", "sell") and sig.strength > 0.45:
                signals.append(sig)

        if not signals:
            return

        # Garder le meilleur signal
        best = max(signals, key=lambda s: s.strength)
        logger.info(f"[{symbol}] Signal détecté: {best.strategy_name} {best.action} "
                   f"strength={best.strength:.2f} | {best.reasoning}")

        # 4. Check corrélation
        all_symbols = list(self.state.positions.keys()) + [symbol]
        correlations = self.market_data.correlation_matrix(all_symbols, "1h", 100) if len(all_symbols) >= 2 else {}
        allow, corr_reason = self.corr_checker.allow_new_position(
            symbol, best.action, self.state.positions, correlations
        )
        if not allow:
            logger.info(f"[{symbol}] Trade refusé (corrélation): {corr_reason}")
            return

        # 5. Sizing
        prices = {s: self.market_data.get_price(s) for s in self.state.positions}
        current_exposure = self.state.total_exposure_usdt(prices)

        sizing = self.sizer.size(
            capital_usdt=capital,
            entry_price=best.entry_price,
            stop_price=best.stop_loss,
            win_rate=self.state.win_rate(),
            avg_win_loss_ratio=self.state.avg_win_loss_ratio(),
            signal_strength=best.strength,
            ai_confidence_mult=assessment.confidence_mult,
            current_exposure_usdt=current_exposure,
        )

        if sizing.qty <= 0:
            logger.info(f"[{symbol}] Sizing refuse: {sizing.rationale}")
            return

        logger.info(f"[{symbol}] Sizing: {sizing.rationale}")

        # 6. Exécution
        exec_res = self.order_manager.execute(
            symbol, best.action, sizing.qty, best.entry_price, urgency="normal"
        )
        if not exec_res.success:
            logger.error(f"[{symbol}] Exec failed: {exec_res.error}")
            return

        # Monitor slippage
        slip_alert = self.slippage_monitor.record(symbol, exec_res.slippage_bps)
        if slip_alert["alert"]:
            logger.warning(f"[{symbol}] {slip_alert['reason']}")

        # 7. Enregistrer la position
        pos = Position(
            symbol=symbol,
            side="long" if best.action == "buy" else "short",
            qty=exec_res.total_filled,
            entry_price=exec_res.avg_fill_price,
            entry_time=time.time(),
            stop_loss=best.stop_loss,
            take_profit=best.take_profit,
            strategy=best.strategy_name,
            ai_context={
                "regime": assessment.regime,
                "regime_score": assessment.regime_score,
                "confidence_mult": assessment.confidence_mult,
                "signal_strength": best.strength,
                "reasoning": best.reasoning,
            },
            order_id=None,
        )
        self.state.open_position(pos)
        logger.info(f"[{symbol}] POSITION OUVERTE {pos.side} {pos.qty:.6f} @ {pos.entry_price:.2f} | "
                   f"SL={pos.stop_loss:.2f} TP={pos.take_profit:.2f}")

    def _manage_position(self, symbol: str, current_price: float, current_atr: float):
        """Check stops/trailing/time-stop sur une position ouverte."""
        pos = self.state.positions[symbol]
        exit_reason = self.stop_manager.check_stop(pos, current_price, current_atr)
        if exit_reason is None:
            return

        # Close position
        close_side = "sell" if pos.side == "long" else "buy"
        exec_res = self.order_manager.execute(
            symbol, close_side, pos.qty, current_price, urgency="high"
        )
        if not exec_res.success:
            logger.error(f"[{symbol}] Close failed: {exec_res.error}")
            return

        trade = self.state.close_position(
            symbol,
            exit_price=exec_res.avg_fill_price,
            fees=exec_res.total_fees,
            exit_reason=exit_reason,
        )
        if trade:
            logger.info(
                f"[{symbol}] POSITION FERMÉE [{exit_reason}] "
                f"PnL={trade.pnl_usdt:+.2f} USDT ({trade.pnl_pct*100:+.2f}%)"
            )

    def _manage_open_positions(self):
        """Appelé même quand le CB est actif, pour continuer à gérer les stops."""
        for symbol in list(self.state.positions.keys()):
            try:
                primary = self.market_data.get_ohlcv(symbol, self.config.trading.primary_tf, 50)
                if len(primary.close) < 20:
                    continue
                price = float(primary.close[-1])
                atr_val = float(atr(primary.high, primary.low, primary.close, 14)[-1])
                self._manage_position(symbol, price, atr_val)
            except Exception as e:
                logger.warning(f"manage_open_positions error {symbol}: {e}")

    def _get_regime_assessment(self, symbol: str, ohlcv_map):
        """Cache IA : réévalue toutes les X minutes seulement."""
        from ai.regime_detector import RegimeAssessment
        cache_ttl = self.config.ai.regime_check_minutes * 60
        now = time.time()

        if symbol in self._regime_cache:
            ts, cached = self._regime_cache[symbol]
            if now - ts < cache_ttl:
                return cached

        if self.regime_detector is None:
            # Fallback sans IA : régime neutre, mult 1.0
            return RegimeAssessment(
                regime="ranging", regime_score=5.0, confidence_mult=1.0,
                kill_switch=False, kill_reason=None,
                reasoning="Mode technique pur, pas de classification IA", raw={},
            )

        # Contexte externe minimal (en v2 on ajouterait news feed)
        external = {}
        try:
            external["spread_bps"] = f"{self.market_data.get_spread_bps(symbol):.1f}"
        except Exception:
            pass

        assessment = self.regime_detector.assess(symbol, ohlcv_map, external)
        self._regime_cache[symbol] = (now, assessment)
        logger.info(f"[{symbol}] IA regime: {assessment.regime} score={assessment.regime_score:.1f} "
                   f"mult={assessment.confidence_mult:.2f} enabled={assessment.enabled_strategies} "
                   f"pause={assessment.pause_minutes}m | {assessment.reasoning}")

        # Buffer pour le dashboard (enrichi NEXT LEVEL)
        self._ai_insights.append({
            "ts": now,
            "symbol": symbol,
            "regime": assessment.regime,
            "regime_score": float(assessment.regime_score),
            "confidence_mult": float(assessment.confidence_mult),
            "reasoning": assessment.reasoning,
            "kill_switch": bool(assessment.kill_switch),
            "enabled_strategies": list(assessment.enabled_strategies),
            "pause_minutes": int(assessment.pause_minutes),
            "conviction_tier": assessment.conviction_tier,
        })
        return assessment

    def _maybe_run_post_mortem(self):
        """Hebdo."""
        if self.post_mortem is None:
            return
        now = time.time()
        if now - self._last_post_mortem < 86400 * 7:
            return
        if len(self.state.closed_trades) < 10:
            return

        logger.info("=== POST-MORTEM HEBDOMADAIRE ===")
        report = self.post_mortem.analyze(self.state, lookback_days=7)
        logger.info(f"Summary: {report.summary}")
        for obs in report.observations:
            logger.info(f"  - {obs}")
        for adj in report.suggested_adjustments:
            logger.info(f"  AJUSTEMENT SUGGÉRÉ: {adj}")
        logger.info("=== FIN POST-MORTEM ===")
        self._last_post_mortem = now

    def _log_status(self, capital: float):
        """Heartbeat log toutes les N iterations."""
        # On log seulement toutes les ~5 minutes pour pas polluer
        if not hasattr(self, "_last_status_log"):
            self._last_status_log = 0
        now = time.time()
        if now - self._last_status_log < 300:
            return
        self._last_status_log = now

        n_open = len(self.state.positions)
        session_pnl = self.state.session_pnl()
        daily_pnl = self.state.daily_pnl()
        wr = self.state.win_rate()
        cost = self.claude.cost.cost_today if self.claude else 0

        logger.info(
            f"[STATUS] capital={capital:.2f} positions={n_open} "
            f"daily_pnl={daily_pnl:+.2f} session_pnl={session_pnl:+.2f} "
            f"win_rate={wr:.1%} trades_closed={len(self.state.closed_trades)} "
            f"ai_cost_today=${cost:.3f}"
        )


def _load_env_overrides(config):
    """Charge les clés API depuis les variables d'environnement si présentes.
    Prioritaire sur ce qui est dans settings.py (évite de committer les secrets)."""
    import os
    exch_key = os.environ.get("BYBIT_API_KEY") or os.environ.get("EXCHANGE_API_KEY")
    exch_secret = os.environ.get("BYBIT_API_SECRET") or os.environ.get("EXCHANGE_API_SECRET")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if exch_key:
        config.exchange.api_key = exch_key
    if exch_secret:
        config.exchange.api_secret = exch_secret
    if anthropic_key:
        config.ai.anthropic_api_key = anthropic_key


def main():
    parser = argparse.ArgumentParser(description="Claude-powered trading bot")
    parser.add_argument("--mode", choices=["paper", "live", "backtest"], default="paper",
                        help="Trading mode (default: paper)")
    args = parser.parse_args()

    if args.mode == "backtest":
        print("Pour backtester, utilise: python run_backtest.py --symbol BTC/USDT:USDT --days 365")
        raise NotImplementedError("Mode backtest pas routé ici — utilise run_backtest.py")

    # Charger les clés depuis env vars (plus sécurisé qu'en hardcode)
    _load_env_overrides(CONFIG)

    if args.mode == "live":
        if CONFIG.exchange.testnet:
            print("⚠️  Mode LIVE demandé mais testnet=True dans settings.py")
            print("   Le bot va tourner sur le testnet Bybit (pas d'argent réel).")
            print("   Pour du vrai live, mets testnet=False et confirme.")
        else:
            print("⚠️  MODE LIVE — Trades réels avec argent réel sur", CONFIG.exchange.name)
            print("Tape 'OUI JE CONFIRME' pour continuer : ", end="")
            if input().strip() != "OUI JE CONFIRME":
                print("Annulé.")
                sys.exit(0)
        if not CONFIG.exchange.api_key or not CONFIG.exchange.api_secret:
            print("❌ Clés API manquantes. Définis BYBIT_API_KEY et BYBIT_API_SECRET en env var.")
            sys.exit(1)

    bot = TradingBot(CONFIG, mode=args.mode)
    bot.run()


if __name__ == "__main__":
    main()
