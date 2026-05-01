"""
Backtester walk-forward.

Principe :
- Fetch historique OHLCV (via ccxt ou CSV)
- Pour chaque bougie i, présente aux stratégies les données [0:i] (pas de lookahead)
- Simule l'exécution avec frais + slippage réalistes
- Gère stops/trailing/time-stop bougie par bougie (intra-bar)
- Produit stats : Sharpe, Sortino, max drawdown, win rate, profit factor, par stratégie/régime

Walk-forward split :
- Train window (ex: 180j) pour stats historiques (win rate, win/loss ratio)
- Test window (ex: 30j) où on trade
- Rolling : on avance de test_window et on recommence
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import numpy as np

from core.market_data import OHLCV
from core.state import BotState, Position
from core.indicators import atr
from strategies.base import Strategy
from risk.position_sizer import PositionSizer
from risk.stop_manager import StopManager

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 1000.0
    fee_rate: float = 0.00055           # 5.5 bps taker (Bybit-like)
    slippage_bps: float = 5             # 5 bps slippage estimé
    train_window_days: int = 180
    test_window_days: int = 30
    primary_tf: str = "1h"
    # Fonction optionnelle: (symbol, ohlcv_window) -> regime_dict
    # Permet d'injecter un régime simulé sans appeler Claude en backtest
    regime_provider: Optional[Callable] = None


@dataclass
class BacktestResult:
    initial_capital: float
    final_capital: float
    total_return_pct: float
    n_trades: int
    win_rate: float
    profit_factor: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    avg_trade_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    by_strategy: Dict[str, dict] = field(default_factory=dict)
    by_regime: Dict[str, dict] = field(default_factory=dict)
    equity_curve: list = field(default_factory=list)  # [(ts, equity)]
    trades: list = field(default_factory=list)

    def __str__(self):
        return (
            f"=== BACKTEST RESULT ===\n"
            f"Capital: ${self.initial_capital:.2f} → ${self.final_capital:.2f} "
            f"({self.total_return_pct:+.2f}%)\n"
            f"Trades: {self.n_trades} | Win rate: {self.win_rate:.1%} | "
            f"Profit factor: {self.profit_factor:.2f}\n"
            f"Sharpe: {self.sharpe:.2f} | Sortino: {self.sortino:.2f} | "
            f"Max DD: {self.max_drawdown_pct:.2f}%\n"
            f"Avg trade: {self.avg_trade_pct:+.2f}% | "
            f"Best: {self.best_trade_pct:+.2f}% | Worst: {self.worst_trade_pct:+.2f}%\n"
        )


class Backtester:
    def __init__(self, config, bt_config: BacktestConfig, strategies: List[Strategy]):
        self.config = config
        self.bt = bt_config
        self.strategies = strategies
        self.sizer = PositionSizer(config)
        self.stop_manager = StopManager(config)

    def run(self, symbol: str, ohlcv: OHLCV,
            min_history_bars: int = 200) -> BacktestResult:
        """
        ohlcv : série historique complète (ex: 1 an de 1h bougies = 8760 points).
        Le backtester présente les données progressivement aux stratégies.
        """
        n = len(ohlcv.close)
        if n < min_history_bars + 50:
            raise ValueError(f"Historique trop court: {n} bougies, besoin >= {min_history_bars + 50}")

        capital = self.bt.initial_capital
        equity_curve = [(float(ohlcv.timestamps[min_history_bars]), capital)]
        open_position: Optional[Position] = None
        trades_log = []
        state_stub = BotState(log_dir="/tmp/backtest_logs")

        # Stats de progression
        import time as _time
        _start_time = _time.time()
        _total_bars = n - min_history_bars
        _last_log_i = min_history_bars

        # Boucle bougie par bougie
        # À l'instant de la bougie i:
        #  - On DÉCIDE en voyant les bougies [0..i-1] (toutes fermées)
        #  - On EXÉCUTE à l'open de la bougie i
        #  - On GÈRE l'intra-bar de i avec high/low/close de i
        for i in range(min_history_bars, n):
            # Log de progression toutes les 500 bougies
            if i - _last_log_i >= 500:
                elapsed = _time.time() - _start_time
                processed = i - min_history_bars
                pct = processed / _total_bars * 100
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (_total_bars - processed) / rate if rate > 0 else 0
                n_trades_so_far = len(trades_log)
                logger.info(
                    f"  [progress] bar {i}/{n} ({pct:.1f}%) | "
                    f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s | "
                    f"trades: {n_trades_so_far} | capital: ${capital:.2f}"
                )
                _last_log_i = i

            # Fenêtre visible par les stratégies : [0..i-1] = bougies fermées
            # ohlcv[-1] de cette fenêtre = bougie i-1 = dernière fermée
            window = OHLCV(
                timestamps=ohlcv.timestamps[:i],
                open=ohlcv.open[:i],
                high=ohlcv.high[:i],
                low=ohlcv.low[:i],
                close=ohlcv.close[:i],
                volume=ohlcv.volume[:i],
            )

            # Prix d'exécution = open de la bougie i (on décide à la clôture de i-1,
            # on exécute à l'open de i). Gestion intra-bar avec high/low de i.
            exec_open = float(ohlcv.open[i])
            bar_high = float(ohlcv.high[i])
            bar_low = float(ohlcv.low[i])
            bar_close = float(ohlcv.close[i])
            bar_ts = float(ohlcv.timestamps[i])

            # 1. Gérer position existante : vérifier si stop ou TP touchés pendant la bougie
            if open_position is not None:
                closed, exit_price, exit_reason = self._check_intrabar_exit(
                    open_position, bar_high, bar_low, bar_close
                )
                if closed:
                    pnl, fees = self._close_position(open_position, exit_price)
                    capital += pnl
                    trades_log.append({
                        "symbol": symbol,
                        "side": open_position.side,
                        "strategy": open_position.strategy,
                        "entry_price": open_position.entry_price,
                        "exit_price": exit_price,
                        "entry_time": open_position.entry_time,
                        "exit_time": bar_ts,
                        "pnl_usdt": pnl,
                        "pnl_pct": pnl / (open_position.entry_price * open_position.qty),
                        "fees": fees,
                        "exit_reason": exit_reason,
                        "regime": open_position.ai_context.get("regime", "unknown"),
                    })
                    open_position = None

                # Trailing / time-stop
                else:
                    # Utiliser ATR de la dernière bougie fermée (i-1)
                    curr_atr = float(atr(window.high, window.low, window.close, 14)[-1])
                    if not np.isnan(curr_atr) and curr_atr > 0:
                        self.stop_manager.check_stop(open_position, bar_close, curr_atr)

            # 2. Si pas de position, essayer d'ouvrir
            if open_position is None:
                # Contexte régime (optionnel)
                regime_info = {"regime": "ranging", "regime_score": 5.0, "confidence_mult": 1.0}
                if self.bt.regime_provider:
                    try:
                        regime_info = self.bt.regime_provider(symbol, window) or regime_info
                    except Exception:
                        pass

                # Stats historiques pour le sizer
                recent_trades = trades_log[-50:]
                wr = (sum(1 for t in recent_trades if t["pnl_usdt"] > 0) / len(recent_trades)) if recent_trades else 0.5
                wins = [t["pnl_usdt"] for t in recent_trades if t["pnl_usdt"] > 0]
                losses = [-t["pnl_usdt"] for t in recent_trades if t["pnl_usdt"] < 0]
                wl_ratio = ((sum(wins) / len(wins)) / (sum(losses) / len(losses))) if wins and losses else 1.0

                # Faire tourner les stratégies
                signals = []
                for strat in self.strategies:
                    # Filtre régime
                    if regime_info["regime"] not in strat.preferred_regimes and regime_info["regime_score"] > 6:
                        continue
                    sig = strat.analyze(symbol, window, regime_info)
                    if sig and sig.strength > 0.45:
                        signals.append(sig)

                if signals:
                    best = max(signals, key=lambda s: s.strength)

                    # Sizer
                    sizing = self.sizer.size(
                        capital_usdt=capital,
                        entry_price=exec_open,  # On exécute à l'open de la bougie i
                        stop_price=best.stop_loss,
                        win_rate=wr,
                        avg_win_loss_ratio=wl_ratio,
                        signal_strength=best.strength,
                        ai_confidence_mult=regime_info["confidence_mult"],
                        current_exposure_usdt=0,
                    )

                    if sizing.qty > 0:
                        # Appliquer slippage à l'entrée
                        slip = self.bt.slippage_bps / 10000
                        fill_price = exec_open * (1 + slip) if best.action == "buy" else exec_open * (1 - slip)
                        entry_fee = fill_price * sizing.qty * self.bt.fee_rate
                        capital -= entry_fee  # On soustrait les fees

                        open_position = Position(
                            symbol=symbol,
                            side="long" if best.action == "buy" else "short",
                            qty=sizing.qty,
                            entry_price=fill_price,
                            entry_time=bar_ts,
                            stop_loss=best.stop_loss,
                            take_profit=best.take_profit,
                            strategy=best.strategy_name,
                            ai_context={
                                "regime": regime_info["regime"],
                                "signal_strength": best.strength,
                            },
                        )

            # 3. Update equity curve (mark-to-market)
            equity = capital
            if open_position is not None:
                equity += open_position.unrealized_pnl(bar_close)
            equity_curve.append((bar_ts, equity))

        # 4. Fermer toute position encore ouverte
        if open_position is not None:
            pnl, fees = self._close_position(open_position, float(ohlcv.close[-1]))
            capital += pnl
            trades_log.append({
                "symbol": symbol,
                "side": open_position.side,
                "strategy": open_position.strategy,
                "entry_price": open_position.entry_price,
                "exit_price": float(ohlcv.close[-1]),
                "entry_time": open_position.entry_time,
                "exit_time": float(ohlcv.timestamps[-1]),
                "pnl_usdt": pnl,
                "pnl_pct": pnl / (open_position.entry_price * open_position.qty),
                "fees": fees,
                "exit_reason": "end_of_data",
                "regime": open_position.ai_context.get("regime", "unknown"),
            })

        return self._compute_stats(capital, trades_log, equity_curve)

    # ---------- Internals ----------

    def _check_intrabar_exit(self, pos: Position, bar_high: float,
                             bar_low: float, bar_close: float):
        """
        Détermine si stop ou TP a été touché pendant la bougie.
        Assumption conservative : si les deux sont dans le range, on assume SL d'abord.
        """
        if pos.side == "long":
            sl_hit = bar_low <= pos.stop_loss
            tp_hit = bar_high >= pos.take_profit
            if sl_hit and tp_hit:
                return True, pos.stop_loss, "sl"  # conservateur
            if sl_hit:
                return True, pos.stop_loss, "sl"
            if tp_hit:
                return True, pos.take_profit, "tp"
        else:
            sl_hit = bar_high >= pos.stop_loss
            tp_hit = bar_low <= pos.take_profit
            if sl_hit and tp_hit:
                return True, pos.stop_loss, "sl"
            if sl_hit:
                return True, pos.stop_loss, "sl"
            if tp_hit:
                return True, pos.take_profit, "tp"
        return False, 0, ""

    def _close_position(self, pos: Position, exit_price: float):
        """Retourne (pnl_net, total_fees)."""
        slip = self.bt.slippage_bps / 10000
        if pos.side == "long":
            fill_price = exit_price * (1 - slip)
            gross_pnl = (fill_price - pos.entry_price) * pos.qty
        else:
            fill_price = exit_price * (1 + slip)
            gross_pnl = (pos.entry_price - fill_price) * pos.qty
        exit_fee = fill_price * pos.qty * self.bt.fee_rate
        net_pnl = gross_pnl - exit_fee
        return net_pnl, exit_fee

    def _compute_stats(self, final_capital: float, trades: list, equity_curve: list) -> BacktestResult:
        if not trades:
            return BacktestResult(
                initial_capital=self.bt.initial_capital,
                final_capital=final_capital,
                total_return_pct=0,
                n_trades=0,
                win_rate=0, profit_factor=0, sharpe=0, sortino=0,
                max_drawdown_pct=0, avg_trade_pct=0,
                best_trade_pct=0, worst_trade_pct=0,
                equity_curve=equity_curve, trades=trades,
            )

        pnls = [t["pnl_usdt"] for t in trades]
        pct_returns = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls)
        profit_factor = (sum(wins) / abs(sum(losses))) if losses else float("inf")

        # Sharpe / Sortino sur returns des trades
        returns_arr = np.array(pct_returns)
        mean_ret = returns_arr.mean()
        std_ret = returns_arr.std()
        sharpe = (mean_ret / std_ret * np.sqrt(len(returns_arr))) if std_ret > 0 else 0
        downside = returns_arr[returns_arr < 0]
        downside_std = downside.std() if len(downside) > 0 else 0
        sortino = (mean_ret / downside_std * np.sqrt(len(returns_arr))) if downside_std > 0 else 0

        # Max drawdown depuis equity curve
        equities = np.array([e for _, e in equity_curve])
        running_max = np.maximum.accumulate(equities)
        drawdowns = (equities - running_max) / running_max
        max_dd = abs(drawdowns.min()) * 100 if len(drawdowns) else 0

        # Stats par stratégie
        by_strat = {}
        from collections import defaultdict
        groups_s = defaultdict(list)
        groups_r = defaultdict(list)
        for t in trades:
            groups_s[t["strategy"]].append(t)
            groups_r[t["regime"]].append(t)
        for s, ts in groups_s.items():
            by_strat[s] = {
                "n": len(ts),
                "win_rate": sum(1 for t in ts if t["pnl_usdt"] > 0) / len(ts),
                "total_pnl": sum(t["pnl_usdt"] for t in ts),
            }
        by_regime = {}
        for r, ts in groups_r.items():
            by_regime[r] = {
                "n": len(ts),
                "win_rate": sum(1 for t in ts if t["pnl_usdt"] > 0) / len(ts),
                "total_pnl": sum(t["pnl_usdt"] for t in ts),
            }

        return BacktestResult(
            initial_capital=self.bt.initial_capital,
            final_capital=final_capital,
            total_return_pct=(final_capital - self.bt.initial_capital) / self.bt.initial_capital * 100,
            n_trades=len(trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown_pct=max_dd,
            avg_trade_pct=mean_ret * 100,
            best_trade_pct=max(pct_returns) * 100 if pct_returns else 0,
            worst_trade_pct=min(pct_returns) * 100 if pct_returns else 0,
            by_strategy=by_strat,
            by_regime=by_regime,
            equity_curve=equity_curve,
            trades=trades,
        )