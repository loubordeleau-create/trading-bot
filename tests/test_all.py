"""
Tests unitaires. Lance avec:
    python -m pytest tests/ -v
ou:
    python tests/test_all.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.indicators import sma, ema, rsi, atr, bollinger_bands, adx, volume_profile_imbalance
from config.settings import BotConfig
from risk.position_sizer import PositionSizer
from risk.stop_manager import StopManager
from risk.circuit_breaker import CircuitBreaker
from core.state import BotState, Position
import time


# ============================================================
# INDICATORS
# ============================================================

def test_sma_basic():
    """SMA d'une constante = la constante."""
    vals = np.full(20, 10.0)
    result = sma(vals, 5)
    assert np.all(np.isnan(result[:4]))
    assert np.allclose(result[4:], 10.0), f"SMA constant failed: {result}"


def test_ema_basic():
    """EMA monte quand les valeurs montent."""
    vals = np.arange(1, 51, dtype=float)
    result = ema(vals, 10)
    # EMA doit être monotone croissant sur une série croissante (après warmup)
    valid = result[~np.isnan(result)]
    assert np.all(np.diff(valid) > 0), "EMA should be monotonically increasing"


def test_rsi_bounds():
    """RSI est dans [0, 100]."""
    np.random.seed(42)
    vals = 100 + np.cumsum(np.random.randn(100))
    result = rsi(vals, 14)
    valid = result[~np.isnan(result)]
    assert np.all(valid >= 0) and np.all(valid <= 100), f"RSI out of bounds: min={valid.min()}, max={valid.max()}"


def test_rsi_extreme_up():
    """Série qui ne fait que monter → RSI ~100."""
    vals = np.arange(1, 51, dtype=float)
    result = rsi(vals, 14)
    assert result[-1] > 95, f"RSI on always-up series should be near 100, got {result[-1]}"


def test_rsi_extreme_down():
    """Série qui ne fait que baisser → RSI ~0."""
    vals = np.arange(50, 0, -1, dtype=float)
    result = rsi(vals, 14)
    assert result[-1] < 5, f"RSI on always-down series should be near 0, got {result[-1]}"


def test_atr_positive():
    """ATR est toujours positif."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100))
    high = close + np.abs(np.random.randn(100))
    low = close - np.abs(np.random.randn(100))
    result = atr(high, low, close, 14)
    valid = result[~np.isnan(result)]
    assert np.all(valid > 0), "ATR must be positive"


def test_bollinger_ordering():
    """Lower <= Middle <= Upper."""
    np.random.seed(42)
    vals = 100 + np.cumsum(np.random.randn(100))
    lower, middle, upper = bollinger_bands(vals, 20, 2.0)
    valid = ~(np.isnan(lower) | np.isnan(upper))
    assert np.all(lower[valid] <= middle[valid]), "Lower must be <= middle"
    assert np.all(middle[valid] <= upper[valid]), "Middle must be <= upper"


def test_adx_bounds():
    """ADX dans [0, 100]."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(200))
    high = close + np.abs(np.random.randn(200))
    low = close - np.abs(np.random.randn(200))
    result = adx(high, low, close, 14)
    valid = result[~np.isnan(result)]
    if len(valid):
        assert np.all(valid >= 0) and np.all(valid <= 100), f"ADX out of bounds: min={valid.min()}, max={valid.max()}"


def test_volume_imbalance_bounds():
    """Volume imbalance dans [-1, 1]."""
    np.random.seed(42)
    volume = np.abs(np.random.randn(50)) * 1000
    close = 100 + np.cumsum(np.random.randn(50))
    result = volume_profile_imbalance(volume, close, 20)
    assert -1 <= result <= 1, f"Volume imbalance out of bounds: {result}"


# ============================================================
# POSITION SIZER
# ============================================================

def test_sizer_basic():
    """Sizer retourne une qty positive avec inputs valides."""
    config = BotConfig()
    sizer = PositionSizer(config)
    res = sizer.size(
        capital_usdt=1000, entry_price=100, stop_price=98,
        win_rate=0.55, avg_win_loss_ratio=1.5,
        signal_strength=0.7, ai_confidence_mult=1.0,
        current_exposure_usdt=0,
    )
    assert res.qty > 0, f"Sizer returned 0: {res.rationale}"


def test_sizer_no_edge():
    """Win rate 0.3 + R=1 → Kelly négatif → qty = 0."""
    config = BotConfig()
    sizer = PositionSizer(config)
    res = sizer.size(
        capital_usdt=1000, entry_price=100, stop_price=98,
        win_rate=0.3, avg_win_loss_ratio=1.0,
        signal_strength=0.7, ai_confidence_mult=1.0,
        current_exposure_usdt=0,
    )
    assert res.qty == 0, f"Expected 0 qty on negative edge, got {res.qty}"


def test_sizer_respects_max_position():
    """Position ne dépasse jamais max_position_pct du capital."""
    config = BotConfig()
    config.trading.max_position_pct = 0.10  # 10% max
    sizer = PositionSizer(config)
    res = sizer.size(
        capital_usdt=1000, entry_price=100, stop_price=95,
        win_rate=0.9, avg_win_loss_ratio=5.0,  # Edge énorme
        signal_strength=1.0, ai_confidence_mult=2.0,
        current_exposure_usdt=0,
    )
    assert res.position_value_usdt <= 100.1, f"Position value {res.position_value_usdt} > max 10% of 1000"


def test_sizer_respects_total_exposure():
    """Refuse si exposition totale dépasserait le cap."""
    config = BotConfig()
    config.trading.max_total_exposure = 0.50
    sizer = PositionSizer(config)
    res = sizer.size(
        capital_usdt=1000, entry_price=100, stop_price=98,
        win_rate=0.6, avg_win_loss_ratio=2.0,
        signal_strength=0.8, ai_confidence_mult=1.5,
        current_exposure_usdt=500,  # Déjà 50% exposé
    )
    assert res.position_value_usdt == 0, f"Should refuse, got {res.position_value_usdt}"


def test_sizer_stop_too_close():
    """Stop trop proche du prix → refuse."""
    config = BotConfig()
    sizer = PositionSizer(config)
    res = sizer.size(
        capital_usdt=1000, entry_price=100, stop_price=99.95,
        win_rate=0.6, avg_win_loss_ratio=2.0,
        signal_strength=0.8, ai_confidence_mult=1.0,
        current_exposure_usdt=0,
    )
    assert res.qty == 0, "Should refuse when stop is too close"


# ============================================================
# STOP MANAGER
# ============================================================

def test_stop_manager_long_sl():
    """Long position + prix sous stop_loss → sl hit."""
    sm = StopManager(BotConfig())
    pos = Position(
        symbol="BTC/USDT", side="long", qty=1.0,
        entry_price=100, entry_time=time.time(),
        stop_loss=95, take_profit=110,
        strategy="test",
    )
    assert sm.check_stop(pos, 94, current_atr=1.0) == "sl"
    assert sm.check_stop(pos, 96, current_atr=1.0) is None


def test_stop_manager_long_tp():
    """Long + prix au-dessus tp → tp hit."""
    sm = StopManager(BotConfig())
    pos = Position(
        symbol="BTC/USDT", side="long", qty=1.0,
        entry_price=100, entry_time=time.time(),
        stop_loss=95, take_profit=110,
        strategy="test",
    )
    assert sm.check_stop(pos, 111, current_atr=1.0) == "tp"


def test_stop_manager_short_sl():
    """Short + prix au-dessus stop_loss → sl hit."""
    sm = StopManager(BotConfig())
    pos = Position(
        symbol="BTC/USDT", side="short", qty=1.0,
        entry_price=100, entry_time=time.time(),
        stop_loss=105, take_profit=90,
        strategy="test",
    )
    assert sm.check_stop(pos, 106, current_atr=1.0) == "sl"


def test_stop_manager_trailing_long():
    """Long qui avance: trailing stop se resserre."""
    sm = StopManager(BotConfig())
    pos = Position(
        symbol="BTC/USDT", side="long", qty=1.0,
        entry_price=100, entry_time=time.time(),
        stop_loss=95, take_profit=110,
        strategy="test",
    )
    initial_stop = pos.stop_loss
    # ATR=2, trail_mult=1.5, on avance à 103 → 103 - 1.5*2 = 100 > 95
    sm.check_stop(pos, 103, current_atr=2.0)
    assert pos.stop_loss > initial_stop, f"Trailing stop should have moved up, got {pos.stop_loss}"


# ============================================================
# CIRCUIT BREAKER
# ============================================================

def test_circuit_breaker_normal():
    """État normal = pas tripped."""
    cb = CircuitBreaker(BotConfig())
    state = BotState("/tmp/bot_test")
    assert not cb.check(state, 1000).tripped


def test_circuit_breaker_kill_switch():
    """kill_switch activé → tripped."""
    cb = CircuitBreaker(BotConfig())
    state = BotState("/tmp/bot_test")
    state.kill_switch = True
    assert cb.check(state, 1000).tripped


def test_circuit_breaker_consecutive_losses():
    """N pertes consécutives → tripped."""
    config = BotConfig()
    config.risk.max_consecutive_losses = 3
    cb = CircuitBreaker(config)
    state = BotState("/tmp/bot_test")
    state.consecutive_losses = 3
    status = cb.check(state, 1000)
    assert status.tripped, f"Should be tripped on {state.consecutive_losses} losses"


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print()
    print(f"=== {passed}/{len(tests)} tests passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
