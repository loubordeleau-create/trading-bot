"""
Indicateurs techniques. Implémentations pures numpy pour perf + zéro surprise.
"""
import numpy as np
from typing import Tuple


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average."""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full(len(values), np.nan)
    cumsum = np.cumsum(values, dtype=float)
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate(([0], cumsum[:-period]))) / period
    return result


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average."""
    alpha = 2.0 / (period + 1)
    result = np.full(len(values), np.nan)
    if len(values) < period:
        return result
    result[period - 1] = values[:period].mean()
    for i in range(period, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder's smoothing)."""
    result = np.full(len(values), np.nan)
    if len(values) < period + 1:
        return result

    diffs = np.diff(values)
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0:
        result[period] = 100
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - 100 / (1 + rs)

    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            result[i] = 100
        else:
            rs = avg_gain / avg_loss
            result[i] = 100 - 100 / (1 + rs)
    return result


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range. Mesure de volatilité."""
    result = np.full(len(close), np.nan)
    if len(close) < period + 1:
        return result

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    result[period - 1] = tr[:period].mean()
    for i in range(period, len(close)):
        result[i] = (result[i - 1] * (period - 1) + tr[i]) / period
    return result


def bollinger_bands(values: np.ndarray, period: int = 20, std_mult: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (lower, middle, upper)."""
    middle = sma(values, period)
    std = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        std[i] = values[i - period + 1:i + 1].std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return lower, middle, upper


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index. Force de tendance (0-100)."""
    n = len(close)
    result = np.full(n, np.nan)
    if n < period * 2:
        return result

    up_move = np.diff(high, prepend=high[0])
    down_move = -np.diff(low, prepend=low[0])

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr_vals = atr(high, low, close, period)

    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)

    for i in range(period, n):
        if atr_vals[i] and atr_vals[i] > 0:
            plus_di[i] = 100 * plus_dm[i - period + 1:i + 1].mean() / atr_vals[i]
            minus_di[i] = 100 * minus_dm[i - period + 1:i + 1].mean() / atr_vals[i]

    dx = np.full(n, np.nan)
    for i in range(period, n):
        denom = plus_di[i] + minus_di[i]
        if denom and denom > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / denom

    # Smooth DX to get ADX
    for i in range(period * 2, n):
        window = dx[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            result[i] = valid.mean()
    return result


def volume_profile_imbalance(volume: np.ndarray, close: np.ndarray, lookback: int = 20) -> float:
    """
    Imbalance buy/sell basé sur volume en hausse vs baisse.
    Retourne un score entre -1 (sell-heavy) et +1 (buy-heavy).
    """
    if len(close) < lookback + 1:
        return 0.0
    recent_vol = volume[-lookback:]
    recent_close = close[-lookback:]
    diffs = np.diff(recent_close, prepend=recent_close[0])
    buy_vol = recent_vol[diffs > 0].sum()
    sell_vol = recent_vol[diffs < 0].sum()
    total = buy_vol + sell_vol
    if total == 0:
        return 0.0
    return (buy_vol - sell_vol) / total
