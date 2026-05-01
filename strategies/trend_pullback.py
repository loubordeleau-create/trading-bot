"""
Strategy F: Trend Pullback - entries sur continuation de trend.

Complementaire de momentum_breakout (= breakout initial) et trend_following (= crossover).
trend_pullback attrape les trades quand:
  - Le trend est deja etabli (EMA20 > EMA50 depuis >= 3 bougies, ADX >= 25)
  - Le prix a pullback sur l'EMA20 (ou juste en-dessous) recemment (1-3 bougies)
  - La bougie courante reprend la direction du trend (close > open pour long)
  - RSI n'est PAS en extreme (evite les entrees late dans un trend fatigue)

Anti-lookahead: evalue sur close[-1] = derniere bougie fermee.
"""
import numpy as np
from typing import Optional
from core.market_data import OHLCV
from core.indicators import atr, adx, ema, rsi
from strategies.base import Strategy, Signal


class TrendPullbackStrategy(Strategy):
    name = "trend_pullback"
    preferred_regimes = ["trending"]

    def __init__(self, fast_period=20, slow_period=50, atr_period=14,
                 adx_min=25, atr_stop_mult=1.5, atr_target_mult=2.5,
                 rsi_period=14, rsi_max_long=75, rsi_min_short=25,
                 pullback_tolerance_atr=0.5):
        """
        pullback_tolerance_atr: combien d'ATRs autour de l'EMA20 compte comme "pullback".
                                 0.5 = price doit etre entre EMA20 - 0.5*ATR et EMA20 + 0.5*ATR
                                 sur au moins une des 3 dernieres bougies.
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.atr_period = atr_period
        self.adx_min = adx_min
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.rsi_period = rsi_period
        self.rsi_max_long = rsi_max_long
        self.rsi_min_short = rsi_min_short
        self.pullback_tolerance_atr = pullback_tolerance_atr

    def analyze(self, symbol, ohlcv: OHLCV, context) -> Optional[Signal]:
        min_len = self.slow_period + 20
        if len(ohlcv.close) < min_len:
            return None

        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low
        open_ = ohlcv.open

        fast = ema(close, self.fast_period)
        slow = ema(close, self.slow_period)
        atr_vals = atr(high, low, close, self.atr_period)
        adx_vals = adx(high, low, close, 14)
        rsi_vals = rsi(close, self.rsi_period)

        price = close[-1]
        curr_open = open_[-1]
        curr_atr = atr_vals[-1]
        curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0
        curr_rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50.0
        curr_fast = fast[-1]
        curr_slow = slow[-1]

        if np.isnan(curr_atr) or curr_atr <= 0:
            return None
        if curr_adx < self.adx_min:
            return None

        # Verifie que le trend est etabli depuis au moins 3 bougies
        # (EMA20 > EMA50 pour uptrend, vice-versa pour downtrend)
        trend_established_bars = 3
        recent_fast = fast[-trend_established_bars - 1:-1]
        recent_slow = slow[-trend_established_bars - 1:-1]

        uptrend = np.all(recent_fast > recent_slow)
        downtrend = np.all(recent_fast < recent_slow)

        # LONG: uptrend etabli + pullback recent sur EMA20 + reprise
        if uptrend and price > curr_slow:
            # Verifie que RSI n'est pas surachete extreme
            if curr_rsi > self.rsi_max_long:
                return None

            # Detecte un pullback recent (1-3 dernieres bougies)
            # Le low d'une des 3 dernieres bougies doit toucher/traverser EMA20
            # (pullback_tolerance_atr * ATR en-dessous de EMA20 compte)
            tolerance = self.pullback_tolerance_atr * curr_atr
            had_pullback = False
            for i in range(-3, 0):
                bar_low = low[i]
                bar_fast = fast[i]
                if bar_low <= bar_fast + tolerance:
                    had_pullback = True
                    break

            if not had_pullback:
                return None

            # Confirmation: bougie courante reprend le trend (close > open)
            if price <= curr_open:
                return None

            # Bonus: price a depasse EMA20 (sortie du pullback)
            if price < curr_fast:
                return None

            strength = self._strength(
                adx_score=min((curr_adx - self.adx_min) / 30, 1.0),
                rsi_room=(self.rsi_max_long - curr_rsi) / self.rsi_max_long,
                bounce_strength=min((price - curr_open) / curr_atr, 1.0),
            )

            return Signal(
                action="buy",
                strength=strength,
                entry_price=price,
                stop_loss=price - self.atr_stop_mult * curr_atr,
                take_profit=price + self.atr_target_mult * curr_atr,
                reasoning=f"Trend pullback long: uptrend established, pulled to EMA20, bounce confirmed, ADX {curr_adx:.1f}, RSI {curr_rsi:.1f}",
                strategy_name=self.name,
            )

        # SHORT: downtrend etabli + pullback sur EMA20 + reprise baissiere
        if downtrend and price < curr_slow:
            if curr_rsi < self.rsi_min_short:
                return None

            tolerance = self.pullback_tolerance_atr * curr_atr
            had_pullback = False
            for i in range(-3, 0):
                bar_high = high[i]
                bar_fast = fast[i]
                if bar_high >= bar_fast - tolerance:
                    had_pullback = True
                    break

            if not had_pullback:
                return None

            if price >= curr_open:
                return None

            if price > curr_fast:
                return None

            strength = self._strength(
                adx_score=min((curr_adx - self.adx_min) / 30, 1.0),
                rsi_room=(curr_rsi - self.rsi_min_short) / (100 - self.rsi_min_short),
                bounce_strength=min((curr_open - price) / curr_atr, 1.0),
            )

            return Signal(
                action="sell",
                strength=strength,
                entry_price=price,
                stop_loss=price + self.atr_stop_mult * curr_atr,
                take_profit=price - self.atr_target_mult * curr_atr,
                reasoning=f"Trend pullback short: downtrend established, pulled to EMA20, rejection confirmed, ADX {curr_adx:.1f}, RSI {curr_rsi:.1f}",
                strategy_name=self.name,
            )

        return None

    def _strength(self, adx_score: float, rsi_room: float, bounce_strength: float) -> float:
        """
        Strength final:
          - 50% ADX (force du trend)
          - 25% RSI room (combien il reste de place avant overbought/oversold)
          - 25% bounce strength (taille de la reprise de la bougie courante)
        Minimum 0.45 pour que le signal soit pris (filtre du main.py).
        """
        raw = 0.50 * adx_score + 0.25 * rsi_room + 0.25 * bounce_strength
        return max(0.0, min(1.0, 0.45 + 0.5 * raw))
