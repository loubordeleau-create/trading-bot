"""
Strategy: Trend Surfer
=======================

Philosophie: Surf les trends confirmes. Pas de timing de reversal.

Logic:
  1. Attendre trend 4h confirme (ADX > 25, EMA aligned, price aligned)
  2. Entree SEULEMENT sur breakout du high/low des N dernieres bougies 1h
  3. Stop initial ATR x 2 (large, respect le noise)
  4. Take profit FIXE tres eloigne (ATR x 5) - le vrai exit sera trailing stop
  5. Trailing stop progressif geré par stop_manager.py:
     - Breakeven apres +1 ATR
     - Lock +0.5 ATR apres +2 ATR
     - Trailing 2 ATR du high apres +3 ATR

Anti-lookahead: utilise close[-1] = derniere bougie fermee.
"""
import numpy as np
from typing import Optional
from core.market_data import OHLCV
from core.indicators import atr, adx, ema, rsi
from strategies.base import Strategy, Signal


class TrendSurferStrategy(Strategy):
    name = "trend_surfer"
    preferred_regimes = ["trending"]

    def __init__(self,
                 breakout_lookback=20,      # nb bougies pour detecter le high/low a casser
                 min_adx=22,                # ADX minimum sur la TF pour considerer trending
                 atr_stop_mult=2.0,         # stop initial = 2x ATR
                 atr_target_mult=5.0,       # target loin (vrai exit = trailing)
                 volume_ratio_min=1.2,      # volume > 1.2x moyenne pour confirmer breakout
                 rsi_max_long=78,           # RSI max pour long (pas d'entree si deja overbought)
                 rsi_min_short=22):
        self.breakout_lookback = breakout_lookback
        self.min_adx = min_adx
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.volume_ratio_min = volume_ratio_min
        self.rsi_max_long = rsi_max_long
        self.rsi_min_short = rsi_min_short

    def analyze(self, symbol: str, ohlcv: OHLCV, context: dict) -> Optional[Signal]:
        min_len = self.breakout_lookback + 50
        if len(ohlcv.close) < min_len:
            return None

        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low
        volume = ohlcv.volume

        # Indicateurs
        atr_vals = atr(high, low, close, 14)
        adx_vals = adx(high, low, close, 14)
        rsi_vals = rsi(close, 14)
        ema20 = ema(close, 20)
        ema50 = ema(close, 50)

        price = close[-1]
        curr_atr = atr_vals[-1]
        curr_adx = adx_vals[-1] if not np.isnan(adx_vals[-1]) else 0
        curr_rsi = rsi_vals[-1] if not np.isnan(rsi_vals[-1]) else 50
        curr_ema20 = ema20[-1]
        curr_ema50 = ema50[-1]

        if np.isnan(curr_atr) or curr_atr <= 0:
            return None

        # === FILTRE 1: Trend confirme sur la TF courante ===
        if curr_adx < self.min_adx:
            return None

        # === FILTRE 1b: ADX en HAUSSE sur les 5 dernieres bougies ===
        # Evite d'entrer sur un trend qui s'essouffle (ADX en declin = trend mature)
        if len(adx_vals) >= 6:
            adx_5_ago = adx_vals[-6]
            if not np.isnan(adx_5_ago) and curr_adx <= adx_5_ago:
                return None  # ADX en baisse ou stagnation = trend faiblissant

        # === FILTRE 2: Alignement EMA STRICT ===
        # Les 2 conditions doivent matcher: EMA cross ET price du bon cote
        # Auparavant on acceptait l'un ou l'autre, maintenant les deux obligatoires
        uptrend = curr_ema20 > curr_ema50 and price > curr_ema50
        downtrend = curr_ema20 < curr_ema50 and price < curr_ema50

        if not (uptrend or downtrend):
            return None  # Configuration mixte (EMA bullish mais price sous EMA50, ou inverse)

        # === FILTRE 2b: EMA20 doit aussi etre du bon cote du prix ===
        # Renforce la conviction directionnelle
        if uptrend and price < curr_ema20:
            return None  # En uptrend, le prix doit etre au-dessus EMA20 (sinon c'est un pullback profond)
        if downtrend and price > curr_ema20:
            return None  # En downtrend, le prix doit etre sous EMA20

        # === FILTRE 3: Breakout du range recent ===
        # On regarde le high/low des N DERNIERES bougies AVANT la courante
        # (la courante c'est [-1], donc on exclut [-1] pour eviter auto-ref)
        lookback_highs = high[-(self.breakout_lookback + 1):-1]
        lookback_lows = low[-(self.breakout_lookback + 1):-1]
        range_high = np.max(lookback_highs)
        range_low = np.min(lookback_lows)

        # === FILTRE 4: Volume confirmant - VRAI breakout ===
        # Les 3 dernieres bougies doivent avoir un volume au-dessus de la moyenne 20
        # (Avant: ratio 3 bars vs 20 bars - facile a fake)
        # (Maintenant: chaque des 3 dernieres bars > moyenne 20 - plus dur a fake)
        if len(volume) >= 23:
            vol_baseline = np.mean(volume[-23:-3])
            recent_3 = volume[-3:]
            # Au moins 2 des 3 dernieres bougies doivent avoir volume > 1.0x baseline
            bars_above = sum(1 for v in recent_3 if v > vol_baseline)
            if bars_above < 2:
                return None
            # ET le dernier bar doit avoir volume > 1.2x baseline (confirmation forte)
            if recent_3[-1] < self.volume_ratio_min * vol_baseline:
                return None
            volume_ratio = recent_3[-1] / vol_baseline
        else:
            return None  # Pas assez de data pour juger

        # === SIGNAL LONG ===
        if uptrend and price > range_high:
            if curr_rsi > self.rsi_max_long:
                return None  # Overbought, on rate

            strength = self._compute_strength(
                adx=curr_adx,
                volume_ratio=volume_ratio,
                trend_strength=(price - curr_ema50) / curr_ema50,  # distance a EMA50
            )

            return Signal(
                action="buy",
                strength=strength,
                entry_price=price,
                stop_loss=price - self.atr_stop_mult * curr_atr,
                take_profit=price + self.atr_target_mult * curr_atr,
                reasoning=f"Trend surfer LONG: 4h bullish (ADX {curr_adx:.0f}, "
                          f"EMA aligned), breakout {range_high:.2f} avec vol {volume_ratio:.2f}x. "
                          f"RSI {curr_rsi:.0f}. Trailing stop progressive activated.",
                strategy_name=self.name,
            )

        # === SIGNAL SHORT ===
        if downtrend and price < range_low:
            if curr_rsi < self.rsi_min_short:
                return None  # Oversold, on rate

            strength = self._compute_strength(
                adx=curr_adx,
                volume_ratio=volume_ratio,
                trend_strength=(curr_ema50 - price) / curr_ema50,
            )

            return Signal(
                action="sell",
                strength=strength,
                entry_price=price,
                stop_loss=price + self.atr_stop_mult * curr_atr,
                take_profit=price - self.atr_target_mult * curr_atr,
                reasoning=f"Trend surfer SHORT: 4h bearish (ADX {curr_adx:.0f}, "
                          f"EMA aligned), breakdown {range_low:.2f} avec vol {volume_ratio:.2f}x. "
                          f"RSI {curr_rsi:.0f}. Trailing stop progressive activated.",
                strategy_name=self.name,
            )

        return None

    def _compute_strength(self, adx: float, volume_ratio: float, trend_strength: float) -> float:
        """
        Strength:
          - 40% ADX (force du trend)
          - 30% volume_ratio (confirmation breakout)
          - 30% trend_strength (ecart a EMA50, plus c'est loin plus c'est convaincu)

        Min 0.50 pour passer le filtre global (0.45 dans main.py).
        """
        adx_score = min((adx - self.min_adx) / 30, 1.0)
        vol_score = min((volume_ratio - self.volume_ratio_min) / 2.0, 1.0)
        trend_score = min(abs(trend_strength) * 20, 1.0)  # 5% d'ecart = max score

        raw = 0.40 * adx_score + 0.30 * vol_score + 0.30 * trend_score
        return max(0.0, min(1.0, 0.50 + 0.45 * raw))
