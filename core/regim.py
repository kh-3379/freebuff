# =================================================================================
# regim.py - ULTIMATE COMPLETE VERSION v15.0
# =================================================================================
# ✅ THREE-SOURCE REGIME FUSION WITH DYNAMIC WEIGHTING
# ✅ COMPLETE CLASSIFICATION TARGETS INTEGRATION
# ✅ PHASE-BASED LOGGING WITH FULL TRACEABILITY
# ✅ ZERO DATA LEAKAGE | PRODUCTION READY
# =================================================================================

from __future__ import annotations
from core.global_method import universal_load, get_active_symbol_timeframe
from core.imports import *
from core.logger import AdvancedTradeLogger

logger = AdvancedTradeLogger(__name__)

from main import folds as NUM_FOLDS
from main import sum_folds as TOTAL_FOLDS
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import hashlib
import json
import traceback
import time
import asyncio
from pathlib import Path
from scipy.stats import entropy

try:
    from Learning_signal_Management import (
        DynamicWeightingSystem as _LSM_DynamicWeightingSystem,
        VolatilityRegimeDetector as _LSM_VolatilityRegimeDetector,
        SymbolDNAEngine as _LSM_SymbolDNAEngine,
    )
    _LEARNING_MGMT_AVAILABLE = True
except Exception as _lsm_err:
    _LSM_DynamicWeightingSystem = None
    _LSM_VolatilityRegimeDetector = None
    _LSM_SymbolDNAEngine = None
    _LEARNING_MGMT_AVAILABLE = False
    logger.warning(f"[regim] Learning_signal_Management not fully available: {_lsm_err}")

# =================================================================================
# CONSTANTS & CONFIGURATION
# =================================================================================

PROJECT_TARGETS = [
    'price_return',
    'price_direction',
    'trend_strength',
    'volatility_regime',
    'market_regime'
]

# وزن‌های هر تارگت برای تشخیص رژیم
TARGET_WEIGHTS = {
    'market_regime': 0.35,
    'trend_strength': 0.25,
    'price_direction': 0.20,
    'volatility_regime': 0.12,
    'price_return': 0.08
}

# وزن‌های سه منبع (پیش‌فرض - در runtime بر اساس عملکرد تنظیم می‌شوند)
DEFAULT_SOURCE_WEIGHTS = {
    'classification': 0.30,
    'adaptive_ta': 0.45,
    'traditional': 0.25
}

# آستانه‌ها
MIN_CONFIDENCE_THRESHOLD = 0.65
HIGH_CONFIDENCE_THRESHOLD = 0.75
CRITICAL_CONFIDENCE_THRESHOLD = 0.85
MIN_SOURCE_CONFIDENCE = 0.25
CACHE_EXPIRY_HOURS = 12
CACHE_DIR = Path("cache")


# =================================================================================
# ENUMS & DATA CLASSES
# =================================================================================

class MarketRegime(Enum):
    """
    ✅ V3 — Enum کامل با پشتیبانی از تمام نگاشت‌های
    regim.py + Learning_signal_Management + FORWARD_TEST_LIVE.
    """
    # ─── Core canonical ──────────────────────────────────────────────────
    BULL_TRENDING      = "bull_trending"
    BEAR_TRENDING      = "bear_trending"
    TRENDING_UP        = "trending_up"
    TRENDING_DOWN      = "trending_down"
    TRENDING           = "trending"
    SIDEWAYS_RANGING   = "sideways_ranging"
    RANGING            = "ranging"
    RANGE              = "range"
    HIGH_VOLATILITY    = "high_volatility"
    LOW_VOLATILITY     = "low_volatility"
    BREAKOUT           = "breakout"
    CRISIS_FLASH       = "crisis_flash"
    VOLATILE           = "volatile"

    # ─── Trending variants (DynamicWeightingSystem) ──────────────────────
    TRENDING_STRONG     = "trending_strong"
    TRENDING_WEAK       = "trending_weak"
    STRONG_TRENDING_UP  = "strong_trending_up"
    STRONG_TRENDING_DOWN= "strong_trending_down"
    WEAK_TRENDING_UP    = "weak_trending_up"
    WEAK_TRENDING_DOWN  = "weak_trending_down"

    # ─── Volatility variants (VolatilityRegimeDetector) ──────────────────
    NORMAL                 = "normal"
    COMPRESSION            = "compression"
    EXPANSION              = "expansion"
    VOLATILITY_EXPANSION   = "volatility_expansion"
    VOLATILITY_COMPRESSION = "volatility_compression"
    SHOCK                  = "shock"
    EXTREME                = "extreme"
    LOW                    = "low"
    HIGH                   = "high"
    VOLATILE_RANGE         = "volatile_range"
    VOLATILE_TRENDING      = "volatile_trending"

    # ─── Reversal / event patterns ──────────────────────────────────────
    PANIC                = "panic"
    OVERBOUGHT_REVERSAL  = "overbought_reversal"
    OVERSOLD_REVERSAL    = "oversold_reversal"
    RECOVERY             = "recovery"
    CRASH                = "crash"

    # ─── Fallback ───────────────────────────────────────────────────────
    UNDEFINED = "undefined"
    UNKNOWN   = "unknown"
    MIXED     = "mixed"

    # ═══════════════════════════════════════════════════════════════════
    @classmethod
    def resolve(cls, raw: Any) -> "MarketRegime":
        if raw is None:
            return cls.UNDEFINED
        if isinstance(raw, cls):
            return raw

        raw_str = str(raw).upper().strip().replace("-", "_").replace(" ", "_")

        mapping = {
            # ─── Core ─────────────────────────────────────────────────
            "BULL_TRENDING":      cls.BULL_TRENDING,
            "BEAR_TRENDING":      cls.BEAR_TRENDING,
            "TRENDING_UP":        cls.TRENDING_UP,
            "TRENDING_DOWN":      cls.TRENDING_DOWN,
            "TRENDING":           cls.TRENDING_UP,
            "SIDEWAYS_RANGING":   cls.SIDEWAYS_RANGING,
            "SIDEWAYS":           cls.SIDEWAYS_RANGING,
            "RANGING":            cls.SIDEWAYS_RANGING,
            "RANGE":              cls.SIDEWAYS_RANGING,
            "HIGH_VOLATILITY":    cls.HIGH_VOLATILITY,
            "LOW_VOLATILITY":     cls.LOW_VOLATILITY,
            "BREAKOUT":           cls.BREAKOUT,
            "CRISIS_FLASH":       cls.CRISIS_FLASH,
            "VOLATILE":           cls.HIGH_VOLATILITY,

            # ─── Trending variants ────────────────────────────────────
            "TRENDING_STRONG":       cls.TRENDING_STRONG,
            "STRONG_TRENDING":       cls.TRENDING_STRONG,
            "STRONG_TRENDING_UP":    cls.STRONG_TRENDING_UP,
            "STRONG_TRENDING_DOWN":  cls.STRONG_TRENDING_DOWN,
            "TRENDING_WEAK":         cls.TRENDING_WEAK,
            "WEAK_TRENDING":         cls.TRENDING_WEAK,
            "WEAK_TRENDING_UP":      cls.WEAK_TRENDING_UP,
            "WEAK_TRENDING_DOWN":    cls.WEAK_TRENDING_DOWN,
            "BULLISH":               cls.TRENDING_UP,
            "STRONG_BULLISH":        cls.BULL_TRENDING,
            "BEARISH":               cls.TRENDING_DOWN,
            "STRONG_BEARISH":        cls.BEAR_TRENDING,
            "UP":                    cls.TRENDING_UP,
            "DOWN":                  cls.TRENDING_DOWN,

            # ─── Volatility variants ──────────────────────────────────
            "NORMAL":                 cls.NORMAL,
            "COMPRESSION":            cls.COMPRESSION,
            "EXPANSION":              cls.EXPANSION,
            "VOLATILITY_EXPANSION":   cls.VOLATILITY_EXPANSION,
            "VOLATILITY_COMPRESSION": cls.VOLATILITY_COMPRESSION,
            "SHOCK":                  cls.SHOCK,
            "EXTREME":                cls.EXTREME,
            "LOW":                    cls.LOW,
            "HIGH":                   cls.HIGH,
            "VOLATILE_RANGE":         cls.VOLATILE_RANGE,
            "VOLATILE_TRENDING":      cls.VOLATILE_TRENDING,

            # ─── Reversal ────────────────────────────────────────────
            "PANIC":                  cls.PANIC,
            "OVERBOUGHT_REVERSAL":    cls.OVERBOUGHT_REVERSAL,
            "OVERSOLD_REVERSAL":      cls.OVERSOLD_REVERSAL,
            "RECOVERY":               cls.RECOVERY,
            "CRASH":                  cls.CRASH,

            # ─── Fallback ────────────────────────────────────────────
            "UNDEFINED": cls.UNDEFINED,
            "UNKNOWN":   cls.UNKNOWN,
            "MIXED":     cls.MIXED,
        }
        return mapping.get(raw_str, cls.UNDEFINED)

    # ═══════════════════════════════════════════════════════════════════
    @property
    def is_tradeable(self) -> bool:
        return self in {
            MarketRegime.BULL_TRENDING,
            MarketRegime.BEAR_TRENDING,
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.TRENDING_STRONG,
            MarketRegime.STRONG_TRENDING_UP,
            MarketRegime.STRONG_TRENDING_DOWN,
            MarketRegime.BREAKOUT,
        }

    @property
    def required_confidence(self) -> float:
        """آستانهٔ confidence لازم برای این رژیم."""
        table = {
            MarketRegime.BULL_TRENDING:       0.65,
            MarketRegime.BEAR_TRENDING:       0.65,
            MarketRegime.TRENDING_UP:         0.60,
            MarketRegime.TRENDING_DOWN:       0.60,
            MarketRegime.TRENDING_STRONG:     0.65,
            MarketRegime.BREAKOUT:            0.70,
            MarketRegime.HIGH_VOLATILITY:     0.75,
            MarketRegime.CRISIS_FLASH:        0.85,
            MarketRegime.PANIC:               0.85,
            MarketRegime.CRASH:               0.85,
            MarketRegime.SIDEWAYS_RANGING:    0.95,
            MarketRegime.NORMAL:              0.70,
        }
        return table.get(self, 0.70)

    @property
    def capital_allocation_multiplier(self) -> float:
        """ضریب تخصیص سرمایه."""
        table = {
            MarketRegime.BULL_TRENDING:       1.00,
            MarketRegime.BEAR_TRENDING:       1.00,
            MarketRegime.TRENDING_STRONG:     1.20,
            MarketRegime.STRONG_TRENDING_UP:  1.20,
            MarketRegime.STRONG_TRENDING_DOWN:1.20,
            MarketRegime.TRENDING_UP:         0.90,
            MarketRegime.TRENDING_DOWN:       0.90,
            MarketRegime.TRENDING_WEAK:       0.80,
            MarketRegime.BREAKOUT:            1.10,
            MarketRegime.SIDEWAYS_RANGING:    0.50,
            MarketRegime.HIGH_VOLATILITY:     0.60,
            MarketRegime.LOW_VOLATILITY:      0.70,
            MarketRegime.CRISIS_FLASH:        0.20,
            MarketRegime.PANIC:               0.15,
            MarketRegime.CRASH:               0.10,
            MarketRegime.NORMAL:              0.85,
            MarketRegime.UNDEFINED:           0.00,
            MarketRegime.UNKNOWN:             0.00,
        }
        return table.get(self, 0.50)
        
@dataclass
class RegimeSignal:
    """Complete regime detection output with confidence"""
    regime: MarketRegime
    confidence: float
    persistence_score: float
    volatility_regime: str
    trend_strength: float
    liquidity_regime: str
    detection_timestamp: datetime
    lookback_window: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_tradeable(self) -> bool:
        if self.regime == MarketRegime.UNDEFINED:
            return False
        return self.confidence >= self.regime.required_confidence

    @property
    def risk_adjusted_capital(self) -> float:
        return self.regime.capital_allocation_multiplier * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'persistence_score': self.persistence_score,
            'volatility_regime': self.volatility_regime,
            'liquidity_regime': self.liquidity_regime,
            'trend_strength': self.trend_strength,
            'detection_timestamp': self.detection_timestamp.isoformat(),
            'lookback_window': self.lookback_window,
            'metadata': self.metadata,
            'is_tradeable': self.is_tradeable,
            'risk_adjusted_capital': self.risk_adjusted_capital
        }


@dataclass
class SourceResult:
    """
    نتیجهٔ یک منبع تشخیص رژیم.

    ✅ FIX H-04:
        - افزودن پرچم صریح `is_available` که تنها بر اساس `status` تعیین می‌شود.
        - `is_valid` همچنان سخت‌گیرانه است (confidence + status).
        - فیلد `raw_regime` برای نگه‌داشتن رژیم قبل از هرگونه fallback.
    """
    name: str
    regime: str
    confidence: float
    weight: float
    details: Dict[str, Any] = field(default_factory=dict)
    status: str = "success"
    raw_regime: Optional[str] = None

    @property
    def is_available(self) -> bool:
        """منبع واقعاً در دسترس بوده است (نه fallback)."""
        return self.status == "success"

    @property
    def is_valid(self) -> bool:
        """منبع قابل رأی‌دادن است."""
        return (
            self.is_available
            and self.confidence >= 0.35
            and self.regime.upper() not in ("UNKNOWN", "UNDEFINED", "")
        )
# =================================================================================
# PHASE 1: CACHE MANAGEMENT FUNCTIONS
# =================================================================================

def _get_cache_key(symbol: str, timeframe: str, cache_type: str = "ultimate_regime") -> str:
    """تولید کلید کش یکتا"""
    return f"{symbol}_{timeframe}_{cache_type}"


def _get_cache_file_path(cache_key: str) -> Path:
    """دریافت مسیر فایل کش"""
    key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"ultimate_regime_{key_hash}.json"


def _get_source_performance_path(symbol: str) -> Path:
    """دریافت مسیر فایل عملکرد منابع"""
    return CACHE_DIR / f"source_performance_{symbol}.json"


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2 (NEW — DIRECT): _detect_market_regime_independent from FORWARD_TEST_LIVE
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_forward_test_regime(
    symbol: str, timeframe: str
) -> Optional[Dict[str, Any]]:
    """
    ✅ SOURCE 2 — FORWARD_TEST_LIVE._detect_market_regime_independent()

    🔧 FIX H-03:
        - اگر `hmmlearn` یا `FORWARD_TEST_LIVE` در دسترس نبود، به جای
          برگرداندن یک fallback جعلی (SIDEWAYS_RANGING با conf=0.35)،
          **`None`** برمی‌گرداند تا در fusion وزنش صفر شود.
        - همچنین یک بررسی صریح برای `hmmlearn` انجام می‌شود تا لاگ شفاف‌تر باشد.
    """
    phase = "SOURCE_2_FORWARD_TEST_INDEPENDENT"

    # ─── بررسی صریح وابستگی‌ها ─────────────────────────────────
    try:
        import hmmlearn  # noqa: F401
        _hmmlearn_ok = True
    except ImportError:
        _hmmlearn_ok = False

    if not _hmmlearn_ok:
        logger.warning(
            f"[{phase}] ⚠️ `hmmlearn` not installed — SOURCE 2 unavailable. "
            f"Install via: pip install hmmlearn"
        )
        return None  # 🔧 برنگرداندن fallback جعلی

    try:
        logger.info(f"[{phase}] 🎯 Running _detect_market_regime_independent for {symbol}")

        try:
            from FORWARD_TEST_LIVE import _detect_market_regime_independent
        except ImportError as ie:
            logger.warning(f"[{phase}] ⚠️ FORWARD_TEST_LIVE unavailable: {ie}")
            return None

        # ─── Load raw OHLCV ────────────────────────────────────────
        df: Optional[pd.DataFrame] = None
        fold_used: Optional[int] = None
        from main import sum_folds as TOTAL_FOLDS

        for fold_num in range(min(TOTAL_FOLDS, 10), 0, -1):
            try:
                raw = await universal_load(
                    data_type="raw", symbol=symbol, timeframe=timeframe,
                    fold=fold_num, split_type="train",
                )
                if raw is None:
                    continue
                if isinstance(raw, pd.DataFrame):
                    df = raw
                elif isinstance(raw, dict) and 'df' in raw:
                    df = raw['df']
                if df is not None and len(df) > 50:
                    df = _ensure_ohlcv_columns(df)
                    fold_used = fold_num
                    break
            except Exception:
                continue

        if df is None or df.empty or len(df) < 50:
            logger.debug(f"[{phase}] Insufficient data")
            return None

        close = df['close'].values.astype(np.float64)
        high = df['high'].values.astype(np.float64)
        low = df['low'].values.astype(np.float64)

        import inspect
        if inspect.iscoroutinefunction(_detect_market_regime_independent):
            raw_result = await _detect_market_regime_independent(
                symbol=symbol,close=close, high=high, low=low, df=df
            )
        else:
            raw_result = await asyncio.to_thread(
                _detect_market_regime_independent,symbol, close, high, low, df,
            )

        if not isinstance(raw_result, dict):
            return None

        # ─── Parse ────────────────────────────────────────────────
        regime_raw = str(raw_result.get('regime', 'SIDEWAYS_RANGING'))
        try:
            confidence = float(raw_result.get('confidence', 0.40) or 0.40)
        except (TypeError, ValueError):
            confidence = 0.40

        adx = float(raw_result.get('adx', 0.0) or 0.0)
        bullish_alignment = bool(raw_result.get('bullish_alignment', False))
        bearish_alignment = bool(raw_result.get('bearish_alignment', False))
        used_unified = bool(raw_result.get('used_unified', False))

        alias_map = {
            'STRONG_BULLISH': 'BULL_TRENDING',
            'BULLISH': 'TRENDING_UP',
            'STRONG_BEARISH': 'BEAR_TRENDING',
            'BEARISH': 'TRENDING_DOWN',
            'RANGE': 'SIDEWAYS_RANGING',
            'RANGING': 'SIDEWAYS_RANGING',
            'SIDEWAYS': 'SIDEWAYS_RANGING',
            'UNKNOWN': 'SIDEWAYS_RANGING',
            'UNDEFINED': 'SIDEWAYS_RANGING',
        }
        regime_final = alias_map.get(regime_raw.upper(), regime_raw.upper())

        if adx > 28:
            if bullish_alignment:
                regime_final = 'BULL_TRENDING'
                confidence = max(0.68, confidence + 0.08)
            elif bearish_alignment:
                regime_final = 'BEAR_TRENDING'
                confidence = max(0.68, confidence + 0.08)

        confidence = float(np.clip(confidence, 0.30, 0.92))

        is_hold = regime_final in (
            'SIDEWAYS_RANGING', 'LOW_VOLATILITY', 'HIGH_VOLATILITY'
        )
        hold_quality = (
            float(np.clip(confidence + 0.08, 0.55, 0.90)) if is_hold else None
        )

        logger.info(
            f"[{phase}] ✅ {regime_final} (conf={confidence:.1%}, "
            f"ADX={adx:.1f}, unified={used_unified})"
        )
        return {
            "regime": regime_final,
            "confidence": confidence,
            "hold_quality": hold_quality,
            "status": "success",
            "source": "forward_test_independent",
            "raw_regime": regime_raw,
            "details": {
                "raw_regime": regime_raw, "fold_used": fold_used,
                "adx": adx, "used_unified": used_unified,
                "bullish_alignment": bullish_alignment,
                "bearish_alignment": bearish_alignment,
            },
        }

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}", exc_info=True)
        return None       
        
async def _get_learning_mgmt_dwm_regime(
    symbol: str,
    timeframe: str,
    connector: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    ✅ SOURCE 3 — DynamicWeightingSystem.detect_market_regime()

    🔧 FIX H-04: در صورت unavailable بودن، `None` برمی‌گرداند نه fallback جعلی.

    🔧 NEW FIX (V2):
        - دریافت `connector` اختیاری برای set_connector روی DWS.
        - اگر connector پاس داده نشود، از رجیستری سراسری استفاده می‌کند.
        - الگوی defensive برای sync/async بودن متدهای DWS.
    """
    phase = "SOURCE_3_DWM_MARKET_REGIME"

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: بررسی در دسترس بودن DWS
    # ═══════════════════════════════════════════════════════════════════
    if not _LEARNING_MGMT_AVAILABLE or _LSM_DynamicWeightingSystem is None:
        logger.warning(
            f"[{phase}] ⚠️ DynamicWeightingSystem unavailable "
            f"(Learning_signal_Management missing)"
        )
        return None

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: Resolve connector
    # ═══════════════════════════════════════════════════════════════════
    resolved_connector = connector
    if resolved_connector is None:
        try:
            resolved_connector = get_global_connector()
        except Exception as conn_err:
            logger.debug(
                f"[{phase}] Could not resolve global connector: {conn_err}"
            )
            resolved_connector = None

    if resolved_connector is None:
        logger.warning(
            f"[{phase}] ⚠️ No connector available. "
            f"DWS may fail if it requires account/market access."
        )

    try:
        logger.info(f"[{phase}] 🎯 Running DynamicWeightingSystem for {symbol}")
        from main import sum_folds as TOTAL_FOLDS

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: بارگذاری DataFrame از کش
        # ═══════════════════════════════════════════════════════════════════
        df: Optional[pd.DataFrame] = None
        fold_used: Optional[int] = None

        for fold_num in range(min(TOTAL_FOLDS, 10), 0, -1):
            try:
                raw = await universal_load(
                    data_type="raw",
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold_num,
                    split_type="train",
                )
                if raw is None:
                    continue
                if isinstance(raw, pd.DataFrame):
                    df = raw
                elif isinstance(raw, dict) and 'df' in raw:
                    df = raw['df']
                if df is not None and len(df) > 50:
                    df = _ensure_ohlcv_columns(df)
                    fold_used = fold_num
                    break
            except Exception:
                continue

        if df is None or df.empty or len(df) < 50:
            logger.debug(
                f"[{phase}] Insufficient data for DWS "
                f"(symbol={symbol}, tf={timeframe})"
            )
            return None

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Instantiate DWS + set_connector
        # ═══════════════════════════════════════════════════════════════════
        try:
            dws = _LSM_DynamicWeightingSystem()
        except Exception as dws_init_err:
            logger.warning(
                f"[{phase}] ⚠️ DWS constructor failed: {dws_init_err}"
            )
            return None

        # 🔧 FIX: تزریق connector (اگر DWS متد set_connector داشته باشد)
        if resolved_connector is not None:
            try:
                if hasattr(dws, 'set_connector') and callable(dws.set_connector):
                    maybe_set = dws.set_connector(resolved_connector)
                    if asyncio.iscoroutine(maybe_set):
                        await maybe_set
                    logger.debug(
                        f"[{phase}] ✅ connector injected into DWS via set_connector()"
                    )
                elif hasattr(dws, 'connector'):
                    dws.connector = resolved_connector
                    logger.debug(
                        f"[{phase}] ✅ connector set via direct attribute"
                    )
                else:
                    logger.debug(
                        f"[{phase}] ℹ️ DWS has no set_connector method; "
                        f"skipping connector injection"
                    )
            except Exception as conn_inj_err:
                logger.warning(
                    f"[{phase}] ⚠️ connector injection failed: {conn_inj_err}"
                )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: بررسی وجود متد detect_market_regime
        # ═══════════════════════════════════════════════════════════════════
        if not hasattr(dws, 'detect_market_regime'):
            logger.debug(f"[{phase}] DWS has no detect_market_regime")
            return None

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 6: فراخوانی detect_market_regime با دفاع sync/async
        # ═══════════════════════════════════════════════════════════════════
        try:
            maybe = dws.detect_market_regime(df)
        except TypeError:
            # اگر امضای متد متفاوت باشد، چند variant را امتحان کن
            maybe = None
            for variant_kwargs in (
                {"df": df, "symbol": symbol, "timeframe": timeframe},
                {"data": df},
                {"market_data": df},
            ):
                try:
                    maybe = dws.detect_market_regime(**variant_kwargs)
                    break
                except TypeError:
                    continue
            if maybe is None:
                raise

        # 🔧 FIX: پشتیبانی از هر دو حالت sync/async
        if inspect.isawaitable(maybe):
            raw_result = await maybe
        else:
            raw_result = maybe

        if not raw_result:
            logger.debug(f"[{phase}] DWS returned falsy result")
            return None

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 7: Parse نتیجه (tuple یا dict)
        # ═══════════════════════════════════════════════════════════════════
        regime_raw = "SIDEWAYS_RANGING"
        confidence = 0.40
        details: Dict[str, Any] = {}

        if isinstance(raw_result, tuple):
            if len(raw_result) >= 1:
                regime_raw = str(raw_result[0] or "SIDEWAYS_RANGING")
            if len(raw_result) >= 2:
                try:
                    confidence = float(raw_result[1] or 0.40)
                except (TypeError, ValueError):
                    confidence = 0.40
            if len(raw_result) >= 3 and isinstance(raw_result[2], dict):
                details = raw_result[2]

        elif isinstance(raw_result, dict):
            regime_raw = str(raw_result.get('regime', 'SIDEWAYS_RANGING'))
            try:
                confidence = float(raw_result.get('confidence', 0.40) or 0.40)
            except (TypeError, ValueError):
                confidence = 0.40
            details = raw_result.get('details', {}) or {}

        else:
            logger.debug(
                f"[{phase}] Unexpected DWS result type: "
                f"{type(raw_result).__name__}"
            )
            return None

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 8: نرمال‌سازی نام رژیم
        # ═══════════════════════════════════════════════════════════════════
        alias_map = {
            'STRONG_BULLISH': 'BULL_TRENDING',
            'BULLISH': 'TRENDING_UP',
            'STRONG_BEARISH': 'BEAR_TRENDING',
            'BEARISH': 'TRENDING_DOWN',
            'RANGE': 'SIDEWAYS_RANGING',
            'RANGING': 'SIDEWAYS_RANGING',
            'SIDEWAYS': 'SIDEWAYS_RANGING',
            'VOLATILE': 'HIGH_VOLATILITY',
            'VOLATILE_RANGE': 'HIGH_VOLATILITY',
            'UNKNOWN': 'SIDEWAYS_RANGING',
            'UNDEFINED': 'SIDEWAYS_RANGING',
        }
        regime_final = alias_map.get(regime_raw.upper(), regime_raw.upper())
        confidence = float(np.clip(confidence, 0.30, 0.92))

        is_hold = regime_final in (
            'SIDEWAYS_RANGING', 'LOW_VOLATILITY', 'HIGH_VOLATILITY'
        )
        hold_quality = (
            float(np.clip(confidence + 0.08, 0.55, 0.90)) if is_hold else None
        )

        logger.info(f"[{phase}] ✅ {regime_final} (conf={confidence:.1%})")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 9: خروجی استاندارد
        # ═══════════════════════════════════════════════════════════════════
        return {
            "regime": regime_final,
            "confidence": confidence,
            "hold_quality": hold_quality,
            "status": "success",
            "source": "dwm_market_regime",
            "raw_regime": regime_raw,
            "details": {
                "raw_regime": regime_raw,
                "fold_used": fold_used,
                "adx": details.get('adx', 0.0),
                "volatility": details.get('atr_volatility', 0.0),
                "connector_injected": resolved_connector is not None,
            },
        }

    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}", exc_info=True)
        return None
        
async def _load_cached_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """✅ FIXED V3: بارگذاری cache با نرمال‌سازی timezone."""
    phase = "CACHE_LOAD_V3"
    try:
        if not CACHE_DIR.exists():
            return None

        cache_file = _get_cache_file_path(cache_key)
        if not cache_file.exists():
            return None

        with open(cache_file, 'r', encoding='utf-8') as f:
            cached = json.load(f)

        timestamp_str = cached.get('timestamp', '')
        if not timestamp_str:
            return None

        # ─── Normalize timestamp ───────────────────────────────────────
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        try:
            cached_time = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            return None

        if cached_time.tzinfo is None:
            cached_time = cached_time.replace(tzinfo=timezone.utc)
        else:
            cached_time = cached_time.astimezone(timezone.utc)

        now_utc = datetime.now(timezone.utc)
        age_hours = (now_utc - cached_time).total_seconds() / 3600.0

        if age_hours > CACHE_EXPIRY_HOURS:
            logger.debug(f"[{phase}] Cache expired: {age_hours:.1f}h")
            return None

        cached['is_valid'] = True
        cached['cache_age_hours'] = age_hours
        cached['cache_file_path'] = str(cache_file)
        cached['used_cache'] = True
        logger.info(f"[{phase}] ✅ Cache hit ({age_hours:.1f}h old)")
        return cached

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}")
        return None

async def _save_cached_result(cache_key: str, result: Dict[str, Any]) -> bool:
    """ذخیره نتیجه در کش"""
    phase = "CACHE_SAVE"
    
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = _get_cache_file_path(cache_key)
        
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now(timezone.utc).isoformat()
        result['used_cache'] = False
        
        temp_file = cache_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        
        if temp_file.exists():
            if cache_file.exists():
                cache_file.unlink()
            temp_file.rename(cache_file)
        
        logger.debug(f"[{phase}] ✅ Cache saved: {cache_file}")
        return True
        
    except Exception as e:
        logger.error(f"[{phase}] ❌ Error saving cache: {e}")
        return False


async def _load_source_performance(symbol: str) -> Dict[str, Dict[str, float]]:
    """
    بارگذاری عملکرد تاریخی منابع با ایمن‌سازی کامل timezone.

    این نسخه از exception ناشی از «naive vs aware datetime» جلوگیری می‌کند:
    هر timestamp ابتدا به UTC نرمال می‌شود و سپس با now(timezone.utc)
    مقایسه می‌گردد.
    """
    phase = "SOURCE_PERF_LOAD"

    try:
        perf_file = _get_source_performance_path(symbol)
        if not perf_file.exists():
            logger.debug(f"[{phase}] No performance file for {symbol}")
            return {}

        with open(perf_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        timestamp_str = data.get('timestamp', '')
        if timestamp_str:
            # ─── Normalize Z → +00:00 ────────────────────────────────
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'

            try:
                cached_time = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                logger.debug(f"[{phase}] Invalid timestamp format: {timestamp_str!r}")
                return data.get('performance', {})

            # ─── Normalize naive → aware UTC ─────────────────────────
            if cached_time.tzinfo is None:
                cached_time = cached_time.replace(tzinfo=timezone.utc)
            else:
                cached_time = cached_time.astimezone(timezone.utc)

            now_utc = datetime.now(timezone.utc)
            age_days = (now_utc - cached_time).total_seconds() / 86400.0

            if age_days > 14:
                logger.debug(
                    f"[{phase}] Performance data expired: "
                    f"{age_days:.1f} days old"
                )
                return {}

        return data.get('performance', {})

    except Exception as e:
        logger.debug(f"[{phase}] Error loading source performance: {e}")
        return {}
        
def _map_target_to_regime(
    target_key: str,
    class_value: Any,
    is_regression: bool = False,
) -> Optional[str]:
    """
    ✅ FIXED V3: نگاشت یکپارچه مقدار تارگت به رژیم استاندارد بازار.

    - از int(class_value) مقاوم‌سازی شده.
    - برای رگرسیون (price_return) علامت مقدار را می‌سنجد.
    - در صورت ناشناخته بودن، None برمی‌گرداند.
    """
    try:
        # ─── Regression branch ──────────────────────────────────────────
        if is_regression or target_key == 'price_return':
            try:
                v = float(class_value)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(v):
                return None
            if v > 0:
                return "BULL_TRENDING"
            if v < 0:
                return "BEAR_TRENDING"
            return "SIDEWAYS_RANGING"

        # ─── Classification branch ──────────────────────────────────────
        try:
            cv = int(round(float(class_value)))
        except (TypeError, ValueError):
            return None

        mappings: Dict[str, Dict[int, str]] = {
            'market_regime': {
                0: "SIDEWAYS_RANGING",
                1: "HIGH_VOLATILITY",
                2: "BULL_TRENDING",
                3: "BEAR_TRENDING",
            },
            'trend_strength': {
                0: "BEAR_TRENDING",
                1: "SIDEWAYS_RANGING",
                2: "BULL_TRENDING",
            },
            'price_direction': {
                0: "BEAR_TRENDING",      # SELL
                1: "SIDEWAYS_RANGING",   # HOLD / NEUTRAL
                2: "BULL_TRENDING",      # BUY
            },
            'volatility_regime': {
                0: "LOW_VOLATILITY",
                1: "SIDEWAYS_RANGING",
                2: "HIGH_VOLATILITY",
            },
        }

        if target_key in mappings:
            return mappings[target_key].get(cv)

        return None

    except Exception as e:
        logger.debug(f"[_map_target_to_regime] Error for {target_key}/{class_value}: {e}")
        return None

async def _save_source_performance(symbol: str, performance: Dict[str, Dict[str, float]]) -> bool:
    """ذخیره عملکرد تاریخی منابع"""
    phase = "SOURCE_PERF_SAVE"
    
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        perf_file = _get_source_performance_path(symbol)
        
        data = {
            "performance": performance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "version": "2.0"
        }
        
        temp_file = perf_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        if temp_file.exists():
            if perf_file.exists():
                perf_file.unlink()
            temp_file.rename(perf_file)
        
        logger.debug(f"[{phase}] ✅ Source performance saved")
        return True
        
    except Exception as e:
        logger.error(f"[{phase}] ❌ Error saving source performance: {e}")
        return False


def _calculate_dynamic_weights(
    source_results: List[SourceResult],
    market_context: Dict[str, Any],
    historical_performance: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """
    ✅ FIX H-04: منابعی که `is_available=False` هستند وزن صفر می‌گیرند
    و در نرمال‌سازی نهایی لحاظ نمی‌شوند.
    """
    phase = "DYNAMIC_WEIGHTS_V7_4REAL_SOURCES"

    BASE_PRIOR: Dict[str, float] = {
        'classification':            0.28,   # کاهش جزئی
        'forward_test_independent':  0.27,
        'dwm_market_regime':         0.20,
        'volatility_detector':       0.10,
        'symbol_dna':                0.15,   # ✅ جدید
    }

    try:
        weights: Dict[str, float] = {}

        for source in source_results:
            # 🔧 منابع unavailable وزن صفر می‌گیرند
            if not source.is_available:
                weights[source.name] = 0.0
                logger.debug(
                    f"[{phase}] Source '{source.name}' unavailable → weight=0.0"
                )
                continue

            conf_w = float(np.clip(source.confidence, 0.20, 0.95))
            if source.regime.upper() in ('UNKNOWN', 'UNDEFINED'):
                conf_w *= 0.55

            hist_acc = 0.60
            if source.name in historical_performance:
                regime_key = market_context.get('current_market_regime', 'default')
                if regime_key in historical_performance[source.name]:
                    hist_acc = float(historical_performance[source.name][regime_key])
                elif 'default' in historical_performance[source.name]:
                    hist_acc = float(historical_performance[source.name]['default'])
            hist_acc = float(np.clip(hist_acc, 0.20, 0.95))

            current_regime = str(
                market_context.get('current_market_regime', 'unknown')
            ).lower()
            is_trending_market = any(x in current_regime for x in ('trend', 'bull', 'bear'))
            is_ranging_market = any(x in current_regime for x in ('range', 'sideways'))
            is_volatile_market = any(x in current_regime for x in ('volatile', 'high_vol', 'panic', 'crisis'))

            src_regime = source.regime.lower()
            source_is_trending = any(x in src_regime for x in ('trend', 'bull', 'bear'))
            source_is_ranging = any(x in src_regime for x in ('range', 'sideways'))
            source_is_volatile = any(x in src_regime for x in ('volatile', 'high_vol', 'crisis', 'panic'))

            if source.regime.upper() in ('UNKNOWN', 'UNDEFINED'):
                regime_fit = 0.35
            elif is_trending_market and source_is_trending:
                regime_fit = 0.92
            elif is_ranging_market and source_is_ranging:
                regime_fit = 0.92
            elif is_volatile_market and source_is_volatile:
                regime_fit = 0.88
            elif is_trending_market and source_is_ranging:
                regime_fit = 0.55
            elif is_ranging_market and source_is_trending:
                regime_fit = 0.60
            else:
                regime_fit = 0.72

            final_w = conf_w * 0.40 + hist_acc * 0.30 + regime_fit * 0.30
            prior = BASE_PRIOR.get(source.name, 0.20)
            final_w = final_w * (1.0 + prior) * 0.5

            if source.regime.upper() in ('UNKNOWN', 'UNDEFINED'):
                final_w *= 0.65

            weights[source.name] = max(0.05, min(0.75, final_w))

        # 🔧 نرمال‌سازی فقط روی منابع available
        total = sum(v for v in weights.values() if v > 0)
        if total > 0:
            for name in weights:
                if weights[name] > 0:
                    weights[name] /= total
        else:
            # هیچ منبع معتبری نیست — همه صفر
            weights = {k: 0.0 for k in weights}

        logger.debug(
            f"[{phase}] Weights → "
            + ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
        )
        return weights

    except Exception as e:
        logger.error(f"[{phase}] Error: {e}", exc_info=True)
        return {k: 0.0 for k in BASE_PRIOR}
        
def _calculate_regime_entropy(
    vote_distribution: Dict[str, float]
) -> Tuple[float, float]:
    """
    محاسبه‌ی آنتروپی توزیع آرا (برای سنجش سطح اجماع).
    """
    try:
        if not vote_distribution:
            return 0.0, 1.0

        weights = np.array(list(vote_distribution.values()), dtype=np.float64)
        weights = weights / (weights.sum() + 1e-10)

        class_entropy = entropy(weights) if len(weights) > 1 else 0.0
        max_entropy = (
            np.log(len(vote_distribution)) if len(vote_distribution) > 1 else 1.0
        )
        normalized = class_entropy / max_entropy if max_entropy > 0 else 0.0
        return float(class_entropy), float(normalized)
    except Exception as e:
        logger.debug(f"Entropy calculation error: {e}")
        return 0.0, 1.0
        
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: CLASSIFICATION TARGETS — get_symbol_regime_classified
# ═══════════════════════════════════════════════════════════════════════════════

async def get_symbol_regime_classified(
    symbol: str,
    timeframe: str,
) -> Optional[Dict[str, Any]]:
    """
    ✅ SOURCE 1 — Classification Targets regime detector.

    دو مسیر استفاده می‌کند:
      A) تلاش برای خواندن پیش‌بینی‌های Classification Targets از
         universal_load (اگر موجود باشد).
      B) در صورت نبود، محاسبهٔ مستقیم features روی OHLCV خام و
         نگاشت آن‌ها به رژیم با استفاده از `_map_target_to_regime`.

    خروجی در قالب SourceResult-compatible dict است:
        {
          "regime": str, "confidence": float,
          "hold_quality": Optional[float],
          "status": "success" | "unavailable",
          "source": "classification_targets",
          "raw_regime": str,
          "details": {...},
        }
    """
    phase = "SOURCE_1_CLASSIFICATION_TARGETS"
    start_time = time.time()

    try:
        # ─── 1) Load raw OHLCV ─────────────────────────────────────────
        df: Optional[pd.DataFrame] = None
        fold_used: Optional[int] = None

        from main import sum_folds as TOTAL_FOLDS

        for fold_num in range(min(TOTAL_FOLDS, 10), 0, -1):
            try:
                raw = await universal_load(
                    data_type="raw",
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold_num,
                    split_type="train",
                )
                if raw is None:
                    continue
                if isinstance(raw, pd.DataFrame):
                    df = raw
                elif isinstance(raw, dict) and "df" in raw:
                    df = raw["df"]
                if df is not None and len(df) >= 50:
                    df = _ensure_ohlcv_columns(df)
                    fold_used = fold_num
                    break
            except Exception:
                continue

        if df is None or df.empty or len(df) < 50:
            logger.debug(f"[{phase}] Insufficient data for {symbol}.{timeframe}")
            return None

        # ═══════════════════════════════════════════════════════════════
        # مسیر A: تلاش برای بارگذاری پیش‌بینی‌های Classification Targets
        # ═══════════════════════════════════════════════════════════════
        predictions: Dict[str, Any] = {}
        for target in PROJECT_TARGETS:
            try:
                pred = await universal_load(
                    data_type="predictions",
                    symbol=symbol,
                    timeframe=timeframe,
                    target=target,
                    fold=fold_used,
                    split_type="train",
                )
                if pred is not None:
                    predictions[target] = pred
            except Exception:
                continue

        # ─── اگر پیش‌بینی‌ها موجود بودند، از نگاشت استفاده کن ──────────
        if predictions:
            return _fuse_classification_predictions(
                predictions, symbol, timeframe, fold_used, phase
            )

        # ═══════════════════════════════════════════════════════════════
        # مسیر B: محاسبهٔ مستقیم روی OHLCV خام
        # ═══════════════════════════════════════════════════════════════
        return await asyncio.to_thread(
            _compute_classification_regime_from_ohlcv,
            df, symbol, timeframe, fold_used,
        )

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}", exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
def _fuse_classification_predictions(
    predictions: Dict[str, Any],
    symbol: str,
    timeframe: str,
    fold_used: Optional[int],
    phase: str,
) -> Optional[Dict[str, Any]]:
    """ادغام پیش‌بینی‌های Classification Targets با وزن‌های TARGET_WEIGHTS."""
    try:
        votes: Dict[str, float] = {}
        total_weight = 0.0
        details: Dict[str, Any] = {}

        for target_key, pred in predictions.items():
            if target_key not in TARGET_WEIGHTS:
                continue

            weight = float(TARGET_WEIGHTS[target_key])

            # استخراج مقدار و نوع
            if isinstance(pred, dict):
                class_value = pred.get(
                    "prediction",
                    pred.get("class", pred.get("value", pred.get("label"))),
                )
                pred_conf = float(
                    pred.get("confidence", pred.get("proba", 0.5)) or 0.5
                )
                is_regression = bool(
                    pred.get("is_regression", target_key == "price_return")
                )
            else:
                class_value = pred
                pred_conf = 0.6
                is_regression = target_key == "price_return"

            regime_str = _map_target_to_regime(
                target_key, class_value, is_regression
            )
            if not regime_str:
                continue

            power = weight * pred_conf
            votes[regime_str] = votes.get(regime_str, 0.0) + power
            total_weight += power
            details[target_key] = {
                "predicted_class": str(class_value),
                "mapped_regime": regime_str,
                "confidence": pred_conf,
                "weight": weight,
            }

        if not votes or total_weight <= 0.0:
            return None

        best_regime = max(votes, key=votes.get)
        raw_conf = votes[best_regime] / total_weight
        confidence = float(np.clip(raw_conf, 0.35, 0.92))

        # بوست اگر چند تارگت هم‌رأی بودند
        agreeing = sum(
            1 for r in votes.values() if r > 0
        )
        if len(votes) >= 2:
            confidence = float(np.clip(confidence * 1.05, 0.35, 0.92))

        is_hold = best_regime in (
            "SIDEWAYS_RANGING", "LOW_VOLATILITY", "HIGH_VOLATILITY"
        )
        hold_quality = (
            float(np.clip(confidence + 0.05, 0.55, 0.90))
            if is_hold else None
        )

        logger.info(
            f"[{phase}] ✅ {best_regime} (conf={confidence:.1%}, "
            f"targets={len(details)})"
        )

        return {
            "regime": best_regime,
            "confidence": confidence,
            "hold_quality": hold_quality,
            "status": "success",
            "source": "classification_targets",
            "raw_regime": best_regime,
            "details": {
                "fold_used": fold_used,
                "votes": votes,
                "target_predictions": details,
                "path": "A_predictions",
            },
        }

    except Exception as e:
        logger.warning(f"[{phase}] fuse failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
def _compute_classification_regime_from_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    fold_used: Optional[int],
) -> Optional[Dict[str, Any]]:
    """
    مسیر B: محاسبهٔ regime از OHLCV خام با استراتژی‌های مشترک
    Classification Targets (ADX/ATR/EMA/BB).
    """
    phase = "SOURCE_1_CLASSIFICATION_RAW"
    try:
        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        n = len(close)

        if n < 50:
            return None

        # ─── ADX (simple) ──────────────────────────────────────────────
        period = 14
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr_series = pd.Series(tr).rolling(period, min_periods=1).mean().values
        atr_last = float(atr_series[-1]) if atr_series.size else 0.0

        up_move = np.zeros(n)
        dn_move = np.zeros(n)
        for i in range(1, n):
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            up_move[i] = up if (up > dn and up > 0) else 0.0
            dn_move[i] = dn if (dn > up and dn > 0) else 0.0

        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        for i in range(period, n):
            if atr_series[i] > 1e-8:
                plus_di[i] = 100.0 * (
                    np.mean(up_move[i - period + 1 : i + 1]) / atr_series[i]
                )
                minus_di[i] = 100.0 * (
                    np.mean(dn_move[i - period + 1 : i + 1]) / atr_series[i]
                )

        dx = np.zeros(n)
        for i in range(period, n):
            s = plus_di[i] + minus_di[i]
            if s > 1e-8:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s
        adx = float(np.mean(dx[-period:])) if n > period else 25.0
        if not np.isfinite(adx):
            adx = 25.0

        # ─── EMA alignment ─────────────────────────────────────────────
        close_s = pd.Series(close)
        ema9  = close_s.ewm(span=9,  adjust=False).mean().iloc[-1]
        ema21 = close_s.ewm(span=21, adjust=False).mean().iloc[-1]
        ema50 = (
            close_s.ewm(span=50, adjust=False).mean().iloc[-1]
            if n >= 50 else ema21
        )
        last = close[-1]
        bullish = (ema9 > ema21 > ema50) and last > ema21
        bearish = (ema9 < ema21 < ema50) and last < ema21

        # ─── Volatility (ATR% و std بازده) ────────────────────────────
        atr_pct = atr_last / (last + 1e-8)
        returns = np.diff(close) / (close[:-1] + 1e-8)
        recent_vol = float(np.std(returns[-50:])) if returns.size >= 10 else 0.0

        # ─── Bollinger width ──────────────────────────────────────────
        bb_period = 20
        if n >= bb_period:
            sma = close_s.rolling(bb_period).mean().iloc[-1]
            std = close_s.rolling(bb_period).std().iloc[-1]
            bb_width = float(
                (4.0 * std) / (sma + 1e-8)
            ) if sma > 0 else 0.0
        else:
            bb_width = 0.0

        # ══════════════════════════════════════════════════════════════
        # قواعد رژیم (هم‌راستا با Classification Targets)
        # ══════════════════════════════════════════════════════════════
        regime_str: str
        confidence: float

        if adx > 28 and bullish:
            regime_str, confidence = "BULL_TRENDING", min(
                0.90, 0.65 + (adx - 28) / 100.0
            )
        elif adx > 28 and bearish:
            regime_str, confidence = "BEAR_TRENDING", min(
                0.90, 0.65 + (adx - 28) / 100.0
            )
        elif adx > 24 and bullish:
            regime_str, confidence = "TRENDING_UP", 0.62
        elif adx > 24 and bearish:
            regime_str, confidence = "TRENDING_DOWN", 0.62
        elif recent_vol > 0.035 or atr_pct > 0.04:
            regime_str, confidence = "HIGH_VOLATILITY", 0.68
        elif recent_vol < 0.008 and atr_pct < 0.01:
            regime_str, confidence = "LOW_VOLATILITY", 0.62
        elif bb_width > 0.10 and recent_vol > 0.02:
            regime_str, confidence = "BREAKOUT", 0.65
        else:
            regime_str, confidence = "SIDEWAYS_RANGING", 0.55

        confidence = float(np.clip(confidence, 0.35, 0.92))

        is_hold = regime_str in (
            "SIDEWAYS_RANGING", "LOW_VOLATILITY", "HIGH_VOLATILITY"
        )
        hold_quality = (
            float(np.clip(confidence + 0.05, 0.55, 0.90))
            if is_hold else None
        )

        logger.info(
            f"[{phase}] ✅ {regime_str} (conf={confidence:.1%}, "
            f"ADX={adx:.1f}, atr%={atr_pct:.3%})"
        )

        return {
            "regime": regime_str,
            "confidence": confidence,
            "hold_quality": hold_quality,
            "status": "success",
            "source": "classification_targets",
            "raw_regime": regime_str,
            "details": {
                "fold_used": fold_used,
                "adx": float(adx),
                "atr_pct": float(atr_pct),
                "recent_vol": float(recent_vol),
                "bb_width": float(bb_width),
                "bullish_alignment": bool(bullish),
                "bearish_alignment": bool(bearish),
                "path": "B_ohlcv_rules",
            },
        }

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}", exc_info=True)
        return None
        
async def get_symbol_regime(
    override_cache: bool = False,
    min_confidence_final: float = 0.65,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    min_available_sources: int = 1,
    regime_source: str = "offline_single",
) -> Dict[str, Any]:
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║      SINGLE-SOURCE OFFLINE REGIME DETECTION — v21.0                 ║
    ║                                                                      ║
    ║  ✅ تنها یک منبع: classification_targets (offline از features)     ║
    ║  ✅ منابع آنلاین (DWM / FORWARD_TEST / VolatilityDetector /         ║
    ║     SymbolDNA) به‌طور کامل غیرفعال شده‌اند.                          ║
    ║  ✅ رژیم فقط از داده‌های ذخیره‌شده روی دیسک (raw OHLCV یا           ║
    ║     features_info ذخیره‌شده) محاسبه می‌شود — هیچ تماس شبکه‌ای.      ║
    ║  ✅ cache 6 ساعته با نرمال‌سازی timezone.                            ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Args:
        override_cache: نادیده‌گرفتن cache و محاسبهٔ مجدد.
        min_confidence_final: آستانهٔ حداقل اطمینان برای trading_ready.
        symbol: نماد. اگر None، از tf_decisions خوانده می‌شود.
        timeframe: تایم‌فریم. اگر None، از tf_decisions خوانده می‌شود.
        min_available_sources: در این نسخه عملاً ۱ است (تک‌منبع). برای سازگاری
                                با caller قدیمی نگه داشته شده.
        regime_source: صرفاً برای لاگ/versioning — مقدار پیش‌فرض "offline_single".

    Returns:
        dict با کلیدهای: symbol, timeframe, regime, confidence, hold_quality,
        trading_ready, recommendation, sources, source_contributions,
        dynamic_weights, regime_votes, sources_available, sources_total,
        entropy, normalized_entropy, processing_time_ms, used_cache, cache_hit,
        timestamp, version.
    """
    phase = "SINGLE_SOURCE_OFFLINE_V21"
    start_time = time.time()
    MIN_CONF = float(min_confidence_final or 0.65)

    try:
        logger.info("=" * 80)
        logger.info(f"[{phase}] 🚀 SINGLE-SOURCE OFFLINE REGIME v21.0")
        logger.info(f"[{phase}]    regime_source={regime_source} | min_conf={MIN_CONF:.2f}")
        logger.info("=" * 80)

        # ─── PHASE 0: Resolve symbol/timeframe ─────────────────────────────
        if symbol is None or timeframe is None:
            try:
                _sym, _tf = get_active_symbol_timeframe()
                symbol = symbol or _sym
                timeframe = timeframe or _tf
            except Exception as e:
                logger.error(f"[{phase}] get_active_symbol_timeframe failed: {e}")

        if not symbol or not timeframe:
            return {
                "symbol": symbol or "UNKNOWN",
                "timeframe": timeframe or "UNKNOWN",
                "regime": "UNDEFINED",
                "confidence": 0.0,
                "hold_quality": 0.50,
                "trading_ready": False,
                "recommendation": "HOLD / REDUCE POSITIONS",
                "sources": {},
                "source_contributions": [],
                "dynamic_weights": {},
                "regime_votes": {},
                "sources_available": 0,
                "sources_total": 0,
                "entropy": 0.0,
                "normalized_entropy": 1.0,
                "processing_time_ms": 0.0,
                "used_cache": False,
                "cache_hit": False,
                "error": "Symbol or timeframe is None",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "21.0_ErrorNoSymbol",
            }

        symbol = symbol.upper().strip()
        logger.info(f"[{phase}] 📊 Symbol: {symbol} | Timeframe: {timeframe}")

        # ─── PHASE 1: Cache check (6h) ─────────────────────────────────────
        cache_key = _get_cache_key(symbol, timeframe, "ultimate_regime_v21")
        if not override_cache:
            try:
                cached = await _load_cached_result(cache_key)
                if (
                    cached
                    and cached.get("is_valid")
                    and cached.get("cache_age_hours", 999.0) < 6.0
                ):
                    cached["used_cache"] = True
                    cached["cache_hit"] = True
                    logger.info(
                        f"[{phase}] ✅ Cache hit "
                        f"({cached.get('cache_age_hours', 0.0):.1f}h old)"
                    )
                    return cached
            except Exception as ce:
                logger.warning(f"[{phase}] Cache check error: {ce}")

        # ─── PHASE 2: Run the SINGLE offline source ────────────────────────
        raw_source_result: Optional[Dict[str, Any]] = None
        try:
            raw_source_result = await _compute_classification_regime_from_ohlcv_offline(
                symbol=symbol,
                timeframe=timeframe,
            )
        except Exception as src_err:
            logger.error(
                f"[{phase}] ❌ single source raised: {src_err}", exc_info=True
            )
            raw_source_result = None

        # ─── PHASE 3: Normalize to SourceResult ────────────────────────────
        if raw_source_result is None or not isinstance(raw_source_result, dict):
            logger.warning(
                f"[{phase}] ⚠️ Single source unavailable → UNDEFINED"
            )
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "regime": "UNDEFINED",
                "confidence": 0.0,
                "hold_quality": 0.30,
                "trading_ready": False,
                "recommendation": "HOLD / REDUCE POSITIONS",
                "fusion_explanation": "Single offline source unavailable",
                "entropy": 0.0,
                "normalized_entropy": 1.0,
                "sources": {
                    "classification": {
                        "regime": "UNDEFINED",
                        "confidence": 0.0,
                        "weight": 0.0,
                        "status": "unavailable",
                        "hold_quality": None,
                    }
                },
                "source_contributions": [],
                "dynamic_weights": {"classification": 0.0},
                "regime_votes": {},
                "sources_available": 0,
                "sources_total": 1,
                "min_available_sources": 1,
                "processing_time_ms": (time.time() - start_time) * 1000.0,
                "used_cache": False,
                "cache_hit": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "21.0_SingleSourceOffline_Unavailable",
            }
            try:
                await _save_cached_result(cache_key, result)
            except Exception:
                pass
            return result

        status = str(raw_source_result.get("status", "success")).lower()
        raw_regime = str(
            raw_source_result.get("regime", "SIDEWAYS_RANGING")
            or "SIDEWAYS_RANGING"
        ).upper()

        alias_map = {
            "UNKNOWN": "SIDEWAYS_RANGING",
            "UNDEFINED": "SIDEWAYS_RANGING",
            "RANGING": "SIDEWAYS_RANGING",
            "SIDEWAYS": "SIDEWAYS_RANGING",
        }
        regime_final = alias_map.get(raw_regime, raw_regime)

        try:
            confidence = float(raw_source_result.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        hold_quality = raw_source_result.get("hold_quality", None)
        if hold_quality is not None:
            try:
                hold_quality = float(hold_quality)
            except (TypeError, ValueError):
                hold_quality = None

        # ─── PHASE 4: Trading ready ────────────────────────────────────────
        is_hold_regime = regime_final in (
            "SIDEWAYS_RANGING",
            "LOW_VOLATILITY",
            "HIGH_VOLATILITY",
            "UNDEFINED",
            "UNKNOWN",
        )
        trading_ready = (
            status == "success"
            and confidence >= MIN_CONF
            and not is_hold_regime
        )

        if hold_quality is None and is_hold_regime:
            hold_quality = float(np.clip(0.55 + confidence * 0.30, 0.55, 0.90))

        # ─── PHASE 5: Entropy (single-source → 0) ─────────────────────────
        regime_votes = {regime_final: float(confidence)} if confidence > 0 else {}

        # ─── PHASE 6: Build result ─────────────────────────────────────────
        elapsed_ms = (time.time() - start_time) * 1000.0
        source_details = raw_source_result.get("details", {}) or {}

        final_result: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": regime_final,
            "confidence": float(confidence),
            "hold_quality": hold_quality,
            "trading_ready": bool(trading_ready),
            "recommendation": (
                "TRADE" if trading_ready else "HOLD / REDUCE POSITIONS"
            ),
            "fusion_explanation": (
                f"Single-source offline | source=classification_targets | "
                f"status={status} | conf={confidence:.1%}"
            ),
            "entropy": 0.0,
            "normalized_entropy": 0.0,
            "sources": {
                "classification": {
                    "regime": regime_final,
                    "raw_regime": raw_source_result.get("raw_regime", raw_regime),
                    "confidence": confidence,
                    "weight": 1.0,
                    "status": status,
                    "hold_quality": hold_quality,
                }
            },
            "source_contributions": [
                {
                    "name": "classification",
                    "regime": regime_final,
                    "confidence": confidence,
                    "weight": 1.0,
                    "vote_power": confidence,
                    "status": status,
                    "hold_quality": hold_quality,
                }
            ],
            "dynamic_weights": {"classification": 1.0},
            "regime_votes": regime_votes,
            "sources_available": 1 if status == "success" else 0,
            "sources_total": 1,
            "min_available_sources": 1,
            "source_details": source_details,
            "processing_time_ms": elapsed_ms,
            "used_cache": False,
            "cache_hit": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "21.0_SingleSourceOffline",
        }

        # ─── PHASE 7: Cache save ──────────────────────────────────────────
        try:
            await _save_cached_result(cache_key, final_result)
        except Exception as ce:
            logger.warning(f"[{phase}] Cache save error: {ce}")

        # ─── PHASE 8: Log ─────────────────────────────────────────────────
        logger.info(f"[{phase}] {'═' * 65}")
        logger.info(
            f"[{phase}] 🎯 FINAL: {regime_final} (conf={confidence:.1%})"
        )
        logger.info(f"[{phase}]    Trading Ready: {trading_ready}")
        logger.info(f"[{phase}]    Hold Quality: {hold_quality}")
        logger.info(f"[{phase}]    ADX: {source_details.get('adx', 'n/a')}")
        logger.info(f"[{phase}]    ATR%: {source_details.get('atr_pct', 'n/a')}")
        logger.info(f"[{phase}]    Time: {elapsed_ms:.1f}ms")
        logger.info(f"[{phase}] {'═' * 65}")

        return final_result

    except Exception as e:
        logger.error(f"[{phase}] ❌ CRITICAL: {e}", exc_info=True)
        return {
            "symbol": symbol or "UNKNOWN",
            "timeframe": timeframe or "UNKNOWN",
            "regime": "UNDEFINED",
            "confidence": 0.0,
            "hold_quality": 0.30,
            "trading_ready": False,
            "recommendation": "HOLD / REDUCE POSITIONS",
            "error": str(e),
            "sources": {},
            "source_contributions": [],
            "dynamic_weights": {},
            "regime_votes": {},
            "sources_available": 0,
            "sources_total": 0,
            "entropy": 0.0,
            "normalized_entropy": 1.0,
            "processing_time_ms": (time.time() - start_time) * 1000.0,
            "used_cache": False,
            "cache_hit": False,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "21.0_Error",
        }

async def _compute_classification_regime_from_ohlcv_offline(
    symbol: str,
    timeframe: str,
) -> Optional[Dict[str, Any]]:
    """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║   SINGLE OFFLINE REGIME SOURCE — v21.0                              ║
    ║                                                                      ║
    ║  ✅ فقط از داده‌های ذخیره‌شده روی دیسک می‌خواند (raw H5 یا           ║
    ║     features_info.json) — هیچ تماس شبکه‌ای.                          ║
    ║  ✅ آستانه‌ها بر اساس percentile واقعی همان نماد (نه ثابت‌های      ║
    ║     هاردکد 28/25 که باعث confidence کاذب می‌شدند).                  ║
    ║  ✅ confidence کالیبره‌شده: cap سخت روی 0.82.                        ║
    ║  ✅ رأی‌گیری داخلی از ۳ سیگنال مستقل (ADX، EMA alignment،           ║
    ║     volatility regime) → رژیم نهایی.                                ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Returns:
        dict سازگار با SourceResult:
            {regime, confidence, hold_quality, status, source,
             raw_regime, details}
        یا None در صورت نبود داده.
    """
    phase = "SINGLE_OFFLINE_CLASSIFICATION_V21"
    try:
        from main import sum_folds as TOTAL_FOLDS

        # ─── PHASE 1: Load raw OHLCV (offline from disk) ──────────────────
        df: Optional[pd.DataFrame] = None
        fold_used: Optional[int] = None

        for fold_num in range(min(TOTAL_FOLDS, 10), 0, -1):
            try:
                raw = await universal_load(
                    data_type="raw",
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold_num,
                    split_type="train",
                )
                if raw is None:
                    continue
                if isinstance(raw, pd.DataFrame):
                    df = raw
                elif isinstance(raw, dict) and "df" in raw:
                    df = raw["df"]
                if df is not None and len(df) >= 100:
                    df = _ensure_ohlcv_columns(df)
                    if df is not None and not df.empty and len(df) >= 100:
                        fold_used = fold_num
                        break
                    df = None
            except Exception:
                continue

        if df is None or df.empty or len(df) < 100:
            logger.warning(
                f"[{phase}] ⚠️ Insufficient offline OHLCV for "
                f"{symbol}.{timeframe}"
            )
            return None

        # ─── PHASE 2: Arrays ──────────────────────────────────────────────
        close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=np.float64)
        high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=np.float64)
        low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=np.float64)

        finite_mask = np.isfinite(close) & np.isfinite(high) & np.isfinite(low)
        if not finite_mask.all():
            close = close[finite_mask]
            high = high[finite_mask]
            low = low[finite_mask]

        n = len(close)
        if n < 100:
            return None

        # ─── PHASE 3: ATR (Wilder) ────────────────────────────────────────
        period = 14
        tr = np.zeros(n, dtype=np.float64)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        atr = np.zeros(n, dtype=np.float64)
        atr[period] = np.mean(tr[1 : period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        atr_last = float(atr[-1]) if np.isfinite(atr[-1]) else 0.0
        last_close = float(close[-1])
        atr_pct = atr_last / (abs(last_close) + 1e-8) if last_close != 0 else 0.0

        # ─── PHASE 4: ADX (Wilder) ────────────────────────────────────────
        plus_dm = np.zeros(n, dtype=np.float64)
        minus_dm = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            plus_dm[i] = up if (up > dn and up > 0) else 0.0
            minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

        smooth_tr = np.zeros(n, dtype=np.float64)
        smooth_plus = np.zeros(n, dtype=np.float64)
        smooth_minus = np.zeros(n, dtype=np.float64)
        smooth_tr[period] = np.sum(tr[1 : period + 1])
        smooth_plus[period] = np.sum(plus_dm[1 : period + 1])
        smooth_minus[period] = np.sum(minus_dm[1 : period + 1])
        for i in range(period + 1, n):
            smooth_tr[i] = smooth_tr[i - 1] - (smooth_tr[i - 1] / period) + tr[i]
            smooth_plus[i] = smooth_plus[i - 1] - (smooth_plus[i - 1] / period) + plus_dm[i]
            smooth_minus[i] = smooth_minus[i - 1] - (smooth_minus[i - 1] / period) + minus_dm[i]

        plus_di = np.zeros(n, dtype=np.float64)
        minus_di = np.zeros(n, dtype=np.float64)
        for i in range(period, n):
            if smooth_tr[i] > 1e-10:
                plus_di[i] = 100.0 * smooth_plus[i] / smooth_tr[i]
                minus_di[i] = 100.0 * smooth_minus[i] / smooth_tr[i]

        dx = np.zeros(n, dtype=np.float64)
        for i in range(period, n):
            s = plus_di[i] + minus_di[i]
            if s > 1e-10:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s

        adx = np.zeros(n, dtype=np.float64)
        start_adx = min(2 * period, n - 1)
        if start_adx > period:
            adx[start_adx] = np.mean(dx[period : start_adx + 1])
        for i in range(start_adx + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

        adx_last = float(adx[-1]) if np.isfinite(adx[-1]) else 25.0

        # ─── PHASE 5: ADX percentile calibration (dynamic, per symbol) ────
        valid_adx = adx[np.isfinite(adx) & (adx > 0)]
        if valid_adx.size >= 50:
            adx_p50 = float(np.percentile(valid_adx, 50))
            adx_p70 = float(np.percentile(valid_adx, 70))
            adx_p85 = float(np.percentile(valid_adx, 85))
        else:
            adx_p50, adx_p70, adx_p85 = 20.0, 25.0, 32.0

        # ─── PHASE 6: EMA alignment ───────────────────────────────────────
        def _ema(arr: np.ndarray, span: int) -> np.ndarray:
            alpha = 2.0 / (span + 1)
            out = np.empty_like(arr)
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
            return out

        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)
        ema50 = _ema(close, 50) if n >= 50 else ema21

        bullish_align = bool(
            ema9[-1] > ema21[-1] > ema50[-1] and close[-1] > ema21[-1]
        )
        bearish_align = bool(
            ema9[-1] < ema21[-1] < ema50[-1] and close[-1] < ema21[-1]
        )

        # ─── PHASE 7: Volatility regime (percentile-based) ────────────────
        returns = np.diff(close) / (np.abs(close[:-1]) + 1e-8)
        roll_vol = pd.Series(returns).rolling(20, min_periods=10).std().to_numpy()
        valid_vol = roll_vol[np.isfinite(roll_vol)]
        if valid_vol.size >= 50:
            vol_p30 = float(np.percentile(valid_vol, 30))
            vol_p70 = float(np.percentile(valid_vol, 70))
        else:
            vol_p30, vol_p70 = 0.008, 0.020

        current_vol = float(roll_vol[-1]) if np.isfinite(roll_vol[-1]) else 0.0

        # ─── PHASE 8: Internal 3-signal voting ────────────────────────────
        votes: Dict[str, float] = {}
        signals: Dict[str, Any] = {}

        # Signal 1: ADX-based trend
        if adx_last >= adx_p85:
            if bullish_align:
                votes["BULL_TRENDING"] = votes.get("BULL_TRENDING", 0.0) + 1.0
                signals["adx_signal"] = "BULL_TRENDING"
            elif bearish_align:
                votes["BEAR_TRENDING"] = votes.get("BEAR_TRENDING", 0.0) + 1.0
                signals["adx_signal"] = "BEAR_TRENDING"
            else:
                votes["SIDEWAYS_RANGING"] = votes.get("SIDEWAYS_RANGING", 0.0) + 0.5
                signals["adx_signal"] = "SIDEWAYS_RANGING(high_adx_no_align)"
        elif adx_last >= adx_p70:
            if bullish_align:
                votes["TRENDING_UP"] = votes.get("TRENDING_UP", 0.0) + 0.8
                signals["adx_signal"] = "TRENDING_UP"
            elif bearish_align:
                votes["TRENDING_DOWN"] = votes.get("TRENDING_DOWN", 0.0) + 0.8
                signals["adx_signal"] = "TRENDING_DOWN"
            else:
                votes["SIDEWAYS_RANGING"] = votes.get("SIDEWAYS_RANGING", 0.0) + 0.6
                signals["adx_signal"] = "SIDEWAYS_RANGING"
        else:
            votes["SIDEWAYS_RANGING"] = votes.get("SIDEWAYS_RANGING", 0.0) + 0.7
            signals["adx_signal"] = "SIDEWAYS_RANGING(low_adx)"

        # Signal 2: EMA alignment
        if bullish_align:
            votes["BULL_TRENDING"] = votes.get("BULL_TRENDING", 0.0) + 0.6
            signals["ema_signal"] = "BULL_TRENDING"
        elif bearish_align:
            votes["BEAR_TRENDING"] = votes.get("BEAR_TRENDING", 0.0) + 0.6
            signals["ema_signal"] = "BEAR_TRENDING"
        else:
            votes["SIDEWAYS_RANGING"] = votes.get("SIDEWAYS_RANGING", 0.0) + 0.6
            signals["ema_signal"] = "SIDEWAYS_RANGING"

        # Signal 3: Volatility regime
        if current_vol > vol_p70:
            votes["HIGH_VOLATILITY"] = votes.get("HIGH_VOLATILITY", 0.0) + 0.6
            signals["vol_signal"] = "HIGH_VOLATILITY"
        elif current_vol < vol_p30:
            votes["LOW_VOLATILITY"] = votes.get("LOW_VOLATILITY", 0.0) + 0.6
            signals["vol_signal"] = "LOW_VOLATILITY"
        else:
            signals["vol_signal"] = "NORMAL_VOL"
            votes["SIDEWAYS_RANGING"] = votes.get("SIDEWAYS_RANGING", 0.0) + 0.3

        # ─── PHASE 9: Final regime + calibrated confidence ────────────────
        total_votes = sum(votes.values())
        if total_votes <= 0.0:
            return {
                "regime": "SIDEWAYS_RANGING",
                "confidence": 0.35,
                "hold_quality": 0.55,
                "status": "success",
                "source": "classification_targets",
                "raw_regime": "SIDEWAYS_RANGING",
                "details": {
                    "fold_used": fold_used,
                    "adx": float(adx_last),
                    "atr_pct": float(atr_pct),
                    "signals": signals,
                    "votes": votes,
                    "path": "offline_v21",
                },
            }

        best_regime = max(votes, key=votes.get)
        best_share = votes[best_regime] / total_votes

        # calibration: پایه بر اساس سهم رأی + تأیید سیگنال‌های هم‌راستا
        agreeing_signals = sum(
            1 for s in signals.values() if isinstance(s, str) and best_regime in s
        )
        base_conf = 0.35 + best_share * 0.50 + agreeing_signals * 0.05

        # جریمه اگر ADX بالا ولی EMA هم‌راستا نبود (جلوگیری از confidence کاذب)
        if best_regime in ("BULL_TRENDING", "BEAR_TRENDING", "TRENDING_UP", "TRENDING_DOWN"):
            if not (bullish_align or bearish_align):
                base_conf *= 0.70

        confidence = float(np.clip(base_conf, 0.30, 0.82))

        is_hold = best_regime in (
            "SIDEWAYS_RANGING",
            "LOW_VOLATILITY",
            "HIGH_VOLATILITY",
        )
        hold_quality = (
            float(np.clip(confidence + 0.10, 0.55, 0.88)) if is_hold else None
        )

        logger.info(
            f"[{phase}] ✅ {best_regime} | conf={confidence:.1%} | "
            f"ADX={adx_last:.1f}(p85={adx_p85:.1f}) | "
            f"ATR%={atr_pct:.3%} | vol={current_vol:.4f} | "
            f"signals={signals}"
        )

        return {
            "regime": best_regime,
            "confidence": confidence,
            "hold_quality": hold_quality,
            "status": "success",
            "source": "classification_targets",
            "raw_regime": best_regime,
            "details": {
                "fold_used": fold_used,
                "bars_analyzed": int(n),
                "adx": float(adx_last),
                "adx_p50": float(adx_p50),
                "adx_p70": float(adx_p70),
                "adx_p85": float(adx_p85),
                "atr_pct": float(atr_pct),
                "current_vol": float(current_vol),
                "vol_p30": float(vol_p30),
                "vol_p70": float(vol_p70),
                "bullish_alignment": bool(bullish_align),
                "bearish_alignment": bool(bearish_align),
                "signals": signals,
                "votes": votes,
                "path": "offline_v21",
            },
        }

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}", exc_info=True)
        return None
        
# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: SymbolDNAEngine — regime from DNA profile
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_symbol_dna_regime(
    symbol: str,
    timeframe: str,
) -> Optional[Dict[str, Any]]:
    """
    ✅ SOURCE 5 — SymbolDNAEngine regime extraction.

    از `SymbolDNAEngine._build_dna()` برای استخراج:
      - `regime_behavior` (argmax → رژیم پایه)
      - `volatility` + `trend_strength` → پالایش رژیم

    این منبع روی «DNA هر نماد» تمرکز دارد و برای تأیید یا تعدیل
    رأی سایر منابع ارزش بالایی دارد.
    """
    phase = "SOURCE_5_SYMBOL_DNA_REGIME"

    if not _LEARNING_MGMT_AVAILABLE or _LSM_SymbolDNAEngine is None:
        logger.warning(f"[{phase}] ⚠️ SymbolDNAEngine unavailable")
        return None

    try:
        from main import sum_folds as TOTAL_FOLDS

        # ─── Load raw OHLCV (DNA needs ≥500 bars) ─────────────────────
        df: Optional[pd.DataFrame] = None
        fold_used: Optional[int] = None

        for fold_num in range(min(TOTAL_FOLDS, 10), 0, -1):
            try:
                raw = await universal_load(
                    data_type="raw",
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold_num,
                    split_type="train",
                )
                if raw is None:
                    continue
                if isinstance(raw, pd.DataFrame):
                    df = raw
                elif isinstance(raw, dict) and "df" in raw:
                    df = raw["df"]
                if df is not None and len(df) >= 500:
                    df = _ensure_ohlcv_columns(df)
                    fold_used = fold_num
                    break
            except Exception:
                continue

        if df is None or df.empty or len(df) < 500:
            logger.debug(
                f"[{phase}] Insufficient data for DNA ({len(df) if df is not None else 0} bars)"
            )
            return None

        # ─── Build DNA (thread-safe) ──────────────────────────────────
        dna_engine = _LSM_SymbolDNAEngine()

        dna_dict = await dna_engine._build_dna(symbol, df)

        if not isinstance(dna_dict, dict):
            return None

        regime_behavior = dna_dict.get("regime_behavior", {}) or {}
        if not isinstance(regime_behavior, dict) or not regime_behavior:
            return None

        volatility = float(dna_dict.get("volatility", 0.02) or 0.02)
        trend_strength = float(dna_dict.get("trend_strength", 0.0) or 0.0)
        liquidity_score = float(dna_dict.get("liquidity_score", 50.0) or 50.0)

        # ─── argmax regime_behavior ──────────────────────────────────
        try:
            best_key = max(regime_behavior, key=lambda k: float(regime_behavior[k]))
            best_score = float(regime_behavior[best_key])
        except Exception:
            return None

        alias = {
            "NORMAL":   "SIDEWAYS_RANGING",
            "RANGING":  "SIDEWAYS_RANGING",
            "RANGE":    "SIDEWAYS_RANGING",
            "TRENDING": "TRENDING_UP",
            "VOLATILE": "HIGH_VOLATILITY",
            "BREAKOUT": "BREAKOUT",
        }
        regime_final = alias.get(best_key.upper(), best_key.upper())

        # ─── پالایش رژیم با volatility و trend_strength ────────────────
        # اگر regime پایه SIDEWAYS است ولی trend_strength بالا است،
        # به رژیم روندی ارتقا بده.
        if regime_final == "SIDEWAYS_RANGING":
            if trend_strength >= 0.6:
                regime_final = "TRENDING_UP"
            elif volatility >= 0.04:
                regime_final = "HIGH_VOLATILITY"

        if regime_final == "TRENDING_UP" and trend_strength >= 0.8:
            regime_final = "BULL_TRENDING"

        # ─── Confidence: از regime_behavior + کیفیت داده ─────────────
        confidence = 0.35 + best_score * 0.55
        if liquidity_score < 30:
            confidence *= 0.85
        confidence = float(np.clip(confidence, 0.30, 0.88))

        is_hold = regime_final in (
            "SIDEWAYS_RANGING", "LOW_VOLATILITY", "HIGH_VOLATILITY"
        )
        hold_quality = (
            float(np.clip(confidence + 0.05, 0.55, 0.88))
            if is_hold else None
        )

        logger.info(
            f"[{phase}] ✅ {regime_final} "
            f"(raw={best_key}, score={best_score:.2f}, "
            f"vol={volatility:.2%}, trend={trend_strength:.2f})"
        )

        return {
            "regime": regime_final,
            "confidence": confidence,
            "hold_quality": hold_quality,
            "status": "success",
            "source": "symbol_dna",
            "raw_regime": best_key,
            "details": {
                "raw_regime": best_key,
                "regime_behavior": regime_behavior,
                "volatility": volatility,
                "trend_strength": trend_strength,
                "liquidity_score": liquidity_score,
                "bars_analyzed": int(dna_dict.get("bars_analyzed", 0) or 0),
                "fold_used": fold_used,
            },
        }

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}", exc_info=True)
        return None
        
async def _get_learning_mgmt_regime(
    symbol: str, timeframe: str
) -> Optional[Dict[str, Any]]:
    """
    ✅ SOURCE 4 — VolatilityRegimeDetector (فقط نوسان)

    🔧 FIX H-04: در صورت unavailable، `None` برمی‌گرداند.
    """
    phase = "SOURCE_4_VOLATILITY_DETECTOR"

    if not _LEARNING_MGMT_AVAILABLE or _LSM_VolatilityRegimeDetector is None:
        logger.warning(f"[{phase}] ⚠️ VolatilityRegimeDetector unavailable")
        return None

    try:
        logger.info(f"[{phase}] 🎯 Running VolatilityRegimeDetector for {symbol}")
        from main import sum_folds as TOTAL_FOLDS

        df: Optional[pd.DataFrame] = None
        for fold_num in range(min(TOTAL_FOLDS, 10), 0, -1):
            try:
                raw = await universal_load(
                    data_type="raw", symbol=symbol, timeframe=timeframe,
                    fold=fold_num, split_type="train",
                )
                if raw is None:
                    continue
                if isinstance(raw, pd.DataFrame):
                    df = raw
                elif isinstance(raw, dict) and 'df' in raw:
                    df = raw['df']
                if df is not None and len(df) > 50:
                    df = _ensure_ohlcv_columns(df)
                    break
            except Exception:
                continue

        if df is None or df.empty or len(df) < 50:
            return None

        detector = _LSM_VolatilityRegimeDetector()
        raw_result: Optional[Dict[str, Any]] = None
        for method_name in ('detect_volatility_regime', 'detect_regime', 'detect'):
            if hasattr(detector, method_name):
                method = getattr(detector, method_name)
                try:
                    maybe = method(df)
                    if asyncio.iscoroutine(maybe):
                        maybe = await maybe
                    if isinstance(maybe, dict):
                        raw_result = maybe
                        break
                except Exception:
                    continue

        if not isinstance(raw_result, dict):
            return None

        vol_regime = str(raw_result.get('regime', 'NORMAL') or 'NORMAL').upper().strip()
        regime_map = {
            'LOW': 'LOW_VOLATILITY',
            'NORMAL': 'SIDEWAYS_RANGING',
            'HIGH': 'HIGH_VOLATILITY',
            'EXTREME': 'CRISIS_FLASH',
            'COMPRESSION': 'LOW_VOLATILITY',
            'EXPANSION': 'HIGH_VOLATILITY',
            'BREAKOUT': 'BREAKOUT',
            'SHOCK': 'CRISIS_FLASH',
        }
        regime = regime_map.get(vol_regime, 'SIDEWAYS_RANGING')

        try:
            raw_conf = float(raw_result.get('confidence', 0.40) or 0.40)
        except (TypeError, ValueError):
            raw_conf = 0.40

        # فقط نوسان → تخفیف 0.70
        confidence = float(np.clip(raw_conf * 0.70, 0.25, 0.70))
        is_hold = regime in ('SIDEWAYS_RANGING', 'LOW_VOLATILITY', 'HIGH_VOLATILITY')
        hold_quality = (
            float(np.clip(confidence + 0.15, 0.55, 0.88)) if is_hold else None
        )

        logger.info(f"[{phase}] ✅ {regime} (conf={confidence:.1%})")
        return {
            "regime": regime,
            "confidence": confidence,
            "hold_quality": hold_quality,
            "status": "success",
            "source": "volatility_detector",
            "raw_regime": vol_regime,
            "details": {
                "raw_vol_regime": vol_regime,
                "historical_volatility": raw_result.get('historical_volatility', 0.0),
            },
        }

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error: {e}", exc_info=True)
        return None
        
     
def _ensure_ohlcv_columns(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    ✅ FIXED V2: اطمینان از وجود ستون‌های OHLCV با تحمل خطای کامل.

    - اگر df=None یا خالی باشد، DataFrame خالی با ستون‌های استاندارد برمی‌گرداند.
    - اگر فقط یک ستون عددی موجود باشد، آن را close در نظر می‌گیرد.
    - هرگز exception پرتاب نمی‌کند؛ در بدترین حالت DataFrame خالی برمی‌گرداند.
    """
    try:
        if df is None:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        if df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        df = df.copy()

        # ─── Map columns case-insensitively ────────────────────────────
        column_mapping: Dict[str, str] = {}
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in ('close', 'c', 'last', 'price', 'adj close', 'adj_close'):
                column_mapping.setdefault(col, 'close')
            elif col_lower in ('open', 'o'):
                column_mapping.setdefault(col, 'open')
            elif col_lower in ('high', 'h'):
                column_mapping.setdefault(col, 'high')
            elif col_lower in ('low', 'l'):
                column_mapping.setdefault(col, 'low')
            elif col_lower in ('volume', 'vol', 'v', 'tick_volume', 'tickvol'):
                column_mapping.setdefault(col, 'volume')

        if column_mapping:
            df = df.rename(columns=column_mapping)

        # ─── Guarantee 'close' ──────────────────────────────────────────
        if 'close' not in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df['close'] = pd.to_numeric(df[numeric_cols[-1]], errors='coerce')
            else:
                df['close'] = np.nan

        # ─── Fill in missing OHLCV ──────────────────────────────────────
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df[['open', 'close']].max(axis=1)
        if 'low' not in df.columns:
            df['low'] = df[['open', 'close']].min(axis=1)
        if 'volume' not in df.columns:
            df['volume'] = 1.0

        # ─── Ensure numeric dtype ──────────────────────────────────────
        for c in ('open', 'high', 'low', 'close', 'volume'):
            if c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                except Exception:
                    pass

        # ─── Sanitize ──────────────────────────────────────────────────
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['close'])

        if df.empty:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        return df

    except Exception as e:
        logger.error(f"[_ensure_ohlcv_columns] ❌ Error: {e}", exc_info=True)
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
       

class MarketRegimeDetector:
    """
    🎯 ULTIMATE MARKET REGIME DETECTOR - FIXED VERSION
    ✅ رفع خطای 'WallStreetQuantFeatureEngine' object has no attribute 'extract_features'
    ✅ استفاده از روش جایگزین برای استخراج فیچرها
    """

    def __init__(self):
        logger.info("▶️ Initializing ULTIMATE Market Regime Detector v10.0 (Fixed)")
        from core.global_method import get_active_symbol_timeframe
        self.symbol, self.timeframe = get_active_symbol_timeframe()
        self.adx_trend_threshold = 22
        self.volatile_threshold = 0.012
        self.ranging_threshold = 0.005
        self.min_bars_required = 50
        self.crash_threshold = 0.12
        self.recovery_threshold = 0.08

        self.ema_fast = 9
        self.ema_slow = 21
        self.rsi_period = 14
        self.atr_period = 14
        self.bb_period = 20

        self.regime_cache = {}
        self.cache_timeout = 43200
        self.cache_hits = 0
        self.cache_misses = 0
        self.expiry_check_hours = 12
        self.expiry_check_seconds = self.expiry_check_hours * 3600

        self.MIN_CONFIDENCE_THRESHOLD = 0.70
        self.HIGH_CONFIDENCE_THRESHOLD = 0.85
        self.CRITICAL_CONFIDENCE_THRESHOLD = 0.50

        self.analyzer_weights = {
            'price': 1.3, 'trend': 1.5, 'volatility': 1.2, 'momentum': 1.3,
            'structure': 1.2, 'moving_averages': 1.4, 'bollinger': 1.1,
            'volume': 1.2, 'support_resistance': 1.1, 'divergence': 1.2,
            'stochastic': 0.9, 'cycle': 0.8, 'statistics': 0.8
        }

        self.feature_engine = None
        self._init_feature_engine()

        logger.info("✅ ULTIMATE Market Regime Detector initialized (v10.0 Fixed)")

    def _init_feature_engine(self):
        """Initialize feature engine - FIXED: استفاده از متد صحیح"""
        try:
            from core.SKEALING_FILE import AntiCorrelationFeatureEngine, WallStreetQuantFeatureEngine
            
            # ✅ FIXED: استفاده از AntiCorrelationFeatureEngine
            # یا WallStreetQuantFeatureEngine با متد صحیح
            self.feature_engine = AntiCorrelationFeatureEngine(self.symbol)
            
            # ✅ همچنین یک نمونه از WallStreetQuantFeatureEngine برای دسترسی به متدها
            self.quant_engine = WallStreetQuantFeatureEngine(self.symbol)
            
            logger.info("✅ Feature engines initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize feature engines: {e}")
            self.feature_engine = None
            self.quant_engine = None

    def set_context(self, symbol: str, timeframe: str = "1H"):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        logger.info(f"[{self.symbol}] ✅ Context set: {timeframe}")

    def _safe_filename(self, text: str) -> str:
        if not text:
            return "unknown"
        safe = str(text)
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '{', '}', "'", ',', '=', '+', '&', '%', '$', '#']:
            safe = safe.replace(char, '_')
        return safe.strip('_')[:100]

    def _get_column(self, df: pd.DataFrame, col_name: str) -> Optional[pd.Series]:
        try:
            if col_name in df.columns:
                return df[col_name]
            for col in df.columns:
                if str(col).lower() == col_name.lower():
                    return df[col]
            for col in df.columns:
                if col_name.lower() in str(col).lower():
                    return df[col]
            return None
        except Exception:
            return None

    def _ensure_ohlcv_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        column_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'close' in col_lower or col_lower == 'c':
                column_mapping[col] = 'close'
            elif 'open' in col_lower or col_lower == 'o':
                column_mapping[col] = 'open'
            elif 'high' in col_lower or col_lower == 'h':
                column_mapping[col] = 'high'
            elif 'low' in col_lower or col_lower == 'l':
                column_mapping[col] = 'low'
            elif 'volume' in col_lower or col_lower == 'v' or col_lower == 'tick_volume':
                column_mapping[col] = 'volume'

        if column_mapping:
            df = df.rename(columns=column_mapping)

        if 'close' not in df.columns:
            if len(df.columns) > 0:
                df['close'] = df.iloc[:, 0]
            else:
                return df
        if 'open' not in df.columns:
            df['open'] = df['close']
        if 'high' not in df.columns:
            df['high'] = df['close']
        if 'low' not in df.columns:
            df['low'] = df['close']
        if 'volume' not in df.columns:
            df['volume'] = 1.0
        return df

    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            result = df.copy()
            close = result['close'] if 'close' in result else result.iloc[:, 0]
            for period in [5, 10, 20, 30, 50, 100]:
                result[f'SMA_{period}'] = close.rolling(window=period, min_periods=1).mean()
            for period in [5, 10, 20, 30, 50]:
                result[f'EMA_{period}'] = close.ewm(span=period).mean()
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1e-10)
            result['RSI_14'] = 100 - (100 / (1 + rs))
            returns = close.pct_change()
            result['Volatility_20'] = returns.rolling(window=20).std()
            return result.fillna(0)
        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ Error in basic indicators: {e}")
            return df

    def _convert_to_dataframe(self, data: Any, symbol: str) -> Optional[pd.DataFrame]:
        try:
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, np.ndarray):
                return pd.DataFrame(data)
            elif isinstance(data, tuple) and len(data) > 0:
                first = data[0]
                if isinstance(first, pd.DataFrame):
                    return first
                elif isinstance(first, np.ndarray):
                    return pd.DataFrame(first)
            elif isinstance(data, dict):
                for key in ['data', 'df', 'prices', 'X']:
                    if key in data:
                        val = data[key]
                        if isinstance(val, pd.DataFrame):
                            return val
                        elif isinstance(val, np.ndarray):
                            return pd.DataFrame(val)
            try:
                df = pd.DataFrame(data)
                if not df.empty:
                    return df
            except:
                pass
            return None
        except Exception as e:
            logger.error(f"[{symbol}] Conversion error: {e}")
            return None

    async def _check_existing_files_within_12h(self, symbol: str, timeframe: str, require_fold_last: bool) -> Optional[Dict[str, Any]]:
        try:
            cache_dir = Path("cache")
            if not cache_dir.exists():
                return None
            current_time = datetime.now(timezone.utc)
            patterns = [
                f"{symbol}_{timeframe}__*.regime",
                f"{symbol.replace('/', '_')}_{timeframe}_*.regime",
                f"{symbol}*{timeframe}*.regime"
            ]
            found_files = []
            for pattern in patterns:
                for file_path in cache_dir.glob(pattern):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                        timestamp_str = file_data.get("timestamp") or file_data.get("detected_at")
                        if not timestamp_str:
                            continue
                        try:
                            file_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        except ValueError:
                            continue
                        time_diff = (current_time - file_time).total_seconds()
                        if time_diff < self.expiry_check_seconds:
                            if file_data.get("status") == "success":
                                found_files.append({
                                    "file_path": str(file_path),
                                    "timestamp": file_time,
                                    "age_hours": time_diff / 3600,
                                    "data": file_data
                                })
                    except Exception:
                        continue
            if found_files:
                found_files.sort(key=lambda x: x["timestamp"], reverse=True)
                latest = found_files[0]
                result = latest['data'].copy()
                result["loaded_from_cache_file"] = True
                result["cache_file_age_hours"] = latest['age_hours']
                result["cache_file_path"] = latest['file_path']
                result["cache_strategy"] = "12_hour_check"
                result["timestamp"] = current_time.isoformat()
                return result
            return None
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Error checking cache files: {e}")
            return None

    async def _save_regime_to_file(self, symbol: str, timeframe: str, result: Dict[str, Any]) -> bool:
        try:
            cache_dir = Path("cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            safe_symbol = self._safe_filename(symbol)
            safe_timeframe = self._safe_filename(timeframe)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_symbol}_{safe_timeframe}_{timestamp}.regime"
            file_path = cache_dir / filename
            result_to_save = result.copy()
            result_to_save["saved_at"] = datetime.now(timezone.utc).isoformat()
            result_to_save["file_path"] = str(file_path)
            result_to_save["cache_valid_hours"] = self.expiry_check_hours
            temp_file = file_path.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(result_to_save, f, ensure_ascii=False, indent=2, default=str)
            if sys.platform == 'win32':
                if file_path.exists():
                    file_path.unlink()
                temp_file.rename(file_path)
            else:
                temp_file.replace(file_path)
            logger.info(f"[{symbol}] 💾 Saved regime file: {filename}")
            return True
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Error saving regime file: {e}")
            return False

    async def detect_regime(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        require_fold_last: bool = False,
    ) -> Dict[str, Any]:
        """
        ✅ SOURCE 5 — Traditional Technical Analysis (FIXED V2)

        از `_detect_market_regime_independent` (FORWARD_TEST_LIVE) به عنوان
        موتور اصلی تشخیص استفاده می‌کند و در صورت عدم دسترسی، fallback
        داخلی را اجرا می‌کند.
        """
        phase = "MARKET_REGIME_DETECTOR_SOURCE5_FIXED_V2"

        try:
            from core.global_method import universal_load
            from main import sum_folds as TOTAL_FOLDS

            logger.info(
                f"[{symbol}] 🎯 Traditional TA (SOURCE 5) for tf={timeframe}"
            )

            symbol = symbol.upper().strip()
            if not symbol:
                raise ValueError("Symbol cannot be empty")

            actual_tf = timeframe or self.timeframe or "1H"
            self.set_context(symbol, actual_tf)

            # ─── PHASE 1: Cache file (12h) ─────────────────────────────
            existing = await self._check_existing_files_within_12h(
                symbol=symbol, timeframe=actual_tf, require_fold_last=require_fold_last
            )
            if existing and existing.get("confidence", 0) >= 0.45:
                self.cache_hits += 1
                existing["detection_strategy"] = "cached_12_hour_file"
                existing["source"] = "MarketRegimeDetector_SOURCE5"
                existing["status"] = "success"
                return existing

            # ─── PHASE 2: Load raw data ────────────────────────────────
            df_raw: Optional[pd.DataFrame] = None
            fold_used: Optional[int] = None

            folds_to_try = (
                [TOTAL_FOLDS] if require_fold_last
                else range(min(TOTAL_FOLDS, 10), 0, -1)
            )

            for fold_num in folds_to_try:
                try:
                    raw_data = await universal_load(
                        data_type="raw",
                        symbol=symbol,
                        timeframe=actual_tf,
                        fold=fold_num,
                        split_type="train",
                    )
                    if raw_data is None:
                        continue
                    if isinstance(raw_data, pd.DataFrame):
                        df_raw = raw_data
                    elif isinstance(raw_data, dict) and 'df' in raw_data:
                        df_raw = raw_data['df']

                    if df_raw is not None and len(df_raw) > 50:
                        df_raw = self._ensure_ohlcv_columns(df_raw)
                        fold_used = fold_num
                        logger.info(
                            f"[{symbol}] ✅ Loaded {len(df_raw)} rows from fold {fold_used}"
                        )
                        break
                except Exception as fold_err:
                    logger.debug(f"[{symbol}] Fold {fold_num} error: {fold_err}")
                    continue

            if df_raw is None or len(df_raw) < 50:
                logger.warning(f"[{symbol}] ⚠️ Insufficient data for traditional TA")
                return self._create_insufficient_data_response(
                    symbol, actual_tf, df_raw, require_fold_last
                )

            # ─── PHASE 3: Detect ───────────────────────────────────────
            try:
                close = df_raw['close'].values.astype(np.float64)
                high = df_raw['high'].values.astype(np.float64)
                low = df_raw['low'].values.astype(np.float64)

                regime_data: Optional[Dict[str, Any]] = None
                try:
                    import inspect
                    from FORWARD_TEST_LIVE import _detect_market_regime_independent

                    if inspect.iscoroutinefunction(_detect_market_regime_independent):
                        regime_data = await _detect_market_regime_independent(
                            symbol,close, high, low
                        )
                    else:
                        regime_data = _detect_market_regime_independent(
                            symbol=symbol, close=close, high=high, low=low, df=df
                        )
                except ImportError as ie:
                    logger.warning(
                        f"[{symbol}] _detect_market_regime_independent unavailable: {ie}"
                    )
                except Exception as ie:
                    logger.warning(f"[{symbol}] Independent detector error: {ie}")

                # ─── Fallback if needed ────────────────────────────────
                if not isinstance(regime_data, dict):
                    logger.info(f"[{symbol}] Using internal fallback regime detection")
                    regime_data = self._internal_regime_detection(close, high, low)

                regime_raw = regime_data.get('regime', 'RANGE')
                confidence = float(regime_data.get('confidence', 0.40) or 0.40)
                adx = float(regime_data.get('adx', 0.0) or 0.0)
                is_range = bool(regime_data.get('is_range', False))
                volatility = float(regime_data.get('volatility', 0.0) or 0.0)
                bullish_alignment = bool(regime_data.get('bullish_alignment', False))
                bearish_alignment = bool(regime_data.get('bearish_alignment', False))

                # ─── Map to canonical names ────────────────────────────
                regime_mapping = {
                    'STRONG_BULLISH': 'BULL_TRENDING',
                    'BULLISH': 'TRENDING_UP',
                    'STRONG_BEARISH': 'BEAR_TRENDING',
                    'BEARISH': 'TRENDING_DOWN',
                    'RANGE': 'SIDEWAYS_RANGING',
                    'RANGING': 'SIDEWAYS_RANGING',
                    'SIDEWAYS': 'SIDEWAYS_RANGING',
                    'VOLATILE_RANGE': 'HIGH_VOLATILITY',
                    'PANIC': 'CRISIS_FLASH',
                    'CRISIS_FLASH': 'CRISIS_FLASH',
                    'VOLATILE_BULLISH': 'BULL_TRENDING',
                    'VOLATILE_BEARISH': 'BEAR_TRENDING',
                    'BREAKOUT': 'BREAKOUT',
                    'UNKNOWN': 'SIDEWAYS_RANGING',
                    'UNDEFINED': 'SIDEWAYS_RANGING',
                    'TRENDING_UP': 'TRENDING_UP',
                    'TRENDING_DOWN': 'TRENDING_DOWN',
                    'BULL_TRENDING': 'BULL_TRENDING',
                    'BEAR_TRENDING': 'BEAR_TRENDING',
                    'SIDEWAYS_RANGING': 'SIDEWAYS_RANGING',
                    'HIGH_VOLATILITY': 'HIGH_VOLATILITY',
                    'LOW_VOLATILITY': 'LOW_VOLATILITY',
                }
                final_regime = regime_mapping.get(regime_raw, 'SIDEWAYS_RANGING')

                # ─── ADX-based override ────────────────────────────────
                if adx > 28:
                    if bullish_alignment:
                        final_regime = 'BULL_TRENDING'
                        confidence = max(0.65, confidence + 0.10)
                    elif bearish_alignment:
                        final_regime = 'BEAR_TRENDING'
                        confidence = max(0.65, confidence + 0.10)
                elif adx > 25:
                    price_change_20 = (
                        (close[-1] - close[-20]) / (close[-20] + 1e-8)
                        if len(close) >= 20 else 0.0
                    )
                    if price_change_20 > 0.02:
                        final_regime = 'TRENDING_UP'
                        confidence = max(0.55, confidence + 0.05)
                    elif price_change_20 < -0.02:
                        final_regime = 'TRENDING_DOWN'
                        confidence = max(0.55, confidence + 0.05)

                # ─── Never return UNKNOWN ──────────────────────────────
                if final_regime in ('UNKNOWN', 'UNDEFINED', None):
                    final_regime = 'SIDEWAYS_RANGING'
                    confidence = max(0.40, confidence)

                confidence = float(np.clip(confidence, 0.35, 0.92))
                is_hold = final_regime in (
                    'SIDEWAYS_RANGING', 'LOW_VOLATILITY', 'HIGH_VOLATILITY'
                )
                trading_ready = (
                    confidence >= 0.65
                    and final_regime not in (
                        'SIDEWAYS_RANGING', 'UNDEFINED', 'UNKNOWN'
                    )
                )
                hold_quality = (
                    float(np.clip(confidence + 0.05, 0.55, 0.90))
                    if is_hold else None
                )

                result = {
                    "symbol": symbol,
                    "regime": final_regime,
                    "timeframe": actual_tf,
                    "confidence": float(confidence),
                    "hold_quality": hold_quality,
                    "source": "MarketRegimeDetector_SOURCE5",
                    "explanation": (
                        f"Traditional TA from {len(df_raw)} bars | "
                        f"ADX={adx:.1f} | raw={regime_raw}"
                    ),
                    "details": {
                        "data_points": len(df_raw),
                        "detection_method": "independent_technical_analysis",
                        "fold_used": fold_used,
                        "adx": float(adx),
                        "is_range": bool(is_range),
                        "volatility": float(volatility),
                        "bullish_alignment": bool(bullish_alignment),
                        "bearish_alignment": bool(bearish_alignment),
                        "raw_regime": regime_raw,
                        "confidence_raw": float(regime_data.get('confidence', 0.40)),
                    },
                    "trading_ready": trading_ready,
                    "recommendation": (
                        "TRADE" if trading_ready else "HOLD / REDUCE POSITIONS"
                    ),
                    "status": "success",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "detector": "MarketRegimeDetector",
                    "version": "16.0_SOURCE5_Fixed",
                }

                await self._save_regime_to_file(symbol, actual_tf, result)
                logger.info(
                    f"[{symbol}] 🎯 SOURCE 5: {final_regime} "
                    f"(conf={confidence:.1%}, ADX={adx:.1f})"
                )
                return result

            except Exception as inner_e:
                logger.error(f"[{symbol}] ❌ Independent detection failed: {inner_e}")
                return self._create_error_response(
                    symbol, actual_tf, inner_e, require_fold_last
                )

        except Exception as e:
            logger.error(f"[{symbol}] ❌ detect_regime (SOURCE 5) error: {e}", exc_info=True)
            return self._create_error_response(
                symbol, timeframe or "1H", e, require_fold_last
            )

    def _internal_regime_detection(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Fallback داخلی اگر `_detect_market_regime_independent` در دسترس نبود.
        """
        try:
            n = len(close)
            if n < 30:
                return {
                    "regime": "SIDEWAYS_RANGING",
                    "confidence": 0.35,
                    "adx": 0.0,
                    "is_range": True,
                    "volatility": 0.0,
                    "bullish_alignment": False,
                    "bearish_alignment": False,
                }

            # ADX
            period = min(14, n // 3)
            tr = np.zeros(n)
            for i in range(1, n):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]),
                )
            atr = np.zeros(n)
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, n):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

            plus_dm = np.zeros(n)
            minus_dm = np.zeros(n)
            for i in range(1, n):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move

            plus_di = np.zeros(n)
            minus_di = np.zeros(n)
            for i in range(period, n):
                if atr[i] > 1e-8:
                    plus_di[i] = 100 * np.mean(plus_dm[i-period+1:i+1]) / atr[i]
                    minus_di[i] = 100 * np.mean(minus_dm[i-period+1:i+1]) / atr[i]

            dx = np.zeros(n)
            for i in range(period, n):
                di_sum = plus_di[i] + minus_di[i]
                if di_sum > 1e-8:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

            adx = np.zeros(n)
            idx = min(2 * period, n - 1)
            adx[idx] = np.mean(dx[period:idx+1]) if idx > period else 25.0
            for i in range(idx+1, n):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

            current_adx = float(adx[-1]) if np.isfinite(adx[-1]) else 25.0

            # EMA alignment
            def _ema(arr: np.ndarray, span: int) -> np.ndarray:
                alpha = 2.0 / (span + 1)
                out = np.zeros_like(arr)
                out[0] = arr[0]
                for i in range(1, len(arr)):
                    out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
                return out

            ema9 = _ema(close, 9)
            ema21 = _ema(close, 21)
            ema50 = _ema(close, 50) if n >= 50 else ema21

            price = close[-1]
            bullish = (ema9[-1] > ema21[-1] > ema50[-1]) and price > ema21[-1]
            bearish = (ema9[-1] < ema21[-1] < ema50[-1]) and price < ema21[-1]

            # Volatility
            returns = np.diff(close) / (close[:-1] + 1e-8)
            volatility = float(np.std(returns[-min(50, len(returns)):]))

            # Decide
            if current_adx > 28 and bullish:
                regime, conf = 'BULL_TRENDING', min(0.85, 0.60 + (current_adx - 28) / 100)
            elif current_adx > 28 and bearish:
                regime, conf = 'BEAR_TRENDING', min(0.85, 0.60 + (current_adx - 28) / 100)
            elif current_adx > 25 and bullish:
                regime, conf = 'TRENDING_UP', 0.62
            elif current_adx > 25 and bearish:
                regime, conf = 'TRENDING_DOWN', 0.62
            elif volatility > 0.035:
                regime, conf = 'HIGH_VOLATILITY', 0.65
            elif volatility < 0.008:
                regime, conf = 'LOW_VOLATILITY', 0.62
            else:
                regime, conf = 'SIDEWAYS_RANGING', 0.55

            return {
                "regime": regime,
                "confidence": conf,
                "adx": current_adx,
                "is_range": regime in ('SIDEWAYS_RANGING', 'HIGH_VOLATILITY', 'LOW_VOLATILITY'),
                "volatility": volatility,
                "bullish_alignment": bullish,
                "bearish_alignment": bearish,
            }
        except Exception as e:
            logger.error(f"[_internal_regime_detection] Error: {e}")
            return {
                "regime": "SIDEWAYS_RANGING",
                "confidence": 0.35,
                "adx": 0.0,
                "is_range": True,
                "volatility": 0.0,
                "bullish_alignment": False,
                "bearish_alignment": False,
            }

# ============================================================================
# EXPIRY DATE MANAGER
# ============================================================================

class ExpiryDateManager:
    """
    🎯 Professional Expiry Date Manager with 12-hour cache
    """

    def __init__(self):
        self.base_expiry_days = {
            "trending_up": 3,
            "trending_down": 3,
            "trending": 3,
            "high_volatility": 1,
            "low_volatility": 5,
            "ranging": 5,
            "crash": 1,
            "recovery": 2,
            "breakout": 2,
            "mixed": 2,
            "unknown": 2
        }

        self.volatility_multipliers = {
            "very_low": (0.0, 0.005, 1.8),
            "low": (0.005, 0.01, 1.4),
            "medium": (0.01, 0.02, 1.0),
            "high": (0.02, 0.05, 0.6),
            "very_high": (0.05, float('inf'), 0.4)
        }

        self.min_expiry_hours = 4
        self.max_expiry_days = 14
        self.expiry_cache = {}
        self.cache_timeout = 43200  # 12 hours
        self.expiry_check_hours = 12
        self.expiry_check_seconds = self.expiry_check_hours * 3600
        self.safety_margin_hours = 6
        self.market_close_days = [5, 6]  # Friday, Saturday

    async def calculate_expiry_date(
            self,
            symbol: str,
            timeframe: str,
            regime_info: Optional[Dict[str, Any]] = None,
            fold_number: int = TOTAL_FOLDS,
            expiry_type: str = "final"
    ) -> Dict[str, Any]:
        """
        Calculate expiry date with 12-hour cache strategy
        """
        logger.info(f"[{symbol}] 📅 Calculating expiry date")

        try:
            # Check cache
            cache_key = f"{symbol}_{timeframe}_{expiry_type}_{fold_number}"
            current_time = time.time()

            if cache_key in self.expiry_cache:
                cached = self.expiry_cache[cache_key]
                cache_age = current_time - cached.get("_cache_time", 0)

                if cache_age < self.cache_timeout and self._validate_cached_expiry(cached):
                    logger.info(f"[{symbol}] ✅ Using cached expiry")
                    return {k: v for k, v in cached.items() if not k.startswith('_')}

            # Get regime if not provided
            if regime_info is None:
                detector = MarketRegimeDetector()
                regime_info = await detector.detect_regime(symbol, timeframe)

            regime = regime_info.get("regime", "unknown")
            confidence = regime_info.get("confidence", 0.5)
            volatility = regime_info.get("details", {}).get("volatility", 0.01)

            # Calculate base days
            base_days = self.base_expiry_days.get(regime, 2)

            # Adjust for confidence
            if confidence < 0.4:
                base_days = min(base_days, 2)

            # Adjust for volatility
            vol_multiplier = self._get_volatility_multiplier(volatility)

            # Calculate final days
            raw_days = base_days * vol_multiplier
            min_days = self.min_expiry_hours / 24
            max_days = self.max_expiry_days * confidence
            final_days = max(min_days, min(raw_days, max_days))

            # Calculate expiry date
            expiry_date = await self._calculate_smart_expiry(
                datetime.now(timezone.utc),
                final_days
            )

            result = {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "expiry_date": expiry_date.isoformat(),
                "expiry_type": expiry_type,
                "days_to_expiry": round(final_days, 2),
                "hours_to_expiry": round(final_days * 24, 1),
                "regime_used": regime,
                "volatility_used": float(volatility),
                "volatility_category": self._get_volatility_category(volatility),
                "regime_confidence": float(confidence),
                "overall_confidence": float(confidence * 0.9),
                "calculation_details": {
                    "base_days": base_days,
                    "volatility_multiplier": vol_multiplier,
                    "raw_days": raw_days,
                    "min_days": min_days,
                    "max_days": max_days,
                    "final_days": final_days,
                    "fold_used": fold_number
                },
                "is_valid": True,
                "valid_until": expiry_date.isoformat(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "version": "3.0",
                "detector": "ExpiryDateManager"
            }

            # Cache result
            result_with_meta = {**result, "_cache_time": current_time}
            self.expiry_cache[cache_key] = result_with_meta

            logger.info(f"[{symbol}] ✅ Expiry: {expiry_date.strftime('%Y-%m-%d %H:%M')} UTC ({final_days:.1f} days)")

            return result

        except Exception as e:
            logger.error(f"[{symbol}] ❌ Expiry calculation error: {e}")
            return await self._create_emergency_expiry(symbol, timeframe)

    def _get_volatility_multiplier(self, volatility: float) -> float:
        """Get multiplier based on volatility"""
        for _, (min_v, max_v, mult) in self.volatility_multipliers.items():
            if min_v <= volatility < max_v:
                return mult
        return 1.0

    def _get_volatility_category(self, volatility: float) -> str:
        """Get volatility category"""
        for cat, (min_v, max_v, _) in self.volatility_multipliers.items():
            if min_v <= volatility < max_v:
                return cat
        return "medium"

    def _validate_cached_expiry(self, cached_data: Dict[str, Any]) -> bool:
        """Validate cached expiry data"""
        try:
            required = ['expiry_date', 'days_to_expiry', 'overall_confidence']
            if not all(k in cached_data for k in required):
                return False

            expiry_str = cached_data.get("expiry_date", "")
            if expiry_str.endswith('Z'):
                expiry_str = expiry_str[:-1] + '+00:00'

            expiry_date = datetime.fromisoformat(expiry_str)
            if expiry_date <= datetime.now(timezone.utc):
                return False

            days = cached_data.get("days_to_expiry", 0)
            if not (0 < days <= self.max_expiry_days):
                return False

            return True

        except Exception:
            return False

    async def _calculate_smart_expiry(self, start_date: datetime, days: float) -> datetime:
        """Calculate smart expiry with market hours consideration"""
        try:
            current = start_date
            remaining_hours = days * 24 + self.safety_margin_hours
            expiry_date = current + timedelta(hours=remaining_hours)

            # Adjust for weekend
            while expiry_date.weekday() in self.market_close_days:
                expiry_date += timedelta(days=1)

            # Adjust for market hours (8:00-20:00 UTC)
            if expiry_date.hour < 8:
                expiry_date = expiry_date.replace(hour=8, minute=0, second=0, microsecond=0)
            elif expiry_date.hour >= 20:
                expiry_date = expiry_date + timedelta(days=1)
                expiry_date = expiry_date.replace(hour=8, minute=0, second=0, microsecond=0)

            return expiry_date

        except Exception:
            return start_date + timedelta(days=days)

    async def _create_emergency_expiry(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Create emergency expiry"""
        emergency_date = datetime.now(timezone.utc) + timedelta(hours=24)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "expiry_date": emergency_date.isoformat(),
            "expiry_type": "emergency",
            "days_to_expiry": 1.0,
            "hours_to_expiry": 24.0,
            "regime_used": "unknown",
            "volatility_used": 0.01,
            "overall_confidence": 0.3,
            "is_valid": True,
            "valid_until": emergency_date.isoformat(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "3.0_emergency",
            "warning": "Emergency expiry - use with caution"
        }

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cache"""
        if symbol:
            keys = [k for k in self.expiry_cache if k.startswith(f"{symbol}_")]
            for key in keys:
                del self.expiry_cache[key]
        else:
            self.expiry_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "cache_size": len(self.expiry_cache),
            "settings": {
                "base_expiry_days": self.base_expiry_days,
                "min_expiry_hours": self.min_expiry_hours,
                "max_expiry_days": self.max_expiry_days
            },
            "timestamp": datetime.now().isoformat()
        }


async def validate_expiry_date(
        symbol: str,
        timeframe: str,
        existing_expiry: Dict[str, Any]
) -> Dict[str, Any]:
    """
    اعتبارسنجی تاریخ انقضای موجود - نسخه اصلاح شده

    🔄 تغییرات:
    - رفع مشکل امضای تابع calculate_expiry_date
    - استفاده صحیح از regime_info به جای existing_expiry
    """
    try:
        logger.info(f"🔍 اعتبارسنجی تاریخ انقضا برای {symbol}.{timeframe}")

        # فاز 1: بررسی آیا داده شامل اطلاعات فایل است
        file_path = existing_expiry.get("file_path", "")
        if file_path and file_path != "none":
            logger.info(f"📁 اعتبارسنجی برای فایل: {file_path}")

            # بررسی وجود فیزیکی فایل
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size
                logger.debug(f"📊 اندازه فایل: {file_size} بایت")
            else:
                logger.warning(f"⚠️ فایل ذکر شده وجود ندارد: {file_path}")

        # فاز 2: استخراج اطلاعات رژیم از existing_expiry
        regime_info = None
        if existing_expiry and isinstance(existing_expiry, dict):
            # ساخت یک دیکشنری رژیم از اطلاعات موجود
            regime_info = {
                "regime": existing_expiry.get("regime_used", "unknown"),
                "confidence": existing_expiry.get("regime_confidence", 0.5),
                "details": {
                    "volatility": existing_expiry.get("volatility_used", 0.01)
                }
            }

        # فاز 3: محاسبه تاریخ انقضای جدید با استفاده از رژیم موجود
        expiry_manager = ExpiryDateManager()

        # اصلاح: استفاده از پارامتر regime_info به جای existing_expiry
        validation_result = await expiry_manager.calculate_expiry_date(
            symbol=symbol,
            timeframe=timeframe,
            regime_info=regime_info  # اینجا اصلاح شد
        )

        # فاز 4: اعتبارسنجی تاریخ موجود در مقابل تاریخ جدید
        if existing_expiry and "expiry_date" in existing_expiry:
            try:
                existing_date_str = existing_expiry.get("expiry_date", "")
                if existing_date_str.endswith('Z'):
                    existing_date_str = existing_date_str[:-1] + '+00:00'

                existing_date = datetime.fromisoformat(existing_date_str)
                new_date = datetime.fromisoformat(validation_result["expiry_date"])

                # محاسبه تفاوت
                days_difference = abs((new_date - existing_date).days)

                # اگر اختلاف کمتر از 2 روز است، تاریخ موجود معتبر است
                if days_difference <= 2:
                    validation_result["is_valid"] = True
                    validation_result["should_recalculate"] = False
                    validation_result["validation_message"] = "Existing expiry is still valid"
                else:
                    validation_result["is_valid"] = False
                    validation_result["should_recalculate"] = True
                    validation_result["validation_message"] = f"Expiry date differs by {days_difference} days"

            except Exception as e:
                logger.warning(f"⚠️ خطا در مقایسه تاریخ‌ها: {e}")
                validation_result["is_valid"] = False
                validation_result["should_recalculate"] = True
                validation_result["validation_message"] = "Could not compare dates"

        # فاز 5: اضافه کردن اطلاعات فایل به نتیجه
        validation_result["original_file_path"] = file_path
        validation_result["file_exists"] = Path(file_path).exists() if file_path and file_path != "none" else False
        validation_result["original_expiry"] = existing_expiry.get("expiry_date",
                                                                   "unknown") if existing_expiry else "unknown"

        status_text = "معتبر" if validation_result.get('is_valid') else "نامعتبر"
        logger.info(f"✅ اعتبارسنجی انجام شد: {status_text}")

        if validation_result.get("should_recalculate"):
            logger.info("🔄 نیاز به محاسبه مجدد تاریخ انقضا")

        return validation_result

    except Exception as e:
        logger.error(f"❌ خطا در اعتبارسنجی تاریخ انقضا: {e}", exc_info=True)

        return {
            "is_valid": False,
            "reason": f"validation_error: {str(e)}",
            "should_recalculate": True,
            "original_file_path": existing_expiry.get("file_path", "unknown") if existing_expiry else "unknown",
            "file_exists": False,
            "confidence": 0.0,
            "error_details": str(e)[:200],
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

async def calculate_expiry_date(
        symbol: str,
        timeframe: str,
        regime_info: Optional[Dict[str, Any]] = None,
        fold_number: int = None,
        expiry_type: str = "final"
) -> Dict[str, Any]:
    """
    Calculate expiry date for symbol

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        regime_info: Optional pre-computed regime info
        fold_number: Fold number (uses latest if None)
        expiry_type: Expiry type

    Returns:
        Dict with expiry information
    """
    try:
        manager = ExpiryDateManager()

        if fold_number is None:
            try:
                from core.global_method import get_latest_training_folds
                latest_folds = get_latest_training_folds(TOTAL_FOLDS, NUM_FOLDS)
                fold_number = latest_folds[-1] if latest_folds else TOTAL_FOLDS
            except:
                fold_number = TOTAL_FOLDS

        result = await manager.calculate_expiry_date(
            symbol=symbol,
            timeframe=timeframe,
            regime_info=regime_info,
            fold_number=fold_number,
            expiry_type=expiry_type
        )

        return result

    except Exception as e:
        logger.error(f"❌ Error calculating expiry: {e}")
        emergency_date = datetime.now(timezone.utc) + timedelta(days=1)

        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "expiry_date": emergency_date.isoformat(),
            "expiry_type": "emergency",
            "days_to_expiry": 1.0,
            "regime_used": "unknown",
            "confidence": 0.0,
            "error": str(e),
            "is_valid": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# ============ اضافه به انتهای regim.py ============

async def _load_source_performance_cache(
    cache_key: str,
) -> Optional[Dict[str, float]]:
    """
    بارگذاری کش عملکرد منابع با ایمن‌سازی کامل timezone.

    این تابع یک alias طولانی‌تر از `_load_source_performance` است و در
    نسخهٔ قبلی دارای همان باگ naive/aware بود. این نسخه:
      - timestamp را به UTC نرمال می‌کند.
      - در صورت خطا، به‌جای exception، مقدار None برمی‌گرداند.
    """
    try:
        cache_dir = Path("cache")
        if not cache_dir.exists():
            return None

        key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"source_performance_{key_hash}.json"

        if not cache_file.exists():
            return None

        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        timestamp_str = data.get('timestamp', '')
        if timestamp_str:
            if timestamp_str.endswith('Z'):
                timestamp_str = timestamp_str[:-1] + '+00:00'

            try:
                cached_time = datetime.fromisoformat(timestamp_str)
            except (ValueError, TypeError):
                logger.debug(
                    f"Invalid timestamp in source performance cache: "
                    f"{timestamp_str!r}"
                )
                return data.get('performance', {})

            if cached_time.tzinfo is None:
                cached_time = cached_time.replace(tzinfo=timezone.utc)
            else:
                cached_time = cached_time.astimezone(timezone.utc)

            now_utc = datetime.now(timezone.utc)
            age_days = (now_utc - cached_time).total_seconds() / 86400.0

            if age_days > 7:
                logger.debug(
                    f"Source performance cache expired: {age_days:.1f} days old"
                )
                return None

        return data.get('performance', {})

    except Exception as e:
        logger.debug(f"Error loading source performance cache: {e}")
        return None


async def _save_source_performance_cache(
    cache_key: str,
    performance_data: Dict[str, Any],
) -> bool:
    """
    ذخیره کش عملکرد منابع — نسخهٔ اصلاح‌شده.
    """
    try:
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_file = cache_dir / f"source_performance_{key_hash}.json"

        cache_content = {
            "performance": performance_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0",
        }

        temp_file = cache_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(cache_content, f, ensure_ascii=False, indent=2)

        if temp_file.exists():
            if cache_file.exists():
                cache_file.unlink()
            temp_file.rename(cache_file)

        return True

    except Exception as e:
        logger.error(f"Error saving source performance cache: {e}")
        return False
     
# =================================================================================
# EXPORTS
# =================================================================================

__all__ = [
    'get_symbol_regime',
    'get_symbol_regime_classified',
    '_compute_classification_regime_from_ohlcv',
    '_compute_classification_regime_from_ohlcv_offline',   # ✅ جدید
    'MarketRegimeDetector',
    'MarketRegime',
    'PROJECT_TARGETS',
    'SourceResult',
    'ExpiryDateManager',
    'calculate_expiry_date',
    'validate_expiry_date',
]