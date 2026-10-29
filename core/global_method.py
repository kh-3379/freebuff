# core/global_method.py
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import time
import traceback
import warnings
import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
from collections import OrderedDict
from collections.abc import Coroutine as CoroutineABC
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

from scipy.stats import (
    chi2_contingency,
    ks_2samp,
    pearsonr,
    spearmanr,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
from pandas.core.dtypes.common import is_datetime64_any_dtype, is_bool_dtype, is_categorical_dtype, is_object_dtype
from scipy.stats import chi2_contingency, pearsonr, spearmanr, ks_2samp
from typing import Any, Optional, Callable, Coroutine, Dict, List, Tuple, Union, Set
from collections.abc import Coroutine as CoroutineABC
from core.imports import *
from core.logger import redirect_prints_to_logger, AdvancedTradeLogger
from main import sum_folds as TOTAL_FOLDS

redirect_prints_to_logger()
logger = AdvancedTradeLogger(__name__)

_project_root_cache = None
if os.name == "nt":
    import msvcrt
else:
    import fcntl
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
TF_DECISION_FILE = CACHE_DIR / "tf_decisions.json"
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

try:
    import tables
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    logger.warning("⚠️ پکیج 'tables' برای HDF5 نصب نیست. استفاده از فرمت‌های جایگزین")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    logger.warning("⚠️ پکیج 'h5py' برای HDF5 نصب نیست")

SAVE_LOCK = asyncio.Lock()
_SAVE_LOCKS: Dict[int, asyncio.Lock] = {}

_DF_CACHE: "OrderedDict[Tuple[str, int, int], pd.DataFrame]" = OrderedDict()
_DF_CACHE_LOCK = threading.Lock()
_DF_CACHE_MAX_ENTRIES = 32


def _get_cache_file_key(file_path: Path) -> Tuple[str, int, int]:
    """
    کلید کش بر اساس مسیر canonical + mtime_ns + size.

    - اگر فایل در دسترس نباشد، کلید ساده (path, 0, 0) برگردانده می‌شود
      تا از exception جلوگیری شود؛ در این حالت cache_put نوشته نمی‌شود
      ولی cache_get برای مسیر نامعتبر هم miss می‌دهد.
    """
    try:
        st = file_path.stat()
        return (str(file_path.resolve()), int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        return (str(file_path), 0, 0)


def _df_cache_get(file_path: Path) -> Optional[pd.DataFrame]:
    """بازیابی DataFrame کش‌شده اگر شناسهٔ فایل (path/mtime/size) یکسان باشد."""
    key = _get_cache_file_key(file_path)
    with _DF_CACHE_LOCK:
        cached = _DF_CACHE.get(key)
        if cached is not None:
            _DF_CACHE.move_to_end(key)
            return cached
    return None


def _df_cache_put(file_path: Path, df: pd.DataFrame) -> None:
    """ذخیره DataFrame در کش LRU؛ در صورت سرریز، قدیمی‌ترین حذف می‌شود."""
    if df is None or not isinstance(df, pd.DataFrame):
        return
    key = _get_cache_file_key(file_path)
    with _DF_CACHE_LOCK:
        _DF_CACHE[key] = df
        _DF_CACHE.move_to_end(key)
        while len(_DF_CACHE) > _DF_CACHE_MAX_ENTRIES:
            _DF_CACHE.popitem(last=False)


def clear_dataframe_cache() -> int:
    """
    پاکسازی کامل کش DataFrame.

    Returns:
        int: تعداد ورودی‌های حذف‌شده
    """
    with _DF_CACHE_LOCK:
        n = len(_DF_CACHE)
        _DF_CACHE.clear()
    return n


def dataframe_cache_stats() -> Dict[str, int]:
    """گزارش آماری از وضعیت کش."""
    with _DF_CACHE_LOCK:
        return {
            "size": len(_DF_CACHE),
            "max_size": _DF_CACHE_MAX_ENTRIES,
        }

def _get_save_lock() -> asyncio.Lock:
    """
    Return an ``asyncio.Lock`` tied to the CURRENT running event loop.

    Why this exists:
        ``asyncio.Lock`` binds to the event loop on which it is first awaited.
        This project spawns a fresh loop per sync/async bridge invocation, so a
        module-level singleton lock eventually explodes with
        ``RuntimeError: ... different event loop`` and silently breaks the save
        pipeline (``universal_save`` catches it and returns False).

    Returns:
        asyncio.Lock: a lock dedicated to the currently running loop. When no
        loop is running, a stable fallback lock (key ``0``) is returned so the
        lock can still mark a critical section for thread-based use.

    Raises:
        RuntimeError: never raised directly by this helper.
    """
    try:
        loop = asyncio.get_running_loop()
        lock_key = id(loop)
    except RuntimeError:
        lock_key = 0

    lock = _SAVE_LOCKS.get(lock_key)
    if lock is None:
        lock = asyncio.Lock()
        _SAVE_LOCKS[lock_key] = lock
    return lock

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


def _mt5_initializer():
    """
    Process initializer for MT5 worker processes.
    Safe for Windows/Linux/Colab.
    """
    if not MT5_AVAILABLE:
        return

    try:
        if not mt5.initialize():
            print(f"⚠️ MT5 initialization failed: {mt5.last_error()}")
    except Exception as ex:
        print(f"⚠️ MT5 initializer exception: {ex}")


if MT5_AVAILABLE:
    _mt5_pool = ProcessPoolExecutor(
        max_workers=2,
        mp_context=mp.get_context("spawn"),
        initializer=_mt5_initializer
    )
else:
    _mt5_pool = None

    try:
        logger.info("🖥️ MetaTrader5 unavailable - Running in Offline Mode")
    except Exception:
        print("🖥️ MetaTrader5 unavailable - Running in Offline Mode")
import numba


# ═══════════════════════════════════════════════════════════════════════════
# PROJECT TARGETS & CLASS DEFINITIONS — v13.0 AUXILIARY-AWARE
# ═══════════════════════════════════════════════════════════════════════════
# سه گروه متمایز:
#   1) MAIN_TARGETS: 5 تارگت اصلی که مدل باید پیش‌بینی کند
#   2) AUXILIARY_TARGETS: تارگت‌های کمکی که backbone را آموزش می‌دهند
#   3) AUXILIARY_WEIGHTS: وزن‌های نمونه (پیش‌بینی نمی‌شوند، فقط در loss)
#
# ✅ v13.0 تغییرات:
#   - momentum_direction به‌عنوان auxiliary رسمی اضافه شد
#   - magnitude_bucket از auxiliary به formal target ارتقا یافت
#   - sample_weight در AUXILIARY_WEIGHTS نگه داشته شد (نه در PROJECT_TARGETS)
#     تا head مدل برایش ساخته نشود
# ═══════════════════════════════════════════════════════════════════════════

MAIN_TARGETS: List[str] = [
    "price_return",       # regression — بازده آتی close[t+h]/close[t] - 1
    "price_direction",    # 2-class classification {0:SELL, 1:BUY}
    "trend_strength",     # 3-class classification {0:WEAK, 1:MOD, 2:STRONG}
    "volatility_regime",  # 3-class classification {0:LOW, 1:NORMAL, 2:HIGH}
    "market_regime",      # 4-class classification {0:SIDE_LOW, 1:SIDE_HIGH, 2:UP, 3:DOWN}
]

AUXILIARY_TARGETS: List[str] = [
    "magnitude_bucket",     # 3-class {0:SMALL, 1:MEDIUM, 2:LARGE}
    "momentum_direction",   # 2-class {0:DOWN_MOM, 1:UP_MOM} — NEW v13.0
]

AUXILIARY_WEIGHTS: List[str] = [
    "sample_weight",        # float32 — وزن نمونه در loss
]

# فهرست کامل تارگت‌های پیش‌بینی‌شده (مدل برای هرکدام یک head دارد)
PROJECT_TARGETS: List[str] = MAIN_TARGETS + AUXILIARY_TARGETS

PROJECT_TARGETS_TYPE: Dict[str, str] = {
    "price_return": "regression",
    "price_direction": "classification",
    "trend_strength": "classification",
    "volatility_regime": "classification",
    "market_regime": "classification",
    "magnitude_bucket": "classification",
    "momentum_direction": "classification",
}

TARGET_CLASS_COUNTS: Dict[str, int] = {
    "price_direction": 2,
    "trend_strength": 3,
    "volatility_regime": 3,
    "market_regime": 4,
    "magnitude_bucket": 3,
    "momentum_direction": 2,
}

CLASSIFICATION_TARGETS: List[str] = [
    name for name, t in PROJECT_TARGETS_TYPE.items() if t == "classification"
]
REGRESSION_TARGETS: List[str] = [
    name for name, t in PROJECT_TARGETS_TYPE.items() if t == "regression"
]
TARGET_ALIASES = {
    'price_return':       ['price_reg_output', 'price_output', 'price'],
    'price_direction':    ['price_cls_output', 'direction_output', 'direction'],
    'trend_strength':     ['trend_output', 'trend'],
    'volatility_regime':  ['volatility_output', 'vol_output', 'vol'],
    'market_regime':      ['market_output', 'market'],
    'magnitude_bucket':   ['magnitude_output', 'mag_bucket', 'magnitude'],
    'momentum_direction': ['momentum_output', 'momentum', 'mom'],
}
# 🔥 GLOBAL — SINGLE SOURCE OF TRUTH برای ترتیب headها در همهٔ ماژول‌ها
# (توسط DYNAMIC_TES.py و سایر فایل‌ها import می‌شود)
OUTPUT_ORDER: List[str] = list(PROJECT_TARGETS)
AUXILIARY_HEADS: List[str] = list(AUXILIARY_TARGETS)

INVALID_LABEL = -1
SEQUENCE_PIPELINE_VERSION = "11.0-auxiliary-formal"
PIPELINE_MANIFEST_VERSION = "v5.0"

# === FIX C-03: pipeline manifest helpers (fail-fast on version mismatch) ===
PIPELINE_MANIFEST_FILENAME = "pipeline_manifest.json"

def _pipeline_manifest_path(symbol: str, timeframe: str, fold: int) -> Path:
    return Path("data/pipelines") / symbol / timeframe / f"fold_{fold}" / PIPELINE_MANIFEST_FILENAME

def _compute_feature_hash(columns, shape) -> str:
    payload = f"{list(columns)}|{list(shape)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

def save_pipeline_manifest(symbol: str, timeframe: str, fold: int,
                           n_features: int, feature_names,
                           train_shape, exception_columns,
                           extra: Optional[dict] = None) -> bool:
    """
    === FIX C-03 / H-01: manifest یکپارچه ===
    ذخیره‌ی manifest با hash، نسخه، نام فیچرها و لیست exception columns
    تا در transform مرحلهٔ VAL/TEST بتوان fail-fast کرد.
    """
    try:
        path = _pipeline_manifest_path(symbol, timeframe, fold)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest = {
            "manifest_version": PIPELINE_MANIFEST_VERSION,
            "pipeline_version": PIPELINE_MANIFEST_VERSION,
            "symbol": symbol,
            "timeframe": timeframe,
            "fold": fold,
            "n_features": int(n_features),
            "feature_names": [str(c) for c in feature_names],
            "train_shape": [int(x) for x in train_shape],
            "exception_columns": [str(c) for c in (exception_columns or [])],
            "feature_hash": _compute_feature_hash(feature_names, train_shape),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            manifest.update(extra)

        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)

        logger.info(f"[{symbol}] ✅ pipeline_manifest saved: n_features={n_features} hash={manifest['feature_hash']}")
        return True
    except Exception as e:
        logger.error(f"[{symbol}] ❌ save_pipeline_manifest failed: {e}")
        return False

def load_pipeline_manifest(symbol: str, timeframe: str, fold: int) -> Optional[dict]:
    """=== FIX C-03: خواندن manifest === """
    try:
        path = _pipeline_manifest_path(symbol, timeframe, fold)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[{symbol}] ⚠️ load_pipeline_manifest failed: {e}")
        return None

def assert_pipeline_matches(manifest: dict, columns, symbol: str = "?") -> None:
    """
    === FIX C-03: fail-fast اگر transform روی فیچرهای ناسازگار اجرا شود. ===
    Raises:
        RuntimeError: در صورت عدم تطابق hash/فیچرها.
    """
    if manifest is None:
        raise RuntimeError(f"[{symbol}] pipeline_manifest missing — cannot transform safely.")

    expected = set(manifest.get("feature_names", []))
    current = set(map(str, columns))

    missing = expected - current
    extra = current - expected

    if missing or extra:
        raise RuntimeError(
            f"[{symbol}] pipeline schema mismatch — "
            f"missing={sorted(missing)[:5]} | extra={sorted(extra)[:5]}"
        )
    logger.info(f"[{symbol}] ✅ pipeline_manifest schema matched (hash={manifest.get('feature_hash')})")

INVALID_LABEL = -1
SEQUENCE_PIPELINE_VERSION = "9.0-strict-temporal-contract"


def get_valid_mask(
    labels_dict: Dict[str, np.ndarray],
    target_names: Optional[List[str]] = None,
    strict: bool = True,
    return_per_target: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    GET_VALID_MASK v3.0 — 2-CLASS price_direction.

    🔥 v3.0 (نسبت به v2.0):
        محدودهٔ معتبر price_direction از [0, 3) به [0, 2) تغییر یافت.
        سایر تارگت‌ها بدون تغییر (trend_strength: 3, volatility_regime: 3,
        market_regime: 4).

    رفتار:
        - price_return: فقط finite.
        - classification: finite + != INVALID_LABEL + در بازهٔ TARGET_CLASS_COUNTS.

    خروجی:
        mask اشتراکی یا dict per-target در صورت return_per_target=True.
    """
    if not isinstance(labels_dict, dict):
        raise TypeError(
            f"labels_dict must be dict, got {type(labels_dict).__name__}"
        )

    expected = list(target_names or PROJECT_TARGETS)

    if strict:
        missing = [n for n in expected if n not in labels_dict]
        if missing:
            raise ValueError(f"Missing required targets: {missing}")

    available = [n for n in expected if n in labels_dict]
    if not available:
        raise ValueError("No target arrays were provided")

    normalized: Dict[str, np.ndarray] = {}
    lengths: Dict[str, int] = {}

    for name in available:
        v = np.asarray(labels_dict[name])
        if v.ndim == 0:
            raise ValueError(
                f"Target {name!r} must contain a sample axis; got scalar"
            )
        if v.ndim > 1:
            if v.ndim == 2 and v.shape[1] == 1:
                v = v[:, 0]
            else:
                raise ValueError(
                    f"Target {name!r} must be 1D or (n,1), got {v.shape}"
                )
        normalized[name] = v.reshape(-1)
        lengths[name] = int(v.shape[0])

    if len(set(lengths.values())) != 1:
        raise ValueError(f"Target lengths are inconsistent: {lengths}")

    n_samples = next(iter(lengths.values()))
    per_target: Dict[str, np.ndarray] = {}

    for name, values in normalized.items():
        if name == "price_return":
            numeric = values.astype(np.float64, copy=False)
            per_target[name] = np.isfinite(numeric)
            continue

        numeric = values.astype(np.float64, copy=False)
        finite = np.isfinite(numeric)
        not_invalid = numeric != float(INVALID_LABEL)
        m = finite & not_invalid

        # 🔥 v3.0: استفاده از TARGET_CLASS_COUNTS برای محدودهٔ پویا
        n_classes = TARGET_CLASS_COUNTS.get(name)
        if n_classes is not None:
            in_range = (numeric >= 0) & (numeric < int(n_classes))
            m &= in_range

        per_target[name] = m

    if return_per_target:
        return per_target

    shared = np.ones(n_samples, dtype=bool)
    for m in per_target.values():
        shared &= m
    return shared
    
def build_temporal_sequences(
    features,
    labels_dict,
    seq_len=60,
    label_horizon=1,
    timestamps=None,
    target_names=None,
    drop_invalid=True,
    dtype=np.float32,
):
    X = np.asarray(features)
    if X.ndim != 2 or not np.issubdtype(X.dtype, np.number):
        raise ValueError("features must be numeric 2D")

    n, f = X.shape
    L, h = int(seq_len), int(label_horizon)
    if L < 1 or h < 1:
        raise ValueError("seq_len and label_horizon must be >=1")

    names = list(target_names or PROJECT_TARGETS)
    y_raw = {}
    for name in names:
        if name not in labels_dict:
            raise ValueError(f"Missing target: {name}")
        a = np.asarray(labels_dict[name])
        if a.ndim == 2 and a.shape[1] == 1:
            a = a[:, 0]
        if a.ndim != 1 or len(a) != n:
            raise ValueError(f"Invalid target shape/length: {name}={a.shape}")
        y_raw[name] = a

    ts = None
    if timestamps is not None:
        ts = validate_strict_timestamp_series(timestamps, "timestamps", True)
        if len(ts) != n:
            raise ValueError("timestamp length mismatch")

    # anchor=end؛ label[t] خودش آینده t→t+h را encode می‌کند
    ends = np.arange(L - 1, n - h, dtype=np.int64)
    starts = ends - L + 1

    row_finite = np.isfinite(X.astype(np.float64, copy=False)).all(axis=1)
    prefix_bad = np.r_[0, np.cumsum(~row_finite)]
    window_ok = (prefix_bad[ends + 1] - prefix_bad[starts]) == 0

    label_ok = get_valid_mask(y_raw, names, strict=True) if drop_invalid else np.ones(n, bool)
    keep = window_ok & label_ok[ends]

    starts, ends = starts[keep], ends[keep]
    event_ends = ends + h

    windows = np.lib.stride_tricks.sliding_window_view(
        X.astype(np.float64, copy=False), L, axis=0
    )
    X_seq = np.ascontiguousarray(windows[starts].transpose(0, 2, 1), dtype=dtype)
    y_seq = {name: y_raw[name][ends].copy() for name in names}

    if ts is not None and np.any(ts.asi8[event_ends] <= ts.asi8[ends]):
        raise RuntimeError("label event end must be after feature anchor")

    meta = {
        "starts": starts,
        "x_end_indices": ends,
        "label_anchor_indices": ends.copy(),
        "label_event_end_indices": event_ends,
        "x_end_timestamps": None if ts is None else ts[ends],
        "label_event_end_timestamps": None if ts is None else ts[event_ends],
        "seq_len": L,
        "label_horizon": h,
        "contract": "X<=t, y=label[t], label_event_end=t+h",
    }
    return X_seq, y_seq, meta
    
def benchmark_label_generator(n_list: Optional[List[int]] = None, seed: int = 7) -> Dict[str, Any]:
    try:
        import time
        if n_list is None:
            n_list = [5000, 20000, 100000]
        rng = np.random.default_rng(int(seed))
        out: Dict[str, Any] = {"hardware": {"cpu_count": int(CPU_COUNT), "has_numba": bool(HAS_NUMBA), "has_scipy": bool(HAS_SCIPY), "has_sklearn": bool(HAS_SKLEARN)}, "runs": []}
        for nn in n_list:
            nn = int(nn)
            rets = rng.normal(0.0, 0.004, size=nn)
            price = np.empty(nn, dtype=np.float64)
            price[0] = 100.0
            for i in range(1, nn):
                price[i] = price[i - 1] * (1.0 + float(rets[i]))
            gen = LabelGenerator(symbol="BENCH", allow_refit=True)
            t0 = time.perf_counter()
            gen.set_caller("train")
            gen.fit(price)
            t1 = time.perf_counter()
            res = gen.make_all_labels(price)
            t2 = time.perf_counter()
            mask = get_valid_mask(res)
            out["runs"].append({"n": int(nn), "fit_sec": float(t1 - t0), "labels_sec": float(t2 - t1), "valid_ratio": float(np.mean(mask)) if mask.size else 0.0})
        return out
    except Exception as e:
        return {"error": str(e)}

def _shift_future_state(state: np.ndarray, horizon: int, invalid=-1) -> np.ndarray:
    a = np.asarray(state).reshape(-1)
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    out = np.full(a.shape[0], invalid, dtype=np.int32)
    if len(a) > h:
        out[:-h] = a[h:].astype(np.int32, copy=False)
    return out

def compute_garch_past_only(returns, omega=1e-6, alpha=0.1, beta=0.85,
                            init_window=20):
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = len(r)
    out = np.full(n, np.nan)
    w = int(init_window)
    if w < 2:
        raise ValueError("init_window must be >=2")

    # variance در t فقط از [t-w,t)؛ هیچ init global وجود ندارد
    for t in range(w, n):
        if t == w:
            hist = r[:t]
            hist = hist[np.isfinite(hist)]
            if len(hist) < w:
                continue
            out[t] = max(float(np.var(hist)), 1e-8)
            continue

        prev_var = out[t - 1]
        prev_ret = r[t - 1]
        if np.isfinite(prev_var) and np.isfinite(prev_ret):
            out[t] = max(omega + alpha * prev_ret**2 + beta * prev_var, 1e-8)
    return out

import re

_TS_PATTERN = re.compile(r"^ts_(\d+)_feat_(\d+)$")


def map_ts_columns_to_feature_indices(columns: List[str]) -> Dict[str, int]:
    """
    نام‌های 'ts_<t>_feat_<i>' را به feature index 'i' نگاشت می‌کند.
    """
    mapping: Dict[str, int] = {}
    for col in columns:
        m = _TS_PATTERN.match(str(col))
        if m:
            mapping[str(col)] = int(m.group(2))
    return mapping


def extract_feature_indices_from_leaky_columns(leaky_columns: List[str]) -> Set[int]:
    """
    از لیست leaky (که ممکن است ts_* باشد)، feature indices را استخراج می‌کند.
    """
    indices: Set[int] = set()
    for col in leaky_columns:
        m = _TS_PATTERN.match(str(col))
        if m:
            indices.add(int(m.group(2)))
    return indices
    
    
class LabelType(Enum):
    UNKNOWN = "unknown"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    PRICE_RETURN = "price_return"
    PRICE_DIRECTION = "price_direction"
    TREND_STRENGTH = "trend_strength"
    VOLATILITY_REGIME = "volatility_regime"
    MARKET_REGIME = "market_regime"

class LeakageSeverity(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
@dataclass
class ClassificationLeakageMetrics:
    accuracy: float = 0.0
    agreement_pct: float = 0.0
    disagreement_pct: float = 0.0
    chi2_statistic: float = 0.0
    chi2_p_value: float = 1.0
    kl_divergence: float = 0.0
    kl_divergence_normalized: float = 0.0
    js_divergence: float = 0.0
    wasserstein_distance: float = 0.0
    n_samples_compared: int = 0
    original_distribution: Dict[int, float] = field(default_factory=dict)
    generated_distribution: Dict[int, float] = field(default_factory=dict)
    class_wise_agreement: Dict[int, float] = field(default_factory=dict)
@dataclass
class RegressionLeakageMetrics:
    pearson_correlation: float = 0.0
    spearman_correlation: float = 0.0
    correlation_leakage_threshold: float = 0.95
    mean_absolute_error: float = 0.0
    mean_squared_error: float = 0.0
    root_mean_squared_error: float = 0.0
    normalized_mae: float = 0.0
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    wasserstein_distance: float = 0.0
    mean_diff_pct: float = 0.0
    std_diff_ratio: float = 0.0
    quantile_comparison: Dict[str, float] = field(default_factory=dict)
    self_correlation: float = 0.0
    self_mae: float = 0.0
    n_samples_compared: int = 0
    original_stats: Dict[str, float] = field(default_factory=dict)
    generated_stats: Dict[str, float] = field(default_factory=dict)

class DirectionSignalBuilder:
    """
    DIRECTION_SIGNAL_BUILDER v1.0 — MOVEMENT-VS-COST AWARE SIGNAL GENERATOR.

    چرا این کلاس لازم است:
        پس از 2-class کردن price_direction، مفهوم NEUTRAL دیگر یک کلاس
        یادگرفته‌شده نیست. این کلاس آن را در لحظهٔ تصمیم‌گیری می‌سازد
        با ترکیب دو فیلتر:
            ۱. اطمینان (confidence) خروجی softmax مدل
            ۲. کفایت حرکت (expected move) نسبت به هزینهٔ معامله

    این معماری، Label Generator را از پارامترهای تجاری جدا نگه می‌دارد:
        - در تغییر بروکر / کارمزد / تایم‌فریم، فقط پارامترهای این کلاس
          را دوباره پیکربندی می‌کنید. نیازی به regenerate لیبل‌ها و
          retrain مدل نیست.
        - در ارزیابی، این کلاس را با سیگنال override صدا بزنید.
        - در ترید زنده، همین کلاس را با بروکر فعال پیکربندی کنید.

    پارامترها:
        confidence_threshold: θ روی max softmax. پیش‌فرض 0.55.
        cost_multiplier     : ضریب ایمنی روی هزینهٔ رفت-و-برگشت. پیش‌فرض 2.0.
        transaction_cost    : کارمزد هر طرف (0.0005 = 5bp).
        slippage            : لغزش هر طرف (0.0002 = 2bp).
        volatility_lookback : پنجرهٔ σ (تعداد کندل اخیر).
        horizon             : افق پیش‌بینی (هم‌راستا با LabelGenerator).
        safety_k            : ضریب محافظه‌کاری روی σ·√h. پیش‌فرض 1.0.

    مثال:
        >>> builder = DirectionSignalBuilder(
        ...     confidence_threshold=0.55,
        ...     cost_multiplier=2.0,
        ...     transaction_cost=0.0005,
        ...     slippage=0.0002,
        ...     volatility_lookback=20,
        ...     horizon=1,
        ... )
        >>> signal = builder.build_signal(
        ...     direction_probs=model_output_probs_2class,  # shape (N, 2)
        ...     recent_prices=close_prices_recent,          # shape (T,)
        ... )
        >>> # signal ∈ {-1, 0, +1}^N ; 0 = NEUTRAL
    """

    def __init__(
        self,
        confidence_threshold: float = 0.55,
        cost_multiplier: float = 2.0,
        transaction_cost: float = 0.0005,
        slippage: float = 0.0002,
        volatility_lookback: int = 20,
        horizon: int = 1,
        safety_k: float = 1.0,
    ):
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.5, 0.95))
        self.cost_multiplier = float(max(1.0, cost_multiplier))
        self.round_trip_cost = 2.0 * (
            max(0.0, float(transaction_cost))
            + max(0.0, float(slippage))
        )
        self.volatility_lookback = int(max(5, volatility_lookback))
        self.horizon = int(max(1, horizon))
        self.safety_k = float(max(0.1, safety_k))

        logger.info(
            "DirectionSignalBuilder | θ_conf=%.3f | cost_mult=%.2f | "
            "round_trip_cost=%.5f (%.1f bp) | σ_win=%d | h=%d | k=%.2f",
            self.confidence_threshold, self.cost_multiplier,
            self.round_trip_cost, self.round_trip_cost * 1e4,
            self.volatility_lookback, self.horizon, self.safety_k,
        )

    def recent_volatility(self, prices: np.ndarray) -> float:
        """σ بازده‌های اخیر (کندل‌به‌کندل)."""
        p = np.asarray(prices, dtype=np.float64).reshape(-1)
        if p.size < 2:
            return 0.0
        rets = np.diff(p) / np.maximum(np.abs(p[:-1]), 1e-12)
        window = rets[-self.volatility_lookback:]
        window = window[np.isfinite(window)]
        if window.size < 5:
            return 0.0
        return float(np.std(window))

    def expected_move(self, prices: np.ndarray) -> float:
        """σ · √h · safety_k به‌عنوان برآورد حرکت در افق h."""
        sigma = self.recent_volatility(prices)
        return self.safety_k * sigma * float(np.sqrt(self.horizon))

    def build_signal(
        self,
        direction_probs: np.ndarray,
        recent_prices: np.ndarray,
        override_confidence: Optional[float] = None,
        override_expected_move: Optional[float] = None,
    ) -> np.ndarray:
        """
        ورودی:
            direction_probs: (N, 2) — softmax روی {0:SELL, 1:BUY}
            recent_prices  : (T,)  — قیمت‌های اخیر برای σ
        خروجی:
            np.ndarray[int64] با مقادیر {-1, 0, +1} در طول N.
        """
        P = np.asarray(direction_probs, dtype=np.float64)
        if P.ndim != 2 or P.shape[1] != 2:
            raise ValueError(
                f"DirectionSignalBuilder expects (N, 2) probs; got {P.shape}"
            )

        conf = (
            float(np.clip(override_confidence, 0.5, 0.95))
            if override_confidence is not None
            else self.confidence_threshold
        )
        emv = (
            float(max(0.0, override_expected_move))
            if override_expected_move is not None
            else self.expected_move(recent_prices)
        )
        cost_barrier = self.cost_multiplier * self.round_trip_cost

        max_p = P.max(axis=1)
        argmax = P.argmax(axis=1)          # 0 = SELL, 1 = BUY

        signal = np.zeros(P.shape[0], dtype=np.int64)

        passes_conf = max_p >= conf
        passes_move = emv >= cost_barrier

        if passes_move:
            signal[(argmax == 0) & passes_conf] = -1
            signal[(argmax == 1) & passes_conf] = +1

        n_traded = int(np.sum(signal != 0))
        n_sell = int(np.sum(signal == -1))
        n_buy = int(np.sum(signal == 1))
        logger.debug(
            "[DirectionSignalBuilder] N=%d | σ=%.6f | emv=%.6f | "
            "cost_barrier=%.6f | pass_move=%s | traded=%d (SELL=%d, BUY=%d)",
            P.shape[0], self.recent_volatility(recent_prices), emv,
            cost_barrier, bool(passes_move), n_traded, n_sell, n_buy,
        )

        return signal
        
class PriceDirectionNames:
    """
    PRICE_DIRECTION_NAMES v2.0 — 2-CLASS (SELL/BUY).

    🔥 v2.0 (نسبت به v1.0):
        کلاس NEUTRAL حذف شد. دلیل: در 3-class قدیمی، NEUTRAL یک ناحیهٔ
        خاکستری وسیع بود که مدل قادر به تفکیک آن از SELL/BUY نبود
        (شاهد از لاگ: bacc≈0.37 روی chance=0.333 در کل ۱۹ اپوک).

        در معماری جدید:
            • LabelGenerator فقط جهت واقعی را برچسب می‌زند (sign of future return).
            • مفهوم «خنثی» در زمان inference توسط:
                - آستانهٔ اطمینان (confidence threshold) روی softmax خروجی
                - فیلتر «حرکت در برابر هزینه» (movement vs cost)
              ساخته می‌شود. کلاس DirectionSignalBuilder این دو فیلتر را
              پیاده می‌کند.

    نگاشت برچسب:
        0 -> SELL / SHORT   (future_return < 0)
        1 -> BUY  / LONG    (future_return >= 0)

    نگاشت قدیمی‌ها (برای resume):
        3-class قدیمی {0: SELL, 1: NEUTRAL, 2: BUY}
            → NEUTRAL به INVALID (-1) نگاشت می‌شود و در get_valid_mask drop.
        5-class قدیمی {0..4: STRONG_DOWN..STRONG_UP}
            → NEUTRAL (کلاس 2) به INVALID (-1) نگاشت می‌شود.
    """

    NAMES = {
        0: "SELL",
        1: "BUY",
    }

    N_CLASSES = 2

    # 3-class legacy: 0=SELL, 1=NEUTRAL, 2=BUY
    LEGACY_3CLASS_MAP = {
        0: 0,   # SELL    -> SELL
        1: -1,  # NEUTRAL -> INVALID (dropped by get_valid_mask)
        2: 1,   # BUY     -> BUY
    }

    # 5-class legacy: 0=STRONG_DOWN, 1=WEAK_DOWN, 2=SIDEWAYS, 3=WEAK_UP, 4=STRONG_UP
    LEGACY_5CLASS_MAP = {
        0: 0,   # STRONG_DOWN -> SELL
        1: 0,   # WEAK_DOWN   -> SELL
        2: -1,  # SIDEWAYS    -> INVALID (dropped)
        3: 1,   # WEAK_UP     -> BUY
        4: 1,   # STRONG_UP   -> BUY
    }

    @classmethod
    def get_name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"UNKNOWN_{code}")

    @classmethod
    def get_trading_signal(cls, code: int) -> str:
        c = int(code)
        if c == 0:
            return "SELL / SHORT"
        if c == 1:
            return "BUY / LONG"
        if c == -1:
            return "INVALID / SKIP"
        return "UNKNOWN / SKIP"

    @classmethod
    def merge_3class_to_2class(cls, arr3) -> np.ndarray:
        """
        نگاشت آرایهٔ ۳کلاسه (0,1,2) به ۲کلاسه (0,1) یا INVALID (-1).
        NEUTRAL (کد 1 در 3-class) به INVALID نگاشت می‌شود.
        """
        a = np.asarray(arr3).astype(np.int64)
        out = np.full(a.shape, INVALID_LABEL, dtype=np.int64)
        lookup = np.array([0, -1, 1], dtype=np.int64)  # index→class
        valid = (a >= 0) & (a <= 2)
        if valid.any():
            out[valid] = lookup[a[valid]]
        return out.astype(np.int32)

    @classmethod
    def merge_5class_to_2class(cls, arr5) -> np.ndarray:
        """
        نگاشت آرایهٔ ۵کلاسه (0..4) به ۲کلاسه (0,1) یا INVALID (-1).
        SIDEWAYS (کد 2 در 5-class) به INVALID نگاشت می‌شود.
        """
        a = np.asarray(arr5).astype(np.int64)
        out = np.full(a.shape, INVALID_LABEL, dtype=np.int64)
        lookup = np.array([0, 0, -1, 1, 1], dtype=np.int64)
        valid = (a >= 0) & (a <= 4)
        if valid.any():
            out[valid] = lookup[a[valid]]
        return out.astype(np.int32)

    # ── backward-compat alias (اسم قدیمی که در جاهای مختلف صدا زده می‌شود) ──
    @classmethod
    def merge_5class_to_3class(cls, arr5) -> np.ndarray:
        """DEPRECATED ALIAS — emits warning و به merge_5class_to_2class منتقل می‌شود."""
        logger.warning(
            "PriceDirectionNames.merge_5class_to_3class is DEPRECATED — "
            "returning 2-class output. Use merge_5class_to_2class."
        )
        return cls.merge_5class_to_2class(arr5)

class MomentumDirectionNames:
    """
    MOMENTUM_DIRECTION_NAMES v1.0 — 2-CLASS (DOWN_MOMENTUM / UP_MOMENTUM).

    ────────────────────────────────────────────────────────────────────────
    چرا این تارگت کمکی لازم است:
    ────────────────────────────────────────────────────────────────────────

    `price_direction` = sign(future_return) روی تایم‌فریم intraday عملاً نویز
    است (شاهد bacc≈0.5175). اما `momentum_direction` یک سیگنال **واقعی**
    است، چون:

        • Momentum Persistence در بازار کریپتو، در بازه‌های چند‌کندلی،
          از نظر آماری معنادار است (autocorrelation مثبت).

        • EMA-spread (ema_5_returns - ema_20_returns) هم نویز microstructure
          را smooth می‌کند و هم جهت را حفظ می‌کند.

        • با یادگیری `momentum_direction` در head جداگانه، backbone
          مشترک مجبور می‌شود نماینده‌ای (representation) بسازد که
          «جهت‌دار» است، و این نماینده به head `price_direction` هم
          منتقل می‌شود (Multi-Task Learning transfer).

    ────────────────────────────────────────────────────────────────────────
    قرارداد زمانی:
    ────────────────────────────────────────────────────────────────────────

    مقدار خام:
        momentum_state[t] = sign(EMA(ret,5)[t] - EMA(ret,20)[t])
        ∈ {0: DOWN, 1: UP, -1: INVALID}

    پس از shift در _generate_targets:
        momentum_direction[t] = momentum_state[t+h]
        (future-anchored، هم‌راستا با price_direction)

    نگاشت برچسب:
        0 → DOWN_MOMENTUM  (EMA-5 returns ≤ EMA-20 returns)
        1 → UP_MOMENTUM    (EMA-5 returns > EMA-20 returns)
    """

    NAMES: Dict[int, str] = {
        0: "DOWN_MOMENTUM",
        1: "UP_MOMENTUM",
    }
    N_CLASSES: int = 2

    @classmethod
    def get_name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"UNKNOWN_{int(code)}")

    @classmethod
    def get_trading_signal(cls, code: int) -> str:
        c = int(code)
        if c == 0:
            return "SELL_CONTEXT (short-term momentum weak)"
        if c == 1:
            return "BUY_CONTEXT (short-term momentum strong)"
        if c == -1:
            return "INVALID / SKIP"
        return "UNKNOWN / SKIP"
        
class TrendStrengthNames:
    NAMES = {0: "WEAK_TREND", 1: "MODERATE_TREND", 2: "STRONG_TREND"}

    @classmethod
    def get_name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"UNKNOWN_{code}")


class VolatilityRegimeNames:
    NAMES = {0: "LOW_VOL", 1: "NORMAL_VOL", 2: "HIGH_VOL"}

    @classmethod
    def get_name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"UNKNOWN_{code}")


class MarketRegimeNames:
    NAMES = {
        0: "SIDEWAYS_LOW_VOL",
        1: "SIDEWAYS_HIGH_VOL",
        2: "TRENDING_UP",
        3: "TRENDING_DOWN",
    }

    @classmethod
    def get_name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), f"UNKNOWN_{code}")

    @classmethod
    def get_trading_action(cls, code: int) -> str:
        return {
            0: "WAIT - No position",
            1: "CAUTION - Reduce position size",
            2: "AGGRESSIVE LONG",
            3: "AGGRESSIVE SHORT",
        }.get(int(code), "HOLD / SKIP")


# ═════════════════════════════════════════════════════════════════════════
# SHARED INFRASTRUCTURE v6.0  (جایگزین _safe_request/_load_cache برای این ۴ کلاس)
# ═════════════════════════════════════════════════════════════════════════
import hashlib
import io
import logging
import os
import pickle
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger("fundamentals")  # اگر logger پروژه دارید این خط را حذف کنید

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; WallStreetQuant/6.0)",
    "Accept": "application/json,text/csv,*/*",
})
_RATE_LOCK = threading.Lock()
_NEXT_SLOT: Dict[str, float] = {}
_MIN_INTERVAL: Dict[str, float] = {
    "www.bitmex.com": 2.1,              # ~30 req/min بدون کلید
    "api.exchange.coinbase.com": 0.12,
    "www.bitstamp.net": 0.15,
    "www.deribit.com": 0.12,
    "futures.kraken.com": 0.3,
    "www.okx.com": 0.25,
    "api.hyperliquid.xyz": 0.3,
    "api.bybit.com": 0.2,
    "fapi.binance.com": 0.2,
    "min-api.cryptocompare.com": 0.1,
    "fred.stlouisfed.org": 0.3,
    "api.blockchain.info": 1.0,
    "community-api.coinmetrics.io": 0.5,
    "stablecoins.llama.fi": 0.3,
    "api.llama.fi": 0.3,
    "api.alternative.me": 0.3,
}
_BLOCKED_HOSTS: set = set()
_NEVER_BLOCK = {"data.binance.vision"}     # S3 گاهی برای فایل ناموجود 403 می‌دهد
_GEO_CODES = (403, 451)
_NOT_FOUND = object()
_MISSING_STR = "__MISSING__"
_BV = "https://data.binance.vision/data"

# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT-AWARE CACHE ROOT (Colab / Local)  — v2.0
# ═══════════════════════════════════════════════════════════════════════════
# قرارداد:
#   ۱. اگر FUND_CACHE_DIR در .env ست شده باشد، همان استفاده می‌شود.
#      الگوی Colab-style ("drive/MyDrive/...") خودکار به "/content/..."
#      بسط داده می‌شود.
#   ۲. در غیر این صورت اگر در Google Colab باشیم:
#         • Drive mount شده → /content/drive/MyDrive/bot_trader/.fund_cache
#         • Drive mount نشده → /content/.fund_cache
#   ۳. در لوکال → ~/.fund_cache
#
# نکته: هر مسیری قبل از استفاده با write-test اعتبارسنجی می‌شود تا خطای
# مجوز در وسط اجرا رخ ندهد و به fallback امن سقوط شود.
# ═══════════════════════════════════════════════════════════════════════════

def _is_colab_environment() -> bool:
    """تشخیص اجرای برنامه در Google Colab."""
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        if "google.colab" in sys.modules:
            return True
    except Exception:
        pass
    try:
        return os.path.isdir("/content")
    except Exception:
        return False


def _resolve_cache_root_path(raw_path: str) -> Optional[str]:
    """
    تبدیل مسیر کش به مسیر مطلق.

    - الگوی Colab: ``drive/MyDrive/...`` → ``/content/drive/MyDrive/...``
    - ``~`` بسط داده می‌شود.
    - مسیرهای نسبی به مطلق تبدیل می‌شوند.
    """
    if not raw_path:
        return None
    try:
        path = os.path.expanduser(str(raw_path).strip())
    except Exception:
        return None
    if not path:
        return None
    norm = path.replace("\\", "/")
    if norm.startswith("drive/"):
        path = os.path.join("/content", norm)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def _is_writable_dir(path: str) -> bool:
    """تست writeability با یک فایل probe موقت."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("ok")
        os.remove(probe)
        return True
    except Exception:
        return False


def _get_fund_cache_root() -> str:
    """
    تعیین مسیر ریشهٔ کش بر اساس محیط اجرا.

    اولویت:
        ۱. FUND_CACHE_DIR (override صریح از .env)
        ۲. Colab + Drive → /content/drive/MyDrive/bot_trader/.fund_cache
        ۳. Colab بدون Drive → /content/.fund_cache
        ۴. لوکال → ~/.fund_cache
    """
    explicit = os.environ.get("FUND_CACHE_DIR")
    if explicit:
        resolved = _resolve_cache_root_path(explicit)
        if resolved and _is_writable_dir(resolved):
            return resolved

    if _is_colab_environment():
        drive_root = "/content/drive/MyDrive"
        if os.path.isdir(drive_root):
            drive_cache = os.path.join(drive_root, "bot_trader", ".fund_cache")
            if _is_writable_dir(drive_cache):
                return drive_cache
        colab_fallback = "/content/.fund_cache"
        if _is_writable_dir(colab_fallback):
            return colab_fallback

    local = os.path.join(os.path.expanduser("~"), ".fund_cache")
    if _is_writable_dir(local):
        return local
    return os.path.join(os.getcwd(), ".fund_cache")


def refresh_fund_cache_root() -> str:
    """
    بازخوانی مسیر کش — مفید در Colab وقتی Drive پس از import شدن
    این ماژول mount می‌شود::

        from google.colab import drive
        drive.mount('/content/drive')

        from core import global_method
        global_method.refresh_fund_cache_root()

    Returns:
        str: مسیر جدید cache root.
    """
    global _CACHE_ROOT
    new_root = _get_fund_cache_root()
    _CACHE_ROOT = new_root
    try:
        logger.info(f"[CACHE] cache root refreshed → {new_root}")
    except Exception:
        pass
    return new_root


def get_cache_root() -> str:
    """قرائت مسیر فعال کش (برای گزارش/دیباگ)."""
    return _CACHE_ROOT


def cache_status() -> Dict[str, Any]:
    """
    گزارش کامل وضعیت کش بنیادی.

    Returns:
        dict با کلیدهای:
            root, ttl_hours, is_colab, namespaces, total_files, total_mb
    """
    namespaces: Dict[str, Dict[str, Any]] = {}
    total_files = 0
    total_bytes = 0
    try:
        if os.path.isdir(_CACHE_ROOT):
            for ns in sorted(os.listdir(_CACHE_ROOT)):
                ns_path = os.path.join(_CACHE_ROOT, ns)
                if not os.path.isdir(ns_path):
                    continue
                files = [f for f in os.listdir(ns_path) if f.endswith(".pkl")]
                ns_bytes = 0
                for f in files:
                    try:
                        ns_bytes += os.path.getsize(os.path.join(ns_path, f))
                    except OSError:
                        pass
                namespaces[ns] = {
                    "files": len(files),
                    "size_mb": round(ns_bytes / (1024 * 1024), 2),
                }
                total_files += len(files)
                total_bytes += ns_bytes
    except Exception as exc:
        logger.warning(f"[CACHE] status scan failed: {exc}")

    return {
        "root": _CACHE_ROOT,
        "ttl_hours": FUNDAMENTAL_CACHE_TTL.total_seconds() / 3600.0,
        "is_colab": _is_colab_environment(),
        "namespaces": namespaces,
        "total_files": total_files,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


# ── مسیر فعال (قابل خواندن از بیرون با get_cache_root()) ──
_CACHE_ROOT: str = _get_fund_cache_root()


# ── TTL سراسری برای کش فیچرهای بنیادی (پیش‌فرض ۱۲ ساعت) ──
try:
    _fund_ttl_hours = float(os.environ.get("FUND_CACHE_TTL_HOURS", "12"))
    if not (0.1 <= _fund_ttl_hours <= 168.0):  # 0.1h .. 7 روز
        _fund_ttl_hours = 12.0
except (TypeError, ValueError):
    _fund_ttl_hours = 12.0

FUNDAMENTAL_CACHE_TTL: timedelta = timedelta(hours=_fund_ttl_hours)


try:
    logger.info(
        "[CACHE] root=%s | FUNDAMENTAL_TTL=%s | colab=%s",
        _CACHE_ROOT, FUNDAMENTAL_CACHE_TTL, _is_colab_environment(),
    )
except Exception:
    pass

def reset_blocked_hosts() -> None:
    """پاک‌کردن لیست hostهای geo-block شده (مثلاً بعد از روشن‌کردن VPN)."""
    _BLOCKED_HOSTS.clear()


def _throttle(host: str) -> None:
    gap = _MIN_INTERVAL.get(host, 0.1)
    if gap <= 0:
        return
    with _RATE_LOCK:
        now = time.monotonic()
        slot = max(now, _NEXT_SLOT.get(host, 0.0))
        _NEXT_SLOT[host] = slot + gap
    wait = slot - now
    if wait > 0:
        time.sleep(wait)


def _http(
    method: str,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    retries: int = 4,
    kind: str = "json",          # json | bytes | text
    nf: bool = False,            # اگر True، 404 → _NOT_FOUND (تفکیک از خطای شبکه)
) -> Any:
    host = url.split("/")[2]
    if host in _BLOCKED_HOSTS:
        return None
    for attempt in range(retries):
        try:
            _throttle(host)
            r = _SESSION.request(
                method, url, params=params, json=json_body,
                headers=headers, timeout=timeout,
            )
        except requests.RequestException:
            time.sleep(min(2 ** attempt, 15))
            continue
        sc = r.status_code
        if sc == 200:
            try:
                if kind == "json":
                    return r.json()
                return r.content if kind == "bytes" else r.text
            except ValueError:
                return None
        if sc == 404 or (sc in _GEO_CODES and host in _NEVER_BLOCK):
            return _NOT_FOUND if nf else None
        if sc in _GEO_CODES:
            _BLOCKED_HOSTS.add(host)
            logger.warning(
                "[HTTP] %s → HTTP %d (geo-block/forbidden) — host disabled "
                "for this session", host, sc,
            )
            return None
        if sc == 429:
            try:
                wait = float(r.headers.get("Retry-After", 0)) or 2 ** (attempt + 1)
            except ValueError:
                wait = 2 ** (attempt + 1)
            time.sleep(min(wait, 60))
            continue
        if sc >= 500:
            time.sleep(min(2 ** attempt, 20))
            continue
        return None
    return None


# ─────────────────────────── Disk cache ────────────────────────────
def _cache_path(ns: str, key: str) -> str:
    d = os.path.join(_CACHE_ROOT, ns)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, hashlib.md5(key.encode()).hexdigest()[:24] + ".pkl")


def _cache_get(ns: str, key: str, ttl: Optional[timedelta] = None) -> Any:
    p = _cache_path(ns, key)
    try:
        if not os.path.exists(p):
            return None
        if ttl is not None and time.time() - os.path.getmtime(p) > ttl.total_seconds():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _cache_put(ns: str, key: str, obj: Any) -> None:
    p = _cache_path(ns, key)
    tmp = f"{p}.{threading.get_ident()}.tmp"
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=4)
        os.replace(tmp, p)
    except Exception as e:
        logger.debug("[CACHE] write failed: %s", e)


# ─────────────────────────── Time helpers ──────────────────────────
def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _epoch_to_ts(values: Any) -> pd.Series:
    """epoch (s / ms / µs) → UTC timestamp؛ واحد از روی بزرگی عدد تشخیص داده می‌شود
    (Binance spot از ۲۰۲۵ به بعد µs است)."""
    s = pd.to_numeric(pd.Series(values), errors="coerce").astype("float64")
    ref = float(s.dropna().abs().median()) if s.notna().any() else 0.0
    unit = "s" if ref < 1e11 else ("ms" if ref < 1e14 else "us")
    return pd.to_datetime(s, unit=unit, utc=True)


def _windows(start: datetime, end: datetime, step: timedelta) -> List[Tuple[datetime, datetime]]:
    out, a = [], start
    while a < end:
        b = min(a + step, end)
        out.append((a, b))
        a = b
    return out


def _last_dom(y: int, m: int) -> date:
    return date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)


def _month_range(start: datetime, end: datetime) -> List[Tuple[int, int]]:
    cur, last, out = date(start.year, start.month, 1), date(end.year, end.month, 1), []
    while cur <= last:
        out.append((cur.year, cur.month))
        cur = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
    return out


def _ts_col(df: pd.DataFrame) -> pd.DataFrame:
    """تضمین ستون 'timestamp' با dtype یکسان datetime64[ns, UTC] (برای merge_asof)."""
    out = df.copy()
    if "timestamp" not in out.columns:
        out = out.reset_index()
        if "timestamp" not in out.columns:
            out = out.rename(columns={out.columns[0]: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out["timestamp"] = out["timestamp"].astype("datetime64[ns, UTC]")
    return out.sort_values("timestamp").reset_index(drop=True)


def _tidy(df: Optional[pd.DataFrame], value_cols: Sequence[str]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    out = df.dropna(subset=["timestamp"]).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.sort_values("timestamp").reset_index(drop=True)
    out = out.dropna(subset=list(value_cols), how="all")
    return out if not out.empty else None


def _clip(df: Optional[pd.DataFrame], start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    m = (df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))
    out = df.loc[m].reset_index(drop=True)
    return out if not out.empty else None


def _fetch_windows(
    ns: str, tag: str, windows: List[Tuple[datetime, datetime]],
    fn: Callable[[datetime, datetime], Optional[pd.DataFrame]], workers: int = 1,
) -> Optional[pd.DataFrame]:
    """
    اجرای fn روی پنجره‌های زمانی با کش دائمی برای پنجره‌های بسته‌شده.
    fn: DataFrame (حتی خالی = «موفق، بدون داده») یا None (= شکست، کش نمی‌شود).
    """
    now = datetime.now(timezone.utc)

    def _one(w: Tuple[datetime, datetime]) -> Optional[pd.DataFrame]:
        a, b = w
        key = f"{tag}|{a:%Y%m%d%H%M}|{b:%Y%m%d%H%M}"
        closed = b < now - timedelta(days=2)
        if closed:
            c = _cache_get(ns, key, None)
            if isinstance(c, pd.DataFrame):
                return c
        try:
            df = fn(a, b)
        except Exception as e:
            logger.debug("[%s] window %s failed: %s", ns, key, e)
            return None
        if closed and isinstance(df, pd.DataFrame):
            _cache_put(ns, key, df)
        return df

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_one, windows))
    else:
        res = [_one(w) for w in windows]

    n_fail = sum(1 for d in res if d is None)
    if n_fail:
        logger.warning("[%s] %s: %d/%d windows failed", ns, tag, n_fail, len(windows))
    frames = [d for d in res if isinstance(d, pd.DataFrame) and not d.empty]
    return pd.concat(frames, ignore_index=True) if frames else None


# ───────────────────── Binance Vision (آرشیو رسمی) ─────────────────────
def _read_zip_csv(blob: bytes, header: Any = "auto") -> Optional[pd.DataFrame]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
        with zf.open(zf.namelist()[0]) as f:
            raw = pd.read_csv(f, header=None, dtype=str)
    except Exception:
        return None
    if raw.empty:
        return None
    if header == "auto":
        header = not re.fullmatch(r"-?\d+(\.\d+)?", str(raw.iloc[0, 0]).strip())
    if header:
        raw.columns = [str(c).strip() for c in raw.iloc[0]]
        raw = raw.iloc[1:].reset_index(drop=True)
    return raw


def _bv_get(path: str, ns: str, header: Any, neg_cache: bool) -> Optional[pd.DataFrame]:
    c = _cache_get(ns, path, None)
    if isinstance(c, pd.DataFrame):
        return c
    if isinstance(c, str) and c == _MISSING_STR:
        return None
    blob = _http("GET", f"{_BV}/{path}", kind="bytes", timeout=60, nf=True)
    if blob is _NOT_FOUND:
        if neg_cache:
            _cache_put(ns, path, _MISSING_STR)
        return None
    if blob is None:
        return None            # خطای گذرا؛ کش نمی‌شود
    raw = _read_zip_csv(blob, header)
    if raw is not None:
        _cache_put(ns, path, raw)
    return raw


def _bv_series(
    market: str, dataset: str, symbol: str, interval: Optional[str],
    start: datetime, end: datetime, parser: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
    ns: str, header: Any = "auto", use_monthly: bool = True,
    daily_fallback: bool = True, workers: int = 16,
) -> Optional[pd.DataFrame]:
    """
    دریافت سری کامل از data.binance.vision:
      ماه‌های کامل از فایل monthly، ماه جاری (و ماه‌های بدون فایل) از فایل daily.
    """
    token = interval or dataset
    base = f"{market}/{{freq}}/{dataset}/{symbol}/" + (f"{interval}/" if interval else "")
    today = datetime.now(timezone.utc).date()

    def m_path(y: int, m: int) -> str:
        return f"{base.format(freq='monthly')}{symbol}-{token}-{y}-{m:02d}.zip"

    def d_path(d: date) -> str:
        return f"{base.format(freq='daily')}{symbol}-{token}-{d:%Y-%m-%d}.zip"

    months = _month_range(start, end)

    def do_month(ym: Tuple[int, int]) -> Tuple[Tuple[int, int], Optional[pd.DataFrame]]:
        if not use_monthly:
            return ym, None
        y, m = ym
        return ym, _bv_get(m_path(y, m), ns, header, (today - _last_dom(y, m)).days > 45)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        month_res = list(ex.map(do_month, months))

    frames: List[pd.DataFrame] = []
    days: List[date] = []
    for (y, m), raw in month_res:
        if raw is not None:
            try:
                p = parser(raw)
            except Exception as e:
                logger.debug("[%s] parse failed %s-%02d: %s", ns, y, m, e)
                p = None
            if p is not None:
                frames.append(p)
            continue
        if not daily_fallback:
            continue
        d0 = max(date(y, m, 1), start.date())
        d1 = min(_last_dom(y, m), end.date(), today - timedelta(days=1))
        d = d0
        while d <= d1:
            days.append(d)
            d += timedelta(days=1)

    def do_day(d: date) -> Optional[pd.DataFrame]:
        raw = _bv_get(d_path(d), ns, header, (today - d).days > 4)
        if raw is None:
            return None
        try:
            return parser(raw)
        except Exception:
            return None

    if days:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            frames.extend([f for f in ex.map(do_day, days) if f is not None])

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _num(raw: pd.DataFrame, i: int) -> pd.Series:
    if raw.shape[1] > i:
        return pd.to_numeric(raw.iloc[:, i], errors="coerce")
    return pd.Series(np.nan, index=raw.index)


def _parse_bv_klines(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    if raw is None or raw.shape[1] < 6:
        return None
    return pd.DataFrame({
        "timestamp": _epoch_to_ts(raw.iloc[:, 0]),          # open-time
        "open": _num(raw, 1), "high": _num(raw, 2), "low": _num(raw, 3),
        "close": _num(raw, 4), "volume": _num(raw, 5), "quote_volume": _num(raw, 7),
    })


def _parse_bv_funding(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    if raw is None or raw.shape[1] < 2:
        return None
    ts = _epoch_to_ts(raw.iloc[:, 0])
    if raw.shape[1] >= 3:
        interval = pd.to_numeric(raw.iloc[:, 1], errors="coerce").fillna(8.0)
        rate = pd.to_numeric(raw.iloc[:, 2], errors="coerce")
    else:
        interval = pd.Series(8.0, index=raw.index)
        rate = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    interval = interval.where(interval > 0, 8.0)
    return pd.DataFrame({"timestamp": ts, "funding_rate": rate * (8.0 / interval)})


def _parse_bv_metrics(raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    if raw is None or "create_time" not in raw.columns:
        return None
    out = pd.DataFrame({"timestamp": pd.to_datetime(
        raw["create_time"], utc=True, errors="coerce")})
    names = {
        "sum_open_interest": "oi_coin",
        "sum_open_interest_value": "oi_usd",
        "count_toptrader_long_short_ratio": "ls_top_accounts",
        "sum_toptrader_long_short_ratio": "ls_top_positions",
        "count_long_short_ratio": "ls_global",
        "sum_taker_long_short_vol_ratio": "taker_ratio",
    }
    for src, dst in names.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan
    return out


# ───────────────────── OHLCV helpers (مشترک) ─────────────────────
_OHLCV = ["open", "high", "low", "close", "volume"]


def _resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    g = df.set_index("timestamp").resample(
        f"{minutes}min", label="left", closed="left")
    out = g.agg({"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"})
    return out.dropna(subset=["close"]).reset_index()


def _coinbase_ohlcv(asset: str, start: datetime, end: datetime,
                    gran: int, quote: str = "USD") -> Optional[pd.DataFrame]:
    """Coinbase candles با pagination کامل (سقف ۳۰۰ کندل در هر درخواست). timestamp = open-time."""
    url = f"https://api.exchange.coinbase.com/products/{asset}-{quote}/candles"
    iso = "%Y-%m-%dT%H:%M:%SZ"

    def fn(a: datetime, b: datetime) -> Optional[pd.DataFrame]:
        p = _http("GET", url, params={
            "start": a.strftime(iso), "end": b.strftime(iso), "granularity": gran})
        if not isinstance(p, list):
            return None
        if not p:
            return pd.DataFrame(columns=["timestamp"] + _OHLCV)
        arr = pd.DataFrame(p).iloc[:, :6]
        arr.columns = ["ts", "low", "high", "open", "close", "volume"]
        arr = arr.apply(pd.to_numeric, errors="coerce")
        arr["timestamp"] = pd.to_datetime(arr["ts"], unit="s", utc=True)
        return arr[["timestamp"] + _OHLCV]

    out = _fetch_windows("coinbase_ohlc", f"{asset}{quote}{gran}",
                         _windows(start, end, timedelta(seconds=gran * 290)), fn, workers=2)
    return _tidy(out, ["close"])
    
class EnhancedDerivativesFetcher:
    """
    ENHANCED_DERIVATIVES_FETCHER v6.0 — Global multi-source, 6+ years.

    منابع funding : Binance(Vision+REST) | Deribit | BitMEX | Kraken | OKX | Hyperliquid | Bybit*
    منابع OI/Flow : Binance Vision metrics (OI, L/S top/global, taker ratio) | OKX rubik
    سایر          : Deribit DVOL | Binance perp klines/premium-index | Coinbase spot
    (*) geo-restricted؛ در صورت 403/451 خودکار غیرفعال می‌شود.

    قراردادها:
      • funding همه‌جا معادل ۸ ساعته (funding_rate = per-8h).
      • بدون leakage: برچسب هر مشاهده = ceil(زمان رویداد) روی grid؛ کندل‌ها با close-time.
      • هیچ صفر جعلی: نبودِ داده = NaN.
      • 'timestamp' به‌صورت column (نه index).
    """

    SOURCE_PRIORITY: Tuple[str, ...] = (
        "binance_funding", "deribit_funding", "bitmex_funding", "kraken_funding",
        "okx_funding", "hyperliquid_funding", "bybit_funding",
        "binance_metrics", "okx_oi", "deribit_dvol",
        "binance_perp_klines", "binance_premium", "coinbase_spot",
    )
    OUTPUT_COLUMNS: Tuple[str, ...] = (
        "funding_rate_median", "funding_rate_mean", "funding_rate_std",
        "funding_rate_max", "funding_rate_min", "n_markets",
        "open_interest_usd_total", "oi_binance_usd", "oi_binance_coin", "n_oi_sources",
        "ls_ratio_top_accounts", "ls_ratio_top_positions", "ls_ratio_global",
        "taker_buysell_ratio", "volume_24h_usd_total",
        "basis_median", "spread_median", "coinbase_premium_pct",
        "dvol", "price_median",
    )
    FUNDING_FFILL_HOURS = 9.0   # funding هر ≤۸ ساعت تسویه می‌شود

    def __init__(
        self,
        api_key: Optional[str] = None,
        cg_api_key: Optional[str] = None,
        cc_api_key: Optional[str] = None,
        enable_binance_fallback: Optional[bool] = None,
        history_years: float = 6.0,
        include_geo_restricted: bool = True,
        max_workers: int = 6,
        vision_workers: int = 16,
    ):
        self.api_key, self.cg_api_key, self.cc_api_key = api_key, cg_api_key, cc_api_key
        self.history_years = float(history_years)
        self.include_geo_restricted = include_geo_restricted
        self.max_workers = int(max_workers)
        self.vision_workers = int(vision_workers)
        logger.info("[ENH-DERIV v6.0] initialized | history_years=%.1f | sources=%s",
                    self.history_years, self.SOURCE_PRIORITY)

    # ═════════════════════════ PUBLIC API ═════════════════════════
    from main import DEFAULT_SYMBOL
    def fetch_all_derivatives_tickers(self, base_asset: str = DEFAULT_SYMBOL) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(365.25 * self.history_years))
        try:
            return self.fetch_historical_derivatives(base_asset, start, end, "1h")
        except Exception as e:
            logger.error("[ENH-DERIV] fetch_all_derivatives_tickers failed: %s", e, exc_info=True)
            return self._empty_result_df()

    def fetch_all_derivatives_tickers_safe(self, base_asset: str = DEFAULT_SYMBOL) -> pd.DataFrame:
        try:
            return self.fetch_all_derivatives_tickers(base_asset)
        except Exception as e:
            logger.error("[ENH-DERIV] safe wrapper failed: %s", e)
            return self._empty_result_df()

    def fetch_historical_derivatives(
        self, base_asset: str, start: datetime, end: datetime,
        resample_freq: str = "1h", symbol_spot: Optional[str] = None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        واکشی و کش‌کردن داده‌های تاریخی مشتقات.

        ✅ v2.0: کش ۱۲ ساعته برای نتیجهٔ نهایی. اگر بازهٔ درخواست‌شده با دقت
        یک ساعت با کش قبلی مطابق باشد، بدون هیچ درخواست شبکه‌ای برمی‌گردد.
        برای دور زدن کش، ``force_refresh=True`` بدهید.
        """
        start, end = _utc(start), _utc(end)
        if end <= start:
            raise ValueError(f"end ({end}) must be after start ({start})")
        asset = str(base_asset).upper()

        # ── 12h cache lookup (hour-granularity key) ──
        cache_key = (
            f"{asset}|"
            f"{start.strftime('%Y%m%d%H')}|"
            f"{end.strftime('%Y%m%d%H')}|"
            f"{resample_freq}"
        )
        if not force_refresh:
            cached = _cache_get("enh_deriv_final", cache_key, FUNDAMENTAL_CACHE_TTL)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                ts_dtype = (
                    str(cached["timestamp"].dtype)
                    if "timestamp" in cached.columns else "n/a"
                )
                logger.info(
                    "[ENH-DERIV] ✅ cache hit (TTL=%s) | key=%s | "
                    "shape=%s | timestamp_dtype=%s",
                    FUNDAMENTAL_CACHE_TTL, cache_key, cached.shape, ts_dtype,
                )
                return cached.copy()
            logger.info(
                "[ENH-DERIV] ❌ cache miss (TTL=%s) | key=%s — fetching",
                FUNDAMENTAL_CACHE_TTL, cache_key,
            )

        tasks: Dict[str, Callable[[], Optional[pd.DataFrame]]] = {
            "funding:binance": lambda: self._funding_binance(asset, start, end),
            "funding:deribit": lambda: self._funding_deribit(asset, start, end),
            "funding:bitmex": lambda: self._funding_bitmex(asset, start, end),
            "funding:kraken": lambda: self._funding_kraken(asset, start, end),
            "funding:okx": lambda: self._funding_okx(asset, start, end),
            "funding:hyperliquid": lambda: self._funding_hyperliquid(asset, start, end),
            "funding:bybit": lambda: self._funding_bybit(asset, start, end),
            "oi:binance": lambda: self._binance_metrics(asset, start, end),
            "oi:okx": lambda: self._okx_oi(asset, start, end),
            "vol:dvol": lambda: self._deribit_dvol(asset, start, end),
            "px:perp": lambda: self._binance_perp_klines(asset, start, end),
            "px:premium": lambda: self._binance_premium(asset, start, end),
            "px:coinbase": lambda: self._coinbase_close(asset, start, end),
        }
        logger.info("[ENH-DERIV] fetching | asset=%s | %s → %s | freq=%s",
                    asset, start.date(), end.date(), resample_freq)

        results: Dict[str, Optional[pd.DataFrame]] = {}
        status: Dict[str, str] = {}

        def _run(item: Tuple[str, Callable]) -> Tuple[str, Optional[pd.DataFrame], str]:
            name, fn = item
            try:
                df = fn()
                if df is None or df.empty:
                    return name, None, "empty"
                return name, df, (
                    f"ok ({len(df)} rows, "
                    f"{df['timestamp'].min():%Y-%m-%d}→{df['timestamp'].max():%Y-%m-%d})"
                )
            except Exception as e:
                logger.warning("[ENH-DERIV] %s failed: %s", name, e)
                return name, None, f"error: {type(e).__name__}"

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for name, df, st in ex.map(_run, list(tasks.items())):
                results[name], status[name] = df, st

        if all(v is None for v in results.values()):
            logger.error("[ENH-DERIV] ❌ NO DATA from any source | %s", status)
            return self._empty_result_df()

        result = self._aggregate_sources(
            results, start, end, resample_freq
        ).reset_index()
        logger.info("[ENH-DERIV] ✅ aggregated | shape=%s | status=%s",
                    result.shape, status)

        # ── 12h cache store ──
        if not force_refresh and result is not None and len(result) > 0:
            try:
                _cache_put("enh_deriv_final", cache_key, result.copy())
                logger.info(
                    "[ENH-DERIV] 💾 cached (TTL=%s) | key=%s | shape=%s",
                    FUNDAMENTAL_CACHE_TTL, cache_key, result.shape,
                )
            except Exception as exc:
                logger.debug("[ENH-DERIV] cache put failed: %s", exc)

        return result
        
    @classmethod
    def _empty_result_df(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", *cls.OUTPUT_COLUMNS])

    # ═════════════════════════ FUNDING SOURCES ═════════════════════════
    def _funding_binance(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        sym = f"{asset}USDT"
        s0 = max(start, datetime(2019, 9, 1, tzinfo=timezone.utc))
        df = _bv_series("futures/um", "fundingRate", sym, None, s0, end,
                        _parse_bv_funding, "bv_funding", daily_fallback=False,
                        workers=self.vision_workers)
        frames = [df] if df is not None else []
        tail_start = (df["timestamp"].max().to_pydatetime() if df is not None else s0)
        if end - tail_start > timedelta(hours=9):          # ماه جاری روی Vision نیست
            tail = self._binance_funding_rest(sym, tail_start, end)
            if tail is not None:
                frames.append(tail)
        if not frames:
            return None
        return _clip(_tidy(pd.concat(frames, ignore_index=True), ["funding_rate"]), start, end)

    @staticmethod
    def _binance_funding_rest(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        url, rows, cursor = "https://fapi.binance.com/fapi/v1/fundingRate", [], _ms(start)
        for _ in range(20):
            p = _http("GET", url, params={"symbol": symbol, "startTime": cursor,
                                          "endTime": _ms(end), "limit": 1000})
            if not isinstance(p, list) or not p:
                break
            rows.extend(p)
            if len(p) < 1000:
                break
            cursor = int(p[-1]["fundingTime"]) + 1
        if not rows:
            return None
        df = pd.DataFrame(rows)
        return pd.DataFrame({"timestamp": _epoch_to_ts(df["fundingTime"]),
                             "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce")})

    def _funding_deribit(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        inst = f"{asset}-PERPETUAL" if asset in ("BTC", "ETH") else f"{asset}_USDC-PERPETUAL"
        url = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
        s0 = max(start, datetime(2019, 4, 1, tzinfo=timezone.utc))

        def fn(a: datetime, b: datetime) -> Optional[pd.DataFrame]:
            p = _http("GET", url, params={"instrument_name": inst,
                                          "start_timestamp": _ms(a), "end_timestamp": _ms(b)})
            if not isinstance(p, dict) or "error" in p:
                return None
            rows = p.get("result") or []
            if not rows:
                return pd.DataFrame(columns=["timestamp", "funding_rate"])
            df = pd.DataFrame(rows)
            col, mult = ("interest_8h", 1.0) if "interest_8h" in df else ("interest_1h", 8.0)
            return pd.DataFrame({"timestamp": _epoch_to_ts(df["timestamp"]),
                                 "funding_rate": pd.to_numeric(df[col], errors="coerce") * mult})

        out = _fetch_windows("deribit_fund", inst, _windows(s0, end, timedelta(days=15)), fn, workers=3)
        return _tidy(out, ["funding_rate"])

    def _funding_bitmex(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        sym = ("XBT" if asset == "BTC" else asset) + "USD"
        url, iso = "https://www.bitmex.com/api/v1/funding", "%Y-%m-%dT%H:%M:%S.000Z"

        def fn(a: datetime, b: datetime) -> Optional[pd.DataFrame]:
            p = _http("GET", url, params={"symbol": sym, "startTime": a.strftime(iso),
                                          "endTime": b.strftime(iso), "count": 500, "reverse": "false"})
            if not isinstance(p, list):
                return None
            if not p:
                return pd.DataFrame(columns=["timestamp", "funding_rate"])
            df = pd.DataFrame(p)
            return pd.DataFrame({"timestamp": pd.to_datetime(df["timestamp"], utc=True),
                                 "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce")})

        # ۱۶۰ روز × ۳ تسویه = ۴۸۰ رکورد (< سقف ۵۰۰)
        out = _fetch_windows("bitmex_fund", sym, _windows(start, end, timedelta(days=160)), fn)
        return _tidy(out, ["funding_rate"])

    def _funding_kraken(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        sym = f"PF_{'XBT' if asset == 'BTC' else asset}USD"
        # ✅ v2.0: کش ۱۲ ساعته (قبلاً ۱ ساعت بود — داده funding هر ۸h می‌آید،
        # ۱۲h TTL هنوز هم داده‌های تازه را از منابع دیگر (Binance/Deribit) تأمین می‌کند)
        c = _cache_get("kraken_fund", sym, FUNDAMENTAL_CACHE_TTL)
        if not isinstance(c, pd.DataFrame):
            p = _http(
                "GET",
                "https://futures.kraken.com/derivatives/api/v3/historicalfundingrates",
                params={"symbol": sym}, timeout=60,
            )
            rates = p.get("rates") if isinstance(p, dict) else None
            if not rates:
                return None
            df = pd.DataFrame(rates)
            # relativeFundingRate = نرخ «ساعتی» → ×۸ برای معادل ۸ ساعته
            c = pd.DataFrame({
                "timestamp": pd.to_datetime(df["timestamp"], utc=True),
                "funding_rate": pd.to_numeric(df["relativeFundingRate"], errors="coerce") * 8.0,
            })
            _cache_put("kraken_fund", sym, c)
            logger.info(
                "[KRAKEN-FUND] 💾 cached (TTL=%s) | sym=%s | rows=%d",
                FUNDAMENTAL_CACHE_TTL, sym, len(c),
            )
        else:
            logger.info(
                "[KRAKEN-FUND] ✅ cache hit (TTL=%s) | sym=%s | rows=%d",
                FUNDAMENTAL_CACHE_TTL, sym, len(c),
            )
        return _clip(_tidy(c, ["funding_rate"]), start, end)
        
    def _funding_okx(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        url, inst = "https://www.okx.com/api/v5/public/funding-rate-history", f"{asset}-USDT-SWAP"
        start_ms, after, rows = _ms(start), _ms(end), []
        for _ in range(80):
            p = _http("GET", url, params={"instId": inst, "after": str(after), "limit": "100"})
            if not isinstance(p, dict) or p.get("code") != "0":
                break
            data = p.get("data") or []
            if not data:
                break
            rows.extend(data)
            try:
                oldest = int(data[-1]["fundingTime"])
            except (KeyError, ValueError, TypeError):
                break
            if oldest <= start_ms:
                break
            after = oldest
        if not rows:
            return None
        df = pd.DataFrame(rows)
        return _clip(_tidy(pd.DataFrame({
            "timestamp": _epoch_to_ts(df["fundingTime"]),
            "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce")}), ["funding_rate"]), start, end)

    def _funding_hyperliquid(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        url = "https://api.hyperliquid.xyz/info"
        s0 = max(start, datetime(2023, 5, 1, tzinfo=timezone.utc))
        if s0 >= end:
            return None

        def fn(a: datetime, b: datetime) -> Optional[pd.DataFrame]:
            p = _http("POST", url, json_body={"type": "fundingHistory", "coin": asset,
                                              "startTime": _ms(a), "endTime": _ms(b)})
            if not isinstance(p, list):
                return None
            if not p:
                return pd.DataFrame(columns=["timestamp", "funding_rate"])
            df = pd.DataFrame(p)
            # نرخ ساعتی → ×۸
            return pd.DataFrame({"timestamp": _epoch_to_ts(df["time"]),
                                 "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce") * 8.0})

        out = _fetch_windows("hl_fund", asset, _windows(s0, end, timedelta(days=20)), fn)  # ≤۵۰۰ رکورد
        return _tidy(out, ["funding_rate"])

    def _funding_bybit(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        if not self.include_geo_restricted:
            return None
        sym, url = f"{asset}USDT", "https://api.bybit.com/v5/market/funding/history"
        s0 = max(start, datetime(2020, 6, 1, tzinfo=timezone.utc))

        def fn(a: datetime, b: datetime) -> Optional[pd.DataFrame]:
            p = _http("GET", url, params={"category": "linear", "symbol": sym,
                                          "startTime": _ms(a), "endTime": _ms(b), "limit": 200})
            if not isinstance(p, dict) or p.get("retCode") != 0:
                return None
            lst = (p.get("result") or {}).get("list") or []
            if not lst:
                return pd.DataFrame(columns=["timestamp", "funding_rate"])
            df = pd.DataFrame(lst)
            return pd.DataFrame({"timestamp": _epoch_to_ts(df["fundingRateTimestamp"]),
                                 "funding_rate": pd.to_numeric(df["fundingRate"], errors="coerce")})

        out = _fetch_windows("bybit_fund", sym, _windows(s0, end, timedelta(days=60)), fn)
        return _tidy(out, ["funding_rate"])

    # ═════════════════════════ OI / FLOW / VOL ═════════════════════════
    def _binance_metrics(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        """OI + Long/Short ratios + taker ratio (۵ دقیقه‌ای، فایل daily). تاریخچه از ~۲۰۲۱-۱۲."""
        s0 = max(start, datetime(2020, 9, 1, tzinfo=timezone.utc))
        df = _bv_series("futures/um", "metrics", f"{asset}USDT", None, s0, end,
                        _parse_bv_metrics, "bv_metrics", header=True, use_monthly=False,
                        workers=self.vision_workers)
        return _clip(_tidy(df, ["oi_usd", "ls_global", "taker_ratio"]), start, end)

    def _okx_oi(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        url = "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history"
        inst, start_ms, cursor, rows = f"{asset}-USDT-SWAP", _ms(start), _ms(end), []
        for _ in range(80):
            p = _http("GET", url, params={"instId": inst, "period": "1H",
                                          "end": str(cursor), "limit": "100"})
            if not isinstance(p, dict) or p.get("code") != "0":
                break
            data = p.get("data") or []
            if not data:
                break
            rows.extend(data)
            try:
                oldest = int(data[-1][0])
            except (IndexError, ValueError, TypeError):
                break
            if oldest <= start_ms:
                break
            cursor = oldest - 1
        if not rows:
            return None
        # ردیف‌ها: [ts, oi(contracts), oiCcy, oiUsd]
        arr = pd.DataFrame([(r[0], r[3]) for r in rows if len(r) >= 4], columns=["ts", "oi_usd"])
        out = pd.DataFrame({"timestamp": _epoch_to_ts(arr["ts"]) + pd.Timedelta(hours=1),
                            "open_interest_usd": pd.to_numeric(arr["oi_usd"], errors="coerce")})
        return _clip(_tidy(out, ["open_interest_usd"]), start, end)

    def _deribit_dvol(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        if asset not in ("BTC", "ETH"):
            return None
        url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
        s0 = max(start, datetime(2021, 3, 1, tzinfo=timezone.utc))

        def fn(a: datetime, b: datetime) -> Optional[pd.DataFrame]:
            p = _http("GET", url, params={"currency": asset, "start_timestamp": _ms(a),
                                          "end_timestamp": _ms(b), "resolution": "3600"})
            if not isinstance(p, dict) or "error" in p:
                return None
            data = (p.get("result") or {}).get("data") or []
            if not data:
                return pd.DataFrame(columns=["timestamp", "dvol"])
            arr = pd.DataFrame(data).iloc[:, :5]
            arr.columns = ["ts", "o", "h", "l", "c"]
            return pd.DataFrame({"timestamp": _epoch_to_ts(arr["ts"]) + pd.Timedelta(hours=1),  # close-time
                                 "dvol": pd.to_numeric(arr["c"], errors="coerce")})

        out = _fetch_windows("deribit_dvol", asset, _windows(s0, end, timedelta(days=30)), fn, workers=2)
        return _tidy(out, ["dvol"])

    # ═════════════════════════ PRICE PANELS ═════════════════════════
    def _binance_perp_klines(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        s0 = max(start, datetime(2019, 9, 1, tzinfo=timezone.utc)) - timedelta(days=2)
        df = _bv_series("futures/um", "klines", f"{asset}USDT", "1h", s0, end,
                        _parse_bv_klines, "bv_perp_1h", workers=self.vision_workers)
        if df is None:
            return None
        out = pd.DataFrame({"timestamp": df["timestamp"] + pd.Timedelta(hours=1),   # close-time
                            "perp_close": df["close"], "perp_quote_vol": df["quote_volume"]})
        return _clip(_tidy(out, ["perp_close"]), start, end)

    def _binance_premium(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        s0 = max(start, datetime(2019, 9, 1, tzinfo=timezone.utc)) - timedelta(days=2)
        df = _bv_series("futures/um", "premiumIndexKlines", f"{asset}USDT", "1h", s0, end,
                        _parse_bv_klines, "bv_premium_1h", workers=self.vision_workers)
        if df is None:
            return None
        out = pd.DataFrame({"timestamp": df["timestamp"] + pd.Timedelta(hours=1),
                            "premium": df["close"]})
        return _clip(_tidy(out, ["premium"]), start, end)

    def _coinbase_close(self, asset: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        df = _coinbase_ohlcv(asset, start - timedelta(hours=2), end, 3600)
        if df is None:
            return None
        out = pd.DataFrame({"timestamp": df["timestamp"] + pd.Timedelta(hours=1),   # close-time
                            "cb_close": df["close"]})
        return _clip(_tidy(out, ["cb_close"]), start, end)

    # ═════════════════════════ AGGREGATION ═════════════════════════
    @staticmethod
    def _ser(df: Optional[pd.DataFrame], col: str) -> Optional[pd.Series]:
        if df is None or df.empty or col not in df.columns:
            return None
        s = df.dropna(subset=[col]).set_index("timestamp")[col].astype("float64")
        s = s[~s.index.duplicated(keep="last")].sort_index()
        return s if len(s) else None

    @staticmethod
    def _to_grid(s: Optional[pd.Series], grid: pd.DatetimeIndex, freq: str,
                 ffill_hours: float = 0.0) -> pd.Series:
        """هر رویداد به ceil(freq) نگاشت می‌شود → در برچسب T فقط اطلاعات ≤ T دیده می‌شود."""
        if s is None:
            return pd.Series(np.nan, index=grid, dtype="float64")
        out = s.groupby(s.index.ceil(freq)).last().reindex(grid)
        if ffill_hours > 0:
            limit = max(1, int(np.ceil(pd.Timedelta(hours=ffill_hours) / pd.Timedelta(freq))))
            out = out.ffill(limit=limit)
        return out

    @staticmethod
    def _check_funding_scale(fund: Dict[str, pd.DataFrame]) -> None:
        meds = {n: float(np.nanmedian(np.abs(d["funding_rate"].to_numpy())))
                for n, d in fund.items() if len(d)}
        if len(meds) < 2:
            return
        ref = float(np.median(list(meds.values())))
        for n, m in meds.items():
            if ref > 0 and (m / ref > 5 or m / ref < 0.2):
                logger.warning("[ENH-DERIV] ⚠️ funding scale mismatch: %s median|rate|=%.2e vs ref=%.2e "
                               "(واحد/interval این منبع را بررسی کنید)", n, m, ref)

    def _aggregate_sources(self, res: Dict[str, Optional[pd.DataFrame]],
                           start: datetime, end: datetime, freq: str) -> pd.DataFrame:
        t0, t1 = pd.Timestamp(start).ceil(freq), pd.Timestamp(end).floor(freq)
        if t1 < t0:
            raise ValueError("range shorter than resample_freq")
        grid = pd.date_range(t0, t1, freq=freq)
        grid.name = "timestamp"
        out = pd.DataFrame(index=grid)
        nan = lambda: pd.Series(np.nan, index=grid, dtype="float64")  # noqa: E731

        # ── Funding ──
        fund = {n.split(":", 1)[1]: d for n, d in res.items()
                if n.startswith("funding:") and d is not None}
        self._check_funding_scale(fund)
        F = pd.DataFrame({n: self._to_grid(self._ser(d, "funding_rate"), grid, freq,
                                           self.FUNDING_FFILL_HOURS) for n, d in fund.items()},
                         index=grid)
        if F.shape[1]:
            out["funding_rate_median"], out["funding_rate_mean"] = F.median(axis=1), F.mean(axis=1)
            out["funding_rate_std"], out["funding_rate_max"] = F.std(axis=1), F.max(axis=1)
            out["funding_rate_min"] = F.min(axis=1)
            out["n_markets"] = F.notna().sum(axis=1).astype(int)
        else:
            for c in ("funding_rate_median", "funding_rate_mean", "funding_rate_std",
                      "funding_rate_max", "funding_rate_min"):
                out[c] = nan()
            out["n_markets"] = 0

        # ── OI / flow ──
        bm, okx = res.get("oi:binance"), res.get("oi:okx")
        g = lambda df, col, h: self._to_grid(self._ser(df, col), grid, freq, h)  # noqa: E731
        oi_b, oi_o = g(bm, "oi_usd", 2.0), g(okx, "open_interest_usd", 3.0)
        oi_df = pd.DataFrame({"b": oi_b, "o": oi_o})
        out["open_interest_usd_total"] = oi_df.sum(axis=1, min_count=1)   # NaN اگر هیچ منبعی نبود
        out["oi_binance_usd"], out["oi_binance_coin"] = oi_b, g(bm, "oi_coin", 2.0)
        out["n_oi_sources"] = oi_df.notna().sum(axis=1).astype(int)
        out["ls_ratio_top_accounts"] = g(bm, "ls_top_accounts", 2.0)
        out["ls_ratio_top_positions"] = g(bm, "ls_top_positions", 2.0)
        out["ls_ratio_global"] = g(bm, "ls_global", 2.0)
        out["taker_buysell_ratio"] = g(bm, "taker_ratio", 2.0)

        # ── Price / basis / premium / volume ──
        perp, prem, cb = res.get("px:perp"), res.get("px:premium"), res.get("px:coinbase")
        p_perp, p_cb = g(perp, "perp_close", 3.0), g(cb, "cb_close", 3.0)
        out["price_median"] = pd.DataFrame({"a": p_perp, "b": p_cb}).median(axis=1)
        out["basis_median"] = g(prem, "premium", 2.0)
        cb_prem = (p_cb / p_perp - 1.0) * 100.0
        out["coinbase_premium_pct"] = cb_prem.replace([np.inf, -np.inf], np.nan)
        out["spread_median"] = out["coinbase_premium_pct"].abs()
        sv = self._ser(perp, "perp_quote_vol")
        out["volume_24h_usd_total"] = self._to_grid(
            sv.rolling("24h", min_periods=12).sum() if sv is not None else None, grid, freq, 2.0)
        out["dvol"] = g(res.get("vol:dvol"), "dvol", 2.0)

        return out[list(self.OUTPUT_COLUMNS)]


class CryptoCompareOHLCVFetcher:
    """
    OHLCV تاریخی چندمنبعی با pagination کامل (پیش‌فرض ۶ سال).

    نام کلاس برای سازگاری حفظ شده؛ CryptoCompare اکنون فقط یکی از منابع است.
    اولویت: Bitstamp(USD) → Coinbase(USD) → Binance Vision(USDT) → CryptoCompare(≥1H)
    شکاف‌های منبع اول از منبع بعدی پر می‌شود (combine_first).
    timestamp = open-time کندل (قرارداد MT5). خروجی: ستون 'timestamp' (یا index با as_index=True).
    """

    BASE_URL = "https://min-api.cryptocompare.com/data/v2"
    SOURCE_PRIORITY: Tuple[str, ...] = ("bitstamp", "coinbase", "binance_vision", "cryptocompare")
    TF_MINUTES = {"1M": 1, "5M": 5, "15M": 15, "30M": 30, "1H": 60, "4H": 240, "1D": 1440}
    _BV_INTERVAL = {1: "1m", 5: "5m", 15: "15m", 30: "30m", 60: "1h", 240: "4h", 1440: "1d"}
    _BITSTAMP_STEPS = {1, 3, 5, 15, 30, 60, 120, 240, 360, 720, 1440}
    _CB_GRAN = {1: (60, 1), 5: (300, 5), 15: (900, 15), 30: (900, 30),
                60: (3600, 60), 240: (3600, 240), 1440: (86400, 1440)}  # (granularity, resample-to)
    TARGET_COVERAGE = 0.995

    def __init__(self, api_key: Optional[str] = None, history_years: float = 6.0,
                 sources: Optional[Sequence[str]] = None, quote: str = "USD"):
        self.api_key = api_key
        self.history_years = float(history_years)
        self.sources = tuple(sources) if sources else self.SOURCE_PRIORITY
        self.quote = quote.upper()
        self._headers: Dict[str, str] = {"authorization": f"Apikey {api_key}"} if api_key else {}
    from main import DEFAULT_SYMBOL
    # ───────────────────────── Public ─────────────────────────
    def fetch_ohlcv(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = "30M",
                    limit: Optional[int] = None, start: Optional[datetime] = None,
                    end: Optional[datetime] = None, as_index: bool = False) -> pd.DataFrame:
        tf = self.TF_MINUTES.get(str(timeframe).upper())
        if tf is None:
            logger.warning("[OHLCV] unsupported timeframe: %s", timeframe)
            return pd.DataFrame()
        symbol = symbol.upper()
        end = _utc(end) if end else datetime.now(timezone.utc)
        if start is not None:
            start = _utc(start)
        elif limit:
            start = end - timedelta(minutes=tf * int(limit))
        else:
            start = end - timedelta(days=int(365.25 * self.history_years))
        if tf < 5 and (end - start) > timedelta(days=400):
            logger.warning("[OHLCV] %s %dm روی بازهٔ طولانی بسیار سنگین است (هزاران درخواست)", symbol, tf)

        result: Optional[pd.DataFrame] = None
        for src in self.sources:
            fn = getattr(self, f"_src_{src}", None)
            if fn is None:
                continue
            try:
                df = fn(symbol, tf, start, end)
            except Exception as e:
                logger.warning("[OHLCV] source %s failed: %s", src, e)
                continue
            df = self._clean(df, start, end)
            if df is None:
                logger.info("[OHLCV] %s: no data", src)
                continue
            result = df if result is None else self._combine(result, df)
            cov = self._coverage(result, tf, start, end)
            logger.info("[OHLCV] %s %s via %s → %d rows | coverage=%.2f%% | %s → %s",
                        symbol, timeframe, src, len(result), cov * 100,
                        result["timestamp"].iloc[0].date(), result["timestamp"].iloc[-1].date())
            if cov >= self.TARGET_COVERAGE:
                break

        if result is None:
            return pd.DataFrame()
        if limit:
            result = result.tail(int(limit)).reset_index(drop=True)
        return result.set_index("timestamp") if as_index else result
    from main import DEFAULT_SYMBOL
    def fetch_ohlcv_safe(self, symbol: str = DEFAULT_SYMBOL, timeframe: str = "30M",
                         limit: Optional[int] = None, **kw: Any) -> pd.DataFrame:
        try:
            return self.fetch_ohlcv(symbol, timeframe, limit, **kw)
        except Exception as exc:
            logger.error("[OHLCV] safe wrapper failed: %s", exc, exc_info=True)
            return pd.DataFrame()

    # ───────────────────────── Sources ─────────────────────────
    def _src_bitstamp(self, symbol: str, tf: int, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        if tf not in self._BITSTAMP_STEPS:
            return None
        step = tf * 60
        pair = f"{symbol.lower()}{self.quote.lower()}"
        url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/"

        def fn(a: datetime, b: datetime) -> Optional[pd.DataFrame]:
            p = _http("GET", url, params={"step": step, "limit": 1000,
                                          "start": int(a.timestamp()), "end": int(b.timestamp())})
            if not isinstance(p, dict) or "data" not in p:
                return None
            rows = (p.get("data") or {}).get("ohlc") or []
            if not rows:
                return pd.DataFrame(columns=["timestamp"] + _OHLCV)
            df = pd.DataFrame(rows)
            out = pd.DataFrame({"timestamp": _epoch_to_ts(df["timestamp"])})
            for c in _OHLCV:
                out[c] = pd.to_numeric(df[c], errors="coerce")
            return out

        return _fetch_windows("bitstamp_ohlc", f"{pair}{step}",
                              _windows(start, end, timedelta(seconds=step * 999)), fn)

    def _src_coinbase(self, symbol: str, tf: int, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        gran, target = self._CB_GRAN[tf]
        df = _coinbase_ohlcv(symbol, start, end, gran, self.quote)
        if df is None:
            return None
        return df if target == gran // 60 else _resample_ohlcv(df, target)

    def _src_binance_vision(self, symbol: str, tf: int, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        df = _bv_series("spot", "klines", f"{symbol}USDT", self._BV_INTERVAL[tf], start, end,
                        _parse_bv_klines, f"bv_spot_{tf}")
        return None if df is None else df[["timestamp"] + _OHLCV]

    def _src_cryptocompare(self, symbol: str, tf: int, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        if tf < 60:        # histominute در Free Tier عمق تاریخی ندارد
            return None
        endpoint, agg = ("histohour", tf // 60) if tf < 1440 else ("histoday", tf // 1440)
        url, cursor, start_s, rows = f"{self.BASE_URL}/{endpoint}", int(end.timestamp()), int(start.timestamp()), []
        for _ in range(300):
            p = _http("GET", url, params={"fsym": symbol, "tsym": "USD", "limit": 2000,
                                          "toTs": cursor, "aggregate": agg},
                      headers=self._headers or None)
            d1 = p.get("Data") if isinstance(p, dict) else None
            data = d1.get("Data") if isinstance(d1, dict) else None
            if not data:
                break
            rows.extend(data)
            first = int(data[0]["time"])
            if first <= start_s:
                break
            cursor = first - 1
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if not {"time", "open", "high", "low", "close", "volumefrom"} <= set(df.columns):
            return None
        out = pd.DataFrame({"timestamp": pd.to_datetime(df["time"], unit="s", utc=True)})
        for c in ("open", "high", "low", "close"):
            out[c] = pd.to_numeric(df[c], errors="coerce")
        out["volume"] = pd.to_numeric(df["volumefrom"], errors="coerce")
        return out

    # ───────────────────────── Helpers ─────────────────────────
    @staticmethod
    def _clean(df: Optional[pd.DataFrame], start: datetime, end: datetime) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        out = df[["timestamp"] + _OHLCV].copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
        for c in _OHLCV:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        out = out.dropna(subset=["open", "high", "low", "close"])
        out = out[(out[["open", "high", "low", "close"]] > 0).all(axis=1)]
        out = out.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
        return _clip(out.reset_index(drop=True), start, end)

    @staticmethod
    def _combine(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
        a, b = primary.set_index("timestamp"), secondary.set_index("timestamp")
        return a.combine_first(b).sort_index().reset_index()

    @staticmethod
    def _coverage(df: pd.DataFrame, tf: int, start: datetime, end: datetime) -> float:
        expected = max(1, int((end - start) / timedelta(minutes=tf)))
        return min(1.0, len(df) / expected)
        
class SentimentFetcher:
    """
    Sentiment + Macro + On-chain + Liquidity (همه رایگان و keyless).

    v3.0:
      • fetch_fear_greed       : alternative.me (limit=0 → کل تاریخچه از ۲۰۱۸)
      • fetch_macro_fred       : FRED (yields, VIX, USD, SP500, NASDAQ, WTI, HY spread, ...)
      • fetch_onchain_daily    : Blockchain.com (BTC) + CoinMetrics Community
      • fetch_stablecoin_supply: DefiLlama (عرضهٔ کل / USDT / USDC)
      • fetch_defi_tvl         : DefiLlama

    همهٔ سری‌های روزانه با lag انتشار شیفت می‌خورند تا leakage نداشته باشند.
    خروجی همه: DataFrame با ستون 'timestamp' (UTC) = لحظهٔ «قابل‌دسترس شدن» داده.
    """

    CACHE_TTL: timedelta = FUNDAMENTAL_CACHE_TTL

    FRED_SERIES: Dict[str, str] = {
        "us10y": "DGS10", "us2y": "DGS2", "curve_10y2y": "T10Y2Y", "fed_funds": "DFF",
        "vix": "VIXCLS", "usd_broad": "DTWEXBGS", "sp500": "SP500", "nasdaq": "NASDAQCOM",
        "wti": "DCOILWTICO", "hy_spread": "BAMLH0A0HYM2", "breakeven10y": "T10YIE",
    }

    BLOCKCHAIN_CHARTS: Dict[str, str] = {
        "hash-rate": "bc_hashrate", "difficulty": "bc_difficulty",
        "n-transactions": "bc_tx_count", "n-unique-addresses": "bc_unique_addr",
        "estimated-transaction-volume-usd": "bc_tx_volume_usd",
        "miners-revenue": "bc_miners_revenue_usd", "transaction-fees-usd": "bc_fees_usd",
        "mempool-size": "bc_mempool_bytes",
    }
    COINMETRICS: Dict[str, str] = {
        "CapMVRVCur": "cm_mvrv", "AdrActCnt": "cm_active_addr", "TxCnt": "cm_tx_count",
        "TxTfrValAdjUSD": "cm_transfer_val_usd", "FeeTotUSD": "cm_fees_usd",
        "HashRate": "cm_hashrate", "NVTAdj": "cm_nvt", "IssTotUSD": "cm_issuance_usd",
    }

    @staticmethod
    def fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
        """limit=0 → همهٔ تاریخچه (v1/v2 با limit=2000 فقط ~۵.۵ سال می‌داد)."""
        key = f"fear_greed_{int(limit)}"
        c = SentimentFetcher._cached(key, SentimentFetcher.CACHE_TTL)
        if c is not None:
            logger.info("[SENTIMENT] ✅ fear_greed cache hit (TTL=%s) | key=%s",
                        SentimentFetcher.CACHE_TTL, key)
            return c
        p = _http("GET", "https://api.alternative.me/fng/",
                  params={"limit": int(limit), "format": "json"}, timeout=45)
        recs = p.get("data") if isinstance(p, dict) else None
        if not recs:
            logger.warning("[SENTIMENT] alternative.me returned no data")
            return SentimentFetcher._empty(["fear_greed"])
        df = pd.DataFrame(recs)
        if "timestamp" not in df.columns or "value" not in df.columns:
            return SentimentFetcher._empty(["fear_greed"])
        out = pd.DataFrame({"timestamp": _epoch_to_ts(df["timestamp"]),
                            "fear_greed": pd.to_numeric(df["value"], errors="coerce")})
        out = _tidy(out, ["fear_greed"])
        if out is None:
            return SentimentFetcher._empty(["fear_greed"])
        _cache_put("sentiment", key, out)
        logger.info("[SENTIMENT] fear_greed: %d rows [%s → %s] | cached TTL=%s",
                    len(out), out["timestamp"].iloc[0].date(),
                    out["timestamp"].iloc[-1].date(), SentimentFetcher.CACHE_TTL)
        return out

    @staticmethod
    def _empty(cols: Sequence[str]) -> pd.DataFrame:
        return pd.DataFrame(columns=["timestamp", *cols])

    @staticmethod
    def _cached(key: str, ttl: timedelta) -> Optional[pd.DataFrame]:
        c = _cache_get("sentiment", key, ttl)
        return c if isinstance(c, pd.DataFrame) and not c.empty else None

    # ─────────────────────────── Fear & Greed ───────────────────────────
    @staticmethod
    def fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
        """limit=0 → همهٔ تاریخچه (v1/v2 با limit=2000 فقط ~۵.۵ سال می‌داد)."""
        key = f"fear_greed_{int(limit)}"
        c = SentimentFetcher._cached(key, timedelta(hours=6))
        if c is not None:
            return c
        p = _http("GET", "https://api.alternative.me/fng/",
                  params={"limit": int(limit), "format": "json"}, timeout=45)
        recs = p.get("data") if isinstance(p, dict) else None
        if not recs:
            logger.warning("[SENTIMENT] alternative.me returned no data")
            return SentimentFetcher._empty(["fear_greed"])
        df = pd.DataFrame(recs)
        if "timestamp" not in df.columns or "value" not in df.columns:
            return SentimentFetcher._empty(["fear_greed"])
        out = pd.DataFrame({"timestamp": _epoch_to_ts(df["timestamp"]),
                            "fear_greed": pd.to_numeric(df["value"], errors="coerce")})
        out = _tidy(out, ["fear_greed"])
        if out is None:
            return SentimentFetcher._empty(["fear_greed"])
        _cache_put("sentiment", key, out)
        logger.info("[SENTIMENT] fear_greed: %d rows [%s → %s]", len(out),
                    out["timestamp"].iloc[0].date(), out["timestamp"].iloc[-1].date())
        return out

    # ─────────────────────────── Macro (FRED) ───────────────────────────
    @staticmethod
    def fetch_macro_fred(start: Optional[datetime] = None,
                         series: Optional[Dict[str, str]] = None,
                         release_lag_days: int = 2) -> pd.DataFrame:
        """
        FRED CSV بدون API key. مقدار تاریخ D با lag (پیش‌فرض ۲ روز) قابل‌دسترس فرض می‌شود
        (H-15/VIX با تأخیر منتشر می‌شوند) → عدم leakage.
        """
        series = series or SentimentFetcher.FRED_SERIES
        cosd = (
            _utc(start) if start
            else datetime(2018, 1, 1, tzinfo=timezone.utc)
        ).strftime("%Y-%m-%d")
        key = f"fred_{cosd}_{','.join(sorted(series))}_{release_lag_days}"
        c = SentimentFetcher._cached(key, SentimentFetcher.CACHE_TTL)
        if c is not None:
            logger.info("[MACRO] ✅ fred cache hit (TTL=%s) | key=%s",
                        SentimentFetcher.CACHE_TTL, key)
            return c

        def one(item: Tuple[str, str]) -> Optional[pd.Series]:
            name, sid = item
            txt = _http("GET", "https://fred.stlouisfed.org/graph/fredgraph.csv",
                        params={"id": sid, "cosd": cosd}, kind="text", timeout=45)
            if not isinstance(txt, str) or len(txt) < 20:
                return None
            try:
                raw = pd.read_csv(io.StringIO(txt))
                d = pd.to_datetime(raw.iloc[:, 0], utc=True, errors="coerce")
                v = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
                s = pd.Series(v.to_numpy(), index=d, name=name).dropna()
                return s[~s.index.isna()]
            except Exception as e:
                logger.debug("[FRED] %s parse failed: %s", sid, e)
                return None

        with ThreadPoolExecutor(max_workers=4) as ex:
            parts = [s for s in ex.map(one, list(series.items()))
                     if s is not None and len(s)]
        if not parts:
            return SentimentFetcher._empty(list(series))
        df = pd.concat(parts, axis=1).sort_index()
        df.index = df.index + pd.Timedelta(days=release_lag_days)
        out = df.reset_index().rename(
            columns={df.index.name or "index": "timestamp"}
        )
        out = out.rename(columns={out.columns[0]: "timestamp"})
        _cache_put("sentiment", key, out)
        logger.info("[MACRO] FRED: %d series | %d rows | cached TTL=%s",
                    len(parts), len(out), SentimentFetcher.CACHE_TTL)
        return out
        
    # ─────────────────────────── On-chain ───────────────────────────
    from main import DEFAULT_SYMBOL
    @staticmethod
    def fetch_onchain_daily(asset: str = DEFAULT_SYMBOL, start: Optional[datetime] = None,
                            lag_days: int = 1) -> pd.DataFrame:
        asset = asset.upper()
        s0 = _utc(start) if start else datetime(2018, 1, 1, tzinfo=timezone.utc)
        key = f"onchain_{asset}_{s0:%Y%m%d}_{lag_days}"
        c = SentimentFetcher._cached(key, SentimentFetcher.CACHE_TTL)
        if c is not None:
            logger.info("[ONCHAIN] ✅ cache hit (TTL=%s) | key=%s",
                        SentimentFetcher.CACHE_TTL, key)
            return c

        frames: List[pd.Series] = []

        # 1) Blockchain.com (فقط BTC)
        if asset == "BTC":
            years = int(np.ceil(
                (datetime.now(timezone.utc) - s0).days / 365.25
            )) + 1
            for chart, name in SentimentFetcher.BLOCKCHAIN_CHARTS.items():
                p = _http("GET", f"https://api.blockchain.info/charts/{chart}",
                          params={"timespan": f"{years}years", "format": "json",
                                  "sampled": "false", "cors": "true"}, timeout=60)
                vals = p.get("values") if isinstance(p, dict) else None
                if not vals:
                    continue
                d = pd.DataFrame(vals)
                s = pd.Series(pd.to_numeric(d["y"], errors="coerce").to_numpy(),
                              index=_epoch_to_ts(d["x"]).dt.floor("D"), name=name)
                frames.append(s[~s.index.duplicated(keep="last")].dropna())

        # 2) CoinMetrics Community
        def cm_fetch(metrics: List[str]) -> Optional[pd.DataFrame]:
            url, params, rows = (
                "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
                {"assets": asset.lower(), "metrics": ",".join(metrics),
                 "frequency": "1d", "start_time": f"{s0:%Y-%m-%d}",
                 "page_size": 10000},
                [],
            )
            for _ in range(20):
                p = _http("GET", url, params=params, timeout=60)
                if not isinstance(p, dict) or "data" not in p:
                    return None if not rows else pd.DataFrame(rows)
                rows.extend(p["data"])
                nxt = p.get("next_page_url")
                if not nxt:
                    break
                url, params = nxt, None
            return pd.DataFrame(rows) if rows else None

        cm = cm_fetch(list(SentimentFetcher.COINMETRICS))
        if cm is None:
            parts = [cm_fetch([m]) for m in SentimentFetcher.COINMETRICS]
            parts = [p for p in parts if p is not None and len(p)]
            if parts:
                cm = pd.concat(
                    [p.set_index("time").drop(columns=["asset"], errors="ignore")
                     for p in parts], axis=1
                ).reset_index()
        if cm is not None and len(cm) and "time" in cm.columns:
            idx = pd.to_datetime(cm["time"], utc=True, errors="coerce").dt.floor("D")
            for m, name in SentimentFetcher.COINMETRICS.items():
                if m in cm.columns:
                    s = pd.Series(
                        pd.to_numeric(cm[m], errors="coerce").to_numpy(),
                        index=idx, name=name,
                    )
                    frames.append(s[~s.index.duplicated(keep="last")].dropna())

        if not frames:
            return SentimentFetcher._empty([])
        df = pd.concat(frames, axis=1).sort_index()
        df.index = df.index + pd.Timedelta(days=1 + lag_days)
        out = df.reset_index()
        out = out.rename(columns={out.columns[0]: "timestamp"})
        _cache_put("sentiment", key, out)
        logger.info("[ONCHAIN] %s: %d columns | %d rows | cached TTL=%s",
                    asset, df.shape[1], len(out), SentimentFetcher.CACHE_TTL)
        return out
    # ─────────────────────────── Liquidity (DefiLlama) ───────────────────────────
    @staticmethod
    def fetch_stablecoin_supply(lag_days: int = 1) -> pd.DataFrame:
        key = f"stable_{lag_days}"
        c = SentimentFetcher._cached(key, SentimentFetcher.CACHE_TTL)
        if c is not None:
            logger.info("[STABLECOIN] ✅ cache hit (TTL=%s) | key=%s",
                        SentimentFetcher.CACHE_TTL, key)
            return c
        cols: List[pd.Series] = []
        for name, sid in (("stablecoin_supply_usd", None),
                          ("usdt_supply_usd", 1),
                          ("usdc_supply_usd", 2)):
            p = _http("GET", "https://stablecoins.llama.fi/stablecoincharts/all",
                      params={"stablecoin": sid} if sid else None, timeout=60)
            if not isinstance(p, list) or not p:
                continue
            idx, vals = [], []
            for r in p:
                tot = r.get("totalCirculatingUSD") or r.get("totalCirculating") or {}
                v = tot.get("peggedUSD") if isinstance(tot, dict) else None
                if v is not None:
                    idx.append(r.get("date"))
                    vals.append(v)
            if idx:
                s = pd.Series(
                    pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(),
                    index=_epoch_to_ts(idx).dt.floor("D"), name=name,
                )
                cols.append(s[~s.index.duplicated(keep="last")].dropna())
        if not cols:
            return SentimentFetcher._empty(["stablecoin_supply_usd"])
        df = pd.concat(cols, axis=1).sort_index()
        df.index = df.index + pd.Timedelta(days=1 + lag_days)
        out = df.reset_index()
        out = out.rename(columns={out.columns[0]: "timestamp"})
        _cache_put("sentiment", key, out)
        logger.info("[STABLECOIN] %d rows | cached TTL=%s",
                    len(out), SentimentFetcher.CACHE_TTL)
        return out
        
    @staticmethod
    def fetch_defi_tvl(lag_days: int = 1) -> pd.DataFrame:
        key = f"tvl_{lag_days}"
        c = SentimentFetcher._cached(key, SentimentFetcher.CACHE_TTL)
        if c is not None:
            logger.info("[TVL] ✅ cache hit (TTL=%s) | key=%s",
                        SentimentFetcher.CACHE_TTL, key)
            return c
        p = _http("GET", "https://api.llama.fi/v2/historicalChainTvl", timeout=60)
        if not isinstance(p, list) or not p:
            return SentimentFetcher._empty(["defi_tvl_usd"])
        d = pd.DataFrame(p)
        out = pd.DataFrame({
            "timestamp": _epoch_to_ts(d["date"]).dt.floor("D")
                         + pd.Timedelta(days=1 + lag_days),
            "defi_tvl_usd": pd.to_numeric(d["tvl"], errors="coerce"),
        })
        out = _tidy(out, ["defi_tvl_usd"])
        if out is None:
            return SentimentFetcher._empty(["defi_tvl_usd"])
        _cache_put("sentiment", key, out)
        logger.info("[TVL] %d rows | cached TTL=%s",
                    len(out), SentimentFetcher.CACHE_TTL)
        return out

class FundamentalFeatureBuilder:
    """
    FundamentalFeatureBuilder v5.0 — Professional Fundamental Stack (6y, leakage-aware).

    لایه‌ها:
      1. Derivatives  : funding(7 منبع)، OI، L/S، taker، basis، Coinbase premium، DVOL، حجم
      2. Sentiment    : Fear & Greed (تاریخچه کامل)
      3. Macro        : FRED (yields/VIX/USD/SP500/NASDAQ/WTI/HY/…)
      4. On-chain     : Blockchain.com + CoinMetrics
      5. Liquidity    : Stablecoin supply + DeFi TVL
      6. Derived      : z-score / تغییرات / RV / VRP / تعاملات (فقط با منبع معتبر)

    قراردادهای ضد-leakage:
      • مشتقات: برچسب = ceil(رویداد) (در fetcher)؛ سری‌های روزانه: با lag انتشار شیفت‌شده
        و merge_asof(backward, tolerance) — بدون forward-fill بی‌نهایت.
      • warm-up: برای پر بودن rolling ها از ابتدای بازه، warmup_days قبل از start هم
        گرفته و در پایان بریده می‌شود.
      • هیچ صفر جعلی؛ ستون‌های all-NaN/ثابت حذف می‌شوند؛ پوشش هر ستون در `last_report`.
    """
    from main import DEFAULT_SYMBOL
    def __init__(self, symbol: str = DEFAULT_SYMBOL, cg_api_key: Optional[str] = None,
                 cc_api_key: Optional[str] = None, history_years: float = 6.0,
                 warmup_days: int = 35, include_macro: bool = True,
                 include_onchain: bool = True, include_liquidity: bool = True,
                 dtype: Any = np.float32, include_geo_restricted: bool = True):
        self.symbol = symbol
        self.base_asset = self._extract_base_asset(symbol)
        self.history_years = float(history_years)
        self.warmup_days = int(warmup_days)
        self.include_macro, self.include_onchain = include_macro, include_onchain
        self.include_liquidity = include_liquidity
        self.dtype = dtype
        self.cg_fetcher = EnhancedDerivativesFetcher(
            cg_api_key=cg_api_key, cc_api_key=cc_api_key, history_years=history_years,
            include_geo_restricted=include_geo_restricted)
        self.cc_fetcher = CryptoCompareOHLCVFetcher(api_key=cc_api_key, history_years=history_years)
        self.last_report: pd.DataFrame = pd.DataFrame()
        logger.info("[FUND-v5.0] initialized | symbol=%s base=%s history=%.1fy",
                    symbol, self.base_asset, self.history_years)

    # ═════════════════════════ PUBLIC API ═════════════════════════
    def collect_all(self, start: datetime, end: datetime, candle_freq: str = "30min",
                    max_staleness: str = "1D", warmup_days: Optional[int] = None,
                    daily_staleness: str = "5D",
                    force_refresh: bool = False) -> pd.DataFrame:
        """
        واکشی و الحاق همهٔ فیچرهای بنیادی.

        ✅ v6.0: کش ۱۲ ساعته در سطح کل مجموعه. اگر بازه با دقت یک ساعت مطابق
        باشد و TTL نگذشته باشد، بدون هیچ درخواست شبکه‌ای، DataFrame ذخیره‌شده
        برمی‌گردد. برای دور زدن کش، ``force_refresh=True`` بدهید.
        """
        start, end = _utc(start), _utc(end)
        if end <= start:
            raise ValueError(f"end ({end}) must be after start ({start})")
        warm = self.warmup_days if warmup_days is None else int(warmup_days)
        ext_start = start - timedelta(days=warm)

        # ── 12h cache lookup (hour-granularity) ──
        cache_key = (
            f"{self.base_asset}|"
            f"{start.strftime('%Y%m%d%H')}|"
            f"{end.strftime('%Y%m%d%H')}|"
            f"{candle_freq}|"
            f"{max_staleness}|"
            f"{daily_staleness}|"
            f"{warm}|"
            f"{int(self.include_macro)}|"
            f"{int(self.include_onchain)}|"
            f"{int(self.include_liquidity)}|"
            f"{int(self.dtype is not None)}"
        )
        if not force_refresh:
            cached = _cache_get("fund_all", cache_key, FUNDAMENTAL_CACHE_TTL)
            if isinstance(cached, pd.DataFrame) and not cached.empty:
                logger.info(
                    "[FUND-v6.0] ✅ cache hit (TTL=%s) | key=%s | shape=%s",
                    FUNDAMENTAL_CACHE_TTL, cache_key, cached.shape,
                )
                return cached.copy()
            logger.info(
                "[FUND-v6.0] ❌ cache miss (TTL=%s) | key=%s — collecting",
                FUNDAMENTAL_CACHE_TTL, cache_key,
            )

        idx = pd.date_range(pd.Timestamp(ext_start).ceil(candle_freq),
                            pd.Timestamp(end).floor(candle_freq), freq=candle_freq)
        idx.name = "timestamp"
        bpd = max(1, int(round(pd.Timedelta("1D") / pd.Timedelta(candle_freq))))

        df = pd.DataFrame(index=idx)
        df["_placeholder"] = np.float32(0.0)
        status: Dict[str, str] = {}

        def step(label: str, fn: Callable[[], Any]) -> Any:
            try:
                r = fn()
                ok = r is not None and len(r) > 0
                status[label] = "success" if ok else "empty"
                return r if ok else None
            except Exception as exc:
                status[label] = f"error: {type(exc).__name__}"
                logger.error("[FUND-v5.0] ❌ %s failed: %s",
                             label, exc, exc_info=True)
                return None

        # 1) Derivatives
        snap = step("derivatives", lambda: self.cg_fetcher.fetch_historical_derivatives(
            self.base_asset, ext_start, end, resample_freq=candle_freq,
            force_refresh=force_refresh))
        if snap is not None:
            df = self._merge_snapshot_derivatives(df, snap, max_staleness)

        # 2) Fear & Greed
        fg = step("fear_greed", lambda: SentimentFetcher.fetch_fear_greed(limit=0))
        if fg is not None:
            df = self._merge_fear_greed(df, fg, daily_staleness)

        # 3..5) Macro / On-chain / Liquidity
        tol = pd.Timedelta(daily_staleness)
        ctx: List[Tuple[str, Callable[[], Any]]] = []
        if self.include_macro:
            ctx.append(("macro", lambda: SentimentFetcher.fetch_macro_fred(start=ext_start)))
        if self.include_onchain:
            ctx.append(("onchain", lambda: SentimentFetcher.fetch_onchain_daily(
                self.base_asset, ext_start)))
        if self.include_liquidity:
            ctx.append(("stablecoins", SentimentFetcher.fetch_stablecoin_supply))
            ctx.append(("defi_tvl", SentimentFetcher.fetch_defi_tvl))
        for label, fn in ctx:
            frame = step(label, fn)
            if frame is not None:
                df = self._asof_merge(df, frame, tol)

        # 6) Derived + interactions
        try:
            df = self._add_derived_features(df, candle_freq, bpd)
            df = self._add_interaction_features(df, bpd)
            status["derived"] = "success"
        except Exception as exc:
            status["derived"] = f"error: {type(exc).__name__}"
            logger.error("[FUND-v5.0] derived features failed: %s",
                         exc, exc_info=True)

        # cleanup
        df = df.drop(columns=["_placeholder"], errors="ignore")
        df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
        df = df.replace([np.inf, -np.inf], np.nan)

        degenerate: List[str] = []
        for col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            fin = vals[np.isfinite(vals)]
            if fin.size == 0 or float(np.std(fin)) < 1e-12:
                degenerate.append(col)
        if degenerate:
            df = df.drop(columns=degenerate)
            logger.info("[FUND-v5.0] 🧹 dropped %d degenerate columns: %s",
                        len(degenerate), degenerate)

        if self.dtype is not None and len(df.columns):
            df = df.astype(self.dtype)

        self.last_report = self._build_report(df)
        logger.info("[FUND-v5.0] finished | %d cols × %d rows | status=%s",
                    df.shape[1], len(df), status)
        if df.shape[1] == 0:
            logger.error(
                "[FUND-v5.0] ❌ NO fundamental features attached — "
                "شبکه/منابع را بررسی کنید."
            )
        else:
            logger.info("[FUND-v5.0] coverage report:\n%s",
                        self.last_report.to_string())

        # ── 12h cache store ──
        if not force_refresh and len(df) > 0 and df.shape[1] > 0:
            try:
                _cache_put("fund_all", cache_key, df.copy())
                logger.info(
                    "[FUND-v6.0] 💾 cached (TTL=%s) | key=%s | shape=%s",
                    FUNDAMENTAL_CACHE_TTL, cache_key, df.shape,
                )
            except Exception as exc:
                logger.debug("[FUND-v6.0] cache put failed: %s", exc)

        return df
        
    def collect_all_safe(self, start: datetime, end: datetime, candle_freq: str = "30min",
                         max_staleness: str = "1D", **kw: Any) -> pd.DataFrame:
        try:
            return self.collect_all(start, end, candle_freq, max_staleness, **kw)
        except Exception as exc:
            logger.error("[FUND-v5.0] collect_all failed: %s", exc, exc_info=True)
            idx = pd.date_range(_utc(start), _utc(end), freq=candle_freq)
            idx.name = "timestamp"
            return pd.DataFrame(index=idx)

    # ═════════════════════════ MERGE HELPERS ═════════════════════════
    @staticmethod
    def _utc(dt: datetime) -> datetime:
        return _utc(dt)

    def _asof_merge(self, left: pd.DataFrame, right: pd.DataFrame, tolerance: pd.Timedelta) -> pd.DataFrame:
        if len(left) == 0 or len(right) == 0:
            return left
        lft, rgt = _ts_col(left), _ts_col(right)
        cols = [c for c in rgt.columns if c != "timestamp" and c not in lft.columns
                and pd.api.types.is_numeric_dtype(rgt[c])]
        if not cols:
            return left
        merged = pd.merge_asof(lft, rgt[["timestamp"] + cols], on="timestamp",
                               direction="backward", tolerance=tolerance)
        merged = merged.set_index("timestamp")
        merged.index.name = "timestamp"
        return merged

    def _merge_snapshot_derivatives(self, left: pd.DataFrame, snapshot: pd.DataFrame,
                                    max_staleness: str) -> pd.DataFrame:
        try:
            tol = pd.Timedelta(max_staleness)
        except Exception:
            tol = pd.Timedelta("1D")
        return self._asof_merge(left, snapshot, tol)

    def _merge_fear_greed(self, left: pd.DataFrame, fg: pd.DataFrame,
                          max_staleness: str = "3D") -> pd.DataFrame:
        if len(left) == 0 or len(fg) == 0 or "fear_greed" not in fg.columns:
            return left
        out = self._asof_merge(left, fg[["timestamp", "fear_greed"]]
                               if "timestamp" in fg.columns else fg[["fear_greed"]],
                               pd.Timedelta(max_staleness))
        if "fear_greed" not in out.columns:
            return left
        bpd = max(1, int(round(pd.Timedelta("1D") / pd.Timedelta(
            pd.infer_freq(out.index[:50]) or "30min")))) if len(out) > 3 else 48
        new = pd.DataFrame(index=out.index)
        new["fear_greed_ma7"] = out["fear_greed"].rolling(7 * bpd, min_periods=bpd).mean()
        new["fear_greed_extreme"] = ((out["fear_greed"] < 20) | (out["fear_greed"] > 80)).astype("float64") \
            .where(out["fear_greed"].notna())
        return pd.concat([out, new], axis=1)

    # ═════════════════════════ DERIVED FEATURES ═════════════════════════
    def _add_derived_features(self, df: pd.DataFrame, freq: str, bpd: int) -> pd.DataFrame:
        if len(df) == 0:
            return df
        n = pd.to_numeric

        def col(name: str) -> Optional[pd.Series]:
            return n(df[name], errors="coerce") if name in df.columns else None

        def z(s: pd.Series, days: int = 30) -> pd.Series:
            w, mp = days * bpd, max(3 * bpd, 10)
            return (s - s.rolling(w, min_periods=mp).mean()) / (s.rolling(w, min_periods=mp).std() + 1e-12)

        new: Dict[str, pd.Series] = {}
        fr = col("funding_rate_median")
        if fr is not None:
            new["funding_ma_1d"] = fr.rolling(bpd, min_periods=max(1, bpd // 2)).mean()
            new["funding_ma_7d"] = fr.rolling(7 * bpd, min_periods=bpd).mean()
            new["funding_z_30d"] = z(fr)
            new["funding_annualized"] = fr * 3.0 * 365.0           # 3 تسویهٔ ۸ ساعته در روز

        oi = col("oi_binance_usd")
        oi = oi if oi is not None else col("open_interest_usd_total")
        if oi is not None:
            oi = oi.replace(0, np.nan)
            new["oi_chg_1d"] = oi.pct_change(bpd, fill_method=None)
            new["oi_chg_7d"] = oi.pct_change(7 * bpd, fill_method=None)
            new["oi_z_30d"] = z(oi)

        px = col("price_median")
        if px is not None:
            lp = np.log(px.where(px > 0))
            h = max(1, int(round(pd.Timedelta("1h") / pd.Timedelta(freq))))
            r1h = lp.diff(h)
            new["ret_1d"] = lp.diff(bpd)
            new["rv_7d_ann"] = r1h.rolling(7 * bpd, min_periods=bpd).std() * np.sqrt(24 * 365)
            if oi is not None:
                new["oi_price_div_1d"] = np.sign(new["oi_chg_1d"]) * np.sign(new["ret_1d"])
            dv = col("dvol")
            if dv is not None:
                new["dvol_minus_rv"] = dv - new["rv_7d_ann"] * 100.0     # VRP تقریبی (درصد)

        for name in ("dvol", "basis_median", "ls_ratio_global", "ls_ratio_top_accounts",
                     "taker_buysell_ratio", "coinbase_premium_pct"):
            s = col(name)
            if s is not None:
                new[f"{name}_z_30d"] = z(s)
        dv = col("dvol")
        if dv is not None:
            new["dvol_chg_1d"] = dv.diff(bpd)

        rate_like = ("us10y", "us2y", "curve_10y2y", "fed_funds", "hy_spread", "breakeven10y", "vix")
        index_like = ("usd_broad", "sp500", "nasdaq", "wti")
        for name in rate_like:
            s = col(name)
            if s is not None:
                new[f"{name}_d5"] = s.diff(5 * bpd)
        for name in index_like:
            s = col(name)
            if s is not None:
                new[f"{name}_r5"] = s.pct_change(5 * bpd, fill_method=None)
        vix = col("vix")
        if vix is not None:
            new["vix_z_30d"] = z(vix)

        for name in ("stablecoin_supply_usd", "defi_tvl_usd"):
            s = col(name)
            if s is not None:
                new[f"{name}_chg_7d"] = s.pct_change(7 * bpd, fill_method=None)
                new[f"{name}_chg_30d"] = s.pct_change(30 * bpd, fill_method=None)
        for name in ("cm_active_addr", "cm_tx_count", "cm_nvt"):
            s = col(name)
            if s is not None:
                new[f"{name}_z_30d"] = z(s)

        if not new:
            return df
        add = pd.DataFrame(new, index=df.index).replace([np.inf, -np.inf], np.nan)
        return pd.concat([df, add], axis=1)

    def _add_interaction_features(self, df: pd.DataFrame, bpd: int = 48) -> pd.DataFrame:
        """تعامل‌ها فقط وقتی هر دو منبع معتبر باشند (≥۳۰٪ finite و std ≥ 1e-12)."""
        if len(df) == 0:
            return df
        n_rows = len(df)

        def usable(name: str, ratio: float = 0.30) -> bool:
            if name not in df.columns:
                return False
            v = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
            fin = v[np.isfinite(v)]
            return fin.size >= max(10, int(n_rows * ratio)) and float(np.std(fin)) >= 1e-12

        if not usable("fear_greed"):
            logger.info("[FUND-v5.0] ⏭️ interactions skipped — fear_greed not usable")
            return df
        fgn = (pd.to_numeric(df["fear_greed"], errors="coerce") - 50.0) / 50.0

        def build_oi() -> Optional[pd.Series]:
            src = "oi_binance_usd" if usable("oi_binance_usd") else "open_interest_usd_total"
            if not usable(src):
                return None
            oi = pd.to_numeric(df[src], errors="coerce").replace(0, np.nan)
            return oi.pct_change(bpd, fill_method=None)

        def build_vol() -> Optional[pd.Series]:
            if not usable("volume_24h_usd_total"):
                return None
            v = pd.to_numeric(df["volume_24h_usd_total"], errors="coerce").replace(0, np.nan)
            return v / (v.rolling(7 * bpd, min_periods=bpd).mean() + 1e-10)

        specs: List[Tuple[str, Callable[[], Optional[pd.Series]]]] = [
            ("funding_x_fg", lambda: pd.to_numeric(df["funding_rate_median"], errors="coerce")
             if usable("funding_rate_median") else None),
            ("oi_change_x_fg", build_oi),
            ("vol_ratio_x_fg", build_vol),
        ]
        new: Dict[str, pd.Series] = {}
        for name, builder in specs:
            base = builder()
            if base is None:
                logger.info("[FUND-v5.0] ⏭️ %s SKIPPED — source not usable", name)
                continue
            inter = (base * fgn).replace([np.inf, -np.inf], np.nan)
            n_fin = int(np.isfinite(inter.to_numpy(dtype=np.float64)).sum())
            if n_fin >= max(10, int(n_rows * 0.20)):
                new[name] = inter
                logger.info("[FUND-v5.0] ✅ %s created | n_finite=%d/%d", name, n_fin, n_rows)
            else:
                logger.info("[FUND-v5.0] ⏭️ %s SKIPPED — insufficient finite (%d/%d)", name, n_fin, n_rows)
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1) if new else df

    @staticmethod
    def _build_report(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for c in df.columns:
            v = df[c]
            fv = v.first_valid_index()
            lv = v.last_valid_index()
            rows.append({"column": c, "coverage_pct": round(float(v.notna().mean()) * 100, 1),
                         "first_valid": fv, "last_valid": lv})
        return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()

    @staticmethod
    def _extract_base_asset(symbol: str) -> str:
        s = str(symbol).upper().replace(".", "").replace("/", "")
        for suffix in ("USDT", "USDC", "USD", "BUSD", "EUR", "GBP"):
            if s.endswith(suffix) and len(s) > len(suffix):
                return s[: -len(suffix)]
        return s[:3] if len(s) >= 3 else s
        
def _as_1d_float64(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).flatten()
    return arr
def _safe_percentile(x: np.ndarray, q: float, fallback: float) -> float:
    try:
        v = np.asarray(x, dtype=np.float64).flatten()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float(fallback)
        return float(np.percentile(v, float(q)))
    except Exception:
        return float(fallback)
def _accuracy_fallback(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        a = np.asarray(y_true).flatten()
        b = np.asarray(y_pred).flatten()
        m = min(a.size, b.size)
        if m == 0:
            return 0.0
        return float(np.mean(a[:m] == b[:m]))
    except Exception:
        return 0.0
def _corr_fallback(a: np.ndarray, b: np.ndarray) -> float:
    try:
        x = np.asarray(a, dtype=np.float64).flatten()
        y = np.asarray(b, dtype=np.float64).flatten()
        m = min(x.size, y.size)
        x = x[:m]
        y = y[:m]
        msk = np.isfinite(x) & np.isfinite(y)
        x = x[msk]
        y = y[msk]
        if x.size < 3:
            return 0.0
        if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            return 0.0
        c = np.corrcoef(x, y)
        v = float(c[0, 1])
        if not np.isfinite(v):
            return 0.0
        return v
    except Exception:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════
# LABEL GENERATOR v14.0 — MINIMAL-INVALID, PRODUCTION-SAFE
# ═══════════════════════════════════════════════════════════════════════════
# این نسخه یک REPLACEMENT کامل برای LabelGenerator موجود است.
# تغییرات اصلی:
#   1. INVALID های price_direction از ۴۰٪ به ~۰٪ کاهش یافت
#   2. warmup تارگت‌های state با best-effort پر می‌شود
#   3. آدیت خودکار توزیع لیبل‌ها برای جلوگیری از دِد-تارگت
#   4. مهاجرت in-place از داده‌های قدیمی (بدون نیاز به regenerate)
# ═══════════════════════════════════════════════════════════════════════════

class LabelGenerator:
    """
    LABEL_GENERATOR v14.0 — MINIMAL-INVALID, LEARNER-FRIENDLY.

    ────────────────────────────────────────────────────────────────────────
    فلسفهٔ v14.0:
    ────────────────────────────────────────────────────────────────────────
    هدف برچسب‌گذاری، «دادن بیشترین سیگنال به یادگیرنده به‌ازای هر نمونه»ست.
    حذف ۴۰٪ نمونه‌ها به بهانهٔ «پاک‌سازی نویز» سه زیان دارد:

        ۱. آن ۴۰٪ هنوز سیگنال واقعی دارند (فقط ضعیف‌تر).
        ۲. نمونه‌های باقی‌مانده اریبیِ «فقط موارد واضح» می‌گیرند.
        ۳. مدل در زمان inference روی نمونه‌های مرزی (که در آموزش
           ندیده) رفتار غیرقابل پیش‌بینی دارد.

    راه‌حل v14.0:
        • حالت پیش‌فرض: labeling_mode = "sign" (صفر INVALID برای جهت)
        • warmup تارگت‌های state با best-effort پر می‌شود
        • فیلتر اختیاری با min_label_abs_return (پیش‌فرض 0 = بدون فیلتر)
        • آدیت خودکار: هشدار در صورت INVALID > max_invalid_ratio_warn
        • محافظت: RuntimeError در صورت دِد-تارگت

    ────────────────────────────────────────────────────────────────────────
    قراردادها:
    ────────────────────────────────────────────────────────────────────────
        • label[t] = sign(future_return[t])     برای قیمت جهت
        • label[t] = state[t+h]                  برای state targets (future-anchored)
        • آستانه‌های magnitude فقط روی TRAIN fit می‌شوند (frozen)
        • کلاس‌ها:
            price_direction    {0: SELL, 1: BUY}
            momentum_direction {0: DOWN, 1: UP}
            magnitude_bucket   {0: SMALL, 1: MED, 2: LARGE}
            trend_strength     {0: WEAK, 1: MOD, 2: STRONG}
            volatility_regime  {0: LOW, 1: NORMAL, 2: HIGH}
            market_regime      {0: SIDE_LOW, 1: SIDE_HIGH, 2: UP, 3: DOWN}

    ────────────────────────────────────────────────────────────────────────
    سازگاری:
    ────────────────────────────────────────────────────────────────────────
        • تمام امضاهای عمومی حفظ شده‌اند
        • sign_with_epsilon همچنان opt-in است (برای سازگاری)
        • get_fitted_state/load_fitted_state سازگار با حالت قدیمی
        • make_all_labels همان 7+1 خروجی را می‌دهد
    """

    INVALID = -1
    VERSION = "14.0-minimal-invalid"

    PRICE_DIRECTION_N_CLASSES = 2
    MOMENTUM_DIRECTION_N_CLASSES = 2
    MAGNITUDE_BUCKET_N_CLASSES = 3

    _CLASS_NAMES = {
        "price_direction":    {0: "SELL", 1: "BUY"},
        "momentum_direction": {0: "DOWN_MOM", 1: "UP_MOM"},
        "magnitude_bucket":   {0: "SMALL", 1: "MED", 2: "LARGE"},
        "trend_strength":     {0: "WEAK", 1: "MODERATE", 2: "STRONG"},
        "volatility_regime":  {0: "LOW", 1: "NORMAL", 2: "HIGH"},
        "market_regime":      {0: "SIDE_LOW", 1: "SIDE_HIGH", 2: "UP", 3: "DOWN"},
    }

    # ──────────────────────────────────────────────────────────────────
    def __init__(
        self,
        symbol=None,
        prediction_horizon: int = 1,
        balance_classes: bool = True,
        clip_outliers: bool = True,
        dynamic_threshold: bool = True,
        regime_aware: bool = True,
        vol_lookback: int = 20,
        strong_min_samples_pct: float = 0.03,
        weak_percentile: float = 40.0,       # legacy — دیگر در تعیین threshold استفاده نمی‌شود
        strong_percentile: float = 60.0,     # legacy
        min_threshold: float = 1e-5,
        max_threshold: float = 0.05,
        allow_refit: bool = False,
        labeling_mode: str = "sign",          # ← پیش‌فرض جدید (بود: sign_with_epsilon)
        epsilon: float = 1e-8,
        momentum_ema_short: int = 5,
        momentum_ema_long: int = 20,
        sample_weight_agreement_power: float = 2.0,
        # ── پارامترهای جدید v14.0 ──
        min_label_abs_return: float = 0.0,   # اگر >0، نمونه‌های |ret|≤این مقدار INVALID می‌شوند
        warmup_targets_best_effort: bool = True,  # warmup با best-effort پر شود (نه INVALID)
        max_invalid_ratio_warn: float = 0.05,     # هشدار اگر INVALID > 5%
        magnitude_q33_percentile: float = 33.0,
        magnitude_q66_percentile: float = 66.0,
        **kwargs,
    ):
        # ── اعتبارسنجی پارامترهای ورودی ──
        try:
            h = int(prediction_horizon)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction_horizon must be int, got {prediction_horizon!r}") from exc
        if h < 1:
            raise ValueError(f"prediction_horizon must be >=1, got {h}")

        weak_p = float(weak_percentile)
        strong_p = float(strong_percentile)
        if not (0.0 < weak_p < strong_p < 100.0):
            raise ValueError(f"percentiles invalid: weak={weak_p}, strong={strong_p}")

        min_t = float(min_threshold)
        max_t = float(max_threshold)
        if not (0.0 < min_t < max_t):
            raise ValueError(f"thresholds invalid: min={min_t}, max={max_t}")

        vol_lb = int(vol_lookback)
        if vol_lb < 2:
            raise ValueError(f"vol_lookback must be >=2, got {vol_lb}")

        mode = str(labeling_mode).lower().strip()
        if mode not in ("sign", "sign_with_epsilon"):
            raise ValueError(
                f"labeling_mode must be 'sign' or 'sign_with_epsilon', got {labeling_mode!r}"
            )

        eps = float(epsilon)
        if eps < 0.0 or not np.isfinite(eps):
            raise ValueError(f"epsilon must be finite and >=0, got {epsilon!r}")

        ema_s = int(momentum_ema_short)
        ema_l = int(momentum_ema_long)
        if ema_s < 2 or ema_l < 2 or ema_s >= ema_l:
            raise ValueError(f"momentum EMA invalid: short={ema_s}, long={ema_l} (need 2<=short<long)")

        agg_pow = float(sample_weight_agreement_power)
        if not (0.0 < agg_pow <= 10.0):
            raise ValueError(f"sample_weight_agreement_power must be in (0,10], got {agg_pow}")

        # ── پارامترهای جدید v14.0 ──
        min_label = float(min_label_abs_return)
        if min_label < 0.0 or not np.isfinite(min_label):
            raise ValueError(f"min_label_abs_return must be finite >=0, got {min_label_abs_return!r}")

        max_warn = float(max_invalid_ratio_warn)
        if not (0.0 < max_warn <= 1.0):
            raise ValueError(f"max_invalid_ratio_warn must be in (0,1], got {max_invalid_ratio_warn}")

        mag_q33_p = float(magnitude_q33_percentile)
        mag_q66_p = float(magnitude_q66_percentile)
        if not (0.0 < mag_q33_p < mag_q66_p < 100.0):
            raise ValueError(
                f"magnitude percentiles invalid: q33={mag_q33_p}, q66={mag_q66_p}"
            )

        # ── ذخیره ──
        self.symbol = symbol
        self.prediction_horizon = h
        self.balance_classes = bool(balance_classes)
        self.clip_outliers = bool(clip_outliers)
        self.dynamic_threshold = bool(dynamic_threshold)
        self.regime_aware = bool(regime_aware)
        self.vol_lookback = vol_lb
        self.strong_min_samples_pct = float(strong_min_samples_pct)
        self.weak_percentile = weak_p
        self.strong_percentile = strong_p
        self.min_threshold = min_t
        self.max_threshold = max_t
        self.allow_refit = bool(allow_refit)
        self.labeling_mode = mode
        self.epsilon = eps

        self.momentum_ema_short = ema_s
        self.momentum_ema_long = ema_l
        self.sample_weight_agreement_power = agg_pow

        # v14.0
        self.min_label_abs_return = min_label
        self.warmup_targets_best_effort = bool(warmup_targets_best_effort)
        self.max_invalid_ratio_warn = max_warn
        self.magnitude_q33_percentile = mag_q33_p
        self.magnitude_q66_percentile = mag_q66_p

        # ── State ──
        self._is_fitted = False
        self._base_threshold: Optional[float] = None
        self._signal_threshold: Optional[float] = None
        self._weak_factor: float = 1.0
        self._strong_factor: float = 1.0
        self._volatility_multiplier: float = 1.0
        self._fitted_volatility_regime: str = "NORMAL_VOL"
        self._thresholds: Optional[np.ndarray] = None
        self._return_clip_low: Optional[float] = None
        self._return_clip_high: Optional[float] = None
        self._fitted_volatility_stats: Optional[Dict[str, Any]] = None
        self._fitted_state: Optional[Dict[str, Any]] = None
        self._caller: Optional[str] = None
        self._n_fitted_samples: int = 0
        self._train_class_dist: Optional[Dict[int, float]] = None

        self._mag_q33_frozen: Optional[float] = None
        self._mag_q66_frozen: Optional[float] = None

        logger.info(
            "LabelGenerator %s init | symbol=%s | horizon=%d | "
            "n_classes_price_dir=%d | mode=%s | min_label_abs_return=%g | "
            "warmup_best_effort=%s | max_invalid_warn=%.1f%% | "
            "momentum_ema=(%d,%d)",
            self.VERSION, str(symbol), self.prediction_horizon,
            self.PRICE_DIRECTION_N_CLASSES, self.labeling_mode,
            self.min_label_abs_return, self.warmup_targets_best_effort,
            self.max_invalid_ratio_warn * 100,
            self.momentum_ema_short, self.momentum_ema_long,
        )

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: fit / transform
    # ══════════════════════════════════════════════════════════════════════

    def fit(
        self,
        prices: np.ndarray,
        volatility_regime: str = "NORMAL_VOL",
        allow_live_fit: bool = False,
    ) -> "LabelGenerator":
        """
        FIT v14.0 — TRAIN-ONLY THRESHOLD FREEZE.

        تغییر اصلی نسبت به v13.0:
            _base_threshold دیگر صدک ۴۰ |بازده| نیست. مقدار جدید:
                max(min_label_abs_return, min_threshold)
            که پیش‌فرض = max(0, 1e-5) = 1e-5 (0.1bp).
            
            دلیل: در sign mode، threshold اصلاً استفاده نمی‌شود. در
            sign_with_epsilon، از min_label_abs_return استفاده می‌شود.
            در هر دو حالت، آستانه از «خواستهٔ کاربر» می‌آید، نه از
            صدک داده.
        """
        try:
            logger.info(
                "LabelGenerator.fit %s n=%s regime=%s caller=%s mode=%s",
                self.VERSION,
                str(len(prices) if prices is not None else "None"),
                str(volatility_regime), str(self._caller), self.labeling_mode,
            )
            if prices is None:
                raise ValueError("prices is None")

            p = _as_1d_float64(prices)
            n = p.shape[0]
            if n < 10:
                raise ValueError(f"need >=10 samples, got {n}")

            if self._is_fitted and not bool(self.allow_refit):
                logger.warning("already fitted, skip refit")
                return self

            if self._caller not in (None, "train", "live_data", "train_split"):
                logger.warning(
                    "fit called with caller=%s — thresholds must be from TRAIN only",
                    str(self._caller),
                )

            fut = self._compute_future_returns(p)
            valid_fut = fut[np.isfinite(fut)]

            # ── FIT magnitude_bucket thresholds روی TRAIN ──
            if valid_fut.size >= 10:
                abs_fut = np.abs(valid_fut)
                abs_fut_clean = abs_fut[abs_fut > 1e-12]
                if abs_fut_clean.size >= 10:
                    self._mag_q33_frozen = float(
                        np.nanpercentile(abs_fut_clean, self.magnitude_q33_percentile)
                    )
                    self._mag_q66_frozen = float(
                        np.nanpercentile(abs_fut_clean, self.magnitude_q66_percentile)
                    )
                    # validate: q33 must be < q66 and both finite positive
                    if (not np.isfinite(self._mag_q33_frozen)
                            or not np.isfinite(self._mag_q66_frozen)
                            or self._mag_q33_frozen <= 0.0
                            or self._mag_q33_frozen >= self._mag_q66_frozen):
                        logger.warning(
                            "[%s] magnitude thresholds degenerate (q33=%s, q66=%s) — "
                            "using fallback (1e-4, 1e-3)",
                            str(self.symbol),
                            self._mag_q33_frozen, self._mag_q66_frozen,
                        )
                        self._mag_q33_frozen = 1e-4
                        self._mag_q66_frozen = 1e-3
                    else:
                        logger.info(
                            "[%s] ✅ magnitude_bucket thresholds frozen: q33=%.8f, q66=%.8f",
                            str(self.symbol), self._mag_q33_frozen, self._mag_q66_frozen,
                        )
                else:
                    logger.warning(
                        "[%s] insufficient |abs_return| for magnitude fit; "
                        "using fallback (1e-4, 1e-3)", str(self.symbol),
                    )
                    self._mag_q33_frozen = 1e-4
                    self._mag_q66_frozen = 1e-3
            else:
                logger.warning(
                    "[%s] insufficient valid future returns; using fallback (1e-4, 1e-3)",
                    str(self.symbol),
                )
                self._mag_q33_frozen = 1e-4
                self._mag_q66_frozen = 1e-3

            # ── CRITICAL FIX v14.0: threshold = user-controlled, NOT percentile ──
            base_thr = float(max(self.min_label_abs_return, self.min_threshold))
            self._base_threshold = base_thr
            self._signal_threshold = base_thr
            self._weak_factor = 1.0
            self._strong_factor = 1.0
            self._volatility_multiplier = 1.0
            self._fitted_volatility_regime = str(volatility_regime)
            self._thresholds = np.full(n, base_thr, dtype=np.float64)

            # ── Clip bounds از TRAIN ──
            if valid_fut.size > 0:
                if bool(self.clip_outliers) and valid_fut.size > 100:
                    lo = float(np.percentile(valid_fut, 0.1))
                    hi = float(np.percentile(valid_fut, 99.9))
                else:
                    lo = float(np.min(valid_fut))
                    hi = float(np.max(valid_fut))
                if not np.isfinite(lo):
                    lo = -0.05
                if not np.isfinite(hi):
                    hi = 0.05
            else:
                lo, hi = -0.05, 0.05

            self._return_clip_low = lo
            self._return_clip_high = hi
            self._n_fitted_samples = int(n)

            # ── train class distribution (برای diagnostics) ──
            try:
                sample_dir = self._apply_thresholds_to_array(
                    fut, self._thresholds, use_strong_classes=False
                )
                vv = sample_dir[sample_dir != self.INVALID]
                if vv.size > 0:
                    uq, ct = np.unique(vv, return_counts=True)
                    self._train_class_dist = {
                        int(a): float(b) / float(vv.size)
                        for a, b in zip(uq, ct)
                    }
            except Exception:
                self._train_class_dist = None

            self._fitted_state = {
                "is_fitted": True,
                "fitted_at": datetime.now().isoformat(),
                "n_samples": int(n),
                "n_valid": int(valid_fut.size),
                "base_threshold": float(base_thr),
                "signal_threshold": float(base_thr),
                "weak_factor": 1.0,
                "strong_factor": 1.0,
                "volatility_multiplier": 1.0,
                "volatility_regime": str(volatility_regime),
                "weak_percentile": float(self.weak_percentile),
                "return_clip_low": float(lo),
                "return_clip_high": float(hi),
                "symbol": self.symbol,
                "version": self.VERSION,
                "n_classes_price_direction": self.PRICE_DIRECTION_N_CLASSES,
                "n_classes_momentum_direction": self.MOMENTUM_DIRECTION_N_CLASSES,
                "n_classes_magnitude_bucket": self.MAGNITUDE_BUCKET_N_CLASSES,
                "labeling_mode": str(self.labeling_mode),
                "train_class_dist": self._train_class_dist,
                "mag_q33_frozen": float(self._mag_q33_frozen),
                "mag_q66_frozen": float(self._mag_q66_frozen),
                "min_label_abs_return": float(self.min_label_abs_return),
                "warmup_targets_best_effort": bool(self.warmup_targets_best_effort),
            }

            logger.info(
                "fitted base=%.6f mode=%s regime=%s mag_q33=%.8f mag_q66=%.8f "
                "class_dist=%s",
                float(base_thr), self.labeling_mode, str(volatility_regime),
                float(self._mag_q33_frozen), float(self._mag_q66_frozen),
                str(self._train_class_dist),
            )
            self._is_fitted = True
            return self

        except Exception as e:
            logger.error("fit failed: %s", str(e))
            logger.error(traceback.format_exc())
            raise

    def transform(
        self,
        prices: np.ndarray,
        volatility_regime: str = "NORMAL_VOL",
    ) -> np.ndarray:
        if not bool(self._is_fitted):
            raise RuntimeError("Must call fit() on TRAIN before transform()")

        p = _as_1d_float64(prices)
        n = p.shape[0]
        if n == 0:
            return np.array([], dtype=np.int32)

        fut = self._compute_future_returns(p)
        base = float(
            self._base_threshold if self._base_threshold is not None else self.min_threshold
        )
        thr = np.full(n, base, dtype=np.float64)

        return self._apply_thresholds_to_array(fut, thr, use_strong_classes=False)

    def fit_transform(self, prices, volatility_regime="NORMAL_VOL"):
        self.set_caller("train")
        self.fit(prices, volatility_regime, allow_live_fit=True)
        return self.transform(prices, volatility_regime)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: generate_* methods
    # ══════════════════════════════════════════════════════════════════════

    def generate_direction_2class(self, prices, volatility_regime="NORMAL_VOL"):
        if not bool(self._is_fitted):
            raise RuntimeError("LabelGenerator not fitted! Call fit() on TRAIN first.")
        return self.transform(prices, volatility_regime)

    def generate_direction_3class(self, *args, **kwargs):
        logger.warning(
            "generate_direction_3class is DEPRECATED — returning 2-class output. "
            "Use generate_direction_2class instead."
        )
        return self.generate_direction_2class(*args, **kwargs)

    def generate_direction_5class(self, *args, **kwargs):
        logger.warning(
            "generate_direction_5class is DEPRECATED — returning 2-class output. "
            "Use generate_direction_2class instead."
        )
        return self.generate_direction_2class(*args, **kwargs)

    def generate_price_return(self, close_prices, clip_percentiles=(0.001, 0.999)):
        try:
            p = _as_1d_float64(close_prices)
            n = p.shape[0]
            if n == 0:
                return np.array([], dtype=np.float64)

            horizon = max(int(getattr(self, "prediction_horizon", 1)), 1)
            out = np.full(n, np.nan, dtype=np.float64)
            if n > horizon:
                base = p[: n - horizon]
                fwd = p[horizon:]
                valid = np.isfinite(base) & np.isfinite(fwd) & (np.abs(base) > 1e-8)
                vals = np.full(base.shape[0], np.nan, dtype=np.float64)
                vals[valid] = (fwd[valid] - base[valid]) / base[valid]
                out[: n - horizon] = vals

            if bool(getattr(self, "clip_outliers", True)):
                fitted_lo = getattr(self, "_return_clip_low", None)
                fitted_hi = getattr(self, "_return_clip_high", None)
                has_fitted = (
                    bool(getattr(self, "_is_fitted", False))
                    and fitted_lo is not None and fitted_hi is not None
                    and np.isfinite(float(fitted_lo)) and np.isfinite(float(fitted_hi))
                    and float(fitted_hi) > float(fitted_lo)
                )
                if has_fitted:
                    lo = float(fitted_lo)
                    hi = float(fitted_hi)
                    m = np.isfinite(out)
                    out[m] = np.clip(out[m], lo, hi)
            return out
        except Exception as e:
            logger.error("generate_price_return failed: %s", str(e))
            try:
                nn = len(close_prices) if close_prices is not None else 0
            except Exception:
                nn = 0
            return np.full(int(nn), np.nan, dtype=np.float64)

    def generate_volatility_regime(self, close_prices, rolling_window=50):
        """
        VOLATILITY_REGIME v3.0 — BEST-EFFORT WARMUP.

        تغییر v3.0: در دورهٔ warmup (زمانی که rolling window پر نیست)
        با یک منطق ساده برچسب می‌زند (NORMAL) — نه INVALID.
        """
        try:
            p = _as_1d_float64(close_prices)
            n = p.shape[0]
            if n == 0:
                return np.array([], dtype=np.int32)

            rw = int(rolling_window)
            if rw < 5:
                rw = 5
            rets = self._compute_past_returns(p)

            labels = np.full(n, np.int32(self.INVALID), dtype=np.int32)

            # Guard: too few samples
            if n < rw + 5:
                if self.warmup_targets_best_effort:
                    # Best-effort: use simple std comparison
                    for i in range(n):
                        window = rets[max(0, i - rw):i]
                        window = window[np.isfinite(window)]
                        if window.size >= 3:
                            v = float(np.std(window))
                            if v < 0.003:
                                labels[i] = 0
                            elif v > 0.015:
                                labels[i] = 2
                            else:
                                labels[i] = 1
                        else:
                            labels[i] = 1  # NORMAL fallback
                return labels

            s_ret = pd.Series(rets)
            roll_vol = s_ret.rolling(rw, min_periods=max(5, rw // 2)).std().shift(1).to_numpy(dtype=np.float64)
            s_vol = pd.Series(roll_vol)
            med = s_vol.rolling(rw, min_periods=20).median().shift(1).to_numpy(dtype=np.float64)
            std = s_vol.rolling(rw, min_periods=20).std().shift(1).to_numpy(dtype=np.float64)

            finite_vol = np.isfinite(roll_vol) & (roll_vol > 0)
            has_stats = finite_vol & np.isfinite(med) & np.isfinite(std)

            denom = np.maximum(std, 0.0001)
            z = np.full(n, 0.0, dtype=np.float64)
            z[has_stats] = (roll_vol[has_stats] - med[has_stats]) / denom[has_stats]

            labels[has_stats & (z < -0.6)] = 0
            labels[has_stats & (z > 0.6)] = 2
            labels[has_stats & (np.abs(z) <= 0.6)] = 1

            # Fallback for finite_vol but not enough stats
            fallback = finite_vol & (~has_stats)
            labels[fallback & (roll_vol < 0.003)] = 0
            labels[fallback & (roll_vol > 0.015)] = 2
            labels[fallback & (roll_vol >= 0.003) & (roll_vol <= 0.015)] = 1

            # ── v14.0: best-effort warmup fill ──
            if self.warmup_targets_best_effort:
                still_invalid = labels == self.INVALID
                if still_invalid.any():
                    for i in np.where(still_invalid)[0]:
                        window = rets[max(0, i - rw):i]
                        window = window[np.isfinite(window)]
                        if window.size >= 3:
                            v = float(np.std(window))
                            if v < 0.003:
                                labels[i] = 0
                            elif v > 0.015:
                                labels[i] = 2
                            else:
                                labels[i] = 1
                        else:
                            labels[i] = 1  # NORMAL

            return labels.astype(np.int32)
        except Exception as e:
            logger.error("generate_volatility_regime failed: %s", str(e))
            logger.error(traceback.format_exc())
            try:
                nn = len(close_prices) if close_prices is not None else 0
            except Exception:
                nn = 0
            return np.full(int(nn), np.int32(self.INVALID), dtype=np.int32)

    def generate_trend_strength(self, close_prices, adx_window=14):
        """
        TREND_STRENGTH v3.0 — BEST-EFFORT WARMUP.

        تغییر v3.0: در warmup از short-term momentum استفاده می‌شود.
        """
        try:
            p = _as_1d_float64(close_prices)
            n = p.shape[0]
            if n == 0:
                return np.array([], dtype=np.int32)

            w = int(adx_window)
            if w < 2:
                w = 2
            long_w = w * 2
            rets = self._compute_past_returns(p)

            labels = np.full(n, np.int32(self.INVALID), dtype=np.int32)

            # Guard: too few samples
            if n < long_w + 5:
                if self.warmup_targets_best_effort:
                    for i in range(n):
                        window = rets[max(0, i - w):i]
                        window = window[np.isfinite(window)]
                        if window.size >= 3:
                            mom = float(np.sum(window))
                            vol = float(np.std(window))
                            score = abs(mom) / (vol * np.sqrt(len(window)) + 1e-12)
                            if score > 1.2:
                                labels[i] = 2
                            elif score > 0.5:
                                labels[i] = 1
                            else:
                                labels[i] = 0
                        else:
                            labels[i] = 1  # MODERATE
                return labels

            s = pd.Series(rets)
            mom_long = s.rolling(long_w, min_periods=long_w).sum().to_numpy(dtype=np.float64)
            vol = s.rolling(w, min_periods=w).std().to_numpy(dtype=np.float64)
            denom = vol * np.sqrt(float(long_w))

            valid = np.isfinite(mom_long) & np.isfinite(vol) & (vol > 1e-8)
            score = np.full(n, np.nan, dtype=np.float64)
            score[valid] = np.abs(mom_long[valid]) / (denom[valid] + 1e-12)

            labels[valid & (score > 1.2)] = 2
            labels[valid & (score > 0.5) & (score <= 1.2)] = 1
            labels[valid & (score <= 0.5)] = 0

            # v14.0: best-effort warmup
            if self.warmup_targets_best_effort:
                still_invalid = labels == self.INVALID
                if still_invalid.any():
                    for i in np.where(still_invalid)[0]:
                        window = rets[max(0, i - w):i]
                        window = window[np.isfinite(window)]
                        if window.size >= 3:
                            mom = float(np.sum(window))
                            vol_val = float(np.std(window))
                            sc = abs(mom) / (vol_val * np.sqrt(len(window)) + 1e-12)
                            if sc > 1.2:
                                labels[i] = 2
                            elif sc > 0.5:
                                labels[i] = 1
                            else:
                                labels[i] = 0
                        else:
                            labels[i] = 1  # MODERATE

            return labels.astype(np.int32)
        except Exception as e:
            logger.error("generate_trend_strength failed: %s", str(e))
            logger.error(traceback.format_exc())
            try:
                nn = len(close_prices) if close_prices is not None else 0
            except Exception:
                nn = 0
            return np.full(int(nn), np.int32(self.INVALID), dtype=np.int32)

    def generate_market_regime(self, close_prices, volatility_regime=None):
        """
        MARKET_REGIME v3.0 — BEST-EFFORT WARMUP.
        """
        try:
            p = _as_1d_float64(close_prices)
            n = p.shape[0]
            if n == 0:
                return np.array([], dtype=np.int32)

            if volatility_regime is None:
                vol_arr = self.generate_volatility_regime(p)
            else:
                try:
                    va = np.asarray(volatility_regime).flatten()
                    if va.shape[0] != n:
                        tmp = np.full(n, 1, dtype=np.int32)
                        m = int(min(n, va.shape[0]))
                        tmp[:m] = va[:m].astype(np.int32)
                        vol_arr = tmp
                    else:
                        vol_arr = va.astype(np.int32)
                except Exception:
                    vol_arr = np.full(n, 1, dtype=np.int32)

            labels = np.full(n, np.int32(self.INVALID), dtype=np.int32)

            # Guard: too few samples
            if n < 30:
                if self.warmup_targets_best_effort:
                    for i in range(n):
                        if i >= 10:
                            recent_avg = float(np.mean(p[max(0, i-10):i+1]))
                            early_avg = float(np.mean(p[max(0, i-30):i+1]))
                            if early_avg > 1e-8:
                                rel = (recent_avg - early_avg) / abs(early_avg)
                                if rel > 0.002:
                                    labels[i] = 2  # UP
                                elif rel < -0.002:
                                    labels[i] = 3  # DOWN
                                else:
                                    labels[i] = 0 if vol_arr[i] != 2 else 1
                            else:
                                labels[i] = 0
                        else:
                            labels[i] = 0  # SIDEWAYS_LOW
                return labels

            s = pd.Series(p)
            sma_fast = s.rolling(10, min_periods=10).mean().shift(1).to_numpy(dtype=np.float64)
            sma_slow = s.rolling(30, min_periods=30).mean().shift(1).to_numpy(dtype=np.float64)

            valid = np.isfinite(sma_fast) & np.isfinite(sma_slow) & (np.abs(sma_slow) > 1e-8)
            rel = np.full(n, 0.0, dtype=np.float64)
            rel[valid] = (sma_fast[valid] - sma_slow[valid]) / np.abs(sma_slow[valid])

            is_up = valid & (rel > 0.002)
            is_down = valid & (rel < -0.002)
            is_side = valid & (~is_up) & (~is_down)

            labels[is_up] = 2
            labels[is_down] = 3
            try:
                is_high = is_side & (vol_arr == 2)
            except Exception:
                is_high = np.zeros(n, dtype=bool)
            labels[is_side & is_high] = 1
            labels[is_side & (~is_high)] = 0

            # v14.0: best-effort warmup
            if self.warmup_targets_best_effort:
                still_invalid = labels == self.INVALID
                if still_invalid.any():
                    for i in np.where(still_invalid)[0]:
                        if i >= 10:
                            recent_avg = float(np.mean(p[max(0, i-10):i+1]))
                            early_avg = float(np.mean(p[max(0, i-30):i+1]))
                            if early_avg > 1e-8:
                                r = (recent_avg - early_avg) / abs(early_avg)
                                if r > 0.002:
                                    labels[i] = 2
                                elif r < -0.002:
                                    labels[i] = 3
                                else:
                                    labels[i] = 0 if vol_arr[i] != 2 else 1
                            else:
                                labels[i] = 0
                        else:
                            labels[i] = 0

            return labels.astype(np.int32)
        except Exception as e:
            logger.error("generate_market_regime failed: %s", str(e))
            logger.error(traceback.format_exc())
            try:
                nn = len(close_prices) if close_prices is not None else 0
            except Exception:
                nn = 0
            return np.full(int(nn), np.int32(self.INVALID), dtype=np.int32)

    def generate_momentum_direction(self, close_prices):
        """
        MOMENTUM_DIRECTION v3.0 — BEST-EFFORT WARMUP.

        تغییر v3.0: در warmup (قبل از پر شدن EMA long) از sign(1-lag return)
        استفاده می‌شود. فقط اولین نمونه (i=0) INVALID است.
        """
        try:
            p = _as_1d_float64(close_prices)
            n = p.shape[0]
            out = np.full(n, np.int32(self.INVALID), dtype=np.int32)

            if n < 2:
                return out

            rets = self._compute_past_returns(p)

            # Warmup: use 1-lag return sign
            warmup_end = min(self.momentum_ema_long, n)

            # First sample cannot be labeled (no prior return)
            out[0] = self.INVALID if not self.warmup_targets_best_effort else 0

            for i in range(1, warmup_end):
                if np.isfinite(rets[i]):
                    out[i] = 1 if rets[i] >= 0.0 else 0
                elif i > 0:
                    out[i] = out[i - 1] if i > 1 else 0

            # After warmup: proper EMA spread
            if n > warmup_end:
                alpha_s = 2.0 / float(self.momentum_ema_short + 1)
                alpha_l = 2.0 / float(self.momentum_ema_long + 1)
                ema_short = self._ema_past_only(rets, alpha_s)
                ema_long = self._ema_past_only(rets, alpha_l)
                spread = ema_short - ema_long
                valid = np.isfinite(spread)

                out[valid & (spread >= 0.0)] = 1
                out[valid & (spread < 0.0)] = 0

            return out
        except Exception as e:
            logger.error("[%s] generate_momentum_direction failed: %s", str(self.symbol), str(e))
            logger.error(traceback.format_exc())
            try:
                nn = len(close_prices) if close_prices is not None else 0
            except Exception:
                nn = 0
            return np.full(int(nn), np.int32(self.INVALID), dtype=np.int32)

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: make_all_labels — CRITICAL PATH
    # ══════════════════════════════════════════════════════════════════════

    def make_all_labels(
        self,
        close_prices,
        high_prices=None,
        low_prices=None,
        drop_invalid: bool = False,
        split_name: str = "train",
    ) -> Dict[str, np.ndarray]:
        """
        MAKE_ALL_LABELS v14.0 — با آدیت خودکار و محافظت از دِد-تارگت.
        """
        if not self._is_fitted:
            raise RuntimeError("Fit LabelGenerator on TRAIN before label transform")

        if split_name in ("val", "test") and (
            self._mag_q33_frozen is None or self._mag_q66_frozen is None
        ):
            raise RuntimeError(
                f"magnitude_bucket thresholds not fitted on TRAIN; "
                f"cannot generate labels for {split_name}"
            )

        p = _as_1d_float64(close_prices)
        n = len(p)
        h = self.prediction_horizon

        # 1) price_return
        price_return = self.generate_price_return(p)

        # 2) price_direction (v14.0: sign mode → 0% INVALID except last h)
        threshold = np.full(n, float(self._base_threshold or self.min_threshold), dtype=np.float64)
        price_direction = self._apply_thresholds_to_array(
            price_return, threshold, use_strong_classes=False
        )

        # 3-5) state variables
        trend_state = self.generate_trend_strength(p)
        vol_state = self.generate_volatility_regime(p)
        market_state = self.generate_market_regime(p, vol_state)

        trend_target = _shift_future_state(trend_state, h, self.INVALID)
        vol_target = _shift_future_state(vol_state, h, self.INVALID)
        market_target = _shift_future_state(market_state, h, self.INVALID)

        # 6) momentum_direction
        mom_state = self.generate_momentum_direction(p)
        mom_target = _shift_future_state(mom_state, h, self.INVALID)

        # 7) magnitude_bucket با Frozen thresholds
        if self._mag_q33_frozen is None or self._mag_q66_frozen is None:
            raise RuntimeError("LabelGenerator must be fitted before magnitude_bucket")
        q33 = float(self._mag_q33_frozen)
        q66 = float(self._mag_q66_frozen)

        abs_ret = np.abs(price_return)
        finite_abs = np.isfinite(abs_ret)
        magnitude_bucket = np.full(n, self.INVALID, dtype=np.int32)
        magnitude_bucket[finite_abs & (abs_ret <= q33)] = 0
        magnitude_bucket[finite_abs & (abs_ret > q33) & (abs_ret <= q66)] = 1
        magnitude_bucket[finite_abs & (abs_ret > q66)] = 2

        # 8) sample_weight جهت‌محور
        sample_weight = self._compute_direction_aware_weight(p, price_return)

        result = {
            "price_return":       price_return.reshape(-1, 1),
            "price_direction":    price_direction.reshape(-1, 1),
            "trend_strength":     trend_target.reshape(-1, 1),
            "volatility_regime":  vol_target.reshape(-1, 1),
            "market_regime":      market_target.reshape(-1, 1),
            "magnitude_bucket":   magnitude_bucket.reshape(-1, 1),
            "momentum_direction": mom_target.reshape(-1, 1),
            "sample_weight":      sample_weight.reshape(-1, 1),
        }

        # ═══════════════════════════════════════════════════════════════
        # v14.0: AUTOMATED AUDIT
        # ═══════════════════════════════════════════════════════════════
        audit_issues = self._audit_labels(
            result=result, n_total=n, split_name=split_name,
        )

        # Guard: any dead target?
        for target_name, info in audit_issues.items():
            if info.get("dead_target", False):
                raise RuntimeError(
                    f"[{self.symbol}][{split_name}] DEAD TARGET '{target_name}': "
                    f"all labels are '{info.get('dominant_class')}' "
                    f"({info.get('dominant_pct', 0):.2%}). "
                    f"This target cannot be learned. Check label generation logic."
                )

        if drop_invalid:
            mask = get_valid_mask(result, strict=True)
            return {k: v[mask] for k, v in result.items()}
        return result

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC: state save/load
    # ══════════════════════════════════════════════════════════════════════

    def get_fitted_state(self):
        if not bool(self._is_fitted):
            return {"is_fitted": False}

        try:
            thr_mean = float(np.mean(self._thresholds)) if self._thresholds is not None else float(self._base_threshold or 0.0)
            thr_std = float(np.std(self._thresholds)) if self._thresholds is not None else 0.0
            thr_min = float(np.min(self._thresholds)) if self._thresholds is not None else float(self._base_threshold or 0.0)
            thr_max = float(np.max(self._thresholds)) if self._thresholds is not None else float(self._base_threshold or 0.0)
        except Exception:
            thr_mean = thr_std = thr_min = thr_max = float(self._base_threshold or 0.0)

        return {
            "is_fitted": True,
            "version": self.VERSION,
            "symbol": self.symbol,
            "prediction_horizon": int(self.prediction_horizon),
            "base_threshold": float(self._base_threshold or 0.0),
            "signal_threshold": float(self._signal_threshold or self._base_threshold or 0.0),
            "weak_factor": float(self._weak_factor),
            "strong_factor": float(self._strong_factor),
            "volatility_multiplier": float(self._volatility_multiplier),
            "volatility_regime": str(self._fitted_volatility_regime),
            "return_clip_low": float(self._return_clip_low) if self._return_clip_low is not None else None,
            "return_clip_high": float(self._return_clip_high) if self._return_clip_high is not None else None,
            "thresholds_mean": float(thr_mean),
            "thresholds_std": float(thr_std),
            "thresholds_min": float(thr_min),
            "thresholds_max": float(thr_max),
            "n_fitted_samples": int(self._n_fitted_samples),
            "weak_percentile": float(self.weak_percentile),
            "strong_percentile": float(self.strong_percentile),
            "n_classes_price_direction": int(self.PRICE_DIRECTION_N_CLASSES),
            "n_classes_momentum_direction": int(self.MOMENTUM_DIRECTION_N_CLASSES),
            "n_classes_magnitude_bucket": int(self.MAGNITUDE_BUCKET_N_CLASSES),
            "labeling_mode": str(self.labeling_mode),
            "epsilon": float(self.epsilon),
            "train_class_dist": (dict(self._train_class_dist) if self._train_class_dist else None),
            "mag_q33_frozen": (float(self._mag_q33_frozen) if self._mag_q33_frozen is not None else None),
            "mag_q66_frozen": (float(self._mag_q66_frozen) if self._mag_q66_frozen is not None else None),
            "momentum_ema_short": int(self.momentum_ema_short),
            "momentum_ema_long": int(self.momentum_ema_long),
            "sample_weight_agreement_power": float(self.sample_weight_agreement_power),
            # v14.0
            "min_label_abs_return": float(self.min_label_abs_return),
            "warmup_targets_best_effort": bool(self.warmup_targets_best_effort),
            "max_invalid_ratio_warn": float(self.max_invalid_ratio_warn),
            "magnitude_q33_percentile": float(self.magnitude_q33_percentile),
            "magnitude_q66_percentile": float(self.magnitude_q66_percentile),
        }

    def load_fitted_state(self, state):
        try:
            if state is None or not bool(state.get("is_fitted", False)):
                raise ValueError("State is not fitted")

            saved_n_classes = state.get("n_classes_price_direction")
            if saved_n_classes is not None and int(saved_n_classes) != int(self.PRICE_DIRECTION_N_CLASSES):
                logger.warning(
                    "load_fitted_state: state n_classes=%s ≠ current n_classes=%d",
                    saved_n_classes, self.PRICE_DIRECTION_N_CLASSES,
                )

            saved_mode = state.get("labeling_mode")
            if saved_mode is not None and str(saved_mode) != str(self.labeling_mode):
                logger.warning(
                    "load_fitted_state: state labeling_mode=%s ≠ current=%s. Using current.",
                    saved_mode, self.labeling_mode,
                )

            self._is_fitted = True
            self._base_threshold = float(state["base_threshold"])
            self._signal_threshold = float(state.get("signal_threshold", state["base_threshold"]))
            self._weak_factor = float(state.get("weak_factor", 1.0))
            self._strong_factor = float(state.get("strong_factor", 1.0))
            self._volatility_multiplier = float(state.get("volatility_multiplier", 1.0))
            self._fitted_volatility_regime = str(
                state.get("volatility_regime", state.get("regime", "NORMAL_VOL"))
            )

            try:
                self._return_clip_low = float(state["return_clip_low"]) if state.get("return_clip_low") is not None else None
            except Exception:
                self._return_clip_low = None
            try:
                self._return_clip_high = float(state["return_clip_high"]) if state.get("return_clip_high") is not None else None
            except Exception:
                self._return_clip_high = None

            try:
                self._n_fitted_samples = int(state.get("n_fitted_samples", 0))
            except Exception:
                self._n_fitted_samples = 0

            try:
                self.symbol = state.get("symbol", self.symbol)
            except Exception:
                pass

            try:
                mq33 = state.get("mag_q33_frozen")
                self._mag_q33_frozen = float(mq33) if mq33 is not None else None
            except Exception:
                self._mag_q33_frozen = None
            try:
                mq66 = state.get("mag_q66_frozen")
                self._mag_q66_frozen = float(mq66) if mq66 is not None else None
            except Exception:
                self._mag_q66_frozen = None

            n_thr = int(self._n_fitted_samples) if int(self._n_fitted_samples) > 0 else 1000
            self._thresholds = np.full(n_thr, float(self._base_threshold), dtype=np.float64)
            self._fitted_state = dict(state)

            # v14.0 optional fields
            try:
                self.min_label_abs_return = float(state.get("min_label_abs_return", self.min_label_abs_return))
            except Exception:
                pass
            try:
                self.warmup_targets_best_effort = bool(state.get("warmup_targets_best_effort", self.warmup_targets_best_effort))
            except Exception:
                pass
            try:
                self.max_invalid_ratio_warn = float(state.get("max_invalid_ratio_warn", self.max_invalid_ratio_warn))
            except Exception:
                pass

            logger.info(
                "load_fitted_state ok base=%.6f mode=%s mag_q33=%.8f mag_q66=%.8f",
                float(self._base_threshold), self.labeling_mode,
                self._mag_q33_frozen if self._mag_q33_frozen is not None else -1.0,
                self._mag_q66_frozen if self._mag_q66_frozen is not None else -1.0,
            )
            return self
        except Exception as e:
            logger.error("load_fitted_state failed: %s", str(e))
            self._is_fitted = False
            raise

    # ══════════════════════════════════════════════════════════════════════
    # MIGRATION helper (جدید در v14.0)
    # ══════════════════════════════════════════════════════════════════════

    def migrate_direction_to_sign_mode(
        self,
        price_return: np.ndarray,
        old_price_direction: np.ndarray,
    ) -> np.ndarray:
        """
        MIGRATE_DIRECTION_TO_SIGN_MODE v1.0 — IN-PLACE CONVERSION.

        از دادهٔ موجود (که با sign_with_epsilon ساخته شده و INVALID دارد)
        یک نسخهٔ sign-based می‌سازد. قابل اجرا روی H5 موجود بدون
        نیاز به regenerate.

        Args:
            price_return: (N,) یا (N,1) — همان price_return موجود در H5
            old_price_direction: (N,) یا (N,1) — لیبل قدیمی

        Returns:
            (N,1) int32 — لیبل جدید مبتنی بر sign(price_return)
        """
        ret = np.asarray(price_return, dtype=np.float64).flatten()
        n = ret.shape[0]

        old = np.asarray(old_price_direction).flatten()
        if old.shape[0] != n:
            logger.warning(
                "[%s] migrate: old_direction length %d != price_return %d",
                str(self.symbol), old.shape[0], n,
            )

        finite = np.isfinite(ret)
        new_dir = np.full(n, self.INVALID, dtype=np.int32)
        new_dir[finite & (ret >= 0.0)] = 1
        new_dir[finite & (ret < 0.0)] = 0

        n_before_invalid = int((old == self.INVALID).sum()) if old.size > 0 else 0
        n_after_invalid = int((new_dir == self.INVALID).sum())
        n_fixed = n_before_invalid - n_after_invalid

        logger.info(
            "[%s] ✅ Direction migrated to sign mode: "
            "before=%d INVALID, after=%d INVALID, fixed=%d samples "
            "(%.2f%% of data recovered)",
            str(self.symbol), n_before_invalid, n_after_invalid, n_fixed,
            100.0 * n_fixed / max(n, 1),
        )

        return new_dir.reshape(-1, 1)

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL: threshold application (KEY FIX)
    # ══════════════════════════════════════════════════════════════════════

    def _apply_thresholds_to_array(
        self,
        future_returns,
        thresholds,
        use_strong_classes: bool = True,
    ):
        """
        _APPLY_THRESHOLDS_TO_ARRAY v14.1 — NOISE-FLOOR / MINIMAL-INVALID.

        ──────────────────────────────────────────────────────────────────────
        تاریخچهٔ تغییرات:
        ──────────────────────────────────────────────────────────────────────
        v14.0:
            شاخهٔ sign_with_epsilon از eps_eff = max(self.epsilon,
            self.min_label_abs_return) استفاده می‌کرد. چون در پایپ‌لاین
            _generate_targets مقدار self.epsilon=0.0015 (۱۵ bp) پاس
            می‌شد، آستانهٔ ثابتِ بزرگی اعمال می‌شد و در بازهٔ ۳۰-دقیقه‌ای
            BTC حدود ۳۵٪ نمونه‌ها INVALID می‌شدند.

        v14.1 (این نسخه):
            منطق آستانه بازنویسی شد تا INVALID از ۳۵٪ به ~۲-۳٪ کاهش یابد
            بدون آنکه ویژگی «فیلترِ نویز» از دست برود.

        ──────────────────────────────────────────────────────────────────────
        قرارداد جدید (فقط در شاخهٔ sign_with_epsilon):
        ──────────────────────────────────────────────────────────────────────
        ۱) اگر self.min_label_abs_return > 0 باشد:
              → آستانهٔ ثابتِ صریحِ کاربر (رفتار قبلی، کاملاً تحت کنترل).
              → مناسب وقتی می‌خواهید حرکت‌های کوچک‌تر از یک مرز مشخص
                حتماً INVALID شوند.

        ۲) اگر self.min_label_abs_return == 0 (پیش‌فرض):
              → کف نویز پیش‌فرض 1e-4 (۱ bp = ۰.۰۱٪).
              → فقط نمونه‌های با |ret| ≤ 1 bp به‌عنوان نویز واقعی
                INVALID می‌شوند. بقیه بر اساس sign به BUY/SELL
                نگاشت می‌شوند (شامل حرکت‌های ضعیف).
              → نمونه‌های INVALID حاصل از این فیلتر معمولاً ~۲-۳٪
                هستند (به‌علاوهٔ h نمونهٔ پایانی که future return
                آن‌ها NaN است).

        ۳) override تنگ‌تر:
              → اگر self.epsilon > 0 و SMALLER از کف نویز باشد
                (مثلاً self.epsilon = 1e-5)، همان مقدار کوچک‌تر
                اعمال می‌شود. یعنی کاربر می‌تواند با کاهش epsilon
                فیلتر را تنگ‌تر کند، ولی نمی‌تواند سهواً آن را
                بزرگ‌تر از کف نویز کند.

        در حالت "sign":
            → ۰٪ INVALID (به‌جز h نمونهٔ پایانی که NaN هستند).

        ──────────────────────────────────────────────────────────────────────
        چرا کف نویز 1 bp امن است؟
        ──────────────────────────────────────────────────────────────────────
        - Std بازده ۳۰-دقیقه‌ای BTC معمولاً ~۳-۵ bp است.
        - |ret| ≤ 1 bp عملاً غیرقابل‌تشخیص از نویز میکروساختار است.
        - نمونه‌های INVALID باقی‌مانده (~۲-۳٪) در loss پایین وزن
          می‌گیرند (sample_weight جهت‌محور) لذا این فیلتر به
          یادگیری مدل آسیب نمی‌زند.
        - برخلاف نسخهٔ v14.0 که ۳۵٪ نمونه را حذف می‌کرد و اریبی
          «فقط موارد واضح» ایجاد می‌کرد، این نسخه ۹۷٪+ داده را
          در اختیار یادگیرنده می‌گذارد.
        """

        try:
            fut = np.asarray(future_returns, dtype=np.float64).flatten()
            n = fut.shape[0]
            if n == 0:
                return np.array([], dtype=np.int32)

            out = np.full(n, np.int32(INVALID_LABEL), dtype=np.int32)
            finite = np.isfinite(fut)

            # ─────────────────────────────────────────────────────────────
            # شاخهٔ sign_with_epsilon
            # ─────────────────────────────────────────────────────────────
            if self.labeling_mode == "sign_with_epsilon":
                eps_user = float(getattr(self, "min_label_abs_return", 0.0) or 0.0)

                if eps_user > 0.0:
                    # (۱) کاربر صریحاً آستانهٔ ثابت خواسته است.
                    eps_eff = eps_user
                    threshold_source = f"user_fixed({eps_eff:.8g})"

                else:
                    # (۲) آستانهٔ پیش‌فرض: کف نویز ۱ bp.
                    DEFAULT_NOISE_FLOOR = 1e-4  # 1 bp = 0.01%
                    eps_eff = DEFAULT_NOISE_FLOOR

                    # (۳) اگر کاربر epsilon کوچک‌تری خواسته باشد، همان اعمال.
                    try:
                        constructor_eps = float(self.epsilon)
                    except (TypeError, ValueError):
                        constructor_eps = 0.0

                    if 0.0 < constructor_eps < DEFAULT_NOISE_FLOOR:
                        eps_eff = constructor_eps
                        threshold_source = (
                            f"constructor_tighter({eps_eff:.8g})"
                        )
                    else:
                        threshold_source = (
                            f"noise_floor({eps_eff:.8g})"
                        )

                eps_arr = np.full(n, float(eps_eff), dtype=np.float64)

                buy_mask = finite & (fut > eps_arr)
                sell_mask = finite & (fut < -eps_arr)
                out[buy_mask] = 1
                out[sell_mask] = 0
                # |fut| <= eps_eff همچنان INVALID می‌ماند (کف نویز).

            # ─────────────────────────────────────────────────────────────
            # شاخهٔ sign (۰٪ INVALID به‌جز NaN futures)
            # ─────────────────────────────────────────────────────────────
            else:
                threshold_source = "sign"
                buy_mask = finite & (fut >= 0.0)
                sell_mask = finite & (fut < 0.0)
                out[buy_mask] = 1
                out[sell_mask] = 0

            # ─────────────────────────────────────────────────────────────
            # لاگ + به‌روزرسانی dist
            # ─────────────────────────────────────────────────────────────
            try:
                vv = out[out != INVALID_LABEL]
                n_invalid = int(np.sum(out == INVALID_LABEL))
                invalid_pct = 100.0 * n_invalid / max(n, 1)

                if vv.size > 0:
                    uq, ct = np.unique(vv, return_counts=True)
                    tot = float(vv.size)
                    msg = ", ".join(
                        f"{int(a)}:{int(b)}({b / tot * 100:.1f}%)"
                        for a, b in zip(uq, ct)
                    )
                    logger.info(
                        "direction(2-class, %s, %s) valid=%d (%.2f%%) "
                        "invalid=%d (%.2f%%) [%s]",
                        self.labeling_mode, threshold_source,
                        int(vv.size), 100.0 * vv.size / max(n, 1),
                        n_invalid, invalid_pct, msg,
                    )
                    self._train_class_dist = {
                        int(a): float(b) / tot for a, b in zip(uq, ct)
                    }
                else:
                    logger.warning(
                        "direction: all %d labels invalid (%s)",
                        n, threshold_source,
                    )
            except Exception as log_err:
                logger.debug(f"direction log step skipped: {log_err}")

            return out

        except Exception as e:
            logger.error(
                "_apply_thresholds_to_array failed: %s", str(e)
            )
            try:
                nn = len(future_returns)
            except Exception:
                nn = 0
            return np.full(int(nn), np.int32(INVALID_LABEL), dtype=np.int32)
            
    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL: audits (NEW v14.0)
    # ══════════════════════════════════════════════════════════════════════

    def _audit_labels(self, result, n_total, split_name):
        """
        AUDIT v14.0 — خودکار بررسی INVALID و توزیع کلاس.
        """
        issues = {}
        logger.info("[%s][%s] ═══════ LABEL AUDIT ═══════", str(self.symbol), split_name)

        for target_name, arr in result.items():
            if target_name == "sample_weight":
                continue
            try:
                a = np.asarray(arr).flatten()
                if target_name == "price_return":
                    n_invalid = int((~np.isfinite(a)).sum())
                    valid = a[np.isfinite(a)]
                    info = {
                        "n_total": n_total,
                        "n_invalid": n_invalid,
                        "invalid_pct": 100.0 * n_invalid / max(n_total, 1),
                        "n_valid": int(valid.size),
                    }
                    if valid.size > 0:
                        info["mean"] = float(np.mean(valid))
                        info["std"] = float(np.std(valid))
                    logger.info(
                        "[%s][%s]   %-22s reg  n=%d, INVALID=%d (%.2f%%), "
                        "μ=%.6f, σ=%.6f",
                        str(self.symbol), split_name, target_name,
                        n_total, n_invalid, info["invalid_pct"],
                        info.get("mean", 0.0), info.get("std", 0.0),
                    )
                else:
                    n_invalid = int((a == self.INVALID).sum())
                    valid = a[a != self.INVALID]
                    n_classes = len(self._CLASS_NAMES.get(target_name, {}))
                    info = {
                        "n_total": n_total,
                        "n_invalid": n_invalid,
                        "invalid_pct": 100.0 * n_invalid / max(n_total, 1),
                        "n_valid": int(valid.size),
                        "n_classes": n_classes,
                    }

                    if valid.size > 0:
                        uq, ct = np.unique(valid, return_counts=True)
                        tot = float(valid.size)
                        dist = {int(k): float(v) / tot for k, v in zip(uq, ct)}
                        info["distribution"] = dist

                        # Dead target check
                        dominant_class = max(dist, key=dist.get)
                        dominant_pct = dist[dominant_class]
                        info["dominant_class"] = int(dominant_class)
                        info["dominant_pct"] = float(dominant_pct)
                        if dominant_pct > 0.98:
                            info["dead_target"] = True
                            info["dead_reason"] = (
                                f"class {dominant_class} accounts for "
                                f"{dominant_pct:.2%}"
                            )
                            issues[target_name] = info
                            logger.error(
                                "[%s][%s]   %-22s cls  n=%d, INVALID=%d (%.2f%%), "
                                "🔴 DEAD TARGET (class %d = %.2f%%)",
                                str(self.symbol), split_name, target_name,
                                n_total, n_invalid, info["invalid_pct"],
                                dominant_class, dominant_pct * 100,
                            )
                            continue

                        # Check class coverage
                        missing_classes = [
                            c for c in range(n_classes)
                            if c not in dist
                        ]
                        info["missing_classes"] = missing_classes

                        cls_names = self._CLASS_NAMES.get(target_name, {})
                        dist_str = ", ".join(
                            f"{k}({cls_names.get(k, '?')}):{v*100:.1f}%"
                            for k, v in sorted(dist.items())
                        )

                        if missing_classes:
                            logger.warning(
                                "[%s][%s]   %-22s cls  n=%d, INVALID=%d (%.2f%%), "
                                "⚠️ missing classes=%s, dist={%s}",
                                str(self.symbol), split_name, target_name,
                                n_total, n_invalid, info["invalid_pct"],
                                missing_classes, dist_str,
                            )
                        else:
                            logger.info(
                                "[%s][%s]   %-22s cls  n=%d, INVALID=%d (%.2f%%), "
                                "dist={%s}",
                                str(self.symbol), split_name, target_name,
                                n_total, n_invalid, info["invalid_pct"],
                                dist_str,
                            )
                    else:
                        info["dead_target"] = True
                        info["dead_reason"] = "all labels INVALID"
                        issues[target_name] = info
                        logger.error(
                            "[%s][%s]   %-22s cls  🔴 DEAD TARGET: all %d INVALID",
                            str(self.symbol), split_name, target_name, n_total,
                        )

                    # v14.0: warning threshold
                    if info["invalid_pct"] > self.max_invalid_ratio_warn * 100:
                        logger.warning(
                            "[%s][%s]   %-22s ⚠️ INVALID ratio %.2f%% > %.2f%%",
                            str(self.symbol), split_name, target_name,
                            info["invalid_pct"], self.max_invalid_ratio_warn * 100,
                        )

                    issues[target_name] = info
            except Exception as e:
                logger.warning(
                    "[%s][%s] audit failed for %s: %s",
                    str(self.symbol), split_name, target_name, e,
                )

        # sample_weight audit
        try:
            sw = np.asarray(result.get("sample_weight", [])).flatten()
            if sw.size > 0:
                valid_sw = sw[sw > 0]
                logger.info(
                    "[%s][%s]   %-22s w    mean=%.4f, std=%.4f, range=[%.4f, %.4f], "
                    "zeros=%d (%.2f%%)",
                    str(self.symbol), split_name, "sample_weight",
                    float(np.mean(valid_sw)) if valid_sw.size else 0.0,
                    float(np.std(valid_sw)) if valid_sw.size else 0.0,
                    float(np.min(valid_sw)) if valid_sw.size else 0.0,
                    float(np.max(valid_sw)) if valid_sw.size else 0.0,
                    int((sw == 0).sum()), 100.0 * (sw == 0).sum() / sw.size,
                )
        except Exception as e:
            logger.debug(f"sample_weight audit failed: {e}")

        logger.info("[%s][%s] ════════════════════════════", str(self.symbol), split_name)
        return issues

    # ══════════════════════════════════════════════════════════════════════
    # INTERNAL: helpers (preserved from v13.0)
    # ══════════════════════════════════════════════════════════════════════

    def set_caller(self, name):
        self._caller = str(name) if name is not None else None
        return self

    def _compute_past_returns(self, prices):
        try:
            p = _as_1d_float64(prices)
            n = p.shape[0]
            out = np.full(n, np.nan, dtype=np.float64)
            if n < 2:
                return out
            prev = p[:-1]
            curr = p[1:]
            valid = np.isfinite(prev) & np.isfinite(curr) & (np.abs(prev) > 1e-8)
            vals = np.full(curr.shape, np.nan, dtype=np.float64)
            pv = prev[valid]
            cv = curr[valid]
            vals[valid] = (cv - pv) / pv
            out[1:] = vals
            return out
        except Exception:
            try:
                n2 = len(prices) if prices is not None else 0
            except Exception:
                n2 = 0
            return np.full(int(n2), np.nan, dtype=np.float64)

    def _compute_future_returns(self, prices):
        try:
            p = _as_1d_float64(prices)
            n = p.shape[0]
            out = np.full(n, np.nan, dtype=np.float64)
            h = int(self.prediction_horizon)
            if h < 1:
                h = 1
            if n <= h:
                return out
            base = p[: n - h]
            fwd = p[h:]
            valid = np.isfinite(base) & np.isfinite(fwd) & (np.abs(base) > 1e-8)
            vals = np.full(base.shape, np.nan, dtype=np.float64)
            vals[valid] = (fwd[valid] - base[valid]) / base[valid]
            out[: n - h] = vals
            return out
        except Exception:
            try:
                n2 = len(prices) if prices is not None else 0
            except Exception:
                n2 = 0
            return np.full(int(n2), np.nan, dtype=np.float64)

    def _ema_past_only(self, arr, alpha):
        try:
            n = int(len(arr))
            out = np.full(n, np.nan, dtype=np.float64)
            if n == 0:
                return out
            arr_f = np.asarray(arr, dtype=np.float64)
            finite_mask = np.isfinite(arr_f)
            if not finite_mask.any():
                return out
            first_idx = int(np.argmax(finite_mask))
            out[first_idx] = arr_f[first_idx]
            one_minus_alpha = 1.0 - float(alpha)
            for i in range(first_idx + 1, n):
                curr = arr_f[i]
                prev = out[i - 1]
                if np.isfinite(curr):
                    out[i] = alpha * curr + one_minus_alpha * prev if np.isfinite(prev) else curr
                else:
                    out[i] = prev
            return out
        except Exception:
            return np.full(len(arr), np.nan, dtype=np.float64)

    def _rolling_mean_past_only(self, arr, window):
        try:
            n = int(len(arr))
            w = int(window)
            out = np.full(n, np.nan, dtype=np.float64)
            if n == 0 or w <= 0:
                return out
            arr_f = np.asarray(arr, dtype=np.float64)
            finite_mask = np.isfinite(arr_f)
            prefix_sum = np.zeros(n + 1, dtype=np.float64)
            prefix_cnt = np.zeros(n + 1, dtype=np.int64)
            for i in range(n):
                if finite_mask[i]:
                    prefix_sum[i + 1] = prefix_sum[i] + arr_f[i]
                    prefix_cnt[i + 1] = prefix_cnt[i] + 1
                else:
                    prefix_sum[i + 1] = prefix_sum[i]
                    prefix_cnt[i + 1] = prefix_cnt[i]
            for i in range(w - 1, n):
                start = i - w + 1
                count = int(prefix_cnt[i + 1] - prefix_cnt[start])
                if count == w:
                    total = prefix_sum[i + 1] - prefix_sum[start]
                    out[i] = total / float(w)
            return out
        except Exception:
            return np.full(len(arr), np.nan, dtype=np.float64)

    def _rolling_rank_past_only(self, arr, window):
        values = np.asarray(arr, dtype=np.float64)
        n = values.shape[0]
        w = int(window)
        out = np.full(n, np.nan, dtype=np.float64)
        if w < 1 or n <= w:
            return out
        for i in range(w, n):
            current = values[i]
            if not np.isfinite(current):
                continue
            history = values[i - w:i]
            finite_history = history[np.isfinite(history)]
            if finite_history.size != w:
                continue
            less = int(np.count_nonzero(finite_history < current))
            equal = int(np.count_nonzero(finite_history == current))
            out[i] = (less + 0.5 * equal) / float(w)
        return out

    def _compute_direction_aware_weight(self, close_prices, price_return):
        """
        _COMPUTE_DIRECTION_AWARE_WEIGHT v14.1 — 2-SIGNAL AGREEMENT.

        (بدون تغییر نسبت به نسخهٔ قبلی، سازگار با v14.0)
        """
        try:
            p = _as_1d_float64(close_prices)
            n = p.shape[0]
            pr = np.asarray(price_return, dtype=np.float64).flatten()

            if len(pr) != n:
                nn = min(len(pr), n)
                p = p[:nn]
                pr = pr[:nn]
                n = nn

            if n < 25:
                finite_mask = np.isfinite(pr)
                return np.where(finite_mask, 1.0, 0.0).astype(np.float32)

            ret_5 = np.full(n, np.nan, dtype=np.float64)
            for i in range(5, n):
                if np.isfinite(p[i]) and np.isfinite(p[i - 5]) and abs(p[i - 5]) > 1e-8:
                    ret_5[i] = (p[i] - p[i - 5]) / p[i - 5]

            ma20 = self._rolling_mean_past_only(p, 20)
            deviation = np.full(n, np.nan, dtype=np.float64)
            valid_ma = np.isfinite(ma20) & (ma20 > 1e-8)
            deviation[valid_ma] = (p[valid_ma] - ma20[valid_ma]) / ma20[valid_ma]

            s_mom = np.sign(ret_5)
            s_mr = -np.sign(deviation)

            valid_signals = (
                np.isfinite(s_mom) & np.isfinite(s_mr)
                & (s_mom != 0) & (s_mr != 0)
            )
            agreement = ((s_mom == s_mr) & valid_signals).astype(np.float64)
            strength = agreement

            power = float(self.sample_weight_agreement_power)
            weight_raw = 1.0 + power * strength

            finite_mask = np.isfinite(pr)
            if int(finite_mask.sum()) < 10:
                return np.where(finite_mask, 1.0, 0.0).astype(np.float32)

            mean_w = float(np.mean(weight_raw[finite_mask]))
            if not np.isfinite(mean_w) or mean_w < 1e-8:
                mean_w = 1.0

            sample_weight = np.where(finite_mask, weight_raw / mean_w, 0.0).astype(np.float32)

            try:
                valid_weights = sample_weight[sample_weight > 0]
                if valid_weights.size > 0:
                    n_agreed = int(agreement[finite_mask].sum())
                    n_total = int(finite_mask.sum())
                    logger.info(
                        "[%s] direction-aware weight | mean=%.4f, std=%.4f, "
                        "min=%.4f, max=%.4f, agreed=%d/%d (%.1f%%)",
                        str(self.symbol),
                        float(valid_weights.mean()), float(valid_weights.std()),
                        float(valid_weights.min()), float(valid_weights.max()),
                        n_agreed, n_total, 100.0 * n_agreed / max(1, n_total),
                    )
            except Exception:
                pass

            return sample_weight
        except Exception as e:
            logger.error("[%s] _compute_direction_aware_weight failed: %s", str(self.symbol), e)
            try:
                pr = np.asarray(price_return, dtype=np.float64).flatten()
                return np.where(np.isfinite(pr), 1.0, 0.0).astype(np.float32)
            except Exception:
                return np.array([], dtype=np.float32)

    def get_valid_mask_from_arrays(
        self,
        price_return,
        price_direction,
        trend_strength,
        volatility_regime,
        market_regime,
        magnitude_bucket=None,
        momentum_direction=None,
    ):
        arrays = {
            "price_return": np.asarray(price_return).reshape(-1),
            "price_direction": np.asarray(price_direction).reshape(-1),
            "trend_strength": np.asarray(trend_strength).reshape(-1),
            "volatility_regime": np.asarray(volatility_regime).reshape(-1),
            "market_regime": np.asarray(market_regime).reshape(-1),
        }
        if magnitude_bucket is not None:
            arrays["magnitude_bucket"] = np.asarray(magnitude_bucket).reshape(-1)
        if momentum_direction is not None:
            arrays["momentum_direction"] = np.asarray(momentum_direction).reshape(-1)
        return get_valid_mask(arrays, strict=True)

    def validate_no_leakage(self, train_df, test_df, use_row_hash=True, use_timestamp_column=None):
        """(بدون تغییر نسبت به v13.0 — همان رفتار)"""
        try:
            if train_df is None or test_df is None or train_df.empty or test_df.empty:
                return {"has_leakage": False, "status": "SKIPPED", "leakage_checks": {}}

            result = {
                "has_leakage": False,
                "status": "CLEAN",
                "leakage_checks": {},
                "suspicious_features": [],
            }

            if bool(use_row_hash):
                try:
                    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
                    if not num_cols:
                        num_cols = list(train_df.columns)
                    use_cols = num_cols[:min(20, len(num_cols))]
                    tr_s = train_df[use_cols].copy().round(6).dropna()
                    te_s = test_df[use_cols].copy().round(6).dropna()
                    if len(tr_s) > 5000:
                        tr_s = tr_s.sample(n=5000, random_state=42)
                    if len(te_s) > 5000:
                        te_s = te_s.sample(n=5000, random_state=42)

                    tr_h = set(pd.util.hash_pandas_object(tr_s, index=False).tolist())
                    te_h = set(pd.util.hash_pandas_object(te_s, index=False).tolist())
                    inter = len(tr_h & te_h)
                    pct = float(inter) / float(max(1, len(tr_h))) * 100.0

                    result["leakage_checks"]["row_hash_overlap"] = {
                        "count": int(inter), "percent": round(float(pct), 2),
                        "status": "LEAKAGE" if pct > 5.0 else "CLEAN",
                    }
                    if pct > 5.0:
                        result["has_leakage"] = True
                        result["status"] = "ROW_DUPLICATES"
                except Exception as e:
                    logger.warning("row hash check failed: %s", str(e))

            ts_col = use_timestamp_column
            if ts_col is None:
                for cand in ["timestamp", "datetime", "date", "time", "open_time"]:
                    if cand in train_df.columns and cand in test_df.columns:
                        ts_col = cand
                        break

            if ts_col is not None and ts_col in train_df.columns and ts_col in test_df.columns:
                try:
                    tr_ts = set(pd.Series(train_df[ts_col].values).dropna().tolist())
                    te_ts = set(pd.Series(test_df[ts_col].values).dropna().tolist())
                    inter_ts = tr_ts & te_ts
                    result["leakage_checks"]["timestamp_overlap"] = {
                        "count": int(len(inter_ts)), "has_overlap": len(inter_ts) > 0,
                    }
                    if len(inter_ts) > 0:
                        result["has_leakage"] = True
                        if result["status"] == "CLEAN":
                            result["status"] = "TIME_OVERLAP"
                except Exception as e:
                    logger.warning("timestamp check failed: %s", str(e))

            return result
        except Exception as e:
            logger.error("validate_no_leakage fatal: %s", str(e))
            return {"has_leakage": False, "status": f"ERROR: {e}", "leakage_checks": {}}
            
@dataclass
class LeakageResult:
    label_type_name: str = "unknown"
    label_type: LabelType = LabelType.UNKNOWN
    has_leakage: bool = False
    severity: LeakageSeverity = LeakageSeverity.NONE
    classification_metrics: Optional[ClassificationLeakageMetrics] = None
    regression_metrics: Optional[RegressionLeakageMetrics] = None
    tolerance: float = 0.05
    n_samples_compared: int = 0
    original_sample: list = field(default_factory=list)
    generated_sample: list = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    @property
    def status(self) -> str:
        mp = {LeakageSeverity.NONE: "CLEAN", LeakageSeverity.LOW: "LOW", LeakageSeverity.MEDIUM: "MEDIUM", LeakageSeverity.HIGH: "HIGH", LeakageSeverity.CRITICAL: "CRITICAL"}
        return mp.get(self.severity, "UNKNOWN")
    @property
    def agreement_pct(self) -> float:
        try:
            if self.classification_metrics is not None:
                return float(self.classification_metrics.agreement_pct)
            if self.regression_metrics is not None:
                c = float(self.regression_metrics.pearson_correlation)
                if not np.isfinite(c):
                    return 0.0
                return float(max(0.0, min(100.0, (c + 1.0) / 2.0 * 100.0)))
        except Exception:
            pass
        return 0.0
    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"label_type_name": str(self.label_type_name), "label_type": self.label_type.value if isinstance(self.label_type, LabelType) else str(self.label_type), "has_leakage": bool(self.has_leakage), "severity": self.severity.value if isinstance(self.severity, LeakageSeverity) else str(self.severity), "status": str(self.status), "agreement_pct": float(self.agreement_pct), "tolerance": float(self.tolerance), "n_samples_compared": int(self.n_samples_compared), "original_sample": list(self.original_sample[:10]) if self.original_sample else [], "generated_sample": list(self.generated_sample[:10]) if self.generated_sample else [], "metadata": dict(self.metadata) if self.metadata else {}}
        if self.classification_metrics is not None:
            cm = self.classification_metrics
            out["classification"] = {"accuracy": float(cm.accuracy), "agreement_pct": float(cm.agreement_pct), "disagreement_pct": float(cm.disagreement_pct), "chi2_statistic": float(cm.chi2_statistic), "chi2_p_value": float(cm.chi2_p_value), "kl_divergence": float(cm.kl_divergence), "kl_divergence_normalized": float(cm.kl_divergence_normalized), "js_divergence": float(cm.js_divergence), "wasserstein_distance": float(cm.wasserstein_distance), "original_distribution": dict(cm.original_distribution), "generated_distribution": dict(cm.generated_distribution), "class_wise_agreement": dict(cm.class_wise_agreement)}
        if self.regression_metrics is not None:
            rm = self.regression_metrics
            out["regression"] = {"pearson_correlation": float(rm.pearson_correlation), "spearman_correlation": float(rm.spearman_correlation), "mean_absolute_error": float(rm.mean_absolute_error), "mean_squared_error": float(rm.mean_squared_error), "root_mean_squared_error": float(rm.root_mean_squared_error), "normalized_mae": float(rm.normalized_mae), "ks_statistic": float(rm.ks_statistic), "ks_p_value": float(rm.ks_p_value), "wasserstein_distance": float(rm.wasserstein_distance), "mean_diff_pct": float(rm.mean_diff_pct), "std_diff_ratio": float(rm.std_diff_ratio), "quantile_comparison": dict(rm.quantile_comparison), "original_stats": dict(rm.original_stats), "generated_stats": dict(rm.generated_stats)}
        return out
class LabelLeakageDetector:
    """
    تشخیص‌دهنده نشتی لیبل‌ها — نسخه یکتا و نهایی (classification + regression).

    تعریف صحیح نشتی:
      - classification: لیبل تولیدشده بیش از حد «شبیه» لیبل اصلی است
        (accuracy >= 1 - tolerance)؛ نه برعکس.
      - regression: همبستگی بسیار بالا (>= correlation_threshold) همراه با
        خطای نرمال‌شده بسیار کوچک (nmae < 0.3).
    """

    def __init__(self, lookback: int = 20, tolerance: float = 0.05,
                 correlation_threshold: float = 0.95, log_level: int = logging.INFO):
        self.lookback = int(lookback)
        self.tolerance = float(tolerance)
        self.correlation_threshold = float(correlation_threshold)
        self.log_level = int(log_level)

    # ------------------------------------------------------------------ #
    # Helpers با import محافظت‌شده (بدون وابستگی سخت به sklearn)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            from sklearn.metrics import accuracy_score
            return float(accuracy_score(y_true, y_pred))
        except Exception:
            a = np.asarray(y_true).flatten()
            b = np.asarray(y_pred).flatten()
            m = int(min(a.size, b.size))
            if m == 0:
                return 0.0
            return float(np.mean(a[:m] == b[:m]))

    @staticmethod
    def _mae_mse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            return float(mean_absolute_error(y_true, y_pred)), float(mean_squared_error(y_true, y_pred))
        except Exception:
            d = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
            return float(np.mean(np.abs(d))), float(np.mean(d * d))

    @staticmethod
    def _pearson(a: np.ndarray, b: np.ndarray) -> float:
        try:
            from scipy.stats import pearsonr
            v = float(pearsonr(a, b)[0])
            return v if np.isfinite(v) else 0.0
        except Exception:
            x = np.asarray(a, dtype=np.float64).flatten()
            y = np.asarray(b, dtype=np.float64).flatten()
            m = int(min(x.size, y.size))
            x = x[:m]
            y = y[:m]
            msk = np.isfinite(x) & np.isfinite(y)
            x = x[msk]
            y = y[msk]
            if x.size < 3 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
                return 0.0
            c = float(np.corrcoef(x, y)[0, 1])
            return c if np.isfinite(c) else 0.0

    @staticmethod
    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        try:
            from scipy.stats import spearmanr
            v = float(spearmanr(a, b)[0])
            return v if np.isfinite(v) else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    # API عمومی
    # ------------------------------------------------------------------ #
    def detect(self, original_labels: np.ndarray, generated_labels: np.ndarray,
               label_type_name: str = "unknown",
               label_type: "LabelType" = None) -> "LeakageResult":
        """
        مقایسه لیبل اصلی و تولیدشده و تشخیص نشتی.

        Exceptions: ورودی غیرآرایه‌ای/None به‌جای کرش، LeakageResult با
        metadata['error'] برمی‌گرداند (رفتار نسخه اول و پایدار برای batch).
        """
        try:
            o = np.asarray(original_labels, dtype=np.float64).flatten()
        except Exception as e:
            logger.warning("LabelLeakageDetector.detect: original_labels invalid: %s", str(e))
            return self._empty_result(label_type_name, label_type, f"original_labels invalid: {e}")
        try:
            g = np.asarray(generated_labels, dtype=np.float64).flatten()
        except Exception as e:
            logger.warning("LabelLeakageDetector.detect: generated_labels invalid: %s", str(e))
            return self._empty_result(label_type_name, label_type, f"generated_labels invalid: {e}")

        m = int(min(o.shape[0], g.shape[0]))
        o = o[:m]
        g = g[:m]
        valid = np.isfinite(o) & np.isfinite(g)
        o = o[valid]
        g = g[valid]
        n = int(o.shape[0])

        if n == 0:
            return self._empty_result(label_type_name, label_type, "No valid samples to compare")

        lt = label_type
        if lt is None or lt == LabelType.UNKNOWN:
            uo = int(np.unique(o).size)
            ug = int(np.unique(g).size)
            lt = LabelType.CLASSIFICATION if (uo <= 10 and ug <= 10) else LabelType.REGRESSION

        if lt == LabelType.CLASSIFICATION:
            return self._detect_classification_leakage(o, g, str(label_type_name), n)
        return self._detect_regression_leakage(o, g, str(label_type_name), n)

    # ------------------------------------------------------------------ #
    # Classification
    # ------------------------------------------------------------------ #
    def _detect_classification_leakage(self, original: np.ndarray, generated: np.ndarray,
                                       label_type_name: str, n_samples: int) -> "LeakageResult":
        oi = original.astype(np.int64)
        gi = generated.astype(np.int64)

        acc = self._accuracy(oi, gi)
        if not np.isfinite(acc):
            acc = 0.0
        agreement_pct = float(acc * 100.0)
        disagreement_pct = float((1.0 - acc) * 100.0)

        unique_classes = np.unique(np.concatenate([oi, gi]))
        class_wise_agreement: Dict[int, float] = {}
        for cls in unique_classes:
            msk = oi == int(cls)
            if int(np.sum(msk)) > 0:
                class_wise_agreement[int(cls)] = float(np.mean(oi[msk] == gi[msk]) * 100.0)

        try:
            nc = int(max(int(oi.max()) + 1 if oi.size else 3,
                         int(gi.max()) + 1 if gi.size else 3, 3))
        except Exception:
            nc = 3
        try:
            od = np.bincount(oi, minlength=nc).astype(np.float64)
            gd = np.bincount(gi, minlength=nc).astype(np.float64)
        except Exception:
            od = np.zeros(nc, dtype=np.float64)
            gd = np.zeros(nc, dtype=np.float64)
        odp = od / (float(od.sum()) + 1e-10) * 100.0
        gdp = gd / (float(gd.sum()) + 1e-10) * 100.0

        chi_s, chi_p = 0.0, 1.0
        try:
            chi_s, chi_p = chi2_contingency([od + 0.5, gd + 0.5])[:2]
            chi_s = float(chi_s)
            chi_p = float(chi_p)
        except Exception as e:
            logger.debug("chi2_contingency failed: %s", str(e))

        kl_div, kl_norm = 0.0, 0.0
        try:
            from scipy.stats import entropy
            kl_div = float(entropy(odp + 1e-10, gdp + 1e-10))
            kl_norm = float(kl_div / (np.log(float(nc)) + 1e-10))
        except Exception as e:
            logger.debug("entropy failed: %s", str(e))

        js_div = 0.0
        try:
            from scipy.spatial.distance import jensenshannon
            js_div = float(jensenshannon(odp + 1e-10, gdp + 1e-10))
        except Exception as e:
            logger.debug("jensenshannon failed: %s", str(e))

        w_dist = 0.0
        try:
            from scipy.stats import wasserstein_distance
            w_dist = float(wasserstein_distance(np.arange(nc), np.arange(nc), odp, gdp))
        except Exception as e:
            logger.debug("wasserstein_distance failed: %s", str(e))

        # ✅ منطق صحیح: نشتی = شباهت بیش از حد (accuracy بالا)
        thresh = float(1.0 - float(self.tolerance))
        has_leakage = bool(acc >= thresh)
        if acc >= 0.99:
            severity = LeakageSeverity.CRITICAL
        elif acc >= 0.95:
            severity = LeakageSeverity.HIGH
        elif acc >= 0.90:
            severity = LeakageSeverity.MEDIUM
        elif acc >= thresh:
            severity = LeakageSeverity.LOW
        else:
            severity = LeakageSeverity.NONE
            has_leakage = False

        metrics = ClassificationLeakageMetrics(
            accuracy=float(acc),
            agreement_pct=float(agreement_pct),
            disagreement_pct=float(disagreement_pct),
            chi2_statistic=float(chi_s),
            chi2_p_value=float(chi_p),
            kl_divergence=float(kl_div),
            kl_divergence_normalized=float(kl_norm),
            js_divergence=float(js_div),
            wasserstein_distance=float(w_dist),
            n_samples_compared=int(n_samples),
            original_distribution=dict(enumerate([float(x) for x in odp])),
            generated_distribution=dict(enumerate([float(x) for x in gdp])),
            class_wise_agreement=dict(class_wise_agreement),
        )
        return LeakageResult(
            label_type_name=str(label_type_name),
            label_type=LabelType.CLASSIFICATION,
            has_leakage=bool(has_leakage),
            severity=severity,
            classification_metrics=metrics,
            n_samples_compared=int(n_samples),
            original_sample=oi[:10].tolist(),
            generated_sample=gi[:10].tolist(),
            tolerance=float(self.tolerance),
            metadata={"threshold": float(thresh)},
        )

    # ------------------------------------------------------------------ #
    # Regression
    # ------------------------------------------------------------------ #
    def _detect_regression_leakage(self, original: np.ndarray, generated: np.ndarray,
                                   label_type_name: str, n_samples: int) -> "LeakageResult":
        pc = self._pearson(original, generated)
        sc = self._spearman(original, generated)
        mae, mse = self._mae_mse(original, generated)
        rmse = float(np.sqrt(max(0.0, mse)))
        ostd = float(np.std(original))
        nmae = float(mae / (ostd + 1e-10))

        ks_s, ks_p = 0.0, 1.0
        try:
            ks_s, ks_p = ks_2samp(original, generated)[:2]
            ks_s = float(ks_s)
            ks_p = float(ks_p)
        except Exception as e:
            logger.debug("ks_2samp failed: %s", str(e))

        w_dist = 0.0
        try:
            from scipy.stats import wasserstein_distance
            w_dist = float(wasserstein_distance(original, generated))
        except Exception as e:
            logger.debug("wasserstein_distance failed: %s", str(e))

        om, gm = float(np.mean(original)), float(np.mean(generated))
        osd, gsd = float(np.std(original)), float(np.std(generated))
        mean_diff_pct = float(abs(om - gm) / (abs(om) + 1e-10) * 100.0)
        std_diff_ratio = float(min(osd, gsd) / (max(osd, gsd) + 1e-10))

        quantile_comparison: Dict[str, float] = {}
        for q in [10, 25, 50, 75, 90]:
            try:
                oq = float(np.percentile(original, q))
                gq = float(np.percentile(generated, q))
                quantile_comparison["q" + str(q)] = float(abs(oq - gq) / (abs(oq) + 1e-10) * 100.0)
            except Exception:
                quantile_comparison["q" + str(q)] = 0.0

        original_stats = {"mean": om, "std": osd, "min": float(np.min(original)),
                          "max": float(np.max(original)), "median": float(np.median(original))}
        generated_stats = {"mean": gm, "std": gsd, "min": float(np.min(generated)),
                           "max": float(np.max(generated)), "median": float(np.median(generated))}

        # ✅ منطق صحیح: نشتی = همبستگی فوق‌العاده بالا + خطای بسیار کوچک
        thr = float(self.correlation_threshold)
        ac = float(abs(pc))
        has_leakage = bool(ac >= thr and nmae < 0.3)
        if ac >= 0.999 and nmae < 0.1:
            severity = LeakageSeverity.CRITICAL
        elif ac >= 0.99 and nmae < 0.2:
            severity = LeakageSeverity.HIGH
        elif ac >= thr and nmae < 0.3:
            severity = LeakageSeverity.MEDIUM
        elif ac >= 0.90:
            severity = LeakageSeverity.LOW
            has_leakage = False
        else:
            severity = LeakageSeverity.NONE
            has_leakage = False

        metrics = RegressionLeakageMetrics(
            pearson_correlation=float(pc),
            spearman_correlation=float(sc),
            correlation_leakage_threshold=float(thr),
            mean_absolute_error=float(mae),
            mean_squared_error=float(mse),
            root_mean_squared_error=float(rmse),
            normalized_mae=float(nmae),
            ks_statistic=float(ks_s),
            ks_p_value=float(ks_p),
            wasserstein_distance=float(w_dist),
            mean_diff_pct=float(mean_diff_pct),
            std_diff_ratio=float(std_diff_ratio),
            quantile_comparison=dict(quantile_comparison),
            self_correlation=1.0,
            self_mae=0.0,
            n_samples_compared=int(n_samples),
            original_stats=dict(original_stats),
            generated_stats=dict(generated_stats),
        )
        return LeakageResult(
            label_type_name=str(label_type_name),
            label_type=LabelType.REGRESSION,
            has_leakage=bool(has_leakage),
            severity=severity,
            regression_metrics=metrics,
            n_samples_compared=int(n_samples),
            original_sample=original[:10].tolist(),
            generated_sample=generated[:10].tolist(),
            tolerance=float(self.tolerance),
            metadata={"correlation_threshold": float(thr)},
        )

    # ------------------------------------------------------------------ #
    def _empty_result(self, label_type_name: str, label_type, err: str) -> "LeakageResult":
        return LeakageResult(
            label_type_name=str(label_type_name),
            label_type=label_type if isinstance(label_type, LabelType) else LabelType.UNKNOWN,
            has_leakage=False,
            severity=LeakageSeverity.NONE,
            n_samples_compared=0,
            tolerance=float(self.tolerance),
            metadata={"error": str(err)},
        )
def remove_highly_correlated_features(
    X: pd.DataFrame,
    correlation_threshold: float = 0.999,
    method: str = 'pearson',
    protected_columns: Optional[List[str]] = None,
    logger_instance=None,
    use_smart_removal: bool = True,
    target_columns: Optional[List[str]] = None,
    fit_mode: bool = True,
    ref_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
    """
    ✅ WALL STREET ZERO-LEAKAGE v7.1 - STRICT TRANSFORM CONTRACT
    ✅ Correlation decisions ONLY on TRAIN data (fit_mode=True)
    ✅ fit_mode=False دیگر روی دادهٔ VAL/TEST همبستگی محاسبه نمی‌کند.

    تغییر حیاتی نسبت به v7.0:
        نسخهٔ قبلی در حالت TRANSFORM (fit_mode=False) ماتریس همبستگی را روی
        خودِ split ارزیابی دوباره محاسبه می‌کرد؛ یعنی «تصمیم حذف فیچر» از توزیع
        VAL/TEST گرفته می‌شد که خود یک form of evaluation-data leakage است و با
        ادعای «ZERO-LEAKAGE» ناسازگار بود. در این نسخه:

          fit_mode=False + ref_columns=None
              → هیچ محاسبه‌ای انجام نمی‌شود؛ هشدار صریح ثبت شده و DataFrame
                بدون تغییر برمی‌گردد (removed_pairs = []).

          fit_mode=False + ref_columns=[...]
              → فقط فیچرهایی حذف می‌شوند که در لیست مرجع (خروجی fit روی TRAIN)
                وجود ندارند. هیچ آماری از split جاری محاسبه نمی‌شود.

    Returns:
        Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
            (دیتافریم پاک‌سازی‌شده, لیست جفت‌های حذف‌شده به فرم (kept, removed, corr))
    """
    log = logger_instance or logging.getLogger(__name__)

    EXEMPT_PAIRS = [
        ('plus_di', 'minus_di'), ('minus_di', 'plus_di'),
        ('plus_di', 'di_spread'), ('di_spread', 'plus_di'),
        ('minus_di', 'di_spread'), ('di_spread', 'minus_di'),
        ('bb_upper', 'bb_middle'), ('bb_middle', 'bb_upper'),
        ('bb_lower', 'bb_middle'), ('bb_middle', 'bb_lower'),
        ('bb_upper', 'bb_lower'), ('bb_lower', 'bb_upper'),
        ('ret', 'log_ret'), ('log_ret', 'ret'),
        ('atr_14', 'atr_28'), ('atr_28', 'atr_14'),
        ('atr_28', 'atr_50'), ('atr_50', 'atr_28'),
    ]

    try:
        if X is None or X.empty:
            log.warning("remove_highly_correlated_features: DataFrame empty or None")
            return X, []

        if protected_columns is None:
            protected_columns = ['open', 'high', 'low', 'close', 'volume', 'ret']

        protected_columns = list(set(protected_columns + PROJECT_TARGETS))

        if target_columns is None:
            target_columns = PROJECT_TARGETS

        log.info(f"   🛡️ Protected columns ({len(protected_columns)}): {protected_columns[:5]}...")
        log.info(f"   🎯 Target columns for scoring: {target_columns[:3]}...")
        log.info(f"   🤝 Exempt pairs from removal: {len(EXEMPT_PAIRS)}")

    except Exception as e:
        log.error(f"❌ remove_correlated: Validation failed: {str(e)}")
        return X, []

    try:
        X_work = X.copy()

        protected_present = [c for c in protected_columns if c in X_work.columns]
        feature_cols = [c for c in X_work.columns if c not in protected_present]
        target_present = [c for c in target_columns if c in X_work.columns]

        if len(feature_cols) < 2:
            log.info(f"   ✅ Only {len(feature_cols)} feature(s), skipping correlation removal")
            return X, []

        log.info(f"   📊 Processing {len(feature_cols)} feature columns...")

    except Exception as e:
        log.error(f"❌ remove_correlated: Column separation failed: {str(e)}")
        return X, []

    # ─────────────────────────────────────────────────────────────────────
    # TRANSFORM MODE: بدون هیچ محاسبه‌ای روی split جاری
    # ─────────────────────────────────────────────────────────────────────
    if not fit_mode:
        if ref_columns is None:
            log.warning(
                "   ⚠️ fit_mode=False و ref_columns ارائه نشده است؛ "
                "برای جلوگیری از نشتی ارزیابی، هیچ حذفی انجام نمی‌شود. "
                "لیست ستون‌های نهایی TRAIN را به‌عنوان ref_columns بدهید."
            )
            return X, []

        ref_set = set(str(c) for c in ref_columns)
        cols_to_remove_transform = {
            c for c in feature_cols
            if c not in ref_set and c not in protected_present
        }

        if cols_to_remove_transform:
            ordered_removed = [c for c in X_work.columns if c in cols_to_remove_transform]
            removed_pairs_transform: List[Tuple[str, str, float]] = [
                ("__ref_alignment__", col, float("nan"))
                for col in ordered_removed
            ]
            X_cleaned = X_work.drop(columns=ordered_removed, errors="raise")
            log.info(f"   🗑️ Transform alignment: removed {len(ordered_removed)} features not present in TRAIN reference")
            log.info(f"   📊 Shape: {X_work.shape[1]} → {X_cleaned.shape[1]} columns")
            return X_cleaned, removed_pairs_transform

        log.info("   ✅ Transform alignment: nothing to remove (all features present in reference)")
        return X_work, []

    # ─────────────────────────────────────────────────────────────────────
    # FIT MODE (TRAIN only)
    # ─────────────────────────────────────────────────────────────────────
    target_corrs: Dict[str, float] = {}

    if use_smart_removal and target_present:
        try:
            primary_target = target_present[0] if target_present else 'price_return'
            if primary_target in X_work.columns:
                log.info(f"   🎯 Computing correlations with target: {primary_target} (fit_mode=True)")

                for col in feature_cols:
                    try:
                        valid_data = X_work[[col, primary_target]].dropna()
                        if len(valid_data) > 10:
                            corr = abs(valid_data[col].corr(valid_data[primary_target]))
                            target_corrs[col] = float(corr) if not np.isnan(corr) else 0.0
                        else:
                            target_corrs[col] = 0.0
                    except Exception:
                        target_corrs[col] = 0.0

                if target_corrs:
                    corr_values = list(target_corrs.values())
                    log.info(f"   📊 Target correlation range: min={min(corr_values):.4f}, max={max(corr_values):.4f}")

        except Exception as e:
            log.warning(f"   ⚠️ Could not compute target correlations: {e}")
            use_smart_removal = False
    else:
        log.info(f"   ⏭️ Skipping target correlation (use_smart_removal={use_smart_removal}, targets={len(target_present)})")

    removed_pairs: List[Tuple[str, str, float]] = []
    cols_to_remove: Set[str] = set()

    try:
        log.info(f"   🔄 Calculating correlation matrix (TRAIN only)...")

        numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(X_work[c])]

        if len(numeric_feature_cols) < 2:
            log.info("   ✅ Not enough numeric features for correlation")
            return X, []

        if method == 'spearman':
            corr_matrix = X_work[numeric_feature_cols].corr(method='spearman').abs()
        else:
            corr_matrix = X_work[numeric_feature_cols].corr(method='pearson').abs()

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        high_corr_pairs = []
        for col in upper.columns:
            for idx in upper.index:
                corr_val = upper.loc[idx, col]
                if pd.notna(corr_val) and corr_val > correlation_threshold:
                    high_corr_pairs.append((col, idx, corr_val))

        log.info(f"   🔍 Found {len(high_corr_pairs)} high-correlation pairs (>{correlation_threshold})")

        skipped_count = 0
        for col_a, col_b, corr_val in high_corr_pairs:
            # جفت‌های مستثنی (همبستگی طبیعی اندیکاتورها)
            is_exempt = False
            for exempt_a, exempt_b in EXEMPT_PAIRS:
                if (col_a == exempt_a and col_b == exempt_b) or (col_a == exempt_b and col_b == exempt_a):
                    is_exempt = True
                    break

            if is_exempt:
                skipped_count += 1
                continue

            # اگر هر دو ستون از قبل برای حذف علامت خورده‌اند، جفت تکراری است
            if col_a in cols_to_remove and col_b in cols_to_remove:
                continue

            if col_a in protected_present:
                if col_b not in cols_to_remove:
                    cols_to_remove.add(col_b)
                    removed_pairs.append((col_a, col_b, float(corr_val)))
            elif col_b in protected_present:
                if col_a not in cols_to_remove:
                    cols_to_remove.add(col_a)
                    removed_pairs.append((col_b, col_a, float(corr_val)))
            elif use_smart_removal and target_corrs:
                corr_a = target_corrs.get(col_a, 0.0)
                corr_b = target_corrs.get(col_b, 0.0)

                if corr_a >= corr_b:
                    if col_b not in cols_to_remove:
                        cols_to_remove.add(col_b)
                        removed_pairs.append((col_a, col_b, float(corr_val)))
                else:
                    if col_a not in cols_to_remove:
                        cols_to_remove.add(col_a)
                        removed_pairs.append((col_b, col_a, float(corr_val)))
            else:
                if col_b not in cols_to_remove:
                    cols_to_remove.add(col_b)
                    removed_pairs.append((col_a, col_b, float(corr_val)))

        if skipped_count > 0:
            log.info(f"   🤝 Skipped {skipped_count} exempt pairs (natural correlations)")

        log.info(f"   ✅ Correlation analysis complete: {len(cols_to_remove)} features to remove")

    except Exception as e:
        log.error(f"❌ remove_correlated: Correlation calculation failed: {str(e)}")
        return X, []

    try:
        if cols_to_remove:
            cols_to_keep = [c for c in X_work.columns if c not in cols_to_remove]
            X_cleaned = X_work[cols_to_keep].copy()

            log.info(f"   🗑️ Removed {len(cols_to_remove)} redundant features:")

            sample_pairs = removed_pairs[:5]
            for i, (kept, removed, corr) in enumerate(sample_pairs):
                log.info(f"      {i+1}. KEPT: '{kept}' | REMOVED: '{removed}' | corr={corr:.6f}")
            if len(removed_pairs) > 5:
                log.info(f"      ... and {len(removed_pairs) - 5} more pairs")

            log.info(f"   📊 Shape: {X_work.shape[1]} → {X_cleaned.shape[1]} columns")
        else:
            X_cleaned = X_work
            log.info(f"   ✅ No correlated features found (threshold={correlation_threshold})")

        return X_cleaned, removed_pairs

    except Exception as e:
        log.error(f"❌ remove_correlated: Final cleanup failed: {str(e)}")
        return X, []

def remove_constant_and_low_variance_features(
    X: pd.DataFrame,
    variance_threshold: float = 1e-8,
    protected_columns: Optional[List[str]] = None,
    logger_instance=None,
    fit_mode: bool = True,
    ref_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    حذف فیچرهای ثابت/کم‌واریانس با قرارداد fit/transform.

    ✅ v2.0 FIX CRITICAL (2026-10):
        نسخهٔ قبلی فقط واریانس «مطلق» را با variance_threshold مقایسه
        می‌کرد. برای فیچرهایی با مقیاس کوچک (مانند funding_rate ~ 1e-4
        یا ma_convergence ~ 1e-3) این باعث حذف کاذب می‌شد، چون
        واریانس مطلق < 1e-3 است در حالی که اطلاعات کامل دارند
        (CV = std/mean ≈ 1.0).

        نسخهٔ جدید سه شرط را ترکیب می‌کند و فقط اگر همه‌شان
        نقض شوند فیچر را حذف می‌کند:

            1) unique_ratio = n_unique / n_samples < 1e-6
               → تقریباً ثابت (near-constant)

            2) abs_var < 1e-14
               → صفرِ مطلقِ واقعی

            3) rel_var = abs_var / (mean_abs² + 1e-12) < variance_threshold
               → کم‌واریانس نسبت به مقیاس طبیعی فیچر

        نتیجه:
            - funding_rate_median (mean=1e-4, var=1e-8, rel_var=1.0) → KEPT
            - ma_convergence (mean=1e-3, var=1e-6, rel_var=1.0) → KEPT
            - price_gap_ratio (mean=0, var=1e-20) → REMOVED (abs_var tiny)
            - true constant → REMOVED (unique_ratio = 0)

    Args:
        X: DataFrame فیچرها.
        variance_threshold: آستانهٔ واریانس «نسبی» (نسبت به mean²).
            مقادیر توصیه‌شده: 1e-6 (default منطقی) تا 1e-3.
        protected_columns: ستون‌هایی که هرگز حذف نمی‌شوند.
        fit_mode: آیا decision از همین X گرفته شود؟
        ref_columns: فقط در fit_mode=False — لیست ستون‌های مجاز TRAIN.

    Returns:
        (X_cleaned, removed_cols)
    """
    log = logger_instance or logger

    if X is None or X.empty:
        log.warning("remove_constant: X empty/None")
        return X, []

    if protected_columns is None:
        protected_columns = ["open", "high", "low", "close", "volume", "ret"]

    protected_columns = list(set(protected_columns) | set(PROJECT_TARGETS))

    if X.columns.duplicated().any():
        dup = X.columns[X.columns.duplicated()].tolist()
        raise ValueError(f"duplicate columns not allowed: {dup}")

    if not fit_mode:
        if ref_columns is None:
            raise ValueError(
                "fit_mode=False requires ref_columns (list of TRAIN's final "
                "columns). This prevents accidental leakage from VAL/TEST."
            )
        ref_set = set(str(c) for c in ref_columns)
        protected_present = set(c for c in protected_columns if c in X.columns)

        to_remove = [
            c for c in X.columns
            if c not in ref_set and c not in protected_present
        ]
        if not to_remove:
            log.info("   ✅ transform: nothing to remove")
            return X.copy(), []

        X_cleaned = X.drop(columns=to_remove, errors="raise")
        log.info(
            f"   🗑️ transform alignment removed {len(to_remove)} features"
        )
        return X_cleaned, list(to_remove)

    # ─────────────── FIT PATH ───────────────
    X_work = X.copy()
    protected_present = [c for c in protected_columns if c in X_work.columns]
    feature_cols = [c for c in X_work.columns if c not in protected_present]

    if not feature_cols:
        return X, []

    n_samples_total = len(X_work)
    removed: List[str] = []
    removed_reasons: Dict[str, str] = {}

    # آستانه‌های ثابت داخلی (مستقل از variance_threshold ورودی)
    ABS_VAR_MIN = 1e-14           # واریانس مطلق کوچک‌تر از این = مرده
    UNIQUE_RATIO_MIN = 1e-6        # کمتر از این نسبت = تقریباً ثابت

    for col in feature_cols:
        try:
            clean = pd.to_numeric(
                X_work[col], errors="coerce"
            ).dropna().to_numpy(dtype=np.float64)

            if clean.size == 0:
                removed.append(col)
                removed_reasons[col] = "all_nan"
                continue

            # ─── بررسی 1: unique ratio ───
            n_unique = int(np.unique(clean).size)
            unique_ratio = n_unique / float(clean.size)

            if n_unique <= 1:
                removed.append(col)
                removed_reasons[col] = "constant_nunique_1"
                continue

            if unique_ratio < UNIQUE_RATIO_MIN:
                removed.append(col)
                removed_reasons[col] = (
                    f"near_constant (unique_ratio={unique_ratio:.2e})"
                )
                continue

            # ─── بررسی 2: واریانس مطلق ───
            var_abs = float(np.var(clean))
            if not np.isfinite(var_abs):
                removed.append(col)
                removed_reasons[col] = "non_finite_variance"
                continue

            if var_abs < ABS_VAR_MIN:
                removed.append(col)
                removed_reasons[col] = (
                    f"abs_variance_zero (var={var_abs:.2e})"
                )
                continue

            # ─── بررسی 3: واریانس نسبی (نسبت به mean²) ───
            mean_val = float(np.mean(clean))
            mean_abs = max(abs(mean_val), 1e-12)
            rel_var = var_abs / (mean_abs ** 2)

            if rel_var < variance_threshold:
                removed.append(col)
                removed_reasons[col] = (
                    f"low_relative_variance "
                    f"(rel_var={rel_var:.2e}, mean={mean_val:.2e}, "
                    f"var={var_abs:.2e})"
                )
                continue

        except Exception as e:
            log.debug(f"   ⚠️ variance check failed for {col}: {e}")
            # در خطا: ستون را نگه‌دار (fail-safe)
            continue

    if removed:
        X_cleaned = X_work.drop(columns=removed, errors="ignore")
        log.info(
            f"   🗑️ fit: removed {len(removed)} low-variance/all-NaN "
            f"features (via relative-variance logic)"
        )
        # نمایش چند نمونه برای debug
        for col in removed[:5]:
            reason = removed_reasons.get(col, "unknown")
            log.info(f"      • '{col}': {reason}")
        return X_cleaned, removed

    log.info("   ✅ fit: no low-variance features")
    return X_work, []
    
def smart_feature_selection(
    df: pd.DataFrame,
    target_columns: Optional[List[str]] = None,
    correlation_threshold: float = 0.98,
    max_features: int = 100,
    logger_instance=None,
    fit_mode: bool = True,
    ref_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    انتخاب هوشمند فیچر با قرارداد fit/transform.

    fit_mode=True:
        - از df محاسبه می‌کند و لیست ستون‌های نهایی را برمی‌گرداند.

    fit_mode=False:
        - بدون هیچ محاسبه‌ای، فقط به ref_columns فیلتر می‌کند.
        - ref_columns از fit اصلی (TRAIN) گرفته می‌شود.

    Returns:
        (df_result, selected_feature_cols)
    """
    log = logger_instance or logger

    if target_columns is None:
        target_columns = list(PROJECT_TARGETS)

    if df is None or df.empty:
        return df, []

    if df.columns.duplicated().any():
        raise ValueError("duplicate columns not allowed")

    present_targets = [t for t in target_columns if t in df.columns]

    # ── TRANSFORM PATH ──
    if not fit_mode:
        if ref_columns is None:
            raise ValueError(
                "fit_mode=False requires ref_columns (TRAIN's final feature "
                "list). Refusing to re-select on VAL/TEST to prevent leakage."
            )
        ref_set = set(str(c) for c in ref_columns)
        selected = [
            c for c in df.columns
            if c in ref_set or c in present_targets
        ]
        missing = [c for c in ref_set if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"transform missing TRAIN-frozen features: {missing[:10]}"
            )
        return df[selected].copy(), list(ref_columns)

    # ── FIT PATH ──
    feature_cols = [c for c in df.columns if c not in present_targets]
    if len(feature_cols) <= max_features:
        log.info(
            f"📊 features={len(feature_cols)} within max={max_features} — "
            f"no selection needed"
        )
        return df.copy(), list(feature_cols)

    # هدف اصلی
    primary_target = "price_return" if "price_return" in present_targets else (
        present_targets[0] if present_targets else None
    )

    target_corrs: Dict[str, float] = {}
    if primary_target is not None:
        for col in feature_cols:
            try:
                pair = df[[col, primary_target]].dropna()
                if len(pair) > 10:
                    c = pair[col].corr(pair[primary_target])
                    target_corrs[col] = float(abs(c)) if np.isfinite(c) else 0.0
                else:
                    target_corrs[col] = 0.0
            except Exception:
                target_corrs[col] = 0.0
    else:
        for col in feature_cols:
            try:
                target_corrs[col] = float(df[col].var())
            except Exception:
                target_corrs[col] = 0.0

    # حذف جفت‌های همبستگی بالا
    if len(feature_cols) > 2:
        corr = df[feature_cols].corr().abs()
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        redundant: List[Tuple[str, str, float]] = []
        for c1 in upper.columns:
            for c2 in upper.index:
                v = upper.loc[c2, c1]
                if pd.notna(v) and v > correlation_threshold:
                    redundant.append((c1, c2, float(v)))

        removed_set: Set[str] = set()
        for c1, c2, _v in redundant:
            if c1 in removed_set or c2 in removed_set:
                continue
            # آن که همبستگی کمتری با هدف دارد حذف شود
            if target_corrs.get(c1, 0.0) < target_corrs.get(c2, 0.0):
                removed_set.add(c1)
            else:
                removed_set.add(c2)

        if removed_set:
            feature_cols = [c for c in feature_cols if c not in removed_set]
            log.info(
                f"   🔄 removed {len(removed_set)} redundant features "
                f"(corr > {correlation_threshold})"
            )

    # محدودیت تعداد
    if len(feature_cols) > max_features:
        ranked = sorted(
            feature_cols,
            key=lambda x: target_corrs.get(x, 0.0),
            reverse=True,
        )
        feature_cols = ranked[:max_features]
        log.info(
            f"   📊 kept top {max_features} features by target correlation"
        )

    final_cols = feature_cols + present_targets
    df_result = df[final_cols].copy()
    log.info(
        f"✅ selection complete: {len(feature_cols)} features + "
        f"{len(present_targets)} targets"
    )
    return df_result, list(feature_cols)
    
def _canonical_column_name(name: Any) -> str:
    """
    تبدیل نام ستون به فرم canonical برای مقایسهٔ ایمن.

    ویژگی‌ها:
        - حذف فاصله‌های ابتدا و انتها
        - تبدیل به حروف کوچک
        - جایگزینی فاصله و خط تیره با underscore
        - حفظ نام اصلی در DataFrame انجام نمی‌شود؛ فقط برای مقایسه است

    Args:
        name: نام ستون

    Returns:
        نام canonical به‌صورت str

    Raises:
        TypeError: اگر نام ستون قابل تبدیل به رشته نباشد.
    """
    if name is None:
        raise TypeError("Column name cannot be None")

    value = str(name).strip().lower()
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value

def _canonical_target_column_map(
    columns: List[Any],
    target_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    نگاشت نام canonical تارگت به نام واقعی ستون.

    این تابع در برابر تفاوت حروف بزرگ و کوچک مقاوم است و در صورت وجود
    چند ستون که پس از canonicalization یکسان شوند، با خطای صریح متوقف می‌شود.

    Args:
        columns: فهرست نام ستون‌های موجود
        target_names: نام تارگت‌های مورد انتظار

    Returns:
        دیکشنری:
            canonical_target_name -> actual_column_name

    Raises:
        ValueError: در صورت وجود نام‌های تکراری یا ابهام در Target.
    """
    expected_targets = list(target_names or PROJECT_TARGETS)

    normalized_columns: Dict[str, str] = {}

    for column in columns:
        actual_name = str(column)
        canonical_name = _canonical_column_name(actual_name)

        if canonical_name in normalized_columns:
            previous = normalized_columns[canonical_name]
            raise ValueError(
                "Ambiguous duplicate columns after canonicalization: "
                f"{previous!r} and {actual_name!r} both map to {canonical_name!r}"
            )

        normalized_columns[canonical_name] = actual_name

    result: Dict[str, str] = {}

    for target_name in expected_targets:
        canonical_target = _canonical_column_name(target_name)
        actual_column = normalized_columns.get(canonical_target)

        if actual_column is not None:
            result[canonical_target] = actual_column

    return result

def _validate_numeric_finite_frame(
    frame: pd.DataFrame,
    frame_name: str,
    allow_nan: bool = False,
) -> None:
    """
    اعتبارسنجی سخت‌گیرانهٔ DataFrame عددی.

    Args:
        frame: DataFrame مورد بررسی
        frame_name: نام DataFrame برای پیام خطا
        allow_nan: آیا NaN مجاز است؟

    Raises:
        TypeError: اگر ورودی DataFrame نباشد.
        ValueError: برای DataFrame خالی، ستون غیرعددی، NaN یا Inf.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"{frame_name} must be pandas.DataFrame, got {type(frame).__name__}"
        )

    if frame.empty:
        raise ValueError(f"{frame_name} is empty")

    duplicated_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicated_columns:
        raise ValueError(
            f"{frame_name} contains duplicate columns: {duplicated_columns}"
        )

    non_numeric_columns = [
        column
        for column in frame.columns
        if not is_numeric_dtype(frame[column])
    ]

    if non_numeric_columns:
        raise TypeError(
            f"{frame_name} contains non-numeric columns: {non_numeric_columns}"
        )

    values = frame.to_numpy(dtype=np.float64, copy=False)

    if not allow_nan and np.isnan(values).any():
        nan_columns = frame.columns[
            frame.isna().any(axis=0)
        ].tolist()
        raise ValueError(
            f"{frame_name} contains NaN values in columns: {nan_columns}"
        )

    if np.isinf(values).any():
        inf_columns = frame.columns[
            np.isinf(frame.to_numpy(dtype=np.float64, copy=False)).any(axis=0)
        ].tolist()
        raise ValueError(
            f"{frame_name} contains infinite values in columns: {inf_columns}"
        )

def validate_strict_timestamp_series(
    timestamps: Union[pd.Series, np.ndarray, List[Any]],
    name: str = "timestamps",
    require_unique: bool = True,
) -> pd.DatetimeIndex:
    """
    تبدیل و اعتبارسنجی سخت‌گیرانهٔ Timestampها.

    الزامات:
        - تمام مقادیر قابل تبدیل به datetime باشند.
        - مقدار NaT وجود نداشته باشد.
        - ترتیب کاملاً صعودی باشد.
        - در صورت require_unique=True هیچ مقدار تکراری وجود نداشته باشد.

    Args:
        timestamps: آرایه یا Series زمانی
        name: نام برای پیام خطا
        require_unique: الزام یکتا بودن زمان‌ها

    Returns:
        pandas.DatetimeIndex با timezone UTC

    Raises:
        ValueError: در صورت وجود NaT، ترتیب نادرست یا timestamp تکراری.
    """
    if timestamps is None:
        raise ValueError(f"{name} cannot be None")

    if isinstance(timestamps, pd.DatetimeIndex):
        raw = timestamps
    else:
        raw = pd.DatetimeIndex(pd.to_datetime(timestamps, errors="coerce"))

    if len(raw) == 0:
        raise ValueError(f"{name} is empty")

    if raw.isna().any():
        invalid_positions = np.flatnonzero(raw.isna())[:10].tolist()
        raise ValueError(
            f"{name} contains invalid datetime values at positions "
            f"{invalid_positions}"
        )

    if raw.tz is None:
        normalized = raw.tz_localize("UTC")
    else:
        normalized = raw.tz_convert("UTC")

    if require_unique and not normalized.is_unique:
        duplicated_values = normalized[normalized.duplicated()].unique()
        raise ValueError(
            f"{name} contains duplicate timestamps: "
            f"{duplicated_values[:10].tolist()}"
        )

    if len(normalized) > 1:
        values = normalized.asi8
        if np.any(values[1:] <= values[:-1]):
            raise ValueError(
                f"{name} must be strictly increasing. "
                "Sort or split the raw data before sequence creation."
            )

    return normalized
    
def final_feature_sanitization(
    df: pd.DataFrame,
    symbol: str,
    variance_threshold: float = 1e-8,
    correlation_threshold: float = 0.999,
    protected_columns: Optional[List[str]] = None,
    logger_instance=None,
    target_columns: Optional[List[str]] = None,
    allow_target_nan: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    پاک‌سازی نهایی بدون ساخت Label جعلی.

    ✅ Featureهای دارای NaN/Inf → hard failure (strict=True)
    ✅ Targetها هرگز با 0 جایگزین نمی‌شوند
    ✅ NaN انتهای Targetهای آینده (horizon) حفظ می‌شود
    ✅ حذف featureهای ثابت/کم‌واریانس (فقط feature)
    ✅ همبستگی فقط در حد داکیومنت؛ این تابع correlation انجام نمی‌دهد
    """
    log = logger_instance or logging.getLogger(__name__)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[{symbol}] df must be pandas.DataFrame, got {type(df).__name__}"
        )
    if df.empty:
        raise ValueError(f"[{symbol}] DataFrame is empty")
    if not np.isfinite(float(variance_threshold)) or variance_threshold < 0:
        raise ValueError(f"[{symbol}] variance_threshold must be finite and >= 0")
    if not np.isfinite(float(correlation_threshold)):
        raise ValueError(f"[{symbol}] correlation_threshold must be finite")
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"[{symbol}] duplicate columns are not allowed: {duplicated}")

    expected_targets = list(target_columns or PROJECT_TARGETS)
    target_map = _canonical_target_column_map(
        columns=list(df.columns),
        target_names=expected_targets,
    )
    actual_target_columns = set(target_map.values())

    protected = set(str(c) for c in (protected_columns or []))
    protected.update(actual_target_columns)

    result = df.copy(deep=True)

    # Non-numeric feature columns -> hard fail
    non_numeric_non_target = [
        c for c in result.columns
        if c not in actual_target_columns and not is_numeric_dtype(result[c])
    ]
    if non_numeric_non_target:
        raise TypeError(
            f"[{symbol}] Non-numeric feature columns are not allowed: "
            f"{non_numeric_non_target}"
        )

    feature_columns = [c for c in result.columns if c not in actual_target_columns]

    # ────────────────────────────────────────────────────────
    # Feature finite check (STRICT)
    # ────────────────────────────────────────────────────────
    if feature_columns:
        feature_values = result[feature_columns].to_numpy(dtype=np.float64, copy=False)

        inf_mask = np.isinf(feature_values)
        if inf_mask.any():
            bad = [
                feature_columns[i]
                for i in np.flatnonzero(inf_mask.any(axis=0))
            ]
            raise ValueError(f"[{symbol}] Infinite values in features: {bad}")

        nan_mask = np.isnan(feature_values)
        if nan_mask.any():
            bad = [
                feature_columns[i]
                for i in np.flatnonzero(nan_mask.any(axis=0))
            ]
            if strict:
                raise ValueError(
                    f"[{symbol}] NaN values found in features: {bad}. "
                    "Fit imputation statistics on TRAIN and transform with "
                    "those frozen statistics before final sanitization."
                )

    # ────────────────────────────────────────────────────────
    # Target checks
    # ────────────────────────────────────────────────────────
    for target_column in actual_target_columns:
        target_series = result[target_column]
        if not is_numeric_dtype(target_series):
            raise TypeError(f"[{symbol}] Target {target_column!r} must be numeric")
        target_values = target_series.to_numpy(dtype=np.float64, copy=False)
        if np.isinf(target_values).any():
            raise ValueError(
                f"[{symbol}] Infinite values found in target {target_column!r}"
            )

        canonical = _canonical_column_name(target_column)
        is_regression = canonical in {
            _canonical_column_name(n) for n in REGRESSION_TARGETS
        }

        if is_regression:
            if not allow_target_nan and np.isnan(target_values).any():
                raise ValueError(
                    f"[{symbol}] NaN in regression target {target_column!r}"
                )
        else:
            if np.isnan(target_values).any() and strict:
                raise ValueError(
                    f"[{symbol}] NaN in classification target {target_column!r}"
                )
            finite_cls = target_values[~np.isnan(target_values)]
            if finite_cls.size > 0:
                if not np.all(np.equal(finite_cls, np.floor(finite_cls))):
                    raise ValueError(
                        f"[{symbol}] Classification target {target_column!r} "
                        "contains non-integer values"
                    )

    # ────────────────────────────────────────────────────────
    # Remove constant / low-variance FEATURES
    # ────────────────────────────────────────────────────────
    removable_columns: List[str] = []
    for column in feature_columns:
        if column in protected:
            continue
        series = result[column]
        finite_values = series.to_numpy(dtype=np.float64, copy=False)
        if finite_values.size == 0:
            removable_columns.append(column)
            continue
        if np.unique(finite_values).size <= 1:
            removable_columns.append(column)
            continue
        variance = float(np.var(finite_values))
        if not np.isfinite(variance):
            raise ValueError(f"[{symbol}] Non-finite variance for feature {column!r}")
        if variance <= variance_threshold:
            removable_columns.append(column)

    if removable_columns:
        result = result.drop(columns=removable_columns, errors="raise")
        log.info(
            "[%s] Removed %d constant/low-variance features",
            symbol, len(removable_columns),
        )

    # ────────────────────────────────────────────────────────
    # Final finite check
    # ────────────────────────────────────────────────────────
    remaining_features = [
        c for c in result.columns if c not in actual_target_columns
    ]
    if remaining_features:
        remaining_values = result[remaining_features].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(remaining_values).all():
            raise ValueError(f"[{symbol}] Final feature matrix contains NaN or Inf")

    log.info(
        "[%s] Final sanitization completed: shape=%s, features=%d, targets=%d",
        symbol,
        result.shape,
        len(remaining_features),
        len(actual_target_columns.intersection(result.columns)),
    )

    return result
    
def validating_leakage_all_target(
    y_dict: Union[Dict[str, np.ndarray], pd.DataFrame],
    data_type: str = "processed",
    split_type: str = "train",
    tolerance: float = 0.05,
    correlation_threshold: float = 0.999,
    max_samples_for_test: int = 10000,
    exempt_pairs: Optional[List[Tuple[str, str]]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    VALIDATING_LEAKAGE_ALL_TARGET v2.0 — 2-CLASS AWARE.

    🔥 v2.0 (نسبت به v1.0):
        • نگاشت قیمت direction از ۵کلاس و ۳کلاس قدیمی به ۲کلاسه.
        • در صورت دیدن مقادیر خارج از [0, 2) → merge خودکار.
    """
    if exempt_pairs is None:
        exempt_pairs = [
            ("price_return", "price_direction"),
            ("price_direction", "price_return"),
            ("trend_strength", "market_regime"),
            ("market_regime", "trend_strength"),
            ("volatility_regime", "market_regime"),
            ("market_regime", "volatility_regime"),
            ("price_direction", "market_regime"),
            ("market_regime", "price_direction"),
        ]

    TARGET_ALIAS_MAP = {
        "price_return": [
            "price_return", "price_reg_output", "price", "regression_target",
        ],
        "price_direction": [
            "price_direction", "price_cls_output", "price_output",
            "direction_target",
        ],
        "trend_strength": ["trend_strength", "trend", "adx_target"],
        "volatility_regime": [
            "volatility_regime", "volatility_output", "volatility",
            "vol_target",
        ],
        "market_regime": ["market_regime", "market", "regime_target"],
    }

    VALID_DATA_TYPES = ("processed", "normalized", "sequenced")
    VALID_SPLIT_TYPES = ("train", "val", "test")

    if data_type not in VALID_DATA_TYPES:
        raise ValueError(f"data_type invalid: {data_type}")
    if split_type not in VALID_SPLIT_TYPES:
        raise ValueError(f"split_type invalid: {split_type}")

    logger.info("=" * 80)
    logger.info("🔍 WALL STREET LEAKAGE VALIDATION (2-class)")
    logger.info(
        "   data_type=%s, split_type=%s, tol=%.3f, corr_thr=%.3f",
        data_type, split_type, tolerance, correlation_threshold,
    )
    logger.info("=" * 80)

    if isinstance(y_dict, pd.DataFrame):
        y_dict = {col: y_dict[col].values for col in y_dict.columns}

    available_keys = list(y_dict.keys())
    logger.info("   📋 Available keys: %s", available_keys)

    standard_key_map = {}
    for std_key, aliases in TARGET_ALIAS_MAP.items():
        for alias in aliases:
            if alias in y_dict:
                standard_key_map[std_key] = alias
                break

    standardized_targets = {}
    for std_key, actual_key in standard_key_map.items():
        data = y_dict[actual_key]
        if len(data.shape) == 2 and data.shape[1] > 1:
            data = np.argmax(data, axis=1)

        if std_key == "price_direction":
            arr = np.asarray(data).flatten()
            finite_vals = arr[np.isfinite(arr) & (arr != INVALID_LABEL)]
            if finite_vals.size > 0:
                max_val = int(finite_vals.max())
                if max_val == 4:
                    logger.warning(
                        "price_direction detected as 5-class (max=%d); "
                        "merging to 2-class", max_val,
                    )
                    data = PriceDirectionNames.merge_5class_to_2class(arr)
                elif max_val == 2:
                    logger.warning(
                        "price_direction detected as 3-class (max=%d); "
                        "merging to 2-class (NEUTRAL → INVALID)",
                        max_val,
                    )
                    data = PriceDirectionNames.merge_3class_to_2class(arr)

        standardized_targets[std_key] = data
        logger.info("   ✅ Mapped '%s' → '%s'", actual_key, std_key)

    targets_to_use = standardized_targets if standardized_targets else y_dict
    available_targets = list(targets_to_use.keys())

    if len(available_targets) < 2:
        logger.warning(
            "⚠️ at least 2 targets required, got %d", len(available_targets)
        )
        return {
            "data_type": data_type, "split_type": split_type,
            "total_pairs_tested": 0, "exempt_pairs": 0,
            "pairs_with_leakage": 0, "clean_pairs": 0,
            "has_any_leakage": False, "pair_results": {},
            "tolerance": tolerance,
            "correlation_threshold": correlation_threshold,
            "exempt_pairs_list": exempt_pairs, "skipped_pairs": [],
        }

    detector = LabelLeakageDetector(
        lookback=20, tolerance=tolerance,
        correlation_threshold=correlation_threshold,
    )

    all_pair_results: Dict[str, Any] = {}
    skipped_pairs: List[str] = []
    pair_index = 0

    for i, target_a in enumerate(available_targets):
        for j, target_b in enumerate(available_targets):
            if i >= j:
                continue
            pair_index += 1
            pair_name = f"pair_{pair_index}_{target_a}_vs_{target_b}"

            is_exempt = any(
                (target_a == ea and target_b == eb)
                or (target_a == eb and target_b == ea)
                for ea, eb in exempt_pairs
            )
            if is_exempt:
                skipped_pairs.append(pair_name)
                all_pair_results[pair_name] = {
                    "label_type_name": pair_name,
                    "has_leakage": False,
                    "severity": "none",
                    "status": "EXEMPT",
                    "agreement_pct": 100.0,
                    "tolerance": tolerance,
                    "n_samples_compared": 0,
                    "exempt": True,
                }
                continue

            arr_a = np.asarray(targets_to_use[target_a], dtype=np.float64).flatten()
            arr_b = np.asarray(targets_to_use[target_b], dtype=np.float64).flatten()
            min_len = min(len(arr_a), len(arr_b))
            arr_a, arr_b = arr_a[:min_len], arr_b[:min_len]

            if min_len > max_samples_for_test:
                np.random.seed(42)
                idx = np.sort(
                    np.random.choice(min_len, max_samples_for_test, replace=False)
                )
                arr_a_s, arr_b_s = arr_a[idx], arr_b[idx]
            else:
                arr_a_s, arr_b_s = arr_a, arr_b

            valid = np.isfinite(arr_a_s) & np.isfinite(arr_b_s)
            a_clean, b_clean = arr_a_s[valid], arr_b_s[valid]

            if len(a_clean) < 30:
                continue

            ua = len(np.unique(a_clean))
            ub = len(np.unique(b_clean))

            if ua <= 10 and ub <= 10:
                result = detector.detect(
                    a_clean.astype(int), b_clean.astype(int),
                    label_type_name=pair_name,
                    label_type=LabelType.CLASSIFICATION,
                )
            else:
                result = detector.detect(
                    a_clean, b_clean,
                    label_type_name=pair_name,
                    label_type=LabelType.REGRESSION,
                )

            all_pair_results[pair_name] = result.to_dict()

    total_pairs = len(all_pair_results)
    leaky = sum(
        1 for r in all_pair_results.values()
        if r.get("has_leakage", False) and not r.get("exempt", False)
    )
    exempt_cnt = sum(
        1 for r in all_pair_results.values() if r.get("exempt", False)
    )
    safe = total_pairs - leaky - exempt_cnt

    logger.info("=" * 80)
    logger.info(
        "📋 LEAKAGE SUMMARY | total=%d exempt=%d leaky=%d safe=%d",
        total_pairs, exempt_cnt, leaky, safe,
    )
    logger.info("=" * 80)

    return {
        "data_type": data_type, "split_type": split_type,
        "total_pairs_tested": total_pairs,
        "exempt_pairs": exempt_cnt,
        "pairs_with_leakage": leaky,
        "clean_pairs": safe,
        "has_any_leakage": leaky > 0,
        "pair_results": all_pair_results,
        "tolerance": tolerance,
        "correlation_threshold": correlation_threshold,
        "exempt_pairs_list": exempt_pairs,
        "skipped_pairs": skipped_pairs,
    }
    
def assert_no_leakage(splits: dict):
    """
    🛡️ بررسی leakage بین train/val/test — نسخهٔ اصلاح‌شدهٔ زمان‌محور

    - row hash based (sha256 full-row)
    - index based (با تشخیص false positive ناشی از index reset)
    - timestamp based اگر ستون زمان موجود باشد؛ مقادیر قبلاً «خام» مقایسه
      می‌شدند و اگر قالب دو طرف متفاوت بود (string vs datetime vs epoch int)
      همپوشانی واقعی از چشم پنهان می‌ماند (false negative). در این نسخه مقادیر
      ابتدا به int64 نانوثانیهٔ UTC نرمال می‌شوند و سپس مقایسه انجام می‌گیرد.

    رفتار در برابر نامتعارف:
        - بخش index: همان منطق قبلی (تشخیص reset-index و نادیده گرفتن آن).
        - بخش row-hash: همان منطق قبلی (خطای بحرانی + raise).
        - بخش timestamp: مثل قبل در صورت همپوشانی raise می‌کند؛ اما حالا با
          نرمال‌سازی، همپوشانی‌های واقعی که قبلاً از دست می‌رفتند نیز گرفته می‌شوند.

    Raises:
        RuntimeError: در صورت تشخیص leakage واقعی.
    """
    logger.info("🛡️ Starting smart leakage detection ...")

    if not isinstance(splits, dict) or not splits:
        raise ValueError("assert_no_leakage: 'splits' must be a non-empty dict of DataFrames")

    for key, frame in splits.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(
                f"assert_no_leakage: split {key!r} must be pandas.DataFrame, "
                f"got {type(frame).__name__}"
            )

    # چاپ جزئیات split
    for k, df in splits.items():
        logger.info(
            f"📦 Split[{k}] | rows={len(df)} | cols={len(df.columns)} | index_range=({df.index.min()} → {df.index.max()})"
        )

    # ─────────────────────────────────────────────────────────────────────
    # 1) بررسی row-hash leakage
    # ─────────────────────────────────────────────────────────────────────
    hashes = {}
    for k, df in splits.items():
        hashes[k] = set(df.apply(hash_row, axis=1))

    split_keys = list(splits.keys())
    n = len(split_keys)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = split_keys[i], split_keys[j]
            overlap = hashes[a] & hashes[b]
            if overlap:
                sample_overlap = list(overlap)[:10]
                logger.critical(
                    f"❌ ROW LEAKAGE DETECTED\nPair: {a} ↔ {b}\nCount: {len(overlap)}\nSample: {sample_overlap}"
                )
                raise RuntimeError(f"❌ Leakage detected: row overlap between {a} and {b}")
            else:
                logger.info(f"✅ Row overlap: none ({a} ↔ {b})")

    # ─────────────────────────────────────────────────────────────────────
    # 2) بررسی index overlap (با تشخیص reset-index false positive)
    # ─────────────────────────────────────────────────────────────────────
    for i in range(n):
        for j in range(i + 1, n):
            a, b = split_keys[i], split_keys[j]
            index_overlap = set(splits[a].index) & set(splits[b].index)
            if index_overlap:
                reset_a = list(splits[a].index) == list(range(len(splits[a])))
                reset_b = list(splits[b].index) == list(range(len(splits[b])))

                if reset_a and reset_b:
                    logger.warning(
                        f"⚠️ Index overlap due to reset (false positive) | Pair: {a} ↔ {b} | Overlap count: {len(index_overlap)}"
                    )
                    continue
                else:
                    sample_index = list(index_overlap)[:10]
                    logger.critical(
                        f"❌ INDEX LEAKAGE DETECTED\nPair          : {a} ↔ {b}\n"
                        f"Overlap count : {len(index_overlap)}\nSample index  : {sample_index}\n"
                        f"{a} range     : {splits[a].index.min()} → {splits[a].index.max()}\n"
                        f"{b} range     : {splits[b].index.min()} → {splits[b].index.max()}"
                    )
                    raise RuntimeError(f"❌ Leakage index between {a} and {b}")

    # ─────────────────────────────────────────────────────────────────────
    # 3) بررسی timestamp overlap (با نرمال‌سازی به int64 نانوثانیهٔ UTC)
    # ─────────────────────────────────────────────────────────────────────
    def _normalize_time_to_int64_ns(col: pd.Series) -> Set[int]:
        """تبدیل هر قالبی به مجموعهٔ int64 نانوثانیهٔ UTC؛ NaT حذف می‌شود."""
        try:
            if is_datetime64_any_dtype(col):
                parsed = pd.to_datetime(col, utc=True, errors="coerce")
            elif is_numeric_dtype(col):
                numeric = pd.to_numeric(col, errors="coerce")
                med = numeric.abs().median()
                if not np.isfinite(med):
                    return set()
                if med >= 1e17:
                    unit = "ns"
                elif med >= 1e14:
                    unit = "us"
                elif med >= 1e11:
                    unit = "ms"
                else:
                    unit = "s"
                parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
            else:
                parsed = pd.to_datetime(col, utc=True, errors="coerce")

            parsed = parsed.dropna()
            if parsed.empty:
                return set()
            return {int(ts.value) for ts in parsed}
        except Exception:
            return set()

    for i in range(n):
        for j in range(i + 1, n):
            a, b = split_keys[i], split_keys[j]
            time_col_a = detect_time_column(splits[a])
            time_col_b = detect_time_column(splits[b])
            if time_col_a and time_col_b:
                times_a = _normalize_time_to_int64_ns(splits[a][time_col_a])
                times_b = _normalize_time_to_int64_ns(splits[b][time_col_b])
                time_overlap = times_a & times_b
                if time_overlap:
                    sample_time = [
                        pd.Timestamp(v, unit="ns", tz="UTC").isoformat()
                        for v in sorted(time_overlap)[:5]
                    ]
                    logger.critical(
                        f"❌ TIME OVERLAP DETECTED\nPair: {a} ↔ {b}\nOverlap count: {len(time_overlap)}\nSample: {sample_time}"
                    )
                    raise RuntimeError(f"❌ Time overlap detected between {a} and {b}")
                else:
                    logger.info(f"✅ Time overlap: none ({a} ↔ {b})")

    logger.info("✅ Smart leakage detection passed for all splits.")

def required_purge_gap(label_horizon: int, embargo: int = 0) -> int:
    h, e = int(label_horizon), int(embargo)
    if h < 1 or e < 0:
        raise ValueError("invalid horizon/embargo")
    return h + e
    
def create_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.25,
    test_ratio: float = 0.15,
    gap_between_splits: Optional[int] = None,
    label_horizon: int = 1,
    embargo: int = 0,
    timestamp_column: Optional[str] = None,
    require_timestamp: bool = False,
    reset_index: bool = False,
    shuffle: bool = False,
) -> Dict[str, Any]:
    """
    ایجاد Split زمانی سخت‌گیرانه با Purge Gap وابسته به label_horizon.

    🔥 تغییر v2.0 (این نسخه):
        نسبت‌های پیش‌فرض از (0.70, 0.20, 0.10) به (0.60, 0.25, 0.15)
        تغییر کرد. دلیل: در اجرای قبلی تعداد نمونه‌های test به‌حدی کوچک
        بود (۳۴۷ نمونه) که نه Sharpe قابل استناد بود، نه win-rate، نه
        ارزیابی ۱۳گانهٔ Edge. با val=25% و test=15% روی مجموعه‌ای در
        حد ۲۳ هزار نمونه، test به ~۳٬۵۰۰ نمونه می‌رسد که به آستانهٔ
        آماری Edge نزدیک می‌شود (هدف نهایی: ≥ 5,000 نمونه out-of-sample).

    قرارداد:
        gap_between_splits >= label_horizon + embargo
        اگر None باشد، به‌طور خودکار برابر minimum_gap قرار می‌گیرد.

    رفتار در قبال Timestamp:
        - اگر ستون زمان یافت شود: یکتایی، ترتیب صعودی و عدم همپوشانی اعتبارسنجی می‌شود.
        - اگر یافت نشود + require_timestamp=True: ValueError.
        - اگر یافت نشود + require_timestamp=False: بررسی یکنوایی index + هشدار در نبود آن.

    ساختار خروجی:
        {
            "train": DataFrame,
            "val": DataFrame,
            "test": DataFrame,
            "metadata": {...}
        }

    Raises:
        TypeError: ورودی غیر DataFrame
        ValueError: نسبت‌های نامعتبر، shuffle=True، نبود timestamp اجباری، یا
                    gap کوچک‌تر از minimum_gap.
        RuntimeError: نقض invariant داخلی یا همپوشانی.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"create_split: df must be pandas.DataFrame, got {type(df).__name__}"
        )
    if df.empty:
        raise ValueError("create_split: Cannot split an empty DataFrame")
    if shuffle:
        raise ValueError(
            "create_split() does not support shuffle=True. "
            "This function enforces strict chronological splitting with purge gaps."
        )

    ratios = {
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
    }
    for name, value in ratios.items():
        if not np.isfinite(value):
            raise ValueError(f"create_split: {name} must be finite")
        if value <= 0.0:
            raise ValueError(f"create_split: {name} must be greater than zero")

    ratio_sum = sum(ratios.values())
    if not np.isclose(ratio_sum, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"create_split: Split ratios must sum to 1.0, got {ratio_sum:.12f}"
        )

    try:
        h_int = int(label_horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"create_split: label_horizon must be int, got {label_horizon!r}"
        ) from exc

    try:
        e_int = int(embargo)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"create_split: embargo must be int, got {embargo!r}"
        ) from exc

    if h_int < 1:
        raise ValueError(f"create_split: label_horizon must be >=1, got {h_int}")
    if e_int < 0:
        raise ValueError(f"create_split: embargo must be >=0, got {e_int}")

    minimum_gap = required_purge_gap(h_int, e_int)

    if gap_between_splits is None:
        gap = int(minimum_gap)
        logger.info(
            "create_split: gap_between_splits auto-set to %d (= horizon=%d + embargo=%d)",
            gap, h_int, e_int,
        )
    else:
        try:
            gap = int(gap_between_splits)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"create_split: gap_between_splits must be int or None, "
                f"got {gap_between_splits!r}"
            ) from exc

        if gap < minimum_gap:
            raise ValueError(
                f"create_split: gap={gap} smaller than required minimum "
                f"{minimum_gap} (= label_horizon {h_int} + embargo {e_int}). "
                "Increase gap_between_splits or set it to None for auto."
            )

    if gap < 0:
        raise ValueError(f"create_split: gap must be >=0, got {gap}")

    column_map = {
        _canonical_column_name(column): str(column)
        for column in df.columns
    }

    selected_timestamp_column: Optional[str] = None

    if timestamp_column is not None:
        canonical_ts = _canonical_column_name(timestamp_column)
        selected_timestamp_column = column_map.get(canonical_ts)
        if selected_timestamp_column is None:
            raise ValueError(
                f"create_split: Timestamp column {timestamp_column!r} not found. "
                f"Available: {list(df.columns)}"
            )
    else:
        for candidate in (
            "timestamp", "datetime", "date", "time", "open_time", "close_time",
        ):
            if candidate in column_map:
                selected_timestamp_column = column_map[candidate]
                break

    if selected_timestamp_column is None:
        if require_timestamp:
            raise ValueError(
                "create_split: A timestamp column is required "
                "(require_timestamp=True) but none was found."
            )
        try:
            index_is_monotonic = bool(df.index.is_monotonic_increasing)
        except Exception:
            index_is_monotonic = False
        if index_is_monotonic:
            logger.debug(
                "create_split: no timestamp; using monotonic index as "
                "implicit order (rows=%d).", len(df),
            )
        else:
            logger.warning(
                "create_split: no timestamp AND index not monotonic. "
                "Cannot verify chronological order; caller is responsible. "
                "rows=%d, index_type=%s", len(df), type(df.index).__name__,
            )

    normalized_timestamp_values: Optional[pd.DatetimeIndex] = None
    if selected_timestamp_column is not None:
        normalized_timestamp_values = validate_strict_timestamp_series(
            df[selected_timestamp_column].to_numpy(),
            name=selected_timestamp_column,
            require_unique=True,
        )

    n_rows = len(df)
    train_size = int(math.floor(n_rows * ratios["train_ratio"]))
    val_size = int(math.floor(n_rows * ratios["val_ratio"]))

    minimum_required = train_size + val_size + 2 * gap + 1
    if n_rows < minimum_required:
        raise ValueError(
            "create_split: Not enough rows for requested ratios and gaps: "
            f"rows={n_rows}, required_at_least={minimum_required}, "
            f"train={train_size}, val={val_size}, gap={gap}"
        )

    train_start = 0
    train_end = train_start + train_size
    val_start = train_end + gap
    val_end = val_start + val_size
    test_start = val_end + gap
    test_end = n_rows

    if not (
        0 <= train_start < train_end <= val_start
        <= val_end <= test_start < test_end <= n_rows
    ):
        raise RuntimeError(
            "create_split: internal boundary invariant failed: "
            f"train=({train_start},{train_end}) "
            f"val=({val_start},{val_end}) "
            f"test=({test_start},{test_end})"
        )

    train_df = df.iloc[train_start:train_end].copy()
    val_df = df.iloc[val_start:val_end].copy()
    test_df = df.iloc[test_start:test_end].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "create_split: All train/val/test splits must have at least one row"
        )

    train_indices = np.arange(train_start, train_end, dtype=np.int64)
    val_indices = np.arange(val_start, val_end, dtype=np.int64)
    test_indices = np.arange(test_start, test_end, dtype=np.int64)

    if (
        np.intersect1d(train_indices, val_indices).size > 0
        or np.intersect1d(train_indices, test_indices).size > 0
        or np.intersect1d(val_indices, test_indices).size > 0
    ):
        raise RuntimeError("create_split: Split index overlap detected")

    if selected_timestamp_column is not None and normalized_timestamp_values is not None:
        train_times = normalized_timestamp_values[train_indices]
        val_times = normalized_timestamp_values[val_indices]
        test_times = normalized_timestamp_values[test_indices]

        if train_times[-1] >= val_times[0]:
            raise RuntimeError(
                "create_split: Train and validation timestamps overlap"
            )
        if val_times[-1] >= test_times[0]:
            raise RuntimeError(
                "create_split: Validation and test timestamps overlap"
            )
        if (
            np.intersect1d(train_times.asi8, val_times.asi8).size > 0
            or np.intersect1d(train_times.asi8, test_times.asi8).size > 0
            or np.intersect1d(val_times.asi8, test_times.asi8).size > 0
        ):
            raise RuntimeError(
                "create_split: Timestamp overlap detected between chronological splits"
            )

        timestamp_metadata = {
            "column": selected_timestamp_column,
            "train_start": train_times[0].isoformat(),
            "train_end": train_times[-1].isoformat(),
            "val_start": val_times[0].isoformat(),
            "val_end": val_times[-1].isoformat(),
            "test_start": test_times[0].isoformat(),
            "test_end": test_times[-1].isoformat(),
        }
    else:
        timestamp_metadata = {
            "column": None,
            "train_start": None, "train_end": None,
            "val_start": None, "val_end": None,
            "test_start": None, "test_end": None,
        }

    if reset_index:
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    actual_train_ratio = float(len(train_df) / n_rows)
    actual_val_ratio = float(len(val_df) / n_rows)
    actual_test_ratio = float(len(test_df) / n_rows)

    metadata: Dict[str, Any] = {
        "pipeline_version": SEQUENCE_PIPELINE_VERSION,
        "n_rows_input": int(n_rows),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_ratio_requested": ratios["train_ratio"],
        "val_ratio_requested": ratios["val_ratio"],
        "test_ratio_requested": ratios["test_ratio"],
        "train_ratio_actual": actual_train_ratio,
        "val_ratio_actual": actual_val_ratio,
        "test_ratio_actual": actual_test_ratio,
        "gap_between_splits": int(gap),
        "label_horizon": int(h_int),
        "embargo": int(e_int),
        "minimum_gap_required": int(minimum_gap),
        "purged_rows": int(2 * gap),
        "original_indices": {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        },
        "timestamp": timestamp_metadata,
    }

    logger.info(
        "create_split: n=%d | train=%.1f%% (%d) | val=%.1f%% (%d) | "
        "test=%.1f%% (%d) | gap=%d | horizon=%d | embargo=%d",
        n_rows,
        actual_train_ratio * 100.0, len(train_df),
        actual_val_ratio * 100.0, len(val_df),
        actual_test_ratio * 100.0, len(test_df),
        gap, h_int, e_int,
    )

    # پس از ساخت train_df/val_df/test_df و metadata:
    try:
        from scipy.stats import ks_2samp
        target_cols = [c for c in df.columns if c in PROJECT_TARGETS]
        ks_report = {}
        for col in target_cols:
            try:
                a = train_df[col].dropna().to_numpy(dtype=np.float64)
                b = val_df[col].dropna().to_numpy(dtype=np.float64)
                c = test_df[col].dropna().to_numpy(dtype=np.float64)
                ks_report[col] = {
                    "train_vs_val": float(ks_2samp(a, b).statistic) if a.size and b.size else None,
                    "train_vs_test": float(ks_2samp(a, c).statistic) if a.size and c.size else None,
                    "val_vs_test": float(ks_2samp(b, c).statistic) if b.size and c.size else None,
                }
                if ks_report[col]["train_vs_val"] is not None and ks_report[col]["train_vs_val"] < 0.01:
                    logger.warning(
                        "create_split: KS(train,val)=%.4f suspicious for target '%s' "
                        "(potential distributional leakage)",
                        ks_report[col]["train_vs_val"], col,
                    )
            except Exception:
                continue
        metadata["target_ks_report"] = ks_report
    except Exception:
        pass

    return {"train": train_df, "val": val_df, "test": test_df, "metadata": metadata}

def validate_pipeline_for_transform(
    symbol: str,
    timeframe: str,
    fold: int,
    actual_columns,
    strict: bool = True,
) -> bool:
    """
    === FIX C-03: fail-fast قبل از transform روی VAL/TEST ===
    اگر manifest وجود نداشت یا schema mismatch بود:
        - strict=True: RuntimeError
        - strict=False: warning و ادامه (فقط برای debug)
    """
    manifest = load_pipeline_manifest(symbol, timeframe, fold)
    try:
        assert_pipeline_matches(manifest, actual_columns, symbol=symbol)
        return True
    except RuntimeError as e:
        if strict:
            raise
        logger.warning(str(e))
        return False

# === FIX M-02: در evalutate_leakage_all_target و سایر نقاطی که 5class انتظار داشتند ===
def _normalize_direction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    _NORMALIZE_DIRECTION_COLUMNS v2.0 — 2-CLASS.

    اگر ستون price_direction از نسخه‌های قدیمی (3-class یا 5-class) آمده باشد،
    به 2-class تبدیل می‌شود. این تابع idempotent است.
    """
    if "price_direction" not in df.columns:
        return df

    col = df["price_direction"].to_numpy()
    finite = col[np.isfinite(col) & (col != INVALID_LABEL)]
    if finite.size == 0:
        return df

    max_val = int(finite.max())

    if max_val == 4:
        df = df.copy()
        df["price_direction"] = PriceDirectionNames.merge_5class_to_2class(col)
        logger.info("price_direction normalized 5→2 classes")
    elif max_val == 2:
        df = df.copy()
        df["price_direction"] = PriceDirectionNames.merge_3class_to_2class(col)
        logger.info("price_direction normalized 3→2 classes")
    # max_val == 1 → already 2-class, no-op

    return df
    
async def universal_save(
        data,
        data_type: str,
        symbol: str = None,
        timeframe: str = None,
        fold: int = None,
        split_type: str = None,
        lookback: int = None,
        metadata: Dict = None,
        **kwargs
) -> bool:
    """
    🚀 GLOBAL ATOMIC SAVE LAYER - نسخه کامل با پشتیبانی از نام نرمالایز شده
    - single writer (per-event-loop lock)
    - atomic
    - retry
    - ✅ FIX: نرمالایز کردن نام نماد برای ذخیره فایل
    - ✅ FIX v2: قفل ذخیره دیگر به یک event loop واحد گره نمی‌خورد؛ هر loop
      قفل مخصوص خود را می‌گیرد و خطای «bound to a different event loop» که
      ذخیره‌سازی را بی‌صدا شکست می‌داد، حذف شده است.
    """

    async with _get_save_lock():   # 🔥 CRITICAL — only one save per loop at a time
        try:
            # ✅ نرمالایز کردن نام نماد
            def normalize_symbol(s: str) -> str:
                if not s:
                    return s
                return s.replace('.', '')

            original_symbol = symbol

            if data_type == "raw":
                logger.info(f"💾 SAVE | {data_type}")
            else:
                logger.info(f"💾 SAVE | {data_type} | {split_type}")

            if symbol is None or timeframe is None:
                symbol, timeframe = get_active_symbol_timeframe()

            clean_symbol = normalize_symbol(symbol)

            if clean_symbol != symbol:
                logger.debug(f"📝 Symbol normalized for save: {symbol} -> {clean_symbol}")

            Path("data").mkdir(exist_ok=True)

            # =========================
            # route
            # =========================
            if data_type == "raw":
                
                return await _save_raw_atomic(data, clean_symbol, timeframe, fold, metadata)

            if data_type in ["processed", "normalized", "sequenced"]:
                if not split_type:
                    logger.error(f"❌ split_type required for {data_type}")
                    return False

                return await _save_split_atomic(
                    data_type, clean_symbol, timeframe, fold, split_type, data, metadata
                )

            logger.error(f"❌ unknown data_type: {data_type}")
            return False

        except Exception as e:
            logger.error(f"❌ universal_save crash: {e}", exc_info=True)
            return False


async def _load_scalers(
    symbol: str,
    timeframe: str,
    fold: int,
    scaler_type: str = "all",
    load_pkl: bool = True,
    load_metadata: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    📦 لود کردن اسکیلرهای ذخیره شده برای فیچرها و تارگت‌ها
    ✅ FIX v5.2: نرمالایز کردن نام نماد (حذف نقطه) قبل از ساخت مسیر
    ✅ FIX v5.2: سازگار با نام‌گذاری SmartAdaptiveScaler.__init__
    """
    import asyncio
    import time
    from pathlib import Path

    # ✅ CRITICAL FIX: نرمالایز کردن نام نماد (حذف نقطه)
    def normalize_symbol(s: str) -> str:
        if not s:
            return s
        return s.replace('.', '')

    clean_symbol = normalize_symbol(symbol)
    clean_timeframe = timeframe.upper() if timeframe else timeframe

    # مسیر اصلی اسکیلرها با نام نرمالایز شده
    base_path = os.path.join("data", "scalers", clean_symbol, clean_timeframe, f"fold_{fold}")

    if not os.path.exists(base_path):
        logger.error(f"❌ مسیر اسکیلرها یافت نشد: {base_path}")
        # Fallback: تلاش با نماد upper
        alt_base_path = os.path.join("data", "scalers", symbol.upper(), clean_timeframe, f"fold_{fold}")
        if os.path.exists(alt_base_path):
            logger.info(f"✅ Fallback: مسیر جایگزین یافت شد: {alt_base_path}")
            base_path = alt_base_path
        else:
            logger.error(f"❌ Fallback نیز ناموفق بود: {alt_base_path}")
            return None

    result = {
        "symbol": clean_symbol,
        "timeframe": clean_timeframe,
        "fold": fold,
        "base_path": base_path,
        "features": {},
        "targets": {}
    }

    # ═══════════════════════════════════════════════════════════════════════
    # لود کردن فیچر اسکیلرها
    # ═══════════════════════════════════════════════════════════════════════
    if scaler_type in ("all", "features"):
        features_path = os.path.join(base_path, "feature_scalers")
        if os.path.exists(features_path):
            result["features"] = await _load_scaler_folder(
                features_path,
                load_pkl=load_pkl,
                load_metadata=load_metadata,
                is_target=False
            )
            logger.info(f"✅ Feature scalers loaded: {len(result['features'].get('feature_names', []))} features")
        else:
            logger.warning(f"⚠️ پوشه feature_scalers یافت نشد: {features_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # لود کردن تارگت اسکیلرها
    # ═══════════════════════════════════════════════════════════════════════
    if scaler_type in ("all", "targets"):
        targets_path = os.path.join(base_path, "target_scalers")
        if os.path.exists(targets_path):
            result["targets"] = await _load_scaler_folder(
                targets_path,
                load_pkl=load_pkl,
                load_metadata=load_metadata,
                is_target=True
            )
            logger.info(f"✅ Target scalers loaded: {len(result['targets'].get('feature_names', []))} targets")
        else:
            logger.warning(f"⚠️ پوشه target_scalers یافت نشد: {targets_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # اطلاعات خلاصه
    # ═══════════════════════════════════════════════════════════════════════
    result["summary"] = {
        "n_features": len(result["features"].get("feature_names", [])) if result["features"] else 0,
        "n_targets": len(result["targets"].get("feature_names", [])) if result["targets"] else 0,
        "feature_names": result["features"].get("feature_names", []) if result["features"] else [],
        "target_names": result["targets"].get("feature_names", []) if result["targets"] else [],
        "has_feature_scaler": "_scaler" in result["features"] if result["features"] else False,
        "has_target_scaler": "_scaler" in result["targets"] if result["targets"] else False,
        "loaded_at": time.time()
    }

    logger.info(f"✅ اسکیلرها لود شدند: {result['summary']['n_features']} فیچر, {result['summary']['n_targets']} تارگت")
    logger.info(f"   Feature scaler object: {result['summary']['has_feature_scaler']}")
    logger.info(f"   Target scaler object: {result['summary']['has_target_scaler']}")

    return result
    
async def _load_scaler_folder(
    folder_path: str,
    load_pkl: bool = True,
    load_metadata: bool = True,
    is_target: bool = False
) -> Dict[str, Any]:
    """
    📁 لود کردن محتویات یک پوشه اسکیلر
    
    ✅ اصلاح شده: لود کامل full_scaler.pkl و برگرداندن خود شیء اسکیلر
    
    Args:
        folder_path: مسیر پوشه اسکیلر (feature_scalers یا target_scalers)
        load_pkl: آیا فایل full_scaler.pkl لود شود؟
        load_metadata: آیا فایل metadata.json لود شود؟
        is_target: آیا این اسکیلر تارگت است؟
    
    Returns:
        دیکشنری حاوی:
        - "_scaler": خود شیء اسکیلر (SmartAdaptiveScaler)
        - "_metadata": متادیتا (در صورت درخواست)
        - feature_names: نام فیچرها/تارگت‌ها
        - feature_profiles: پروفایل فیچرها
        - و سایر کلیدهای موجود در metadata
    """
    import asyncio
    import pickle
    import json
    from pathlib import Path
    
    result = {}
    folder_type = "target" if is_target else "feature"
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: لود فایل full_scaler.pkl (مهمترین بخش)
    # ═══════════════════════════════════════════════════════════════════════
    pkl_path = os.path.join(folder_path, "full_scaler.pkl")
    scaler_obj = None
    
    if load_pkl and os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                scaler_obj = pickle.load(f)
            
            result["_scaler"] = scaler_obj
            logger.debug(f"📦 {folder_type}_scalers: full_scaler.pkl loaded successfully")
            
            # استخراج اطلاعات از شیء اسکیلر
            if hasattr(scaler_obj, 'feature_names'):
                result["feature_names"] = scaler_obj.feature_names
                logger.debug(f"   ├─ feature_names: {len(scaler_obj.feature_names)} items")
            
            if hasattr(scaler_obj, 'feature_profiles'):
                result["feature_profiles"] = scaler_obj.feature_profiles
                logger.debug(f"   ├─ feature_profiles: {len(scaler_obj.feature_profiles)} items")
            
            if hasattr(scaler_obj, 'target_types'):
                result["target_types"] = scaler_obj.target_types
            
            if hasattr(scaler_obj, 'classification_targets'):
                result["classification_targets"] = scaler_obj.classification_targets
            
            if hasattr(scaler_obj, 'regression_targets'):
                result["regression_targets"] = scaler_obj.regression_targets
            
            if hasattr(scaler_obj, 'fitted'):
                result["fitted"] = scaler_obj.fitted
            
            if hasattr(scaler_obj, 'input_dim'):
                result["input_dim"] = scaler_obj.input_dim
                
        except Exception as e:
            logger.error(f"❌ خطا در لود full_scaler.pkl از {folder_path}: {e}")
            # ادامه می‌دهیم تا شاید metadata.json کار کند
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: لود metadata.json (به عنوان پشتیبان یا تکمیل‌کننده)
    # ═══════════════════════════════════════════════════════════════════════
    metadata_path = os.path.join(folder_path, "metadata.json")
    
    if load_metadata and os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            result["_metadata"] = metadata
            
            # پر کردن اطلاعات缺失 از metadata (اگر در scaler نبود)
            if "feature_names" not in result and "feature_names" in metadata:
                result["feature_names"] = metadata["feature_names"]
                logger.debug(f"   ├─ feature_names from metadata: {len(metadata['feature_names'])} items")
            
            if "feature_profiles" not in result and "feature_profiles" in metadata:
                result["feature_profiles"] = metadata["feature_profiles"]
                logger.debug(f"   ├─ feature_profiles from metadata: {len(metadata['feature_profiles'])} items")
            
            if "target_types" not in result and "target_types" in metadata:
                result["target_types"] = metadata["target_types"]
            
            if "classification_targets" not in result and "classification_targets" in metadata:
                result["classification_targets"] = metadata["classification_targets"]
            
            if "regression_targets" not in result and "regression_targets" in metadata:
                result["regression_targets"] = metadata["regression_targets"]
            
            if "fitted" not in result and "fitted" in metadata:
                result["fitted"] = metadata["fitted"]
            
            if "input_dim" not in result and "input_dim" in metadata:
                result["input_dim"] = metadata["input_dim"]
            
            # اطلاعات اضافی
            result["symbol"] = metadata.get("symbol")
            result["timeframe"] = metadata.get("timeframe")
            result["fold_num"] = metadata.get("fold_num")
            result["is_target"] = metadata.get("is_target", is_target)
            result["saved_at"] = metadata.get("saved_at")
            result["engine_version"] = metadata.get("engine_version")
            
            logger.debug(f"📋 {folder_type}_scalers: metadata.json loaded")
            
        except Exception as e:
            logger.error(f"❌ خطا در لود metadata.json از {folder_path}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: لود فایل‌های JSON تکی (به عنوان پشتیبان نهایی)
    # ═══════════════════════════════════════════════════════════════════════
    if not result.get("feature_names") and not result.get("feature_profiles"):
        try:
            files = os.listdir(folder_path)
        except Exception as e:
            logger.error(f"❌ خطا در خواندن پوشه {folder_path}: {e}")
            return result
        
        feature_profiles_temp = {}
        feature_names_temp = []
        
        for filename in files:
            if filename in ("metadata.json", "full_scaler.pkl"):
                continue
            
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path) and filename.endswith('.json'):
                feature_name = filename.replace('.json', '')
                feature_names_temp.append(feature_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        feature_data = json.load(f)
                    feature_profiles_temp[feature_name] = feature_data
                    logger.debug(f"  ├─ {feature_name}: {feature_data.get('scaler_type', 'unknown')}")
                except Exception as e:
                    logger.error(f"❌ خطا در لود {filename}: {e}")
        
        if feature_names_temp:
            result["feature_names"] = feature_names_temp
            result["feature_profiles"] = feature_profiles_temp
            logger.debug(f"   ├─ Loaded {len(feature_names_temp)} individual JSON files")
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: گزارش نهایی
    # ═══════════════════════════════════════════════════════════════════════
    n_features = len(result.get("feature_names", []))
    has_scaler = "_scaler" in result
    has_metadata = "_metadata" in result
    
    logger.debug(f"✅ {folder_type}_scalers: {n_features} features | scaler={has_scaler} | metadata={has_metadata}")
    
    return result


async def universal_load(
        data_type: str,
        symbol: str = None,
        timeframe: str = None,
        fold: int = None,
        split_type: str = None,
        lookback: int = None,
        **kwargs
):
    """
    📦 Universal Loader - نسخه اصلاح شده با بازگشت دیکشنری برای sequenced data

    ✅ FIX v3.0 (CRITICAL):
        - PROJECT_TARGETS_LIST از لیست هاردکد (که momentum_direction را
          نداشت) به `list(PROJECT_TARGETS) + list(AUXILIARY_WEIGHTS)` تغییر
          کرد. حالا هر تارگت جدیدی که در PROJECT_TARGETS تعریف شود،
          به‌طور خودکار در این لیست هم لحاظ می‌شود.
        - این باگ باعث می‌شد momentum_direction ساخته شود، در H5 ذخیره شود،
          اما هرگز به مدل نرسد.
    """
    import time
    from collections import OrderedDict
    import numpy as np

    DATA_TYPE_ALIASES = {
        "sequence": "sequenced",
        "seq": "sequenced",
        "scaler": "scalers",
        "scaler_loader": "scalers",
        "processed_data": "processed",
        "norm": "normalized"
    }
    data_type = DATA_TYPE_ALIASES.get(data_type.lower(), data_type.lower())
    start_time = time.time()

    def normalize_symbol(s: str) -> str:
        if not s:
            return s
        return s.replace('.', '')

    multi_output = kwargs.get("multi_output", False)
    return_targets_only = kwargs.get("return_targets_only", False)
    return_dataframe = kwargs.get("return_dataframe", False)

    try:
        # ========== 2. RESOLVE SYMBOL/TIMEFRAME ==========
        if symbol is None or timeframe is None:
            try:
                active_symbol, active_timeframe = get_active_symbol_timeframe()

                if active_symbol is None or active_timeframe is None:
                    error_msg = (
                        "❌ CRITICAL: Could not resolve active symbol/timeframe from tf_decisions.json. "
                        "Please ensure the file exists and contains valid data."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

                if symbol is None:
                    symbol = active_symbol
                    logger.info(f"📋 Using active symbol from tf_decisions: {symbol}")

                if timeframe is None:
                    timeframe = active_timeframe
                    logger.info(f"📋 Using active timeframe from tf_decisions: {timeframe}")

            except Exception as e:
                logger.error(f"❌ Failed to get active symbol/timeframe: {e}")
                return None

        original_symbol = symbol
        clean_symbol = normalize_symbol(symbol)

        if clean_symbol != original_symbol:
            logger.debug(f"[{original_symbol}] 📝 Symbol normalized for path: {original_symbol} -> {clean_symbol}")

        symbol = original_symbol  # برای لاگ و نمایش، نام اصلی نگهداری می‌شود
        timeframe = timeframe.upper()

        # مقدار پیش‌فرض برای fold
        if fold is None and data_type not in ["scalers", "model_dims", "regime", "project_info"]:
            try:
                from main import sum_folds as TOTAL_FOLDS
                fold = TOTAL_FOLDS
                logger.debug(f"[{symbol}] Using default fold: {fold}")
            except ImportError:
                fold = 5
                logger.debug(f"[{symbol}] Using fallback fold: {fold}")

        # ========== 3. LOAD BASED ON DATA TYPE ==========
        result = None

        # ---------- 3.1 SCALERS ----------
        if data_type == "scalers":
            try:
                scaler_type = kwargs.get("scaler_type", "all")
                load_pkl = kwargs.get("load_pkl", True)
                load_metadata = kwargs.get("load_metadata", True)

                result = await _load_scalers(
                    symbol=clean_symbol,
                    timeframe=timeframe,
                    fold=fold,
                    scaler_type=scaler_type,
                    load_pkl=load_pkl,
                    load_metadata=load_metadata
                )

                if result:
                    n_features = result.get('summary', {}).get('n_features', 0)
                    n_targets = result.get('summary', {}).get('n_targets', 0)
                    logger.info(f"[{symbol}] ✅ Scalers loaded: {n_features} features, {n_targets} targets")

                    if kwargs.get("return_scaler_only", False):
                        if scaler_type == "features" and result.get("features") and "_scaler" in result["features"]:
                            return result["features"]["_scaler"]
                        elif scaler_type == "targets" and result.get("targets") and "_scaler" in result["targets"]:
                            return result["targets"]["_scaler"]
                        elif result.get("features") and "_scaler" in result["features"]:
                            return result["features"]["_scaler"]

                    return result
                else:
                    logger.warning(f"[{symbol}] ⚠️ Scalers not found for fold {fold}")
                    return None

            except Exception as e:
                logger.error(f"[{symbol}] ❌ Error loading scalers: {e}", exc_info=True)
                return None

        # ---------- 3.2 SEQUENCED ----------
        elif data_type == "sequenced":
            try:
                logger.info(f"[{symbol}] 📥 Loading sequenced data: split={split_type}, fold={fold}")

                data = await _load_sequenced_data(clean_symbol, timeframe, fold, split_type, lookback)

                if data is None:
                    logger.warning(f"[{symbol}] ⚠️ Sequenced data is None")
                    return None

                X = data.get('X')
                if X is None:
                    logger.error(f"[{symbol}] ❌ X not found in data")
                    return None

                # استخراج تارگت‌ها
                y_dict = {}

                # ✅ FIX v3.0 (CRITICAL):
                # استفاده از module-level constants به‌جای hardcode.
                # نسخهٔ قبلی 'momentum_direction' را نداشت، بنابراین
                # تارگت الحاقی جدید هرگز به مدل نمی‌رسید.
                PROJECT_TARGETS_LIST = list(PROJECT_TARGETS) + list(AUXILIARY_WEIGHTS)

                for target in PROJECT_TARGETS_LIST:
                    if target in data and data[target] is not None:
                        y_arr = data[target]
                        if hasattr(y_arr, 'ndim'):
                            if y_arr.ndim == 1:
                                y_arr = y_arr.reshape(-1, 1)
                            elif y_arr.ndim == 2 and y_arr.shape[1] == 5:
                                try:
                                    y_arr = np.argmax(y_arr, axis=1).reshape(-1, 1)
                                    logger.info(f"[{symbol}]   ✅ Converted one-hot to indices for {target}")
                                except Exception:
                                    pass
                        y_dict[target] = y_arr
                        logger.info(f"[{symbol}]   ✅ Target/Aux loaded: {target} | shape={y_arr.shape}")

                if not y_dict and 'y' in data and data['y'] is not None:
                    y_data = data['y']
                    if isinstance(y_data, dict):
                        for target in PROJECT_TARGETS_LIST:
                            if target in y_data:
                                y_dict[target] = y_data[target]
                                logger.info(f"[{symbol}]   ✅ Target from data['y']: {target}")

                # ✅ برگرداندن دیکشنری
                exec_time = time.time() - start_time

                logger.info(
                    f"[{symbol}] ✅ Loaded sequenced data (dict format): "
                    f"X={X.shape if hasattr(X, 'shape') else '?'}, "
                    f"targets={len(y_dict)} | {exec_time:.2f}s"
                )

                result_dict = {
                    'X': X,
                    'y': y_dict if y_dict else None,
                    'metadata': data.get('metadata', {}),
                    'file_path': data.get('file_path'),
                    'split_type': split_type,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'fold': fold
                }

                for target_name, target_arr in y_dict.items():
                    result_dict[target_name] = target_arr

                if return_targets_only:
                    return y_dict

                if kwargs.get("return_X_only", False):
                    return X

                if multi_output or kwargs.get("return_dict", False):
                    return X, y_dict

                return result_dict

            except Exception as e:
                logger.error(f"[{symbol}] ❌ Error in universal_load (sequenced): {e}", exc_info=True)
                return None

        # ---------- 3.3 RAW, PROCESSED, NORMALIZED ----------
        elif data_type in {"raw", "processed", "normalized"}:
            result = await _load_processed_optimized(
                data_type=data_type,
                symbol=clean_symbol,
                original_symbol=symbol,
                timeframe=timeframe,
                fold=fold,
                split_type=split_type
            )

            if result is not None and isinstance(result, dict) and return_dataframe:
                import pandas as pd
                try:
                    if 'df' in result:
                        result = pd.DataFrame(result['df'])
                    else:
                        result = pd.DataFrame(result)
                    logger.info(f"[{symbol}] ✅ Converted dict to DataFrame: {result.shape}")
                except Exception as e:
                    logger.warning(f"[{symbol}] ⚠️ Could not convert to DataFrame: {e}")

        # ---------- 3.4 OTHER TYPES ----------
        elif data_type == "model_dims":
            result = await load_model_dims(clean_symbol, fold)

        elif data_type == "regime":
            from core.regim import get_symbol_regime
            result = await get_symbol_regime()

        elif data_type == "expiry":
            fold_to_use = fold if fold is not None else 5
            from core.regim import calculate_expiry_date
            result = await calculate_expiry_date(
                symbol=clean_symbol,
                timeframe=timeframe,
                fold_number=fold_to_use,
                regime_info=kwargs.get("regime_info")
            )

        elif data_type == "project_info":
            result = load_project_info()

        elif data_type == "metadata":
            effective_split = split_type if split_type is not None else "train"
            result = await _load_sequence_metadata(
                clean_symbol,
                timeframe,
                fold,
                split_type=effective_split
            )

        elif data_type == "features_info":
            result = await _load_features_info(clean_symbol, timeframe, fold)

        elif data_type == "split":
            result = await _load_split_data(clean_symbol, timeframe, fold, split_type)

        else:
            logger.warning(f"[{symbol}] ⚠️ Unknown data type: {data_type}")
            result = None

        # ========== 4. FINAL OUTPUT ==========
        exec_time = time.time() - start_time

        if result is not None:
            if hasattr(result, 'shape'):
                logger.info(f"[{symbol}] ✅ Loaded {data_type} | shape={result.shape} | {exec_time:.2f}s")
            elif isinstance(result, dict) and len(result) > 0:
                logger.info(f"[{symbol}] ✅ Loaded {data_type} | {len(result)} keys | {exec_time:.2f}s")
            elif isinstance(result, tuple) and len(result) == 2:
                x_shape = result[0].shape if hasattr(result[0], 'shape') else '?'
                y_shape = result[1].shape if hasattr(result[1], 'shape') else '?'
                logger.info(f"[{symbol}] ✅ Loaded {data_type} | X={x_shape}, y={y_shape} | {exec_time:.2f}s")
            else:
                logger.info(f"[{symbol}] ✅ Loaded {data_type} | {exec_time:.2f}s")
        else:
            logger.warning(f"[{symbol}] ⚠️ {data_type} not found (fold={fold}, split={split_type})")

        return result

    except Exception as e:
        logger.error(f"[{symbol}] ❌ universal_load error for {data_type}: {e}", exc_info=True)
        return None
        
async def _load_processed_optimized(
    data_type: str,
    symbol: str,
    timeframe: str,
    fold: int,
    split_type: Optional[str] = None,
    original_symbol: str = None
) -> Optional[Union[pd.DataFrame, np.ndarray]]:
    """
    🔧 بارگذاری بهینه‌شده داده‌های پردازش شده - با پشتیبانی از نام نرمالایز شده
    ✅ FIX: استفاده از نام نماد بدون نقطه برای جستجو
    ✅ FIX: جستجوی چندگانه با الگوهای مختلف
    
    Args:
        data_type: نوع داده ('raw', 'processed', 'normalized')
        symbol: نام نماد نرمالایز شده (بدون نقطه)
        timeframe: تایم فریم
        fold: شماره فولد
        split_type: نوع اسپلیت ('train', 'val', 'test') - برای processed/normalized لازم است
        original_symbol: نام اصلی نماد (برای لاگ)
    """
    try:
        from pathlib import Path
        import re
        
        display_symbol = original_symbol or symbol
        
        logger.debug(f"[{display_symbol}] 🔍 _load_processed_optimized: data_type={data_type}, symbol={symbol}, fold={fold}, split={split_type}")
        
        # تعیین مسیر پایه با نام نرمالایز شده
        if data_type == "raw":
            base_dir = Path("data/raw/mt5")
            # الگوهای مختلف نام فایل برای raw
            patterns = [
                f"{symbol}_{timeframe}_fold{fold}_raw.h5",
                f"{symbol}_{timeframe}_fold{fold}.h5",
                f"{symbol}_{timeframe}_raw_fold{fold}.h5",
                f"*{symbol}*{timeframe}*fold{fold}*raw*.h5",
            ]
        elif data_type in ["processed", "normalized"]:
            if not split_type:
                logger.error(f"[{display_symbol}] ❌ split_type required for {data_type}")
                return None
            base_dir = Path("data") / data_type / symbol / timeframe / f"fold_{fold}"
            base_dir.mkdir(parents=True, exist_ok=True)
            
            # الگوهای مختلف نام فایل
            patterns = [
                # اولویت 1: نام استاندارد نرمالایز شده
                f"{symbol}_{timeframe}_fold{fold}_{split_type}_{data_type}.h5",
                # اولویت 2: بدون data_type در نام
                f"{symbol}_{timeframe}_fold{fold}_{split_type}.h5",
                # اولویت 3: با فرمت مختلف
                f"{symbol}_{timeframe}_fold{fold}_{split_type}_*.h5",
                # اولویت 4: جستجوی wildcard
                f"*{split_type}_{data_type}.h5",
                f"*fold{fold}*{split_type}*.h5",
            ]
        elif data_type == "sequenced":
            if not split_type:
                logger.error(f"[{display_symbol}] ❌ split_type required for {data_type}")
                return None
            base_dir = Path("data") / data_type / symbol / timeframe / f"fold_{fold}"
            base_dir.mkdir(parents=True, exist_ok=True)
            
            patterns = [
                f"{symbol}_{timeframe}_fold{fold}_{split_type}_{data_type}_lookback*.h5",
                f"{symbol}_{timeframe}_fold{fold}_{split_type}_{data_type}.h5",
                f"{symbol}_{timeframe}_fold{fold}_{split_type}.h5",
                f"*{split_type}_sequenced*.h5",
            ]
        else:
            logger.error(f"[{display_symbol}] ❌ Unknown data type: {data_type}")
            return None
        
        logger.debug(f"[{display_symbol}] 📁 Searching in: {base_dir}")
        logger.debug(f"[{display_symbol}] 📁 Patterns: {patterns}")
        
        # جستجوی فایل
        for pattern in patterns:
            file_path = base_dir / pattern if not any(c in pattern for c in ['*', '?']) else None
            
            if file_path and file_path.exists():
                if file_path.stat().st_size > 100:
                    logger.info(f"[{display_symbol}] 📁 Loading: {file_path.name}")
                    return await _load_hdf5(file_path, data_type, display_symbol, timeframe, fold)
            else:
                # جستجوی wildcard
                try:
                    matches = list(base_dir.glob(pattern))
                    for match in matches:
                        if match.stat().st_size > 100:
                            logger.info(f"[{display_symbol}] 📁 Found via pattern: {match.name}")
                            return await _load_hdf5(match, data_type, display_symbol, timeframe, fold)
                except Exception as glob_err:
                    logger.debug(f"[{display_symbol}] Pattern error {pattern}: {glob_err}")
        
        # اگر فایل پیدا نشد، جستجوی بازگشتی عمیق‌تر
        if base_dir.exists():
            logger.debug(f"[{display_symbol}] 🔍 Deep search in {base_dir}")
            all_h5_files = list(base_dir.rglob("*.h5"))
            for h5_file in all_h5_files:
                file_lower = h5_file.name.lower()
                if split_type and split_type.lower() in file_lower:
                    if data_type.lower() in file_lower or (data_type == "processed" and "proc" in file_lower):
                        logger.info(f"[{display_symbol}] 📁 Found via deep search: {h5_file.name}")
                        return await _load_hdf5(h5_file, data_type, display_symbol, timeframe, fold)
        
        logger.warning(f"[{display_symbol}] ⚠️ File not found: {data_type} | fold={fold} | split={split_type}")
        return None
        
    except Exception as e:
        logger.error(f"[{display_symbol or symbol}] ❌ Error in _load_processed_optimized: {e}", exc_info=True)
        return None
        


async def _load_sequenced_data(
        symbol: str,
        timeframe: str,
        fold: int,
        split_type: str = "train",
        lookback: Optional[int] = None,
        **kwargs
) -> Optional[Dict[str, Any]]:
    """
    📥 بارگذاری داده‌های sequenced — v3.0 (AUDIT-HARDENED).

    ────────────────────────────────────────────────────────────────────────
    تغییرات v3.0 نسبت به v2.1:
    ────────────────────────────────────────────────────────────────────────
    ✅ FIX P0 (fallback lookback):
        اگر `lookback` صریحاً داده شود و فایل مربوط به همان lookback نباشد،
        دیگر به‌طور بی‌صدا به lookback دیگر fallback نمی‌شود. خطای صریح با
        لیست lookbackهای موجود لاگ و `None` برگردانده می‌شود.
        این جلوی آموزش مدل با dimension اشتباه را می‌گیرد.

    ✅ FIX P0 (integrity gate):
        `check_sequenced_files_structure` که هر سه split را چک می‌کرد، به
        soft-warning تبدیل شد. نبود `val`/`test` نباید load کردن `train`
        را بلاک کند. این تابع خودش وجود فایل `split_type` را چک می‌کند.

    ✅ FIX P1 (one-hot conversion):
        تبدیل one-hot → indices برای `price_direction` حالا ۲/۳/۵ کلاسه را
        پشتیبانی می‌کند (قبلاً فقط ۵ کلاسه). با `np.allclose(sum==1)` از
        false-positive روی داده‌های باینری/پیوسته جلوگیری می‌شود.

    ✅ FIX P1 (missing targets):
        در صورت غیبت تارگت‌های اجباری، فایل بارگذاری می‌شود اما با
        فلگ‌های `_integrity_ok=False` و `_missing_targets=[...]` برگردانده
        می‌شود تا caller بتواند fail-fast کند.

    ✅ FIX P2 (perf):
        بررسی one-hot فقط روی نمونهٔ head (تا ۱۰۰۰ ردیف) انجام می‌شود،
        نه کل آرایه.

    ✅ FIX P2 (metadata):
        attrهایی که به‌صورت JSON-string ذخیره شده‌اند، اکنون parse
        می‌شوند (columns/shape/…).

    ✅ FIX P2 (dependency):
        بررسی صریح `H5PY_AVAILABLE` در ابتدای تابع.

    ────────────────────────────────────────────────────────────────────────
    Returns:
        Dict یا None. کلیدهای خروجی:
            X, y (dict), metadata, file_path, split_type, symbol,
            timeframe, fold, _integrity_ok (bool), _missing_targets (list)
            + یک کلید top-level برای هر تارگت/aux موجود.
    """
    logger.info(
        f"[{symbol}] 📥 Loading sequenced | split={split_type} | "
        f"fold={fold} | lookback={lookback}"
    )

    # ── FIX P2: بررسی صریح وابستگی h5py ──
    if not H5PY_AVAILABLE:
        logger.error(
            f"[{symbol}] ❌ h5py در دسترس نیست — بارگذاری sequenced ناممکن است"
        )
        return None

    # ── fold validation ──
    try:
        fold_int = int(fold)
        if fold_int <= 0:
            logger.error(f"[{symbol}] ❌ Invalid fold: {fold}")
            return None
    except (ValueError, TypeError):
        logger.error(
            f"[{symbol}] ❌ Fold must be int, got: "
            f"{type(fold).__name__}={fold}"
        )
        return None

    # ── FIX P0: integrity check به soft-warning تبدیل شد ──
    try:
        structure_ok = check_sequenced_files_structure(
            symbol, timeframe, fold_int, logger
        )
        if not structure_ok:
            logger.warning(
                f"[{symbol}] ⚠️ Integrity check برای fold {fold_int} ناقص است "
                f"(احتمالاً val یا test ساخته نشده). ادامه با split={split_type}."
            )
    except Exception as check_err:
        logger.warning(
            f"[{symbol}] ⚠️ Integrity check exception (ignored): {check_err}"
        )

    base_dir = Path("data/sequenced") / symbol / timeframe / f"fold_{fold_int}"
    if not base_dir.exists():
        logger.error(f"[{symbol}] ❌ Directory not found: {base_dir}")
        return None

    # ── FIX P0: resolve file (بدون fallback بی‌صدا) ──
    file_path: Optional[Path] = None

    if lookback is not None:
        try:
            lb_int = int(lookback)
        except (ValueError, TypeError):
            logger.error(
                f"[{symbol}] ❌ lookback must be int, got {lookback!r}"
            )
            return None

        exact = list(
            base_dir.glob(f"*{split_type}*sequenced_lookback{lb_int}.h5")
        )
        if exact:
            file_path = exact[0]
        else:
            available = sorted({
                int(m.group(1))
                for f in base_dir.glob(f"*{split_type}*sequenced*.h5")
                for m in [re.search(r"lookback(\d+)", f.name)]
                if m
            })
            logger.error(
                f"[{symbol}] ❌ No file with lookback={lb_int} for "
                f"split={split_type}. "
                f"Available lookbacks: {available if available else 'none'}. "
                f"Refusing silent fallback to a different lookback."
            )
            return None
    else:
        candidates = list(base_dir.glob(f"*{split_type}*sequenced*.h5"))
        if not candidates:
            candidates = list(base_dir.glob(f"*{split_type}*.h5"))
        if not candidates:
            logger.error(
                f"[{symbol}] ❌ File not found for {split_type} in {base_dir}"
            )
            return None
        file_path = candidates[0]

    logger.info(f"[{symbol}] 📂 Reading: {file_path.name}")

    EXPECTED_TARGETS = list(PROJECT_TARGETS)
    AUXILIARY_COLUMNS = list(AUXILIARY_WEIGHTS)

    try:
        with h5py.File(file_path, 'r') as hf:
            # ── X ──
            if 'X' in hf:
                X = hf['X'][:]
            elif 'X_seq' in hf:
                X = hf['X_seq'][:]
            else:
                logger.error(
                    f"[{symbol}] ❌ Neither 'X' nor 'X_seq' in "
                    f"{file_path.name}. keys={list(hf.keys())}"
                )
                return None

            logger.info(f"[{symbol}] 📊 X shape: {X.shape}")

            target_dict: Dict[str, np.ndarray] = {}
            missing_targets: List[str] = []

            # ── targets ──
            for target in EXPECTED_TARGETS:
                if target not in hf:
                    missing_targets.append(target)
                    logger.warning(
                        f"[{symbol}]   ⚠️ Target '{target}' NOT found in H5"
                    )
                    continue

                target_data = hf[target][:]
                if target_data.ndim == 1:
                    target_data = target_data.reshape(-1, 1)

                # ── FIX P1: one-hot → indices برای 2/3/5 کلاس ──
                # ── FIX P2: بررسی فقط روی نمونهٔ head ──
                if target == 'price_direction' and target_data.ndim == 2:
                    n_cols = target_data.shape[1]
                    if n_cols in (2, 3, 5):
                        sample = target_data[: min(1000, len(target_data))]
                        is_onehot_like = (
                            np.all((sample >= 0) & (sample <= 1))
                            and np.allclose(sample.sum(axis=1), 1.0)
                        )
                        if is_onehot_like:
                            target_data = np.argmax(
                                target_data, axis=1
                            ).reshape(-1, 1)
                            logger.info(
                                f"[{symbol}]   🔄 Converted one-hot({n_cols}) "
                                f"→ indices for {target}"
                            )

                target_dict[target] = target_data
                logger.info(
                    f"[{symbol}]   ✅ Target loaded: {target} | "
                    f"shape={target_data.shape} | dtype={target_data.dtype}"
                )

            # ── auxiliary weights ──
            aux_loaded = 0
            for aux in AUXILIARY_COLUMNS:
                if aux in hf:
                    aux_data = hf[aux][:]
                    if aux_data.ndim == 1:
                        aux_data = aux_data.reshape(-1, 1)
                    target_dict[aux] = aux_data
                    aux_loaded += 1
                    logger.info(
                        f"[{symbol}]   ✅ Auxiliary loaded: {aux} | "
                        f"shape={aux_data.shape} | dtype={aux_data.dtype}"
                    )
                else:
                    logger.warning(
                        f"[{symbol}]   ⚠️ Auxiliary '{aux}' NOT found in H5"
                    )

            logger.info(
                f"[{symbol}] ✅ Loaded {len(target_dict)} columns total "
                f"({len(EXPECTED_TARGETS) - len(missing_targets)}/"
                f"{len(EXPECTED_TARGETS)} targets + "
                f"{aux_loaded} auxiliary weights)"
            )

            # ── FIX P1: missing targets فقط flag می‌شوند، نه return None ──
            if missing_targets:
                logger.error(
                    f"[{symbol}] ❌ CRITICAL: {len(missing_targets)} required "
                    f"targets MISSING from H5: {missing_targets}. "
                    f"File likely generated with an OLD version of "
                    f"Sequence_Factory. Setting _integrity_ok=False so "
                    f"downstream consumers can fail-fast."
                )

            # ── legacy fallback ──
            if not target_dict:
                for legacy_key in ('y_seq', 'y'):
                    if legacy_key in hf:
                        t = hf[legacy_key][:]
                        if t.ndim == 1:
                            t = t.reshape(-1, 1)
                        target_dict['target'] = t
                        logger.warning(
                            f"[{symbol}] ⚠️ Using legacy '{legacy_key}' as "
                            f"fallback target (no formal targets found)"
                        )
                        break

            if not target_dict:
                logger.warning(
                    f"[{symbol}] ⚠️ NO columns loaded. "
                    f"keys={list(hf.keys())}"
                )
                return {
                    'X': X,
                    'y': {},
                    'metadata': {},
                    'file_path': str(file_path),
                    'split_type': split_type,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'fold': fold_int,
                    '_missing_targets': missing_targets,
                    '_integrity_ok': False,
                }

            # ── FIX P2: metadata با parse کردن JSON attrs ──
            metadata: Dict[str, Any] = {}
            for attr in hf.attrs:
                try:
                    raw = hf.attrs[attr]
                    if isinstance(raw, bytes):
                        raw = raw.decode('utf-8')
                    if isinstance(raw, str):
                        try:
                            raw = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            pass  # اگر JSON نبود، رشته را نگه دار
                    metadata[attr] = raw
                except Exception as attr_err:
                    metadata[attr] = str(hf.attrs[attr])
                    logger.debug(
                        f"[{symbol}] metadata attr '{attr}' fallback: "
                        f"{attr_err}"
                    )

            result: Dict[str, Any] = {
                'X': X,
                'y': target_dict,
                'metadata': metadata,
                'file_path': str(file_path),
                'split_type': split_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'fold': fold_int,
                '_missing_targets': missing_targets,
                '_integrity_ok': len(missing_targets) == 0,
            }

            for name, arr in target_dict.items():
                result[name] = arr

            logger.info(
                f"[{symbol}] ✅ Loaded: X={X.shape}, "
                f"columns={list(target_dict.keys())} | "
                f"integrity_ok={result['_integrity_ok']}"
            )
            return result

    except Exception as e:
        logger.error(
            f"[{symbol}] ❌ Error loading sequenced: {e}", exc_info=True
        )
        return None 

def discover_available_lookback(
    symbol: str,
    timeframe: str,
    fold: int,
    split_type: str = "train",
) -> Optional[int]:
    """
    DISCOVER_AVAILABLE_LOOKBACK v1.0 — DYNAMIC LOOKBACK RESOLUTION.

    چرا لازم است:
        نسخهٔ قبلی `run_tests` مقدار `_lookback = 64` را هاردکد کرده بود.
        در لاگ اجرا، فایل‌های موجود lookback=65 داشتند و
        `_load_sequenced_data` به‌درستی fail-fast کرد:
            "No file with lookback=64 for split=train. Available lookbacks: [65]"
        این fail-fast برای جلوگیری از آموزش مدل با dimension اشتباه ضروری
        است، اما خودِ عدد 64 غلط بود. باید از خودِ داده کشف شود.

    استراتژی (کاملاً dynamic — هیچ هاردکدی):
        ۱. نرمال‌سازی symbol (حذف نقطه) و timeframe (uppercase)
        ۲. اسکن دایرکتوری `data/sequenced/{symbol}/{timeframe}/fold_{fold}/`
        ۳. جستجوی فایل‌های `*_{split_type}_sequenced*.h5`
        ۴. استخراج lookback از نام فایل با regex `lookback(\d+)`
        ۵. اگر نام فایل تگ lookback نداشت → خواندن shape از H5 (کلید X_seq یا X)
        ۶. اگر چند مقدار یافت شد → انتخاب پرتکرارترین (tiebreak = بزرگ‌ترین)
        ۷. اگر هیچ مقداری یافت نشد → None (caller باید خطای صریح بدهد)

    قرارداد:
        • این تابع تنها منبع معتبر lookback است. هیچ caller نباید
          مقدار را هاردکد کند.
        • خروجی None یعنی هیچ فایل معتبری یافت نشد؛ caller موظف است
          RuntimeError بدهد (نه fallback ساکت).

    Args:
        symbol: نماد معاملاتی (با یا بدون نقطه — هر دو پشتیبانی می‌شوند).
        timeframe: تایم‌فریم (حساس به بزرگی/کوچکی نیست).
        fold: شمارهٔ فولد (int).
        split_type: 'train' | 'val' | 'test' (پیش‌فرض 'train').

    Returns:
        Optional[int]: lookback کشف‌شده یا None در صورت نبود فایل معتبر.
    """
    import re

    if symbol is None or timeframe is None:
        logger.warning(
            "[discover_lookback] symbol or timeframe is None — cannot scan"
        )
        return None

    clean_symbol = str(symbol).replace(".", "").strip()
    clean_timeframe = str(timeframe).upper().strip()
    try:
        fold_int = int(fold)
    except (TypeError, ValueError):
        logger.warning(f"[discover_lookback] invalid fold={fold!r}")
        return None

    base_dir = (
        Path("data") / "sequenced" / clean_symbol / clean_timeframe
        / f"fold_{fold_int}"
    )
    if not base_dir.exists():
        logger.warning(
            f"[discover_lookback] directory missing: {base_dir}"
        )
        return None

    candidates = list(base_dir.glob(f"*_{split_type}_sequenced*.h5"))
    if not candidates:
        candidates = list(base_dir.glob(f"*{split_type}*sequenced*.h5"))
    if not candidates:
        logger.warning(
            f"[discover_lookback] no '{split_type}' sequenced files in "
            f"{base_dir} | available files: "
            f"{[p.name for p in list(base_dir.glob('*.h5'))[:5]]}"
        )
        return None

    lookback_counts: Dict[int, int] = {}

    for fpath in candidates:
        m = re.search(r"lookback(\d+)", fpath.name)
        if m:
            try:
                lb = int(m.group(1))
                if 1 <= lb <= 8192:
                    lookback_counts[lb] = lookback_counts.get(lb, 0) + 1
            except (TypeError, ValueError):
                continue
            continue

        # fallback: inspect H5 shape
        try:
            import h5py as _h5
            with _h5.File(fpath, "r") as hf:
                for key in ("X_seq", "X", "data"):
                    if key in hf:
                        shape = hf[key].shape
                        if len(shape) >= 2:
                            lb = int(shape[1])
                            if 1 <= lb <= 8192:
                                lookback_counts[lb] = (
                                    lookback_counts.get(lb, 0) + 1
                                )
                            break
        except Exception as e:
            logger.debug(
                f"[discover_lookback] could not read {fpath.name}: {e}"
            )

    if not lookback_counts:
        logger.warning(
            f"[discover_lookback] no lookback discovered from "
            f"{len(candidates)} files under {base_dir}"
        )
        return None

    best_lb = max(
        lookback_counts.items(),
        key=lambda kv: (kv[1], kv[0]),
    )[0]

    logger.info(
        "[discover_lookback] %s/%s/fold%d split=%s → lookback=%d "
        "(distribution=%s)",
        clean_symbol, clean_timeframe, fold_int, split_type, best_lb,
        dict(sorted(lookback_counts.items())),
    )
    return int(best_lb)
    
def _hdf5_safe_dataframe(df: pd.DataFrame):
    df = df.copy()
    datetime_columns = []

    for col in df.columns:

        if is_datetime64_any_dtype(df[col]):
            datetime_columns.append(col)

            # حذف timezone
            if hasattr(df[col].dtype, "tz") and df[col].dtype.tz is not None:
                df[col] = df[col].dt.tz_convert("UTC").dt.tz_localize(None)

            # همیشه با ns ذخیره می‌کنیم
            df[col] = df[col].astype("int64")

        elif is_bool_dtype(df[col]):
            df[col] = df[col].astype("uint8")

        elif is_categorical_dtype(df[col]):
            df[col] = df[col].astype("string")

        elif is_object_dtype(df[col]):
            df[col] = df[col].astype("string")

    return df, datetime_columns


def _sync_save(data, file_path: Path, metadata: dict = None, logger=None):

    import h5py
    import json
    import gc
    import numpy as np
    import pandas as pd
    from pandas.api.types import is_datetime64_any_dtype

    temp_path = file_path.with_suffix('.tmp')

    try:

        # -----------------------------
        # Convert to DataFrame
        # -----------------------------
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # -----------------------------
        # Handle datetime columns safely
        # -----------------------------
        datetime_columns = []

        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                datetime_columns.append(col)
                df[col] = df[col].astype("int64")

        # -----------------------------
        # Save HDF5
        # -----------------------------
        with h5py.File(temp_path, "w") as f:

            data_grp = f.create_group("data")

            for col in df.columns:
                dset = data_grp.create_dataset(
                    col,
                    data=df[col].values,
                    compression="gzip",
                    compression_opts=6,
                    shuffle=True
                )

                dset.attrs["original_dtype"] = str(data[col].dtype)

                # ⭐ ثبت صریح اینکه datetime بوده
                if col in datetime_columns:
                    dset.attrs["is_datetime"] = True
                    dset.attrs["datetime_unit"] = "ns"

            # -----------------------------
            # Save index
            # -----------------------------
            index_grp = f.create_group("index")

            index_values = df.index.values
            index_is_datetime = False

            if is_datetime64_any_dtype(df.index):
                index_values = index_values.astype("int64")
                index_is_datetime = True

            index_grp.create_dataset("values", data=index_values)
            index_grp.attrs["dtype"] = str(df.index.dtype)
            index_grp.attrs["is_datetime"] = index_is_datetime
            index_grp.attrs["datetime_unit"] = "ns"

            # -----------------------------
            # Metadata
            # -----------------------------
            f.attrs["columns"] = json.dumps(list(df.columns))
            f.attrs["shape"] = json.dumps(list(df.shape))
            f.attrs["format_version"] = "3.0"
            f.attrs["datetime_columns"] = json.dumps(datetime_columns)

            if metadata:
                f.attrs["metadata"] = json.dumps(metadata, default=str)

            f.flush()

        temp_path.replace(file_path)

        if logger:
            logger.info(f"✅ Saved (v3): {file_path.name} | shape={df.shape}")

        return True

    except Exception as e:
        if logger:
            logger.error(f"❌ Error in atomic save: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False

    finally:
        gc.collect()


async def _save_raw_atomic(
    data,
    symbol: str,
    timeframe: str,
    fold: str,
    metadata: dict = None,
):
    """
    RAW DATA WRAPPER
    Async-safe
    """

    import asyncio
    from pathlib import Path

    base_path = Path("data/raw/mt5")
    base_path.mkdir(parents=True, exist_ok=True)

    filename = f"{symbol}_{timeframe}_fold{fold}_raw.h5"
    final_file = base_path / filename

    logger.info(f"📁 RAW SAVE PATH: {final_file}")

    # Build metadata
    full_metadata = {
        "symbol": symbol,
        "timeframe": timeframe,
        "fold": str(fold),
        "data_type": "raw"
    }

    if metadata:
        full_metadata.update(metadata)

    loop = asyncio.get_running_loop()

    # 🔥 فقط این خط مهمه
    result = await loop.run_in_executor(
        None,
        _sync_save,
        data,
        final_file,
        full_metadata,
    )

    return result

async def _load_hdf5(
        file_path: Path,
        data_type: str,
        symbol: str,
        timeframe: str,
        fold: int
):
    """
    ✅ FIXED LOADER v2.0 — با کش LRU DataFrame.

    رفتار:
      1. بررسی کش بر اساس (path, mtime_ns, size) → در صورت hit، بازگشت فوری.
      2. در صورت miss، تشخیص خودکار ساختار فایل (new/raw/old/fallback)
         و بارگذاری کامل.
      3. ذخیره در کش فقط اگر نتیجه DataFrame معتبر باشد.

    ⚠️ بازگشتی در صورت cache-hit بین فراخوانی‌ها share می‌شود. برای
    جلوگیری از mutation ناخواسته، مصرف‌کننده در صورت نیاز باید copy بگیرد.

    Args:
        file_path: مسیر کامل فایل HDF5.
        data_type: نوع داده ('raw' | 'processed' | 'normalized' | 'sequenced').
        symbol: نام نماد (برای لاگ).
        timeframe: تایم‌فریم (برای لاگ).
        fold: شماره فولد (برای لاگ).

    Returns:
        pd.DataFrame یا None در صورت خطا.
    """
    import h5py
    import pandas as pd
    import numpy as np

    try:
        if not file_path.exists():
            logger.warning(f"⚠️ فایل یافت نشد: {file_path}")
            return None

        # ─── Phase 1: Cache lookup ──────────────────────────────────
        cached = _df_cache_get(file_path)
        if cached is not None:
            logger.info(
                f"📂 [CACHE HIT] {file_path.name} | "
                f"shape={cached.shape if hasattr(cached, 'shape') else '?'} | "
                f"cache_size={dataframe_cache_stats()['size']}"
            )
            return cached

        # ─── Phase 2: File metadata ─────────────────────────────────
        logger.info(f"📂 بارگذاری فایل: {file_path.name}")
        logger.info(f"   📍 مسیر کامل: {file_path}")
        logger.info(f"   📊 حجم فایل: {file_path.stat().st_size:,} bytes")

        result: Optional[pd.DataFrame] = None

        with h5py.File(file_path, "r") as hf:
            all_keys = list(hf.keys())
            logger.info(f"📋 کلیدهای موجود در فایل: {all_keys}")

            # ─── Priority 1: New format (index + data groups) ──────
            if "index" in all_keys and "data" in all_keys:
                logger.info("📀 تشخیص ساختار جدید فایل (با ایندکس مجزا)")
                result = await _load_new_hdf5_format(hf, file_path)

            # ─── Priority 2: RAW format (data group with dataset) ───
            elif (
                "data" in all_keys
                and isinstance(hf["data"], h5py.Group)
                and "dataset" in hf["data"]
            ):
                logger.info("📀 تشخیص ساختار raw (با دیتاست واحد)")
                result = await _load_raw_hdf5_format(hf, file_path)

            # ─── Priority 3: Old format (df group) ──────────────────
            elif "df" in all_keys:
                logger.info("📀 تشخیص ساختار قدیمی فایل (با گروه 'df')")
                result = await _load_old_hdf5_format(hf, file_path)

            # ─── Priority 4: Fallback ───────────────────────────────
            else:
                logger.warning(
                    f"⚠️ ساختار ناشناخته، تلاش با fallback: {all_keys}"
                )
                result = await _load_fallback_hdf5_format(hf, file_path)

        # ─── Phase 3: Cache store (only on valid DataFrame) ─────────
        if result is not None and isinstance(result, pd.DataFrame):
            _df_cache_put(file_path, result)
            logger.debug(
                f"💾 [CACHE PUT] {file_path.name} | "
                f"cache_size={dataframe_cache_stats()['size']}"
            )

        return result

    except Exception as e:
        logger.error(f"❌ load failed {file_path.name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
        
async def _load_new_hdf5_format(hf, file_path):

    import pandas as pd
    import json

    try:

        data_grp = hf["data"]
        index_grp = hf["index"]

        # -----------------------------
        # Load index
        # -----------------------------
        index_values = index_grp["values"][:]

        if index_grp.attrs.get("is_datetime", False):
            unit = index_grp.attrs.get("datetime_unit", "ns")
            index = pd.to_datetime(index_values, unit=unit)
        else:
            index = pd.Index(index_values)

        # -----------------------------
        # Load columns
        # -----------------------------
        columns = json.loads(hf.attrs["columns"])
        datetime_columns = json.loads(hf.attrs.get("datetime_columns", "[]"))

        data_dict = {}

        for col in columns:
            dset = data_grp[col]
            values = dset[:]

            if col in datetime_columns or dset.attrs.get("is_datetime", False):
                unit = dset.attrs.get("datetime_unit", "ns")
                values = pd.to_datetime(values, unit=unit)

            data_dict[col] = values

        df = pd.DataFrame(data_dict, index=index)

        return df

    except Exception as e:
        logger.error(f"❌ Load new format failed: {e}")
        return None

async def _load_raw_hdf5_format(hf: h5py.File, file_path: Path) -> pd.DataFrame:
    """
    بارگذاری ساختار raw (با گروه data و دیتاست dataset)
    """
    try:
        logger.info("🔍 شروع بارگذاری با فرمت raw")

        data_grp = hf["data"]
        dataset = data_grp["dataset"][:]

        logger.info(f"   📊 دیتاست - shape: {dataset.shape}")
        logger.info(f"   📊 دیتاست - dtype: {dataset.dtype}")

        # تبدیل به DataFrame
        if isinstance(dataset, np.ndarray):
            if dataset.dtype.names is not None:
                # این یک record array است
                df = pd.DataFrame.from_records(dataset)
                logger.info(f"   ✅ تبدیل record array به DataFrame")
            else:
                # آرایه ساده
                df = pd.DataFrame(dataset)
                logger.info(f"   ✅ تبدیل آرایه ساده به DataFrame")

        # تلاش برای بازیابی نام ستون‌ها از metadata
        if "columns" in hf.attrs:
            columns = json.loads(hf.attrs["columns"])
            if len(columns) == len(df.columns):
                df.columns = columns
                logger.info(f"   ✅ نام ستون‌ها از metadata بازیابی شد: {columns[:5]}")

        logger.info(f"📊 DataFrame نهایی - shape: {df.shape}")
        logger.info(f"📊 DataFrame نهایی - columns: {list(df.columns)}")

        # بررسی NaN
        if not df.empty:
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"⚠️ {nan_count} مقدار NaN در DataFrame")

        return df

    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری ساختار raw: {e}")
        return None


async def _load_fallback_hdf5_format(hf: h5py.File, file_path: Path) -> pd.DataFrame:
    """
    روش fallback برای بارگذاری فایل‌های HDF5 با ساختار نامشخص
    """
    try:
        logger.info("🔍 شروع بارگذاری با روش fallback")

        # اگر فقط یک دیتاست وجود دارد
        datasets = []
        for key in hf.keys():
            if isinstance(hf[key], h5py.Dataset):
                datasets.append((key, hf[key]))

        logger.info(f"   📊 {len(datasets)} دیتاست یافت شد")

        if len(datasets) == 1:
            # فقط یک دیتاست - آن را به عنوان DataFrame بارگذاری کن
            name, dset = datasets[0]
            logger.info(f"   📊 دیتاست واحد: {name}, shape={dset.shape}")

            data = dset[:]

            if isinstance(data, np.ndarray):
                if data.dtype.names is not None:
                    df = pd.DataFrame.from_records(data)
                    logger.info(f"   ✅ record array به DataFrame تبدیل شد")
                else:
                    df = pd.DataFrame(data)
                    logger.info(f"   ✅ آرایه ساده به DataFrame تبدیل شد")

                logger.info(f"📊 DataFrame نهایی - shape: {df.shape}")
                return df

        # اگر چند دیتاست وجود دارد، هرکدام را به عنوان ستون در نظر بگیر
        data_dict = {}
        for key in hf.keys():
            if isinstance(hf[key], h5py.Dataset):
                try:
                    data_dict[key] = hf[key][:]
                    logger.info(f"   📊 دیتاست {key}: shape={hf[key].shape}")
                except Exception as e:
                    logger.warning(f"   ⚠️ خطا در بارگذاری دیتاست {key}: {e}")

        if data_dict:
            df = pd.DataFrame(data_dict)
            logger.info(f"✅ Fallback: بارگذاری {len(data_dict)} دیتاست به عنوان ستون")
            logger.info(f"📊 DataFrame نهایی - shape: {df.shape}")
            return df

        logger.error("❌ هیچ دیتاستی برای بارگذاری یافت نشد")
        return None

    except Exception as e:
        logger.error(f"❌ خطا در fallback: {e}")
        return None


async def _load_old_hdf5_format(hf, file_path: Path):
    """
    Load legacy HDF5 format where data is stored under group 'df'.
    ✅ FIX v2.2: decode ستون‌های bytes — h5py 3.x رشته‌های variable-length را
       به‌صورت bytes برمی‌گرداند و نسخهٔ قبلی فقط dtype نوع 'S' را پوشش می‌داد؛
       ستون‌های object حاوی bytes نیز حالا به str تبدیل می‌شوند (در غیر این
       صورت تبدیل datetime بعدی روی bytes خام می‌خورد).
    ✅ FIX v2.1: تبدیل ستون‌های datetime از string به datetime64
    ✅ FIX v2.1: حذف duplicate columns پس از لود
    """
    import pandas as pd
    import numpy as np

    logger.info("=" * 60)
    logger.info("📊 FINAL LOAD REPORT (OLD FORMAT)")
    logger.info("=" * 60)

    try:
        # -------------------------------------------------
        # 1️⃣ Validate structure
        # -------------------------------------------------
        if "df" not in hf:
            raise ValueError("Group 'df' not found in HDF5 file.")

        group = hf["df"]
        columns = list(group.keys())

        if not columns:
            raise ValueError("Group 'df' contains no datasets.")

        logger.info(f"📦 Columns detected: {len(columns)}")

        # -------------------------------------------------
        # 2️⃣ Load columns safely (با decode صحیح bytes)
        # -------------------------------------------------
        data_dict = {}

        for col in columns:
            try:
                data = group[col][()]

                if isinstance(data, np.ndarray):
                    if data.dtype.kind == "S":
                        # fixed-width bytes → string
                        data = data.astype(str)
                    elif data.dtype == object and data.size > 0:
                        # variable-length bytes (h5py 3.x) → decode هر عنصر
                        first_element = next(
                            (item for item in data.flat if item is not None),
                            None,
                        )
                        if isinstance(first_element, (bytes, bytearray)):
                            data = np.array(
                                [
                                    item.decode("utf-8", errors="replace")
                                    if isinstance(item, (bytes, bytearray))
                                    else item
                                    for item in data.flat
                                ],
                                dtype=object,
                            ).reshape(data.shape)

                data_dict[col] = data

            except Exception as col_error:
                logger.warning(f"⚠️ Failed loading column '{col}': {col_error}")

        if not data_dict:
            raise ValueError("No columns could be loaded from HDF5.")

        # -------------------------------------------------
        # 3️⃣ Create DataFrame
        # -------------------------------------------------
        df = pd.DataFrame(data_dict)

        # -------------------------------------------------
        # 4️⃣ Safety check
        # -------------------------------------------------
        if df is None or not isinstance(df, pd.DataFrame):
            raise TypeError(f"Loaded object is not DataFrame. Type: {type(df)}")

        if df.empty:
            raise ValueError("Loaded DataFrame is empty.")

        # -------------------------------------------------
        # 5️⃣ Convert datetime string columns back to datetime64
        # -------------------------------------------------
        datetime_col_converted = []
        for col in df.columns:
            col_lower = str(col).lower()

            is_date_col = any(
                kw in col_lower
                for kw in ['datetime', 'timestamp', 'date', 'time']
            )

            if not is_date_col:
                continue

            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue

            try:
                converted = pd.to_datetime(df[col], errors='coerce', utc=True)
                if converted.notna().mean() > 0.5:
                    df[col] = converted
                    datetime_col_converted.append(col)
            except Exception as conv_err:
                logger.debug(f"   ⚠️ Could not convert '{col}' to datetime: {conv_err}")

        if datetime_col_converted:
            logger.info(f"   ✅ Converted {len(datetime_col_converted)} datetime columns: {datetime_col_converted}")

        # -------------------------------------------------
        # 6️⃣ Remove duplicate columns after loading
        # -------------------------------------------------
        if df.columns.duplicated().any():
            dup_cols = df.columns[df.columns.duplicated()].tolist()
            logger.warning(f"   ⚠️ Found {len(dup_cols)} duplicate columns after load, removing: {dup_cols[:5]}")
            df = df.loc[:, ~df.columns.duplicated(keep='first')]

        # -------------------------------------------------
        # 7️⃣ Final report
        # -------------------------------------------------
        logger.info(f"✅ Loaded successfully: {file_path.name}")
        logger.info(f"📐 Shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"❌ Old HDF5 load failed: {file_path.name} | {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

async def _save_split_atomic(
        data_type,
        symbol,
        timeframe,
        fold,
        split_type,
        data,
        metadata
):
    """
    ✅ FIXED V7: استفاده از تایم فریم واقعی از tf_decisions
    ✅ نرمالایز کردن نام نماد برای ذخیره فایل
    ✅ FIX: پشتیبانی از هر دو فرمت نام (با و بدون نقطه)
    ✅ FIX: لاگ دقیق مسیر ذخیره‌سازی
    """

    # ✅ نرمالایز کردن نام نماد (حذف نقطه)
    def normalize_symbol(s: str) -> str:
        if not s:
            return s
        return s.replace('.', '')
    
    clean_symbol = normalize_symbol(symbol)
    
    # ⭐⭐⭐⭐⭐ CRITICAL FIX: دریافت تایم فریم واقعی از tf_decisions
    try:
        active_symbol, active_timeframe = get_active_symbol_timeframe()
        # اگر تایم فریم ارسال شده با تایم فریم فعال متفاوت است، هشدار بده
        if timeframe != active_timeframe:
            logger.warning(f"[{symbol}] ⚠️ ذخیره فایل: timeframe ارسالی ({timeframe}) با تایم فریم فعال ({active_timeframe}) متفاوت است!")
            logger.warning(f"[{symbol}]    → استفاده از تایم فریم فعال برای مسیر: {active_timeframe}")
        # ⭐ استفاده از تایم فریم فعال برای مسیر
        actual_timeframe = active_timeframe
    except Exception as e:
        logger.warning(f"[{symbol}] ⚠️ خطا در دریافت تایم فریم فعال: {e} - استفاده از مقدار ارسالی: {timeframe}")
        actual_timeframe = timeframe
    
    if not split_type:
        logger.error("split_type required")
        return False

    # ساخت مسیر با نام نرمالایز شده و تایم فریم واقعی
    base = Path("data") / data_type / clean_symbol / actual_timeframe / f"fold_{fold}"
    base.mkdir(parents=True, exist_ok=True)

    # نام فایل استاندارد با نام نرمالایز شده و تایم فریم واقعی
    final_path = base / f"{clean_symbol}_{actual_timeframe}_fold{fold}_{split_type}_{data_type}.h5"
    tmp_path = base / f".tmp_{uuid.uuid4().hex}.h5"

    logger.info(f"[{symbol}] 💾 Saving {data_type}/{split_type} to: {final_path}")
    logger.info(f"[{symbol}]    Clean symbol used: {clean_symbol}")
    logger.info(f"[{symbol}]    Timeframe used: {actual_timeframe} (from tf_decisions)")

    # =========================
    # Normalize -> DataFrame
    # =========================
    try:
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame({f"col0": data})
            else:
                df = pd.DataFrame(data, columns=[f"col{i}" for i in range(data.shape[1])])
        elif isinstance(data, dict):
            processed_dict = {}
            for key, val in data.items():
                if isinstance(val, np.ndarray):
                    processed_dict[key] = val
                elif isinstance(val, (int, float, str, bool)):
                    processed_dict[key] = np.array([val])
                elif isinstance(val, (list, tuple)):
                    processed_dict[key] = np.array(val)
                else:
                    processed_dict[key] = np.array([val])
            df = pd.DataFrame(processed_dict)
        else:
            logger.error(f"unsupported type: {type(data)}")
            return False

        if df.empty or df.shape[0] == 0:
            logger.warning(f"⚠️ Empty DataFrame for {split_type}, skipping save")
            return True

        columns = list(df.columns)
        logger.debug(f"[{symbol}] DataFrame shape: {df.shape}, columns: {len(columns)}")
        
    except Exception as e:
        logger.error(f"[{symbol}] ❌ Error converting to DataFrame: {e}")
        return False

    # =========================
    # Atomic Write (HDF5 Safe)
    # =========================
    for attempt in range(5):
        try:
            with h5py.File(tmp_path, "w") as hf:
                grp = hf.create_group("df")

                for col in columns:
                    col_data = df[col]
                    
                    # Convert to string for HDF5 if needed
                    if pd.api.types.is_datetime64_any_dtype(col_data):
                        col_data = col_data.astype(str)
                    elif col_data.dtype == 'object':
                        col_data = col_data.astype(str)
                    elif pd.api.types.is_categorical_dtype(col_data):
                        col_data = col_data.astype(str)
                    elif col_data.dtype == bool:
                        col_data = col_data.astype('int8')
                    
                    # Convert to numpy array
                    try:
                        arr = col_data.to_numpy()
                    except Exception:
                        arr = np.array(col_data)
                    
                    # Ensure 1D
                    if arr.ndim > 1:
                        arr = arr.flatten()
                    
                    grp.create_dataset(
                        col,
                        data=arr,
                        compression="gzip",
                        chunks=True
                    )

                # Convert ALL attributes to string-safe format
                def to_hdf5_safe(value):
                    if isinstance(value, (str, bytes)):
                        return value
                    elif isinstance(value, (int, float, bool)):
                        return str(value)
                    elif isinstance(value, (list, tuple, dict)):
                        return json.dumps(value, default=str, ensure_ascii=False)
                    elif value is None:
                        return "null"
                    else:
                        return str(value)

                hf.attrs["columns"] = json.dumps(columns, ensure_ascii=False)
                hf.attrs["shape"] = json.dumps(list(df.shape))
                hf.attrs["symbol"] = to_hdf5_safe(clean_symbol)  # ✅ استفاده از نام نرمالایز شده
                hf.attrs["original_symbol"] = to_hdf5_safe(symbol)  # ✅ نگهداری نام اصلی برای مرجع
                hf.attrs["timeframe"] = to_hdf5_safe(actual_timeframe)  # ✅ استفاده از تایم فریم واقعی
                hf.attrs["fold"] = to_hdf5_safe(fold)
                hf.attrs["split_type"] = to_hdf5_safe(split_type)
                hf.attrs["data_type"] = to_hdf5_safe(data_type)
                hf.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()

                if metadata:
                    safe_metadata = {}
                    for k, v in metadata.items():
                        safe_metadata[k] = to_hdf5_safe(v)
                    hf.attrs["metadata"] = json.dumps(safe_metadata, default=str, ensure_ascii=False)

            gc.collect()
            
            # Atomic rename
            tmp_path.replace(final_path)
            
            file_size = final_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ {split_type} saved safely ({df.shape[0]} rows, {df.shape[1]} cols) -> {final_path.name} ({file_size:.2f} MB)")
            return True

        except PermissionError:
            await asyncio.sleep(0.25 * (attempt + 1))
            logger.warning(f"[{symbol}] ⚠️ Permission error on attempt {attempt + 1}/5")
        except Exception as e:
            logger.error(f"[{symbol}] ❌ Error during atomic save attempt {attempt + 1}/5: {e}")
            if attempt < 4:
                await asyncio.sleep(0.5 * (attempt + 1))

    logger.error(f"[{symbol}] Failed to save {split_type} after 5 retries: {final_path}")
    
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except:
            pass
    
    return False

   
async def _save_with_h5py(
        hdf5_file: Path,
        data,
        metadata: dict,
        logger
) -> bool:
    """
    ✅ SAFE HDF5 SAVE
    unified format with _save_split_atomic
    """

    import h5py, json, uuid, gc
    import pandas as pd
    import numpy as np

    tmp_path = hdf5_file.with_name(f".tmp_{uuid.uuid4().hex}.h5")

    try:
        # -------------------------
        # normalize → DataFrame
        # -------------------------
        if isinstance(data, pd.DataFrame):
            df = data.copy()

        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data, columns=[f"col{i}" for i in range(data.shape[1])])

        elif isinstance(data, dict):
            df = pd.DataFrame(data)

        else:
            logger.error(f"unsupported type: {type(data)}")
            return False

        columns = list(df.columns)

        # -------------------------
        # atomic write
        # -------------------------
        with h5py.File(tmp_path, "w") as hf:

            grp = hf.create_group("df")

            for col in columns:
                grp.create_dataset(
                    col,
                    data=df[col].to_numpy(),
                    compression="gzip",
                    chunks=True
                )

            hf.attrs["columns"] = json.dumps(columns)
            hf.attrs["shape"] = df.shape

            if metadata:
                hf.attrs["metadata"] = json.dumps(metadata, default=str)

        gc.collect()
        tmp_path.replace(hdf5_file)

        return True

    except Exception as e:
        logger.warning(f"⚠️ save failed: {e}")
        return False


##==================


def get_latest_training_folds(total: int, num: int) -> List[int]:
    """
    🎯 فقط جدیدترین فولدها را برمی‌گرداند
    مثال:
        total=17, num=3 → [15,16,17]
    """
    if total <= 0 or num <= 0:
        return []

    num = min(num, total)
    start = total - num + 1
    return list(range(start, total + 1))

async def get_file_path(
    data_type: str,
    symbol: str,
    timeframe: str,
    fold: int,
    split_type: str = None,
    lookback: int = None
) -> Path:
    """
    دریافت مسیر کامل فایل بر اساس پارامترها
    ✅ نرمالایز کردن نام نماد (حذف نقطه) برای مسیرهای فایل
    ✅ ⭐⭐⭐⭐⭐ CRITICAL FIX: استفاده از تایم فریم از tf_decisions (نه هاردکد)
    ✅ پشتیبانی از هر دو فرمت نام
    """

    # ✅ نرمالایز کردن نام نماد - حذف نقطه
    def normalize_symbol(s: str) -> str:
        if not s:
            return s
        return s.replace('.', '')
    
    clean_symbol = normalize_symbol(symbol)
    
    # ⭐⭐⭐⭐⭐ CRITICAL FIX: دریافت تایم فریم واقعی از tf_decisions
    try:
        active_symbol, active_timeframe = get_active_symbol_timeframe()
        # اگر تایم فریم ارسال شده با تایم فریم فعال متفاوت است، هشدار بده
        if timeframe != active_timeframe:
            logger.warning(f"⚠️ دریافت مسیر فایل: timeframe ارسالی ({timeframe}) با تایم فریم فعال ({active_timeframe}) متفاوت است!")
            logger.warning(f"   → استفاده از تایم فریم فعال برای مسیر: {active_timeframe}")
        # ⭐ استفاده از تایم فریم فعال برای مسیر
        actual_timeframe = active_timeframe
    except Exception as e:
        logger.warning(f"⚠️ خطا در دریافت تایم فریم فعال: {e} - استفاده از مقدار ارسالی: {timeframe}")
        actual_timeframe = timeframe
    
    data_dir = Path("data")
    
    if data_type == "raw":
        file_name = f"{clean_symbol}_{actual_timeframe}_fold{fold}_raw.h5"
        return data_dir / "raw" / "mt5" / file_name
        
    elif data_type == "processed":
        file_name = f"{clean_symbol}_{actual_timeframe}_fold{fold}_{split_type}_processed.h5"
        return data_dir / "processed" / clean_symbol / actual_timeframe / f"fold_{fold}" / file_name
        
    elif data_type == "normalized":
        file_name = f"{clean_symbol}_{actual_timeframe}_fold{fold}_{split_type}_normalized.h5"
        return data_dir / "normalized" / clean_symbol / actual_timeframe / f"fold_{fold}" / file_name
        
    elif data_type == "sequenced":
        if lookback:
            file_name = f"{clean_symbol}_{actual_timeframe}_fold{fold}_{split_type}_sequenced_lookback{lookback}.h5"
        else:
            file_name = f"{clean_symbol}_{actual_timeframe}_fold{fold}_{split_type}_sequenced.h5"
        return data_dir / "sequenced" / clean_symbol / actual_timeframe / f"fold_{fold}" / file_name
        
    elif data_type == "scalers":
        return data_dir / "scalers" / clean_symbol / actual_timeframe / f"fold_{fold}"
    
    return None
    
def hash_row(row) -> str:
    """🔹 Hash یک سطر به صورت امن و تطبیق‌پذیر"""
    row_bytes = str(tuple(row.values)).encode('utf-8')
    return hashlib.sha256(row_bytes).hexdigest()

def detect_time_column(df: pd.DataFrame):
    """
    🔹 تشخیص ستون زمان بدون هیچ side-effect روی DataFrame ورودی.

    نسخهٔ قبلی داخل حلقهٔ try عبارت ``df[c] = pd.to_datetime(df[c])`` را اجرا
    می‌کرد؛ یعنی ستون Caller بی‌صدا تغییر می‌نمود (mutation پنهان) که می‌توانست
    مراحل بعدی پایپ‌لاین (مقایسه/ذخیره/هدف‌گذاری) را مختل کند. این نسخه فقط روی
    یک SAMPLE پارس می‌کند و هیچ تغییری در df ایجاد نمی‌نماید.

    Args:
        df: دیتافریم کاندیدا.

    Returns:
        Optional[str]: نام ستون زمان در صورت یافت شدن، وگرنه None.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None

    candidates = [
        c for c in df.columns
        if isinstance(c, str) and ('time' in c.lower() or 'date' in c.lower())
    ]

    for c in candidates:
        try:
            col = df[c]

            if is_datetime64_any_dtype(col):
                return c

            probe = col.dropna().head(10000)
            if probe.empty:
                continue

            if is_numeric_dtype(probe):
                numeric_probe = pd.to_numeric(probe, errors="coerce")
                med = numeric_probe.abs().median()
                if not np.isfinite(med):
                    continue
                if med >= 1e17:
                    unit = "ns"
                elif med >= 1e14:
                    unit = "us"
                elif med >= 1e11:
                    unit = "ms"
                else:
                    unit = "s"
                parsed = pd.to_datetime(numeric_probe, unit=unit, utc=True, errors="coerce")
            else:
                parsed = pd.to_datetime(probe, utc=True, errors="coerce")

            if parsed.notna().mean() >= 0.8:
                return c

        except Exception:
            continue

    logger.info("ℹ️ No timestamp column detected. Skipping time-based leakage check.")
    return None
from main import DEFAULT_SYMBOL
def attach_fundamental_features(
    df: pd.DataFrame,
    symbol: str = DEFAULT_SYMBOL,
    timestamp_col: str = "timestamp",
    max_staleness: str = "1D",
    require_success: bool = False,
    cg_api_key: Optional[str] = None,
    cc_api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    الحاق فیچرهای بنیادی به DataFrame.

    ✅ v3.1 — Auto-detect timestamp column:

        نسخهٔ قبلی به timestamp_col="timestamp" هاردکد نیاز داشت. اما
        داده RAW از MT5 ستون را "datetime" می‌نامد (نه "timestamp").
        این نسخه به‌طور خودکار هر ستون datetime/timestamp/date/time را
        پیدا می‌کند.

    قرارداد:
        - merge_asof با direction='backward' → بدون leakage زمانی.
        - در صورت شکست، df بدون تغییر برگردانده می‌شود (مگر
          require_success=True که RuntimeError پرتاب می‌شود).

    Args:
        df: DataFrame با ستون datetime.
        symbol: نماد پروژه.
        timestamp_col: نام ستون زمان (اگر در df نبود، auto-detect).
        max_staleness: حداکثر فاصلهٔ forward-fill.
        require_success: خطا در صورت شکست.
        cg_api_key: CoinGecko API Key (اختیاری).
        cc_api_key: CryptoCompare API Key (اختیاری).

    Returns:
        DataFrame با ستون‌های بنیادی اضافه‌شده (یا df اصلی در صورت شکست).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"[FUND-v3.1] df must be DataFrame, got {type(df).__name__}"
        )
    if df.empty:
        logger.warning("[FUND-v3.1] df is empty — skipping")
        return df

    # ═══════════════════════════════════════════════════════════════════
    # ✅ FIX: Auto-detect timestamp column
    # ═══════════════════════════════════════════════════════════════════
    def _find_timestamp_column(frame: pd.DataFrame) -> Optional[str]:
        # اولویت 1: ستون دلخواه caller
        if timestamp_col in frame.columns:
            return timestamp_col
        # اولویت 2: نام‌های رایج
        for candidate in (
            "timestamp", "datetime", "date", "time",
            "open_time", "close_time",
        ):
            if candidate in frame.columns:
                return candidate
        # اولویت 3: dtype datetime
        for col in frame.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    return str(col)
            except Exception:
                continue
        return None

    detected_ts_col = _find_timestamp_column(df)

    if detected_ts_col is None:
        message = (
            f"[FUND-v3.1] no datetime column found in {list(df.columns)[:10]}"
        )
        if require_success:
            raise RuntimeError(message)
        logger.warning(message)
        return df

    logger.info(
        "[FUND-v3.1] using timestamp column: '%s'", detected_ts_col,
    )

    try:
        ts_series = pd.to_datetime(
            df[detected_ts_col], utc=True, errors="coerce",
        )
    except Exception as exc:
        message = f"[FUND-v3.1] timestamp conversion failed: {exc}"
        if require_success:
            raise RuntimeError(message) from exc
        logger.warning(message)
        return df

    if ts_series.isna().all():
        message = "[FUND-v3.1] all timestamps invalid — skipping"
        if require_success:
            raise RuntimeError(message)
        logger.warning(message)
        return df

    valid_ts = ts_series.dropna()
    if valid_ts.empty:
        message = "[FUND-v3.1] no valid timestamps — skipping"
        if require_success:
            raise RuntimeError(message)
        logger.warning(message)
        return df

    start = valid_ts.min().to_pydatetime()
    end = valid_ts.max().to_pydatetime()

    logger.info(
        "[FUND-v3.1] fetch range: %s → %s (%d rows)",
        start.date(), end.date(), len(df),
    )

    builder = FundamentalFeatureBuilder(
        symbol=symbol,
        cg_api_key=cg_api_key,
        cc_api_key=cc_api_key,
    )

    try:
        fund_df = builder.collect_all_safe(
            start=start, end=end,
            candle_freq="30min",
            max_staleness=max_staleness,
        )
    except Exception as exc:
        message = f"[FUND-v3.1] collection failed: {exc}"
        if require_success:
            raise RuntimeError(message) from exc
        logger.error(message, exc_info=True)
        return df

    if fund_df is None or len(fund_df) == 0 or fund_df.shape[1] == 0:
        message = "[FUND-v3.1] collection returned no features"
        if require_success:
            raise RuntimeError(message)
        logger.warning(message)
        return df

    df_work = df.copy()
    df_work["_ts_for_merge"] = ts_series

    fund_reset = fund_df.reset_index().rename(columns={"index": "timestamp"})
    fund_reset["timestamp"] = pd.to_datetime(fund_reset["timestamp"], utc=True)
    fund_reset = fund_reset.sort_values("timestamp").reset_index(drop=True)

    new_cols = [c for c in fund_reset.columns if c != "timestamp"]
    if not new_cols:
        logger.warning("[FUND-v3.1] no new columns to merge")
        return df

    try:
        df_sorted = df_work.sort_values("_ts_for_merge").reset_index(drop=True)

        # حذف هر ستون بنیادی قدیمی برای جلوگیری از تداخل
        overlap_cols = [c for c in new_cols if c in df_sorted.columns]
        if overlap_cols:
            logger.warning(
                "[FUND-v3.1] removing pre-existing fundamental columns: %s",
                overlap_cols[:5],
            )
            df_sorted = df_sorted.drop(columns=overlap_cols, errors="ignore")
            new_cols = [c for c in new_cols if c not in overlap_cols]

        merged = pd.merge_asof(
            df_sorted,
            fund_reset,
            left_on="_ts_for_merge",
            right_on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(max_staleness),
        )
        merged = merged.drop(columns=["timestamp"], errors="ignore")
        merged = merged.drop(columns=["_ts_for_merge"], errors="ignore")

        # اعتبارسنجی نهایی
        non_null_cols = [
            c for c in new_cols
            if c in merged.columns and merged[c].notna().any()
        ]

        logger.info(
            "[FUND-v3.1] attached %d/%d columns (non-null): %s",
            len(non_null_cols), len(new_cols), non_null_cols[:8],
        )

        if not non_null_cols:
            message = (
                "[FUND-v3.1] all fundamental columns are entirely NaN. "
                "Likely API rate-limit or geo-block."
            )
            if require_success:
                raise RuntimeError(message)
            logger.error(message)
            return df

        return merged

    except Exception as exc:
        message = f"[FUND-v3.1] merge_asof failed: {exc}"
        if require_success:
            raise RuntimeError(message) from exc
        logger.error(message, exc_info=True)
        return df
        
def prepare_final_dict_with_dates(X_scaled, y_scaled, feature_names,
                                  date_column_name=None, date_data=None, split_name=None):
    """
    🎯 آماده‌سازی دیکشنری نهایی با تضمین اضافه شدن ستون تاریخ
    نسخه نهایی و بدون نقص
    """
    logger.info(f"[{split_name.upper()}] {'=' * 60}")
    logger.info(f"[{split_name.upper()}] 📅 Preparing final dict for {split_name} with date column")
    logger.info(f"[{split_name.upper()}] {'=' * 60}")

    # =====================================================
    # 1️⃣ Convert X to DataFrame
    # =====================================================
    if isinstance(X_scaled, pd.DataFrame):
        X_df = X_scaled.copy()
        logger.info(f"[{split_name.upper()}]   ✅ X_scaled is already DataFrame with {len(X_df.columns)} columns")
    else:
        if X_scaled.ndim == 3:
            n_samples, n_timesteps, n_features = X_scaled.shape
            column_names = []
            for t in range(n_timesteps):
                for f_idx, f_name in enumerate(feature_names[:n_features]):
                    column_names.append(f"{f_name}_t{t}")
            X_2d = X_scaled.reshape(n_samples, n_timesteps * n_features)
            X_df = pd.DataFrame(X_2d, columns=column_names, index=range(n_samples))
            logger.info(f"[{split_name.upper()}]   ✅ Converted 3D to 2D: {X_df.shape}")
        else:
            n_features = min(len(feature_names), X_scaled.shape[1]) if X_scaled.ndim == 2 else X_scaled.shape[0]
            X_df = pd.DataFrame(X_scaled, columns=feature_names[:n_features], index=range(X_scaled.shape[0]))
            logger.info(f"[{split_name.upper()}]   ✅ Converted numpy to DataFrame: {X_df.shape}")

    # =====================================================
    # 2️⃣ Convert y to DataFrame
    # =====================================================
    if y_scaled is not None:
        if isinstance(y_scaled, pd.DataFrame):
            y_df = y_scaled.copy()
        else:
            if y_scaled.ndim == 3:
                n_samples, n_timesteps, n_targets = y_scaled.shape
                target_names = ["price_output", "volatility_output"][:n_targets]
                column_names = []
                for t in range(n_timesteps):
                    for t_idx, t_name in enumerate(target_names):
                        column_names.append(f"{t_name}_t{t}")
                y_2d = y_scaled.reshape(n_samples, n_timesteps * n_targets)
                y_df = pd.DataFrame(y_2d, columns=column_names, index=range(n_samples))
            else:
                target_names = ["price_output", "volatility_output"][:y_scaled.shape[1]]
                y_df = pd.DataFrame(y_scaled, columns=target_names, index=range(y_scaled.shape[0]))
        logger.info(f"[{split_name.upper()}]   ✅ y DataFrame created: {y_df.shape}")
    else:
        y_df = pd.DataFrame()
        logger.warning(f"[{split_name.upper()}]   ⚠️ y_scaled is None")

    # =====================================================
    # 3️⃣ اضافه کردن ستون تاریخ - با اولویت بالا
    # =====================================================
    data_dict = {}
    date_added = False

    if date_column_name and date_data:
        logger.info(f"[{split_name.upper()}]   📅 Adding date column '{date_column_name}'")

        if split_name in date_data:
            dates = date_data[split_name]
            n_samples = len(X_df)

            logger.info(f"[{split_name.upper()}]     • X samples: {n_samples}")
            logger.info(f"[{split_name.upper()}]     • Available dates: {len(dates)}")
            logger.info(f"[{split_name.upper()}]     • Date type: {type(dates)}")

            # استراتژی تطبیق برای train/val/test
            if split_name == 'train':
                # برای train: از ابتدای تاریخ‌ها استفاده کن
                if hasattr(dates, 'iloc'):
                    date_values = dates.iloc[:n_samples].values
                else:
                    date_values = dates[:n_samples]
                logger.info(f"[{split_name.upper()}]     • Using first {n_samples} dates for {split_name}")
            else:
                # برای val/test: از انتهای تاریخ‌ها استفاده کن
                if len(dates) >= n_samples:
                    start_idx = len(dates) - n_samples
                    if hasattr(dates, 'iloc'):
                        date_values = dates.iloc[start_idx:].values
                    else:
                        date_values = dates[start_idx:]
                    logger.info(f"[{split_name.upper()}]     • Using last {n_samples} dates for {split_name}")
                else:
                    # اگر تعداد تاریخ‌ها کمتر است، از ابتدا استفاده کن
                    if hasattr(dates, 'iloc'):
                        date_values = dates.iloc[:n_samples].values
                    else:
                        date_values = dates[:n_samples]
                    logger.warning(f"[{split_name.upper()}]     ⚠️ Not enough dates, using first {n_samples}")

            # تبدیل به فرمت مناسب برای HDF5
            if hasattr(date_values, 'dtype') and pd.api.types.is_datetime64_any_dtype(date_values):
                date_values = date_values.astype('int64')
                logger.info(f"[{split_name.upper()}]     • Converted datetime to int64 for HDF5")

            # اضافه کردن به دیکشنری
            data_dict[date_column_name] = date_values
            date_added = True
            logger.info(f"[{split_name.upper()}]     ✅ Date column added successfully")

            if len(date_values) > 0:
                logger.info(f"[{split_name.upper()}]     • First date: {date_values[0]}")
                if len(date_values) > 1:
                    logger.info(f"[{split_name.upper()}]     • Last date: {date_values[-1]}")
        else:
            logger.error(f"[{split_name.upper()}]     ❌ Split '{split_name}' not found in date_data")
            logger.info(f"[{split_name.upper()}]     Available splits: {list(date_data.keys())}")
    else:
        logger.warning(f"[{split_name.upper()}]   ⚠️ No date data available")
        if not date_column_name:
            logger.warning(f"[{split_name.upper()}]     date_column_name is None")
        if not date_data:
            logger.warning(f"[{split_name.upper()}]     date_data is None")

    # =====================================================
    # 4️⃣ اضافه کردن X features
    # =====================================================
    feature_count = 0
    for col in X_df.columns:
        data_dict[col] = X_df[col].values.astype(np.float32)
        feature_count += 1

    logger.info(f"[{split_name.upper()}]   ✅ Added {feature_count} feature columns")

    # =====================================================
    # 5️⃣ اضافه کردن y targets
    # =====================================================
    if not y_df.empty:
        target_count = 0
        for col in y_df.columns:
            data_dict[col] = y_df[col].values.astype(np.float32)
            target_count += 1
        logger.info(f"[{split_name.upper()}]   ✅ Added {target_count} target columns")

    # =====================================================
    # 6️⃣ اعتبارسنجی نهایی
    # =====================================================
    logger.info(f"[{split_name.upper()}]   📊 Final dictionary has {len(data_dict)} keys")

    # بررسی تطابق طول همه آرایه‌ها
    lengths = [len(v) for v in data_dict.values()]
    unique_lengths = set(lengths)

    if len(unique_lengths) == 1:
        logger.info(f"[{split_name.upper()}]   ✅ All arrays have same length: {lengths[0]}")
    else:
        logger.error(f"[{split_name.upper()}]   ❌ Inconsistent lengths: {unique_lengths}")
        # تلاش برای اصلاح
        target_len = max(set(lengths), key=lengths.count)
        logger.warning(f"[{split_name.upper()}]   ⚠️ Attempting to fix by trimming to length {target_len}")
        for key, value in data_dict.items():
            if len(value) != target_len:
                data_dict[key] = value[:target_len]
        logger.info(f"[{split_name.upper()}]   ✅ Fixed all arrays to length {target_len}")

    # بررسی نهایی ستون تاریخ
    if date_column_name and date_column_name in data_dict:
        logger.info(f"[{split_name.upper()}]   ✅ Date column '{date_column_name}' preserved in final dict")
        logger.info(f"[{split_name.upper()}]     Final date count: {len(data_dict[date_column_name])}")
        if len(data_dict[date_column_name]) > 0:
            logger.info(f"[{split_name.upper()}]     Final first date: {data_dict[date_column_name][0]}")
            if len(data_dict[date_column_name]) > 1:
                logger.info(f"[{split_name.upper()}]     Final last date: {data_dict[date_column_name][-1]}")
    elif date_column_name:
        logger.error(f"[{split_name.upper()}]   ❌ Date column '{date_column_name}' NOT in final dict!")

    logger.info(f"[{split_name.upper()}] {'=' * 60}")
    logger.info(f"[{split_name.upper()}] ✅ Final dict preparation complete for {split_name}")
    logger.info(f"[{split_name.upper()}] {'=' * 60}")

    return data_dict



async def save_model_dims(
        symbol: str,
        fold: int,
        lookback: int = None,
        features: int = None,
        hidden_size: int = None,
        metadata: Dict = None
) -> bool:
    """
    💾 VERSION 5.1 - COMPLETE FIX FOR FEATURE EXTRACTION
    ✅ FIX: Support for X_seq key in H5 files (primary key in your files)
    ✅ FIX: Support for both with-dot and without-dot symbol names
    ✅ FIX: Multiple fallback methods for feature extraction
    ✅ FIX: Complete logging for each extraction attempt
    """
    
    start_time = time.time()
    
    # نرمالایز کردن نماد برای مسیر
    def normalize_symbol_for_path(s: str) -> str:
        if not s:
            return s
        return s.replace('.', '')
    
    clean_symbol = normalize_symbol_for_path(symbol)
    
    logger.info(f"[{clean_symbol}] 🚀 ====== SAVE MODEL DIMS PHASE {fold} =====")
    logger.info(f"[{clean_symbol}] 📍 Starting save_model_dims for fold {fold}")

    extraction_methods = []

    try:
        # ==================== فاز 1: اعتبارسنجی ====================
        if not symbol or not isinstance(symbol, str):
            logger.error(f"[{clean_symbol}] ❌ Invalid symbol: {symbol}")
            return False

        if fold <= 0:
            logger.error(f"[{clean_symbol}] ❌ Invalid fold: {fold}")
            return False

        # ==================== فاز 2: تشخیص تایم‌فریم ====================
        timeframe = "1H"
        try:
            active_symbol, tf_raw = get_active_symbol_timeframe()
            if tf_raw:
                if isinstance(tf_raw, dict):
                    timeframe = tf_raw.get('timeframe', '1H')
                elif isinstance(tf_raw, str):
                    timeframe = tf_raw
        except Exception as tf_err:
            logger.warning(f"[{clean_symbol}] ⚠️ Using default timeframe: {timeframe}")

        extraction_methods.append(f"timeframe: {timeframe}")

        # ==================== فاز 3: استخراج LOOKBACK ====================
        actual_lookback = lookback
        lookback_source = "parameter"

        # روش 1: از فایل sequenced (اولویت اول)
        if actual_lookback is None:
            # ✅ تلاش با نام نرمالایز شده (بدون نقطه)
            h5_dir = Path(f"data/sequenced/{clean_symbol}/{timeframe}/fold_{fold}")
            
            logger.info(f"[{clean_symbol}] 🔍 Searching sequenced files in: {h5_dir}")
            
            if h5_dir.exists():
                # الگوهای مختلف نام فایل
                search_patterns = [
                    f"{clean_symbol}_{timeframe}_fold{fold}_train_sequenced*.h5",
                    f"{clean_symbol}_{timeframe}_fold{fold}_train*.h5",
                    f"*train_sequenced*.h5",
                    f"*sequenced*.h5"
                ]
                
                h5_files = []
                for pattern in search_patterns:
                    matches = list(h5_dir.glob(pattern))
                    if matches:
                        h5_files.extend(matches)
                        logger.info(f"[{clean_symbol}] Found {len(matches)} files matching pattern: {pattern}")
                
                # همچنین با نام اصلی (با نقطه) جستجو کن
                original_symbol = symbol.replace('.', '')
                if original_symbol != clean_symbol:
                    alt_pattern = f"{original_symbol}_{timeframe}_fold{fold}_train_sequenced*.h5"
                    alt_files = list(Path(f"data/sequenced/{original_symbol}/{timeframe}/fold_{fold}").glob(alt_pattern))
                    if alt_files:
                        h5_files.extend(alt_files)
                        logger.info(f"[{clean_symbol}] Found {len(alt_files)} files with original name: {original_symbol}")
                
                for h5_file in h5_files:
                    import re
                    match = re.search(r'lookback(\d+)', str(h5_file))
                    if match:
                        actual_lookback = int(match.group(1))
                        lookback_source = f"filename: {h5_file.name}"
                        extraction_methods.append(f"lookback: {actual_lookback} from filename")
                        logger.info(f"[{clean_symbol}] ✅ Found lookback={actual_lookback} from {h5_file.name}")
                        break
                    
                    # روش جایگزین: خواندن از H5
                    if actual_lookback is None:
                        try:
                            import h5py
                            with h5py.File(h5_file, 'r') as f:
                                # ✅ اولویت 1: X_seq
                                if 'X_seq' in f:
                                    shape = f['X_seq'].shape
                                    if len(shape) >= 2:
                                        actual_lookback = shape[1]
                                        lookback_source = f"X_seq.shape[1] from {h5_file.name}"
                                        extraction_methods.append(f"lookback: {actual_lookback} from X_seq.shape")
                                        logger.info(f"[{clean_symbol}] ✅ Found lookback={actual_lookback} from X_seq in {h5_file.name}")
                                        break
                                # ✅ اولویت 2: X
                                elif 'X' in f:
                                    shape = f['X'].shape
                                    if len(shape) >= 2:
                                        actual_lookback = shape[1]
                                        lookback_source = f"X.shape[1] from {h5_file.name}"
                                        extraction_methods.append(f"lookback: {actual_lookback} from X.shape")
                                        logger.info(f"[{clean_symbol}] ✅ Found lookback={actual_lookback} from X in {h5_file.name}")
                                        break
                        except Exception as e:
                            logger.debug(f"[{clean_symbol}] Could not read lookback from H5: {e}")
            else:
                logger.warning(f"[{clean_symbol}] ⚠️ Sequenced directory does not exist: {h5_dir}")

        # مقدار پیش‌فرض
        if actual_lookback is None:
            actual_lookback = 190
            lookback_source = "default"
            extraction_methods.append(f"lookback: {actual_lookback} (default)")
            logger.warning(f"[{clean_symbol}] ⚠️ Using default lookback: {actual_lookback}")

        # ==================== فاز 4: استخراج FEATURES ====================
        actual_features = features
        features_source = "parameter"
        feature_names = []

        # روش 1: از فایل sequenced با کلید X_seq (اولویت اول)
        if actual_features is None or actual_features == 0:
            h5_dir = Path(f"data/sequenced/{clean_symbol}/{timeframe}/fold_{fold}")
            logger.info(f"[{clean_symbol}] 🔍 Searching for feature count in: {h5_dir}")
            
            if h5_dir.exists():
                search_patterns = [
                    f"{clean_symbol}_{timeframe}_fold{fold}_train_sequenced*.h5",
                    f"{clean_symbol}_{timeframe}_fold{fold}_train*.h5",
                    f"*train_sequenced*.h5"
                ]
                
                train_files = []
                for pattern in search_patterns:
                    matches = list(h5_dir.glob(pattern))
                    if matches:
                        train_files.extend(matches)
                
                # همچنین با نام اصلی
                original_symbol = symbol.replace('.', '')
                if original_symbol != clean_symbol:
                    alt_dir = Path(f"data/sequenced/{original_symbol}/{timeframe}/fold_{fold}")
                    if alt_dir.exists():
                        alt_files = list(alt_dir.glob(f"{original_symbol}_{timeframe}_fold{fold}_train_sequenced*.h5"))
                        train_files.extend(alt_files)
                
                for h5_path in train_files:
                    try:
                        import h5py
                        with h5py.File(h5_path, 'r') as f:
                            logger.info(f"[{clean_symbol}] 📂 Checking file: {h5_path.name}")
                            logger.info(f"[{clean_symbol}]    Available keys: {list(f.keys())}")
                            
                            # ✅ اولویت 1: X_seq (کلید اصلی در فایل شما)
                            if 'X_seq' in f:
                                shape = f['X_seq'].shape
                                logger.info(f"[{clean_symbol}]    X_seq.shape = {shape}")
                                if len(shape) >= 3:
                                    actual_features = shape[2]
                                    features_source = f"X_seq[2] from {h5_path.name}"
                                    extraction_methods.append(f"features: {actual_features} from X_seq[2]")
                                    logger.info(f"[{clean_symbol}] ✅✅✅ {actual_features} features from X_seq[2]")
                                    break
                            
                            # ✅ اولویت 2: X
                            if actual_features is None and 'X' in f:
                                shape = f['X'].shape
                                logger.info(f"[{clean_symbol}]    X.shape = {shape}")
                                if len(shape) >= 3:
                                    actual_features = shape[2]
                                    features_source = f"X[2] from {h5_path.name}"
                                    extraction_methods.append(f"features: {actual_features} from X[2]")
                                    logger.info(f"[{clean_symbol}] ✅✅✅ {actual_features} features from X[2]")
                                    break
                                elif len(shape) == 2:
                                    # اگر X 2 بعدی است (n_samples, n_features)
                                    actual_features = shape[1]
                                    features_source = f"X[1] from {h5_path.name}"
                                    extraction_methods.append(f"features: {actual_features} from X[1]")
                                    logger.info(f"[{clean_symbol}] ✅✅✅ {actual_features} features from X[1]")
                                    break
                            
                            # ✅ اولویت 3: data
                            if actual_features is None and 'data' in f:
                                shape = f['data'].shape
                                logger.info(f"[{clean_symbol}]    data.shape = {shape}")
                                if len(shape) >= 3:
                                    actual_features = shape[2]
                                    features_source = f"data[2] from {h5_path.name}"
                                    extraction_methods.append(f"features: {actual_features} from data[2]")
                                    logger.info(f"[{clean_symbol}] ✅✅✅ {actual_features} features from data[2]")
                                    break
                    except Exception as e:
                        logger.warning(f"[{clean_symbol}] ⚠️ Error reading {h5_path.name}: {e}")

        # روش 2: از فایل normalized
        if actual_features is None or actual_features == 0:
            norm_dir = Path(f"data/normalized/{clean_symbol}/{timeframe}/fold_{fold}")
            logger.info(f"[{clean_symbol}] 🔍 Looking for feature count in normalized: {norm_dir}")
            
            if norm_dir.exists():
                norm_patterns = [
                    f"{clean_symbol}_{timeframe}_fold{fold}_train_normalized*.h5",
                    f"{clean_symbol}_{timeframe}_fold{fold}_train*.h5",
                    f"*train_normalized*.h5"
                ]
                
                norm_files = []
                for pattern in norm_patterns:
                    matches = list(norm_dir.glob(pattern))
                    if matches:
                        norm_files.extend(matches)
                
                for norm_path in norm_files:
                    try:
                        import h5py
                        with h5py.File(norm_path, 'r') as f:
                            logger.info(f"[{clean_symbol}] 📂 Checking normalized file: {norm_path.name}")
                            logger.info(f"[{clean_symbol}]    Available keys: {list(f.keys())}")
                            
                            if 'X' in f:
                                shape = f['X'].shape
                                if len(shape) >= 2:
                                    actual_features = shape[1]
                                    features_source = f"normalized X[1] from {norm_path.name}"
                                    extraction_methods.append(f"features: {actual_features} from normalized X")
                                    logger.info(f"[{clean_symbol}] ✅ {actual_features} features from normalized X")
                                    break
                    except Exception as e:
                        logger.debug(f"[{clean_symbol}] Could not read normalized: {e}")

        # روش 3: از فایل processed
        if actual_features is None or actual_features == 0:
            proc_dir = Path(f"data/processed/{clean_symbol}/{timeframe}/fold_{fold}")
            logger.info(f"[{clean_symbol}] 🔍 Looking for feature count in processed: {proc_dir}")
            
            if proc_dir.exists():
                proc_patterns = [
                    f"{clean_symbol}_{timeframe}_fold{fold}_train_processed*.h5",
                    f"{clean_symbol}_{timeframe}_fold{fold}_train*.h5",
                    f"*train_processed*.h5"
                ]
                
                proc_files = []
                for pattern in proc_patterns:
                    matches = list(proc_dir.glob(pattern))
                    if matches:
                        proc_files.extend(matches)
                
                for proc_path in proc_files:
                    try:
                        import h5py
                        with h5py.File(proc_path, 'r') as f:
                            logger.info(f"[{clean_symbol}] 📂 Checking processed file: {proc_path.name}")
                            logger.info(f"[{clean_symbol}]    Available keys: {list(f.keys())}")
                            
                            if 'X' in f:
                                shape = f['X'].shape
                                if len(shape) >= 2:
                                    actual_features = shape[1]
                                    features_source = f"processed X[1] from {proc_path.name}"
                                    extraction_methods.append(f"features: {actual_features} from processed X")
                                    logger.info(f"[{clean_symbol}] ✅ {actual_features} features from processed X")
                                    break
                    except Exception as e:
                        logger.debug(f"[{clean_symbol}] Could not read processed: {e}")

        # ❌ عدم موفقیت در استخراج features
        if actual_features is None or actual_features == 0:
            logger.error(f"[{clean_symbol}] ❌❌ Could not extract features for fold {fold}")
            logger.error(f"[{clean_symbol}]    Extraction attempts: {extraction_methods}")
            logger.error(f"[{clean_symbol}]    Checked directories:")
            logger.error(f"[{clean_symbol}]       - Sequenced: data/sequenced/{clean_symbol}/{timeframe}/fold_{fold}")
            logger.error(f"[{clean_symbol}]       - Normalized: data/normalized/{clean_symbol}/{timeframe}/fold_{fold}")
            logger.error(f"[{clean_symbol}]       - Processed: data/processed/{clean_symbol}/{timeframe}/fold_{fold}")
            return False

        # ==================== فاز 5: تعیین HIDDEN_SIZE ====================
        actual_hidden_size = hidden_size
        if actual_hidden_size is None:
            if actual_features:
                if actual_features <= 50:
                    actual_hidden_size = 64
                elif actual_features <= 100:
                    actual_hidden_size = 128
                elif actual_features <= 200:
                    actual_hidden_size = 256
                else:
                    actual_hidden_size = 512
            else:
                actual_hidden_size = 128
            extraction_methods.append(f"hidden_size: {actual_hidden_size}")

        # ==================== فاز 6: ساخت PAYLOAD و ذخیره ====================
        input_shape = [int(actual_lookback), int(actual_features)]

        payload = {
            "symbol": clean_symbol,
            "original_symbol": symbol,
            "fold": fold,
            "timeframe": timeframe,
            "lookback": int(actual_lookback),
            "features": int(actual_features),
            "hidden_size": int(actual_hidden_size),
            "dimensions": {
                "input_shape": input_shape,
                "total_input_elements": input_shape[0] * input_shape[1],
                "lookback_steps": int(actual_lookback),
                "feature_dimension": int(actual_features)
            },
            "extraction": {
                "lookback": {"value": int(actual_lookback), "source": lookback_source},
                "features": {"value": int(actual_features), "source": features_source},
                "hidden_size": {"value": int(actual_hidden_size), "source": "dynamic_calculation" if hidden_size is None else "parameter"}
            },
            "extraction_methods": extraction_methods,
            "metadata": metadata or {},
            "timestamps": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "extraction_duration": time.time() - start_time
            }
        }

        dims_dir = Path("cache") / "model_dims"
        dims_dir.mkdir(parents=True, exist_ok=True)

        file_path = dims_dir / f"{clean_symbol}_{timeframe}_fold{fold}_dims.json"
        
        # ذخیره اتمی
        temp_file = file_path.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            
            # جایگزینی اتمی
            if sys.platform == 'win32':
                try:
                    os.remove(file_path)
                except FileNotFoundError:
                    pass
                os.rename(temp_file, file_path)
            else:
                os.replace(temp_file, file_path)
            
            logger.info(f"[{clean_symbol}] ✅ Model dimensions saved: lookback={actual_lookback}, features={actual_features}, hidden_size={actual_hidden_size}")
            logger.info(f"[{clean_symbol}]    File: {file_path}")
            return True
            
        except Exception as write_error:
            logger.error(f"[{clean_symbol}] ❌ Error writing file: {write_error}")
            if temp_file.exists():
                temp_file.unlink()
            return False

    except Exception as e:
        logger.error(f"[{clean_symbol}] ❌ Critical error in save_model_dims: {e}", exc_info=True)
        return False
        
import pandas as pd
import numpy as np
import json
from pathlib import Path
import asyncio
import gc
import re
import uuid




import asyncio
import uuid
import time
import gc
import re
import h5py

MAX_RETRIES = 5


async def _save_split(
        data_type: str,
        symbol: str,
        timeframe: str,
        fold: int,
        split_type: str,
        data,
        metadata: Dict = None
) -> bool:

    try:
        base_dir = Path("data") / data_type / symbol / timeframe / f"fold_{fold}"
        base_dir.mkdir(parents=True, exist_ok=True)

        final_path = base_dir / f"{symbol}_{timeframe}_fold{fold}_{split_type}_{data_type}.h5"

        # =====================================================
        # 1️⃣ SANITIZE + NUMERIC ONLY
        # =====================================================
        safe_cols = [
            re.sub(r"[^0-9a-zA-Z_]", "_", str(c))
            for c in data.columns
        ]

        data = data.copy()
        data.columns = safe_cols
        data = data.select_dtypes(include=[np.number])

        matrix = data.to_numpy(dtype=np.float32, copy=False)

        # =====================================================
        # 2️⃣ ATOMIC TMP PATH (unique)
        # =====================================================
        tmp_path = base_dir / f".tmp_{uuid.uuid4().hex}.h5"

        # =====================================================
        # 3️⃣ RETRY LOOP (permission safe)
        # =====================================================
        for attempt in range(MAX_RETRIES):

            try:
                # ---------- write tmp ----------
                with h5py.File(tmp_path, "w") as hf:

                    hf.create_dataset(
                        "X",
                        data=matrix,
                        compression="gzip",
                        compression_opts=4,
                        chunks=True
                    )

                    hf.attrs["columns"] = json.dumps(list(data.columns))
                    hf.attrs["shape"] = matrix.shape
                    hf.attrs["symbol"] = symbol
                    hf.attrs["timeframe"] = timeframe
                    hf.attrs["fold"] = fold
                    hf.attrs["split_type"] = split_type
                    hf.attrs["data_type"] = data_type

                    if metadata:
                        hf.attrs["metadata"] = json.dumps(metadata, default=str)

                # force release
                gc.collect()

                # ---------- atomic rename ----------
                tmp_path.replace(final_path)

                logger.info(f"[{symbol}] ✅ saved atomic | {split_type} | {matrix.shape}")
                return True

            except PermissionError:

                logger.warning(
                    f"[{symbol}] ⚠️ file locked (attempt {attempt+1}/{MAX_RETRIES})"
                )

                await asyncio.sleep(0.3 * (attempt + 1))
                gc.collect()

            except Exception as e:
                logger.error(f"[{symbol}] ❌ save crash: {e}", exc_info=True)
                return False

        return False

    except Exception as e:
        logger.error(f"[{symbol}] ❌ split fatal: {e}", exc_info=True)
        return False


async def _save_processed_data(
        data_type: str,
        symbol: str,
        timeframe: str,
        fold: int,
        data,
        metadata: Dict = None,
        split_type: str = None
) -> bool:
    """
    📦 Smart universal saver
    raw -> no split
    processed/normalized/sequenced -> require split

    ✅ FIX v3.0 (round-trip با loader):
        شاخهٔ raw قبلاً ایندکس را به‌صورت دیتاست سطح-بالا (``index``) می‌نوشت،
        در حالی‌که ``_load_new_hdf5_format`` ساختار گروه ``index`` با دیتاست
        ``values`` انتظار دارد؛ نتیجه: فایل raw این تابع قابل خواندن برگشت نبود
        (KeyError('values') → universal_load برمی‌گرداند None). این نسخه دقیقاً
        همان قرارداد فرمت v3 تابع ``_sync_save`` را تولید می‌کند:
            group 'data' (dataset به‌ازای هر ستون + attrs)
            group 'index' (dataset 'values' + attrs)
            attrs: columns / shape / format_version / datetime_columns / metadata
        و ستون‌های datetime را به int64(ns) تبدیل می‌کند تا round-trip کامل شود.
    """
    try:
        logger.info(f"[{symbol}] 💾 شروع ذخیره‌سازی | نوع: {data_type} | split: {split_type}")

        if data_type == "raw":
            base_dir = Path("data/raw/mt5")
            base_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{symbol}_{timeframe}_fold{fold}_raw.h5"
            file_path = base_dir / filename

            logger.info(f"[{symbol}] 💾 ذخیره RAW → {filename}")

            import h5py as _h5py_for_raw_save

            # ---------- آماده‌سازی DataFrame ----------
            if isinstance(data, pd.DataFrame):
                df = data.copy()
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                logger.error(f"[{symbol}] ❌ Unsupported raw data type: {type(data).__name__}")
                return False

            datetime_columns: List[str] = []

            with _h5py_for_raw_save.File(file_path, "w") as hf:
                data_grp = hf.create_group("data")

                for col in df.columns:
                    col_data = df[col]

                    if is_datetime64_any_dtype(col_data):
                        datetime_columns.append(col)
                        try:
                            tz_info = getattr(col_data.dt, "tz", None)
                        except Exception:
                            tz_info = None
                        if tz_info is not None:
                            col_data = col_data.dt.tz_convert("UTC").dt.tz_localize(None)

                        values = col_data.astype("int64").to_numpy()
                    else:
                        values = col_data.to_numpy()

                    dset = data_grp.create_dataset(
                        col,
                        data=values,
                        compression="gzip",
                        compression_opts=6,
                        shuffle=True,
                    )
                    dset.attrs["original_dtype"] = str(df[col].dtype)
                    if col in datetime_columns:
                        dset.attrs["is_datetime"] = True
                        dset.attrs["datetime_unit"] = "ns"

                # ---------- this is the loader-compatible 'index' group ----------
                index_grp = hf.create_group("index")

                idx = df.index
                if is_datetime64_any_dtype(idx):
                    index_values = idx.asi8
                    index_is_datetime = True
                else:
                    index_values = np.asarray(idx)
                    index_is_datetime = False

                index_grp.create_dataset("values", data=index_values)
                index_grp.attrs["dtype"] = str(idx.dtype)
                index_grp.attrs["is_datetime"] = index_is_datetime
                index_grp.attrs["datetime_unit"] = "ns"

                # ---------- metadata / attrs ----------
                hf.attrs["columns"] = json.dumps(list(df.columns))
                hf.attrs["shape"] = json.dumps(list(df.shape))
                hf.attrs["format_version"] = "3.0"
                hf.attrs["datetime_columns"] = json.dumps(datetime_columns)
                hf.attrs["symbol"] = str(symbol)
                hf.attrs["timeframe"] = str(timeframe)
                hf.attrs["fold"] = str(fold)
                hf.attrs["data_type"] = "raw"
                hf.attrs["saved_at"] = datetime.now(timezone.utc).isoformat()

                if metadata:
                    hf.attrs["metadata"] = json.dumps(metadata, default=str)

                hf.flush()

            return file_path.exists() and file_path.stat().st_size > 0

        else:
            # 🔹 PROCESSED/NORMALIZED/SEQUENCED: نیازمند split
            if not split_type:
                logger.error(f"❌ {data_type} نیاز به split_type دارد")
                return False

            return await _save_split(
                data_type=data_type,
                symbol=symbol,
                timeframe=timeframe,
                fold=fold,
                split_type=split_type,
                data=data,
                metadata=metadata
            )

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطا در _save_processed_data: {e}", exc_info=True)
        return False




async def _discover_existing_files(
        symbol: str,
        timeframe: str,
        fold: int,
        data_type: str,
        split_type: str = None
) -> List[Path]:
    """
    🔍 کشف خودکار فایل‌های موجود با الگوهای مختلف
    """
    import re
    from pathlib import Path

    discovered_files = []

    # 🔥 الگوهای جستجوی اصلی
    search_patterns = []

    # ساختارهای مختلف دایرکتوری
    base_dirs = [
        Path("data") / data_type,
        Path("data") / data_type / symbol / timeframe,
        Path("data") / data_type / symbol,
        Path("data") / data_type / f"{symbol}_{timeframe}",
    ]

    if fold is not None:
        base_dirs.extend([
            Path("data") / data_type / symbol / timeframe / f"fold_{fold}",
            Path("data") / data_type / symbol / f"fold_{fold}",
        ])

    # 🔥 الگوهای نام فایل
    filename_patterns = []

    if split_type:
        # الگوهای split-specific
        filename_patterns.extend([
            f"{symbol}_{timeframe}_fold{fold}_{split_type}_{data_type}.h5",
            f"{symbol}_{timeframe}_fold{fold}_{split_type}.h5",
            f"{symbol}_fold{fold}_{split_type}_{data_type}.h5",
        ])
    else:
        # الگوهای عمومی
        filename_patterns.extend([
            f"{symbol}_{timeframe}_fold{fold}_{data_type}.h5",
            f"{symbol}_{timeframe}_fold{fold}.h5",
            f"{symbol}_fold{fold}_{data_type}.h5",
            f"{symbol}_{timeframe}_{data_type}_fold{fold}.h5",
        ])

    # جستجو در همه ترکیبات
    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        for pattern in filename_patterns:
            try:
                # جستجوی مستقیم
                file_path = base_dir / pattern
                if file_path.exists() and file_path.stat().st_size > 100:
                    discovered_files.append(file_path)
                    logger.debug(f"✅ فایل یافت شد: {file_path}")

                # جستجوی با wildcard
                if "*" in pattern:
                    for matched_file in base_dir.glob(pattern):
                        if matched_file.exists() and matched_file.stat().st_size > 100:
                            discovered_files.append(matched_file)
                            logger.debug(f"✅ فایل (wildcard) یافت شد: {matched_file}")
            except Exception as e:
                logger.debug(f"⚠️ خطا در بررسی الگو {pattern}: {e}")


    # حذف تکراری‌ها و مرتب‌سازی
    if discovered_files:
        # حذف duplicate بر اساس مسیر واقعی
        unique_files = []
        seen_paths = set()

        for file_path in discovered_files:
            real_path = str(file_path.resolve())
            if real_path not in seen_paths:
                seen_paths.add(real_path)
                unique_files.append(file_path)

        # مرتب‌سازی: اول فایل‌های کامل، سپس جدیدترین
        unique_files.sort(key=lambda x: (
            0 if "train" in x.name.lower() else 1 if "val" in x.name.lower() else 2 if "test" in x.name.lower() else 3,
            x.stat().st_mtime if x.exists() else 0
        ), reverse=True)

        logger.info(f"🔍 تعداد {len(unique_files)} فایل {data_type} برای fold {fold} یافت شد")
        for i, file_path in enumerate(unique_files[:3]):  # فقط 3 فایل اول
            file_size = file_path.stat().st_size / 1024 / 1024
            logger.info(f"  {i + 1}. {file_path.name} ({file_size:.2f} MB)")

    return discovered_files


async def load_processed_data(
        data_type: str,
        symbol: str,
        timeframe: str,
        fold: int,
        split_type: str = None
) -> Optional[Union[pd.DataFrame, Tuple, Dict, np.ndarray]]:

    try:
        logger.info(f"📥 شروع بارگذاری {data_type} | {symbol} - {timeframe} - fold={fold}")


        data_dir = Path("data")
        folder_map = {
            "raw": data_dir / "raw" / "mt5",
            "processed": data_dir / "processed",
            "normalized": data_dir / "normalized",
        }

        if data_type not in folder_map:
            logger.error(f"❌ نوع داده پردازش شده ناشناخته: {data_type}")
            return None

        base_dir = folder_map[data_type]

        # 🔍 جستجوی فایل
        base_name = f"{symbol}_{timeframe}_fold{fold}_{data_type}"
        search_patterns = [f"{base_name}.h5"]

        data_file = None
        for pattern in search_patterns:
            files = list(base_dir.glob(pattern))
            if files:
                data_file = files[0]
                break

        if not data_file or not data_file.exists():
            logger.warning(f"⚠️ فایل {data_type} برای {symbol} - {timeframe} - fold={fold} یافت نشد")
            return None

        # بارگذاری فایل
        file_ext = data_file.suffix.lower()

        if file_ext in ['.h5', '.hdf5']:
            return await _load_hdf5(data_file, data_type, symbol, timeframe, fold)
        else:
            logger.warning(f"⚠️ فرمت فایل پشتیبانی نمی‌شود: {file_ext}")
            return None

    except Exception as e:
        logger.error(f"❌ خطا در load_processed_data برای {symbol} fold={fold}: {e}")
        return None


def check_sequenced_files_structure(symbol: str, timeframe: str, fold: int, logger=None) -> bool:
    """
    🔍 بررسی صحت و انسجام فایل‌های sequenced (پشتیبانی صحیح از چندتارگتی)

    ✅ FIX v2.0 (شمارش درست تارگت‌ها):
        نسخهٔ قبلی فقط کلیدهایی با پیشوند ``y_`` یا کلیدهای legacy
        ``y_seq/y/target`` را می‌شمرد؛ عملاً با فایل‌های استاندارد امروزی که
        پنج کلید مستقیم ``price_return/price_direction/trend_strength/
        volatility_regime/market_regime`` دارند، «1 تارگت» گزارش می‌کرد (آن
        یک فقط 'y' بود) و این لاگ گمراه‌کننده‌ای برای ممیزی چندتارگتی بود.
        در این نسخه هر سه خانوادهٔ کلید شمرده می‌شوند:
          1) کلیدهای مستقیم EXPECTED_TARGETS
          2) کلیدهای y_<name>
          3) کلیدهای legacy (y_seq, y, target)

    Returns:
        bool: True اگر هر ۳ split فایل سالم با X معتبر داشته باشند.
    """
    log = logger or logging.getLogger(__name__)

    try:
        if not H5PY_AVAILABLE:
            log.warning(
                f"[{symbol}] ❌ h5py در دسترس نیست؛ امکان بررسی ساختار sequenced وجود ندارد"
            )
            return False

        base_dir = Path("data/sequenced") / symbol / timeframe / f"fold_{fold}"
        if not base_dir.exists():
            log.warning(f"[{symbol}] ❌ پوشه sequenced وجود ندارد: {str(base_dir)}")
            return False

        expected_targets = tuple(PROJECT_TARGETS)
        x_candidates = ("X_seq", "X", "data")
        legacy_target_keys = {"y_seq", "y", "target"}

        splits = ["train", "val", "test"]
        success = True

        for split in splits:
            files = list(base_dir.glob(f"*{split}*sequenced*.h5"))
            if not files:
                success = False
                log.warning(f"[{symbol}] ❌ {split}: فایل یافت نشد")
                continue

            file = files[0]
            try:
                with h5py.File(file, 'r') as hf:
                    # ---------- یافتن X ----------
                    X_shape = None
                    for x_key in x_candidates:
                        if x_key in hf:
                            X_shape = hf[x_key].shape
                            break

                    if X_shape is None:
                        success = False
                        log.warning(
                            f"[{symbol}] ❌ {split}: هیچ‌کدام از کلیدهای X_seq/X/data "
                            f"در {file.name} یافت نشد | keys={list(hf.keys())}"
                        )
                        continue

                    # ---------- شمارش دقیق تارگت‌ها ----------
                    direct_targets = [k for k in hf.keys() if k in expected_targets]
                    prefixed_targets = [k for k in hf.keys() if k.startswith("y_")]
                    legacy_targets = [k for k in hf.keys() if k in legacy_target_keys]

                    total_targets = len(set(direct_targets)) + len(prefixed_targets) + len(legacy_targets)

                    if total_targets > 0:
                        log.info(
                            f"[{symbol}] ✅ {split}: X={X_shape}, {total_targets} تارگت در {file.name}"
                        )
                        if len(direct_targets) < len(expected_targets):
                            missing = sorted(set(expected_targets) - set(direct_targets))
                            log.info(
                                f"[{symbol}]    ⚠️ {split}: {len(direct_targets)}/{len(expected_targets)} "
                                f"تارگت مستقیم موجود است؛ موارد غایب: {missing}"
                            )
                    else:
                        log.info(
                            f"[{symbol}] ⚠️ {split}: X={X_shape} اما هیچ کلید تارگتی "
                            f"در {file.name} یافت نشد | keys={list(hf.keys())}"
                        )

            except Exception as e:
                success = False
                log.warning(
                    f"[{symbol}] ❌ {split}: خطا در خواندن {file.name} ({str(e)[:100]})"
                )

        return success

    except Exception as e:
        log.error(f"[{symbol}] ❌ خطا در بررسی ساختار: {str(e)}")
        return False

def audit_sequenced_target_scales(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    fold: Optional[int] = None,
    target_name: str = "price_return",
    lookback: Optional[int] = None,
    splits: Tuple[str, ...] = ("train", "val", "test"),
    sigma_ratio_warn: float = 2.0,
    sigma_ratio_fail: float = 5.0,
    sigma_abs_min: float = 1e-4,
    sigma_abs_max: float = 100.0,
    raise_on_mismatch: bool = False,
    logger_instance=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    ممیزی مقیاس تارگت بین splitهای sequenced — کاملاً dynamic.

    هدف:
        با یک نگاه، تأیید شود که train/val/test روی یک مقیاس واحد ذخیره
        شده‌اند. این تنها راهی است که «AI» آینده می‌تواند بدون حدس‌زدن
        تشخیص دهد داده مشکل دارد یا نه.

    منطق تصمیم (بدون فرض absolute-scale):
        ۱. σ هر split باید در بازهٔ [sigma_abs_min, sigma_abs_max] باشد.
           ⇒ اگر کسی داده را دوبار نرمال کند (σ≈1e-6)، یا مقیاس را فراموش
             کند (σ≈1e4)، این گیت فوراً می‌گیرد.
        ۲. نسبت σ_max/σ_min بین سه split باید < sigma_ratio_warn باشد.
           ⇒ این «قانون طلایی» است؛ فرقی نمی‌کند z-score باشد یا raw، نسبت
             باید نزدیک ۱ باشد چون تارگت مشترک از یک scaler واحد می‌آید.
        ۳. σ_max/σ_min بین sigma_ratio_warn و sigma_ratio_fail ⇒ warning.
        ۴. σ_max/σ_min > sigma_ratio_fail ⇒ fail.

    Dynamic resolution:
        • symbol/timeframe اگر None باشند از tf_decisions.json خوانده می‌شوند.
        • fold اگر None باشد از main.sum_folds گرفته می‌شود.
        • lookback اگر None باشد از نام فایل (regex lookback\\d+) استخراج می‌شود.
        • مسیر فایل با glob ساخته می‌شود؛ هیچ نام فایل هاردکد نمی‌شود.

    Args:
        symbol, timeframe, fold: اگر None، خودکار resolve می‌شوند.
        target_name: نام تارگت در فایل HDF5 (پیش‌فرض price_return).
        lookback: اگر داده شود، فقط فایل‌های با این lookback بررسی می‌شوند.
            اگر None، هر lookback موجود پذیرفته می‌شود.
        splits: لیست splitهایی که باید ممیزی شوند.
        sigma_ratio_warn / sigma_ratio_fail: آستانه‌های نسبت σ.
        sigma_abs_min / sigma_abs_max: بازهٔ مجاز مطلق σ برای هر split.
        raise_on_mismatch: اگر True و مشکل جدی دیده شد، RuntimeError پرتاب شود.
        logger_instance: override لاگر ماژول.
        verbose: چاپ جدول خلاصه در لاگ.

    Returns:
        dict با کلیدهای:
            ok (bool)               — verdict نهایی
            symbol, timeframe, fold
            target_name
            lookback_resolved
            seq_dir
            splits                  — {split: {path, n_finite, mean, std, ...}}
            sigma_min, sigma_max, sigma_ratio_max
            issues                  — لیست مشکلات
            recommendation          — متن پیشنهاد
    """
    log = logger_instance or logger

    # ── 0) resolve symbol/timeframe ──
    if symbol is None or timeframe is None:
        try:
            s_resolved, t_resolved = get_active_symbol_timeframe()
            if symbol is None:
                symbol = s_resolved
            if timeframe is None:
                timeframe = t_resolved
        except Exception as e:
            log.warning(
                f"[SCALE_AUDIT] active symbol/timeframe resolution failed: {e}"
            )

    if symbol is None or timeframe is None:
        raise ValueError(
            "[SCALE_AUDIT] symbol/timeframe could not be resolved — "
            "provide them explicitly or ensure tf_decisions.json exists."
        )

    # ── normalize symbol (dot-free) to match disk paths ──
    clean_symbol = str(symbol).replace(".", "")
    clean_timeframe = str(timeframe).upper()

    # ── 1) resolve fold ──
    if fold is None:
        try:
            fold = int(TOTAL_FOLDS)
        except Exception:
            fold = 5
    fold = int(fold)

    # ── 2) h5py availability ──
    try:
        import h5py as _h5py
    except ImportError:
        msg = "[SCALE_AUDIT] h5py is not installed — cannot audit sequenced files."
        log.error(msg)
        return {
            "ok": False,
            "symbol": clean_symbol,
            "timeframe": clean_timeframe,
            "fold": fold,
            "target_name": str(target_name),
            "issues": ["h5py_unavailable"],
            "recommendation": msg,
            "splits": {},
        }

    # ── 3) locate directory ──
    seq_dir = (
        Path("data") / "sequenced" / clean_symbol / clean_timeframe / f"fold_{fold}"
    )

    result: Dict[str, Any] = {
        "ok": False,
        "symbol": clean_symbol,
        "timeframe": clean_timeframe,
        "fold": fold,
        "target_name": str(target_name),
        "lookback_requested": lookback,
        "lookback_resolved": None,
        "seq_dir": str(seq_dir),
        "splits": {},
        "sigma_min": None,
        "sigma_max": None,
        "sigma_ratio_max": None,
        "issues": [],
        "recommendation": "",
    }

    if not seq_dir.exists():
        msg = f"[SCALE_AUDIT] sequenced directory not found: {seq_dir}"
        log.warning(msg)
        result["issues"].append("directory_not_found")
        result["recommendation"] = msg
        if raise_on_mismatch:
            raise FileNotFoundError(msg)
        return result

    # ── 4) discover and read each split ──
    discovered_lookbacks: Set[int] = set()

    for split in splits:
        # exact match if lookback was requested, otherwise any
        if lookback is not None:
            candidates = list(
                seq_dir.glob(f"*_{split}_sequenced_lookback{int(lookback)}.h5")
            )
            if not candidates:
                candidates = list(seq_dir.glob(f"*_{split}_sequenced*.h5"))
        else:
            candidates = list(seq_dir.glob(f"*_{split}_sequenced*.h5"))

        if not candidates:
            # looser fallback (some old layouts)
            candidates = list(seq_dir.glob(f"*{split}*sequenced*.h5"))

        if not candidates:
            result["splits"][split] = {
                "found": False,
                "error": "file_not_found",
            }
            result["issues"].append(f"{split}:file_not_found")
            continue

        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        file_path = candidates[0]

        # extract lookback from filename
        match_lb = re.search(r"lookback(\d+)", file_path.name)
        lb_from_name = int(match_lb.group(1)) if match_lb else None
        if lb_from_name is not None:
            discovered_lookbacks.add(lb_from_name)

        entry: Dict[str, Any] = {
            "found": True,
            "path": str(file_path),
            "lookback_from_name": lb_from_name,
        }

        try:
            with _h5py.File(file_path, "r") as hf:
                if target_name not in hf:
                    entry["error"] = "target_not_in_file"
                    entry["available_keys"] = list(hf.keys())
                    result["splits"][split] = entry
                    result["issues"].append(f"{split}:target_not_found")
                    continue

                dset = hf[target_name]
                arr = np.asarray(dset[:]).reshape(-1).astype(np.float64)
                finite = np.isfinite(arr)
                n_total = int(arr.size)
                n_finite = int(finite.sum())

                if n_finite == 0:
                    entry["error"] = "no_finite_values"
                    entry["n_total"] = n_total
                    result["splits"][split] = entry
                    result["issues"].append(f"{split}:no_finite_values")
                    continue

                arr_f = arr[finite]
                entry.update({
                    "n_total": n_total,
                    "n_finite": n_finite,
                    "mean": float(np.mean(arr_f)),
                    "std": float(np.std(arr_f)),
                    "min": float(np.min(arr_f)),
                    "max": float(np.max(arr_f)),
                    "dtype": str(dset.dtype),
                })

        except Exception as e:
            entry["error"] = f"read_failed:{type(e).__name__}"
            entry["error_detail"] = str(e)[:200]
            result["splits"][split] = entry
            result["issues"].append(f"{split}:read_failed")
            log.warning(
                f"[SCALE_AUDIT] could not read {file_path.name}: {e}"
            )
            continue

        result["splits"][split] = entry

    # ── 5) resolve lookback consistency ──
    if len(discovered_lookbacks) == 1:
        result["lookback_resolved"] = int(next(iter(discovered_lookbacks)))
    elif len(discovered_lookbacks) > 1:
        result["issues"].append(
            f"multiple_lookbacks_found:{sorted(discovered_lookbacks)}"
        )
        log.error(
            f"[SCALE_AUDIT] multiple lookbacks found across splits: "
            f"{sorted(discovered_lookbacks)} — data generation may be inconsistent."
        )

    # ── 6) collect valid splits ──
    valid_splits = [
        (s, e) for s, e in result["splits"].items()
        if e.get("found") and ("std" in e)
    ]

    if len(valid_splits) < 2:
        msg = (
            f"[SCALE_AUDIT] only {len(valid_splits)} valid split(s) — "
            f"cross-split check skipped."
        )
        log.warning(msg)
        result["recommendation"] = msg
        if verbose:
            log.info("=" * 78)
            log.info(msg)
            log.info("=" * 78)
        return result

    # ── 7) cross-split sigma statistics ──
    stds = [float(e["std"]) for _, e in valid_splits]
    result["sigma_min"] = float(min(stds))
    result["sigma_max"] = float(max(stds))
    result["sigma_ratio_max"] = float(
        max(stds) / max(min(stds), 1e-12)
    )

    # ── 8) apply checks ──
    issues: List[str] = list(result["issues"])
    hard_issues: List[str] = []

    for split_name, entry in valid_splits:
        s = float(entry["std"])
        if s < sigma_abs_min:
            hard_issues.append(
                f"{split_name}:sigma_too_small({s:.3e} < {sigma_abs_min})"
            )
        elif s > sigma_abs_max:
            hard_issues.append(
                f"{split_name}:sigma_too_large({s:.3e} > {sigma_abs_max})"
            )

    ratio = float(result["sigma_ratio_max"])
    if ratio > sigma_ratio_fail:
        hard_issues.append(f"sigma_ratio_critical({ratio:.2f}x)")
    elif ratio > sigma_ratio_warn:
        issues.append(f"sigma_ratio_high({ratio:.2f}x)")

    result["issues"] = issues + hard_issues
    result["ok"] = len(hard_issues) == 0

    # ── 9) self-describing log ──
    if verbose:
        log.info("=" * 78)
        log.info(
            f"[SCALE_AUDIT] target='{target_name}' | "
            f"{clean_symbol}/{clean_timeframe}/fold{fold}"
        )
        log.info(f"[SCALE_AUDIT] seq_dir = {seq_dir}")
        log.info(
            f"[SCALE_AUDIT] lookback: "
            f"requested={result['lookback_requested']} "
            f"resolved={result['lookback_resolved']}"
        )
        log.info("-" * 78)
        log.info(
            f"[SCALE_AUDIT] {'split':<8} {'n':>10} {'μ':>14} {'σ':>14} "
            f"{'min':>12} {'max':>12}  dtype"
        )
        log.info("-" * 78)

        for split in splits:
            e = result["splits"].get(split, {})
            if not e.get("found"):
                log.info(
                    f"[SCALE_AUDIT] {split:<8} — "
                    f"NOT FOUND ({e.get('error', 'unknown')})"
                )
                continue
            if "std" not in e:
                log.info(
                    f"[SCALE_AUDIT] {split:<8} — "
                    f"READ FAILED ({e.get('error', 'unknown')})"
                )
                continue
            log.info(
                f"[SCALE_AUDIT] {split:<8} "
                f"{e['n_finite']:>10,} "
                f"{e['mean']:>+14.6f} "
                f"{e['std']:>14.6f} "
                f"{e['min']:>+12.4f} "
                f"{e['max']:>+12.4f}  "
                f"{e['dtype']}"
            )

        log.info("-" * 78)
        log.info(
            f"[SCALE_AUDIT] σ ratio (max/min) = {result['sigma_ratio_max']:.3f}x "
            f"(warn>{sigma_ratio_warn}x, fail>{sigma_ratio_fail}x)"
        )

        if result["ok"]:
            log.info("[SCALE_AUDIT] verdict: ✅ OK")
        else:
            log.error("[SCALE_AUDIT] verdict: ❌ DATA SCALE PROBLEM DETECTED")

        if result["issues"]:
            log.warning(f"[SCALE_AUDIT] issues: {result['issues']}")
        log.info("=" * 78)

    # ── 10) recommendation ──
    if result["ok"]:
        result["recommendation"] = (
            f"[SCALE_AUDIT] ✅ All splits on consistent scale. "
            f"σ ratio {result['sigma_ratio_max']:.2f}x < {sigma_ratio_warn}x. "
            f"Safe to proceed with training/evaluation."
        )
    else:
        result["recommendation"] = (
            f"[SCALE_AUDIT] ❌ Data scale problem detected: {hard_issues}. "
            f"Regenerate sequenced files with a single scaler fitted on TRAIN."
        )
        log.error(result["recommendation"])

        if raise_on_mismatch:
            raise RuntimeError(result["recommendation"])

    return result

def get_scale_audit_reports(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    fold: Optional[int] = None,
    target_names: Optional[List[str]] = None,
    raise_on_mismatch: bool = False,
    verbose: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    GLOBAL FUNCTION — ممیزی مقیاس تارگت‌ها برای همهٔ تارگت‌های پروژه، به‌صورت lazy.

    این تابع جایگزین بلوک module-level نسخهٔ قبلی است که به‌شکل زیر در
    global_method.py قرار داشت::

        reports = {
            t: audit_sequenced_target_scales(target_name=t, verbose=False)
            for t in PROJECT_TARGETS
        }

    سه مشکل بحرانی آن بلوک:

      ۱. در لحظهٔ import اجرا می‌شد، پیش از آنکه فایل tf_decisions.json
         ساخته شده باشد.

      ۲. به get_active_symbol_timeframe ارجاع می‌داد که در همان لحظه هنوز
         در namespace ماژول تعریف نشده بود (forward reference)
         ⇒ NameError که در try/except به warning تبدیل می‌شد و در نهایت
         به ValueError منتهی می‌گردید.

      ۳. متغیر reports در هیچ جای دیگر ماژول مصرف نمی‌شد؛ صرفاً یک
         side-effect در زمان import و منبع crash بود.

    این نسخه فقط زمانی اجرا می‌شود که صریحاً فراخوانی شود، تمام استثناهای
    ممکن را می‌گیرد و در بدترین حالت دیکشنری گزارش با کلید error برمی‌گرداند
    (بدون هیچ‌گاه crash کردن فراخواننده).

    Args:
        symbol: نام نماد. اگر None باشد، از tf_decisions.json خوانده می‌شود.
        timeframe: تایم‌فریم. اگر None باشد، از tf_decisions.json خوانده می‌شود.
        fold: شمارهٔ فولد. اگر None باشد، از main.sum_folds گرفته می‌شود.
        target_names: فهرست تارگت‌ها؛ پیش‌فرض PROJECT_TARGETS.
        raise_on_mismatch: اگر True باشد، عدم تطابق مقیاس در
            audit_sequenced_target_scales استثنا پرتاب می‌کند.
            در غیر این‌صورت فقط در گزارش ثبت می‌شود.
        verbose: چاپ جدول خلاصه در لاگ.

    Returns:
        Dict[str, Dict[str, Any]] — {target_name: report_dict}.
        در صورت شکست در هر تارگت، دیکشنری آن تارگت شامل
        {"ok": False, "error": "...", "target_name": ...} خواهد بود.
    """
    targets = list(target_names or PROJECT_TARGETS)
    reports: Dict[str, Dict[str, Any]] = {}

    if not targets:
        logger.warning(
            "[get_scale_audit_reports] target_names خالی است — گزارش تولید نشد."
        )
        return reports

    logger.info(
        f"[get_scale_audit_reports] شروع ممیزی مقیاس برای {len(targets)} تارگت "
        f"| symbol={symbol!r}, timeframe={timeframe!r}, fold={fold!r}"
    )

    for target in targets:
        try:
            report = audit_sequenced_target_scales(
                symbol=symbol,
                timeframe=timeframe,
                fold=fold,
                target_name=target,
                raise_on_mismatch=raise_on_mismatch,
                verbose=verbose,
            )
            reports[target] = report

            if isinstance(report, dict):
                ok_flag = report.get("ok")
                ratio = report.get("sigma_ratio_max")
                if ok_flag is True:
                    logger.info(
                        f"[get_scale_audit_reports] ✅ target '{target}': "
                        f"OK (σ_ratio={ratio})"
                    )
                elif ok_flag is False:
                    logger.warning(
                        f"[get_scale_audit_reports] ⚠️ target '{target}': "
                        f"scale problem detected — issues="
                        f"{report.get('issues')}"
                    )
                else:
                    logger.warning(
                        f"[get_scale_audit_reports] ⚠️ target '{target}': "
                        f"verdict indeterminate (ok={ok_flag!r})"
                    )

        except FileNotFoundError as fnf_err:
            reports[target] = {
                "ok": False,
                "error": f"directory_not_found: {fnf_err}",
                "target_name": target,
                "symbol": symbol,
                "timeframe": timeframe,
                "fold": fold,
            }
            logger.warning(
                f"[get_scale_audit_reports] target '{target}': "
                f"پوشهٔ sequenced یافت نشد ({fnf_err}) — skip"
            )

        except ValueError as val_err:
            reports[target] = {
                "ok": False,
                "error": f"configuration_error: {val_err}",
                "target_name": target,
                "symbol": symbol,
                "timeframe": timeframe,
                "fold": fold,
            }
            logger.warning(
                f"[get_scale_audit_reports] target '{target}': "
                f"خطای پیکربندی ({val_err}) — skip. "
                f"tf_decisions.json را بررسی کنید یا symbol/timeframe را "
                f"صریحاً به این تابع بدهید."
            )

        except Exception as exc:
            reports[target] = {
                "ok": False,
                "error": f"unexpected: {type(exc).__name__}: {exc}",
                "target_name": target,
                "symbol": symbol,
                "timeframe": timeframe,
                "fold": fold,
            }
            logger.error(
                f"[get_scale_audit_reports] target '{target}': "
                f"خطای پیش‌بینی‌نشده ({exc!r}) — skip",
                exc_info=True,
            )

    n_ok = sum(
        1 for r in reports.values()
        if isinstance(r, dict) and r.get("ok") is True
    )
    n_failed = sum(
        1 for r in reports.values()
        if isinstance(r, dict) and r.get("ok") is False
    )

    logger.info(
        f"[get_scale_audit_reports] پایان ممیزی — "
        f"{n_ok}/{len(targets)} OK, {n_failed} مشکل‌دار"
    )

    return reports


async def _save_features_info(symbol: str, timeframe: str, fold: int, features_info: Dict) -> bool:
    """
    💾 ذخیره اطلاعات ویژگی‌ها
    """
    try:
        logger.info(f"[{symbol}] 💾 شروع ذخیره features info برای فولد {fold}")

        # ایجاد دایرکتوری
        features_dir = Path("cache") / "features_info"
        features_dir.mkdir(parents=True, exist_ok=True)

        # نام فایل
        filename = f"{symbol}_{timeframe}_fold{fold}_features.json"
        filepath = features_dir / filename

        # اضافه کردن metadata
        features_info.update({
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "timeframe": timeframe,
            "fold": fold,
            "version": "1.0"
        })

        # ذخیره اتمی
        temp_file = filepath.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(features_info, f, indent=2, ensure_ascii=False, default=str)

            # جایگزینی اتمی
            if sys.platform == 'win32':
                try:
                    os.remove(filepath)
                except FileNotFoundError:
                    pass
                os.rename(temp_file, filepath)
            else:
                os.replace(temp_file, filepath)

            logger.info(f"[{symbol}] ✅ features info ذخیره شد: {filepath}")
            return True

        except Exception as write_error:
            logger.error(f"[{symbol}] ❌ خطا در نوشتن فایل: {write_error}")
            if temp_file.exists():
                temp_file.unlink()
            return False

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطای کلی در ذخیره features info: {e}")
        return False

def _save_tf_decision(symbol: str, tf: str) -> None:
    TF_DECISION_FILE.parent.mkdir(parents=True, exist_ok=True)
    decisions = json.loads(TF_DECISION_FILE.read_text()) if TF_DECISION_FILE.exists() else {}
    decisions[symbol] = tf
    TF_DECISION_FILE.write_text(json.dumps(decisions, indent=2))

def _md5_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return MD5 hex-digest of any file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def get_active_symbol_timeframe() -> Tuple[str, str]:
    """
    دریافت نماد و تایم‌فریم فعال از فایل tf_decisions.json
    ✅ FIX: جستجوی هوشمند در مسیرهای مختلف
    ✅ FIX: بدون هاردکد
    ✅ FIX: برگرداندن None, None در صورت عدم وجود فایل
    """
    import json
    from pathlib import Path
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"🔍 شروع دریافت نماد و تایم‌فریم فعال از tf_decisions.json")
        
        def find_root():
            current = Path(__file__).resolve()
            for p in [current] + list(current.parents):
                if (p / "cache").exists() or (p / "data").exists():
                    return p
            # اگر هیچ نشانه‌ای پیدا نشد، به دایرکتوری جاری برگرد
            return Path.cwd()
        
        ROOT = find_root()
        
        # ⭐⭐⭐⭐⭐ CRITICAL FIX: جستجو در مسیرهای مختلف
        possible_paths = [
            ROOT / "cache" / "tf_decisions.json",
            Path.cwd() / "cache" / "tf_decisions.json",
            Path(__file__).resolve().parent / "cache" / "tf_decisions.json",
            Path(__file__).resolve().parent.parent / "cache" / "tf_decisions.json",
            Path("cache") / "tf_decisions.json",
        ]
        
        # همچنین جستجو در کل پروژه
        for search_root in [ROOT, Path.cwd(), Path(__file__).resolve().parent]:
            cache_dir = search_root / "cache"
            if cache_dir.exists():
                tf_file = cache_dir / "tf_decisions.json"
                if tf_file.exists():
                    possible_paths.insert(0, tf_file)
        
        tf_decisions_path = None
        for path in possible_paths:
            if path.exists():
                tf_decisions_path = path
                logger.info(f"✅ فایل tf_decisions یافت شد: {tf_decisions_path}")
                break
        
        # ✅ نرمالایز کردن نماد (حذف نقطه)
        def normalize_symbol(s: str) -> str:
            if not s:
                return s
            return s.replace('.', '')
        
        # ================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL: بررسی وجود فایل - بدون پیش‌فرض
        # ================================================================
        if tf_decisions_path is None or not tf_decisions_path.exists():
            logger.warning(f"❌ فایل tf_decisions یافت نشد در هیچ یک از مسیرهای:")
            for path in possible_paths:
                logger.warning(f"   • {path}")
            logger.warning(f"   لطفا ابتدا فایل پیکربندی را در مسیر cache/tf_decisions.json ایجاد کنید")
            return None, None
        
        # ================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL: خواندن و اعتبارسنجی فایل
        # ================================================================
        try:
            with open(tf_decisions_path, 'r', encoding='utf-8') as f:
                decisions = json.load(f)
            
            logger.debug(f"📋 محتوای tf_decisions: {decisions}")
            
        except json.JSONDecodeError as json_error:
            logger.error(f"❌ فایل tf_decisions خراب است (JSON معتبر نیست): {json_error}")
            logger.error(f"   مسیر فایل: {tf_decisions_path}")
            return None, None
        
        except Exception as read_error:
            logger.error(f"❌ خطا در خواندن فایل tf_decisions: {read_error}")
            return None, None
        
        # ================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL: اعتبارسنجی محتوا
        # ================================================================
        if not decisions:
            logger.error(f"❌ فایل tf_decisions خالی است: {tf_decisions_path}")
            return None, None
        
        if not isinstance(decisions, dict):
            logger.error(f"❌ فایل tf_decisions باید دیکشنری باشد، نوع: {type(decisions).__name__}")
            return None, None
        
        # ================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL: استخراج نماد
        # ================================================================
        if len(decisions) == 0:
            logger.error(f"❌ فایل tf_decisions دیکشنری خالی است")
            return None, None
        
        symbol = list(decisions.keys())[0]
        
        if not symbol or not isinstance(symbol, str):
            logger.error(f"❌ کلید نماد نامعتبر: {symbol}")
            return None, None
        
        # ================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL: استخراج تایم فریم
        # ================================================================
        timeframe_value = decisions[symbol]
        
        if timeframe_value is None:
            logger.error(f"❌ مقدار تایم‌فریم برای '{symbol}' None است")
            return None, None
        
        # استخراج timeframe - پشتیبانی از هر دو فرمت
        if isinstance(timeframe_value, dict):
            timeframe = timeframe_value.get('timeframe')
            if timeframe is None:
                # تلاش برای یافتن کلیدهای جایگزین
                for key in ['tf', 'time_frame', 'period']:
                    if key in timeframe_value:
                        timeframe = timeframe_value[key]
                        break
            if timeframe is None:
                logger.error(f"❌ در دیکشنری تایم‌فریم برای '{symbol}'، کلید 'timeframe' یافت نشد")
                return None, None
        else:
            timeframe = str(timeframe_value)
        
        if not timeframe or not isinstance(timeframe, str):
            logger.error(f"❌ تایم‌فریم برای '{symbol}' نامعتبر: {timeframe}")
            return None, None
        
        # ================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL: اعتبارسنجی تایم فریم
        # ================================================================
        valid_timeframes = ["1M", "5M", "15M", "30M", "1H", "2H", "4H", "6H", "8H", "12H", "1D", "1W", "1MN"]
        
        # نرمالایز کردن تایم فریم
        tf_upper = timeframe.upper()
        if tf_upper in valid_timeframes:
            timeframe = tf_upper
        else:
            # تلاش برای استخراج عدد از تایم فریم
            import re
            match = re.match(r'(\d+)([MHDW])', tf_upper)
            if match:
                num, unit = match.groups()
                num = int(num)
                if unit == 'M':
                    if num == 1:
                        timeframe = "1M"
                    elif num == 5:
                        timeframe = "5M"
                    elif num == 15:
                        timeframe = "15M"
                    elif num == 30:
                        timeframe = "30M"
                    else:
                        timeframe = f"{num}M"
                elif unit == 'H':
                    if num == 1:
                        timeframe = "1H"
                    elif num == 2:
                        timeframe = "2H"
                    elif num == 4:
                        timeframe = "4H"
                    elif num == 6:
                        timeframe = "6H"
                    elif num == 8:
                        timeframe = "8H"
                    elif num == 12:
                        timeframe = "12H"
                    else:
                        timeframe = f"{num}H"
                elif unit == 'D':
                    if num == 1:
                        timeframe = "1D"
                    else:
                        timeframe = f"{num}D"
                elif unit == 'W':
                    timeframe = "1W"
            else:
                logger.error(f"❌ تایم‌فریم '{timeframe}' معتبر نیست")
                logger.error(f"   تایم‌فریم‌های معتبر: {valid_timeframes}")
                return None, None
        
        # ================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL: نرمالایز کردن نماد
        # ================================================================
        clean_symbol = normalize_symbol(symbol)
        
        if not clean_symbol:
            logger.error(f"❌ پس از نرمالایز کردن، نماد '{symbol}' خالی شد")
            return None, None
        
        if clean_symbol != symbol:
            logger.info(f"🔧 Symbol normalized: '{symbol}' -> '{clean_symbol}'")
        
        logger.info(f"✅ نماد و تایم‌فریم فعال: {clean_symbol} - {timeframe}")
        logger.info(f"   منبع فایل: {tf_decisions_path}")
        return clean_symbol, timeframe
        
    except Exception as e:
        logger.error(f"❌ خطای غیرمنتظره در get_active_symbol_timeframe: {e}", exc_info=True)
        return None, None

def save_project_info(project_data: Dict) -> bool:
    """
    ذخیره اطلاعات پروژه با مدیریت int64 - نسخه اصلاح شده
    """
    try:
        project_file = Path("cache") / "project_info.json"
        project_file.parent.mkdir(parents=True, exist_ok=True)

        # تابع تبدیل انواع داده
        def prepare_for_json(obj):
            """آماده‌سازی داده برای JSON"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                # برای آرایه‌های بزرگ، فقط metadata برمی‌گردانیم
                if obj.size > 50:
                    return {
                        "type": "ndarray",
                        "shape": list(obj.shape),
                        "dtype": str(obj.dtype),
                        "size": obj.size,
                        "sample": obj.flatten()[:10].tolist() if obj.size > 0 else []
                    }
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, pd.DataFrame):
                return {
                    "type": "DataFrame",
                    "shape": list(obj.shape),
                    "columns": list(obj.columns),
                    "dtypes": {col: str(dtype) for col, dtype in obj.dtypes.items()}
                }
            elif isinstance(obj, dict):
                return {k: prepare_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [prepare_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # برای اشیاء، فقط attributes مهم
                return str(obj)
            else:
                return obj

        # خواندن اطلاعات موجود
        existing_data = {}
        if project_file.exists():
            try:
                with open(project_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.warning(f"⚠️ فایل project_info خراب است ({e})، ایجاد مجدد")
                existing_data = {}

        # آماده‌سازی داده جدید
        safe_project_data = prepare_for_json(project_data)

        # به‌روزرسانی داده‌ها
        existing_data.update(safe_project_data)
        existing_data["last_update"] = datetime.now(timezone.utc).isoformat()
        existing_data["_metadata"] = {
            "saved_by": "save_project_info",
            "python_version": sys.version,
            "platform": sys.platform,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # ذخیره اتمی با try-catch دقیق
        temp_file = project_file.with_suffix('.tmp')
        try:
            # نوشتن به فایل موقت
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f,
                          ensure_ascii=False,
                          indent=2,
                          default=str,
                          separators=(',', ': '))

            # تایید نوشتن
            if not temp_file.exists() or temp_file.stat().st_size == 0:
                raise IOError("فایل موقت خالی ایجاد شد")

            # جایگزینی اتمی
            backup_file = None
            if project_file.exists():
                backup_file = project_file.with_suffix('.bak')
                try:
                    shutil.copy2(project_file, backup_file)
                except Exception as backup_error:
                    logger.debug(f"⚠️ ایجاد backup ناموفق: {backup_error}")

            try:
                # در ویندوز نیاز به حذف قبلی داریم
                if sys.platform == 'win32':
                    try:
                        os.remove(project_file)
                    except FileNotFoundError:
                        pass
                    os.rename(temp_file, project_file)
                else:
                    os.replace(temp_file, project_file)

                # تایید فایل نهایی
                if project_file.exists() and project_file.stat().st_size > 0:
                    logger.info(f"✅ اطلاعات پروژه ذخیره شد: {project_file} ({project_file.stat().st_size} bytes)")

                    # پاکسازی backup قدیمی
                    if backup_file and backup_file.exists():
                        try:
                            backup_file.unlink()
                        except Exception:
                            pass

                    return True
                else:
                    raise IOError("فایل نهایی ایجاد نشد یا خالی است")

            except Exception as replace_error:
                logger.error(f"❌ خطا در جایگزینی فایل: {replace_error}")

                # بازیابی از backup
                if backup_file and backup_file.exists():
                    try:
                        shutil.copy2(backup_file, project_file)
                        logger.info("✅ بازیابی از backup انجام شد")
                    except Exception as restore_error:
                        logger.error(f"❌ خطا در بازیابی از backup: {restore_error}")

                return False

        except Exception as write_error:
            logger.error(f"❌ خطا در نوشتن فایل: {write_error}", exc_info=True)
            return False

        finally:
            # پاکسازی فایل موقت در صورت باقی ماندن
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception as cleanup_error:
                    logger.debug(f"⚠️ خطا در پاکسازی فایل موقت: {cleanup_error}")

    except Exception as e:
        logger.error(f"❌ خطای کلی در ذخیره اطلاعات پروژه: {e}", exc_info=True)
        return False

def load_project_info() -> Dict[str, Any]:
    """
    بارگذاری اطلاعات پروژه
    """
    try:
        project_file = Path("cache") / "project_info.json"

        if project_file.exists():
            with open(project_file, 'r', encoding='utf-8') as f:
                project_data = json.load(f)

            logger.info("✅ اطلاعات پروژه بارگیری شد")
            return project_data
        else:
            logger.warning("⚠️ فایل project_info یافت نشد")
            return {}

    except json.JSONDecodeError as e:
        logger.error(f"❌ فایل project_info خراب است: {e}")
        # ایجاد فایل خالی
        default_data = {
            "created_at": datetime.now().isoformat(),
            "error": "corrupted_file_recreated"
        }
        save_project_info(default_data)
        return default_data

    except Exception as e:
        logger.error(f"❌ خطا در بارگذاری اطلاعات پروژه: {e}")
        return {}






def get_project_info() -> Dict[str, Any]:
    """
    دریافت اطلاعات پروژه

    Returns:
        Dict با اطلاعات پروژه
    """
    try:
        project_info_path = Path("cache") / "project_info.json"
        if project_info_path.exists():
            with open(project_info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"⚠️ خطا در خواندن اطلاعات پروژه: {e}")
        return {}




def _load_tf_decision(symbol: str) -> str | None:
    if TF_DECISION_FILE.exists():
        return json.loads(TF_DECISION_FILE.read_text()).get(symbol)
    return None



async def _load_sequence_info(symbol: str, timeframe: str, fold: int = None) -> Optional[Dict]:
    """
    🔢 بارگذاری اطلاعات sequence با پشتیبانی از fold خودکار

    Args:
        symbol: نماد
        timeframe: تایم‌فریم
        fold: شماره فولد (اختیاری) - اگر None باشد، از آخرین فولد استفاده می‌کند

    Returns:
        Dict با اطلاعات sequence
    """
    try:
        logger.info(f"[{symbol}] 🔢 مرحله 1: شروع بارگذاری sequence info")

        # مرحله 1: تعیین فولد صحیح
        fold_info = {}
        try:
            from main import sum_folds as TOTAL_FOLDS
            from main import folds as NUM_FOLDS

            if fold is None:
                # استفاده از آخرین فولدهای آموزشی
                latest_folds = get_latest_training_folds(TOTAL_FOLDS, NUM_FOLDS)
                if latest_folds:
                    fold = latest_folds[-1]  # آخرین فولد
                    logger.info(f"[{symbol}] ✅ استفاده از آخرین فولد آموزشی: {fold}")
                else:
                    fold = TOTAL_FOLDS  # fallback به آخرین فولد
                    logger.warning(f"[{symbol}] ⚠️ استفاده از fallback فولد: {fold}")

            fold_info = {
                'fold_used': fold,
                'total_folds': TOTAL_FOLDS,
                'num_training_folds': NUM_FOLDS,
                'selection_method': 'auto' if fold is None else 'manual',
                'latest_training_folds': latest_folds if 'latest_folds' in locals() else []
            }

        except Exception as fold_error:
            logger.error(f"[{symbol}] ❌ خطا در تعیین فولد: {fold_error}")

        logger.info(f"[{symbol}] 📊 فولد تعیین شده: {fold}")

        # مرحله 2: بارگذاری sequence
        logger.info(f"[{symbol}] 🔄 مرحله 2: بارگذاری داده‌های sequence")
        sequence_data = await universal_load("sequence", symbol=symbol, timeframe=timeframe, fold=fold)

        if sequence_data is None:
            logger.warning(f"[{symbol}] ⚠️ sequence data یافت نشد برای فولد {fold}")
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "fold_info": fold_info,
                "sequence_status": "not_found",
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "error": "sequence_data_not_found"
            }

        info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "fold_info": fold_info,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "sequence_status": "loaded"
        }

        # مرحله 3: استخراج اطلاعات از sequence
        try:
            logger.info(f"[{symbol}] 🔄 مرحله 3: استخراج اطلاعات از sequence")

            if isinstance(sequence_data, tuple) and len(sequence_data) >= 2:
                X, y = sequence_data

                # اطلاعات X
                if hasattr(X, 'shape'):
                    info["X_shape"] = X.shape
                    info["X_dims"] = X.ndim
                    info["X_size"] = X.size
                    info["X_dtype"] = str(X.dtype) if hasattr(X, 'dtype') else "unknown"

                    # محاسبات اضافی برای شکل‌های مختلف
                    if X.ndim == 3:
                        info["sequence_info"] = {
                            "samples": X.shape[0],
                            "sequence_length": X.shape[1],
                            "features": X.shape[2],
                            "total_sequences": X.shape[0]
                        }
                    elif X.ndim == 2:
                        info["sequence_info"] = {
                            "samples": X.shape[0],
                            "features": X.shape[1],
                            "is_flat": True
                        }

                # اطلاعات y
                if hasattr(y, 'shape'):
                    info["y_shape"] = y.shape
                    info["y_dims"] = y.ndim
                    info["y_size"] = y.size
                    info["y_dtype"] = str(y.dtype) if hasattr(y, 'dtype') else "unknown"

                    # بررسی consistency بین X و y
                    if hasattr(X, 'shape') and hasattr(y, 'shape'):
                        if X.shape[0] == y.shape[0]:
                            info["consistency"] = {
                                "samples_match": True,
                                "sample_count": X.shape[0]
                            }
                        else:
                            info["consistency"] = {
                                "samples_match": False,
                                "X_samples": X.shape[0],
                                "y_samples": y.shape[0],
                                "warning": "تعداد نمونه‌ها در X و y مطابقت ندارد"
                            }

                # آمار توصیفی (اگر numpy array باشد)
                if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.number):
                    try:
                        info["X_statistics"] = {
                            "mean": float(np.nanmean(X)),
                            "std": float(np.nanstd(X)),
                            "min": float(np.nanmin(X)),
                            "max": float(np.nanmax(X)),
                            "has_nan": bool(np.any(np.isnan(X))),
                            "has_inf": bool(np.any(np.isinf(X)))
                        }
                    except Exception as stat_error:
                        info["X_statistics"] = {"error": str(stat_error)}

                if hasattr(y, 'dtype') and np.issubdtype(y.dtype, np.number):
                    try:
                        info["y_statistics"] = {
                            "mean": float(np.nanmean(y)),
                            "std": float(np.nanstd(y)),
                            "min": float(np.nanmin(y)),
                            "max": float(np.nanmax(y)),
                            "has_nan": bool(np.any(np.isnan(y))),
                            "has_inf": bool(np.any(np.isinf(y)))
                        }
                    except Exception as stat_error:
                        info["y_statistics"] = {"error": str(stat_error)}

                logger.info(
                    f"[{symbol}] ✅ sequence info بارگیری شد: X{info.get('X_shape', 'unknown')}, y{info.get('y_shape', 'unknown')}")

            else:
                # فرمت غیراستاندارد
                info["data_type"] = str(type(sequence_data))
                info["data_info"] = "non_standard_format"

                if isinstance(sequence_data, dict):
                    info["dict_keys"] = list(sequence_data.keys())
                elif hasattr(sequence_data, '__len__'):
                    info["length"] = len(sequence_data)

                logger.info(f"[{symbol}] ⚠️ sequence info (فرمت غیراستاندارد): {info['data_type']}")

        except Exception as extract_error:
            logger.error(f"[{symbol}] ❌ خطا در استخراج اطلاعات sequence: {extract_error}")
            info["extraction_error"] = str(extract_error)
            info["sequence_status"] = "extraction_failed"

        # مرحله 4: اضافه کردن metadata اضافی
        try:
            logger.info(f"[{symbol}] 🔄 مرحله 4: اضافه کردن metadata اضافی")

            # بارگذاری metadata مرتبط
            metadata = await universal_load("metadata", symbol=symbol,
                                            timeframe=timeframe, fold=fold)
            if metadata:
                info["related_metadata"] = metadata

            # بارگذاری features info
            features_info = await universal_load("features_info", symbol=symbol,
                                                 timeframe=timeframe, fold=fold)
            if features_info:
                info["features_info"] = features_info

            # بارگذاری model_dims
            model_dims = await universal_load("model_dims", symbol=symbol, fold=fold)
            if model_dims:
                info["model_dims_reference"] = model_dims

        except Exception as meta_error:
            logger.warning(f"[{symbol}] ⚠️ خطا در بارگذاری metadata اضافی: {meta_error}")

        # مرحله 5: اعتبارسنجی نهایی
        try:
            logger.info(f"[{symbol}] 🔄 مرحله 5: اعتبارسنجی نهایی")

            validation_checks = []

            # بررسی وجود داده
            if 'X_shape' in info:
                validation_checks.append(("data_exists", True))
            else:
                validation_checks.append(("data_exists", False))

            # بررسی consistency
            if 'consistency' in info and info['consistency'].get('samples_match', False):
                validation_checks.append(("consistency", True))
            else:
                validation_checks.append(("consistency", False))

            # بررسی shape معتبر
            if 'X_shape' in info and all(s > 0 for s in info['X_shape']):
                validation_checks.append(("valid_shape", True))
            else:
                validation_checks.append(("valid_shape", False))

            info["validation"] = {
                "checks": dict(validation_checks),
                "passed": all(check[1] for check in validation_checks),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            if info["validation"]["passed"]:
                info["sequence_status"] = "validated"
                logger.info(f"[{symbol}] ✅ اعتبارسنجی sequence موفق")
            else:
                info["sequence_status"] = "validation_failed"
                logger.warning(f"[{symbol}] ⚠️ اعتبارسنجی sequence ناموفق")

        except Exception as validation_error:
            logger.error(f"[{symbol}] ❌ خطا در اعتبارسنجی: {validation_error}")
            info["validation_error"] = str(validation_error)

        # گزارش نهایی
        logger.info(f"[{symbol}] 📊 گزارش sequence info:")
        logger.info(f"    • فولد: {fold}")
        logger.info(f"    • وضعیت: {info.get('sequence_status', 'unknown')}")
        logger.info(f"    • شکل X: {info.get('X_shape', 'unknown')}")
        logger.info(f"    • شکل y: {info.get('y_shape', 'unknown')}")
        logger.info(f"    • اعتبارسنجی: {'✅' if info.get('validation', {}).get('passed', False) else '❌'}")

        return info

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطا در بارگذاری sequence info: {e}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "fold": fold,
            "error": str(e),
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "sequence_status": "error"
        }


async def _load_features_info(symbol: str, timeframe: str, fold: int) -> Optional[Dict]:
    """
    🎯 بارگذاری اطلاعات ویژگی‌ها (features info) - نسخه اصلاح شده
    ✅ FIX: اضافه شدن return features_info در تمام مسیرها
    """
    try:
        logger.info(f"[{symbol}] 🎯 شروع بارگذاری features info برای فولد {fold}")

        features_info = {
            "symbol": symbol,
            "timeframe": timeframe,
            "fold": fold,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "sources_checked": [],
            "feature_count": 0,
            "feature_names": [],
            "source": "not_found",
        }

        # 🔍 روش ۱: از metadata سکانس
        try:
            logger.debug(f"[{symbol}] 🔍 روش ۱: بارگذاری از metadata")
            metadata = await universal_load("metadata", symbol=symbol,
                                            timeframe=timeframe, fold=fold)
            if metadata:
                features_info["sources_checked"].append("metadata")
                if "feature_names" in metadata:
                    feature_names = metadata["feature_names"]
                    if isinstance(feature_names, str):
                        feature_names = [f.strip() for f in feature_names.split(",") if f.strip()]
                    features_info["feature_names"] = feature_names
                    features_info["feature_count"] = len(feature_names)
                    features_info["source"] = "metadata_feature_names"
                    logger.info(f"[{symbol}] ✅ features از metadata: {len(feature_names)} ویژگی")
                    return features_info
                elif "features_count" in metadata:
                    features_info["feature_count"] = metadata["features_count"]
                    features_info["source"] = "metadata_features_count"
                    return features_info
        except Exception as meta_error:
            logger.debug(f"[{symbol}] ⚠️ خطا در بارگذاری از metadata: {meta_error}")

        # 🔍 روش ۲: از model_dims
        try:
            logger.debug(f"[{symbol}] 🔍 روش ۲: استخراج از model_dims")
            model_dims = await universal_load("model_dims", symbol=symbol, fold=fold)
            if model_dims and isinstance(model_dims, dict):
                features_info["sources_checked"].append("model_dims")
                if "features" in model_dims and model_dims["features"]:
                    features_info["feature_count"] = int(model_dims["features"])
                    features_info["lookback"] = model_dims.get("lookback")
                    features_info["hidden_size"] = model_dims.get("hidden_size")
                    features_info["source"] = "model_dims_dict"
                    logger.info(f"[{symbol}] ✅ features از model_dims: {model_dims['features']} ویژگی")
                    return features_info
                elif "n_features" in model_dims:
                    features_info["feature_count"] = int(model_dims["n_features"])
                    features_info["source"] = "model_dims_n_features"
                    return features_info
        except Exception as dims_error:
            logger.debug(f"[{symbol}] ⚠️ خطا در بارگذاری model_dims: {dims_error}")

        # 🔍 روش ۳: خواندن مستقیم از فایل‌های sequenced
        try:
            logger.debug(f"[{symbol}] 🔍 روش ۳: خواندن مستقیم از فایل‌های sequenced")
            sequenced_dir = Path("data") / "sequenced" / symbol / timeframe / f"fold_{fold}"
            if sequenced_dir.exists():
                train_files = list(sequenced_dir.glob("*train*sequenced*.h5"))
                if train_files:
                    train_file = train_files[0]
                    try:
                        import h5py
                        with h5py.File(train_file, 'r') as f:
                            # ✅ FIX: پشتیبانی از X و X_seq
                            if 'X_seq' in f:
                                X_shape = f['X_seq'].shape
                            elif 'X' in f:
                                X_shape = f['X'].shape
                            else:
                                X_shape = None
                            if X_shape is not None and len(X_shape) >= 3:
                                features_count = X_shape[2]
                                features_info["feature_count"] = int(features_count)
                                features_info["source_file"] = str(train_file)
                                features_info["source"] = "direct_h5_read"
                                features_info["X_shape"] = list(X_shape)
                                logger.info(f"[{symbol}] ✅ features از فایل H5: {features_count} ویژگی")
                                return features_info
                    except Exception as h5_error:
                        logger.debug(f"[{symbol}] ⚠️ خطا در خواندن H5: {h5_error}")
        except Exception as file_error:
            logger.debug(f"[{symbol}] ⚠️ خطا در خواندن فایل: {file_error}")

        # 🔍 روش ۴: جستجو در فایل‌های JSON metadata
        try:
            logger.debug(f"[{symbol}] 🔍 روش ۴: جستجو در فایل‌های JSON")
            metadata_files = list(Path(".").rglob(f"*{symbol}*fold{fold}*metadata*.json"))
            if metadata_files:
                latest_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                features_info["sources_checked"].append(f"file_{latest_file.name}")
                if "feature_names" in file_data:
                    feature_names = file_data["feature_names"]
                    if isinstance(feature_names, str):
                        feature_names = [x.strip() for x in feature_names.split(",") if x.strip()]
                    features_info["feature_names"] = feature_names
                    features_info["feature_count"] = len(feature_names)
                    features_info["source_file"] = str(latest_file)
                    features_info["source"] = "direct_json_read"
                    return features_info
                elif "shapes" in file_data and "X_train" in file_data["shapes"]:
                    shapes = file_data["shapes"]["X_train"]
                    if len(shapes) >= 3:
                        features_info["feature_count"] = shapes[2]
                        features_info["source_file"] = str(latest_file)
                        features_info["source"] = "json_shapes"
                        return features_info
        except Exception as json_error:
            logger.debug(f"[{symbol}] ⚠️ خطا در خواندن JSON: {json_error}")

        # ✅ FIX: بازگشت مقدار در صورت عدم یافتن هیچ منبعی
        logger.warning(f"[{symbol}] ⚠️ هیچ منبعی برای features info یافت نشد")
        return features_info

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطای کلی در _load_features_info: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "fold": fold,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "source": "error",
            "feature_count": 0,
        }

async def _load_sequence_metadata(
    symbol: str,
    timeframe: str,
    fold: int,
    split_type: str = "train"
) -> Optional[Dict[str, Any]]:
    """
    📋 بارگذاری metadata توالی - نسخه اصلاح شده
    ✅ FIX v2.0: پشتیبانی از split_type برای یافتن val/test metadata
    ✅ FIX v2.0: جستجوی هوشمند در مسیرهای مختلف
    """
    try:
        symbol = symbol.upper()
        logger.info(f"[{symbol}] 📋 شروع بارگذاری metadata برای فولد {fold} | split={split_type}")

        # ✅ FIX: شامل split_type در الگوهای جستجو
        search_patterns = [
            # اولویت ۱: مسیر استاندارد با split
            f"data/sequenced/{symbol}/{timeframe}/fold_{fold}/{symbol}_{timeframe}_fold{fold}_{split_type}_metadata.json",
            # اولویت ۲: بدون split در مسیر دوم
            f"data/sequenced/{symbol}/{timeframe}/fold_{fold}/{symbol}_{timeframe}_fold{fold}_metadata.json",
            # اولویت ۳: مسیر جایگزین
            f"data/sequenced/{symbol}/{timeframe}/{symbol}_{timeframe}_fold{fold}_{split_type}_metadata.json",
            f"data/sequenced/{symbol}/{timeframe}/{symbol}_{timeframe}_fold{fold}_metadata.json",
            # اولویت ۴: مسیر قدیمی
            f"data/sequences/{symbol}/{timeframe}/fold_{fold}/{symbol}_{timeframe}_fold{fold}_{split_type}_metadata.json",
            f"data/sequences/{symbol}/{timeframe}/fold_{fold}/{symbol}_{timeframe}_fold{fold}_metadata.json",
        ]

        found_file = None
        for pattern in search_patterns:
            path = Path(pattern)
            if path.exists():
                found_file = path
                logger.info(f"[{symbol}] 📍 فایل metadata یافت شد: {path}")
                break

        # ✅ جستجوی بازگشتی در صورت یافت نشدن
        if not found_file:
            logger.info(f"[{symbol}] 🔍 جستجوی بازگشتی برای فایل‌های metadata (split={split_type})")

            search_dirs = [Path("cache"), Path("data"), Path(".")]

            # ✅ FIX: الگوهای دقیق‌تر شامل split_type
            file_patterns = [
                f"*{symbol}*fold{fold}*{split_type}*metadata*.json",
                f"*{symbol}*{split_type}*metadata*.json",
                f"*{symbol}*fold{fold}*metadata*.json",
                f"*{symbol}*metadata*.json",
            ]

            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                for pattern in file_patterns:
                    for file_path in search_dir.rglob(pattern):
                        if file_path.is_file() and file_path.suffix.lower() == '.json':
                            # ✅ اطمینان از تطابق split_type (اگر در نام هست)
                            fname = file_path.name.lower()
                            if split_type.lower() in fname or 'metadata' in fname:
                                found_file = file_path
                                logger.info(f"[{symbol}] 📍 فایل یافت شد (جستجوی بازگشتی): {file_path}")
                                break
                    if found_file:
                        break
                if found_file:
                    break

        if found_file:
            try:
                with open(found_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                metadata["metadata_source"] = str(found_file)
                metadata["loaded_at"] = datetime.now(timezone.utc).isoformat()
                metadata["split_type_requested"] = split_type
                logger.info(f"[{symbol}] ✅ metadata بارگذاری شد ({found_file.stat().st_size} بایت)")
                return metadata
            except json.JSONDecodeError as e:
                logger.error(f"[{symbol}] ❌ فایل metadata معتبر نیست: {e}")
                return None
            except Exception as e:
                logger.error(f"[{symbol}] ❌ خطا در خواندن metadata: {e}")
                return None
        else:
            logger.warning(f"[{symbol}] ⚠️ فایل metadata یافت نشد برای split={split_type}")
            return None

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطای کلی در بارگذاری metadata: {e}", exc_info=True)
        return None
        
async def _load_split_data(
        symbol: str,
        timeframe: str,
        fold: int,
        split_type: str
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    📦 بارگذاری داده‌های split از فایل‌های HDF5
    """
    try:
        from pathlib import Path

        # ساخت مسیر فایل
        if split_type in ["train", "val", "test"]:
            file_name = f"{symbol}_{timeframe}_fold{fold}_{split_type}_processed.h5"
            file_path = Path("data/processed") / file_name

            if not file_path.exists():
                # جستجوی جایگزین
                alt_path = Path("data/processed") / symbol / timeframe / f"fold_{fold}" / file_name
                if alt_path.exists():
                    file_path = alt_path
                else:
                    logger.warning(f"[{symbol}] ⚠️ فایل split {split_type} یافت نشد: {file_name}")
                    return None

            # بارگذاری از HDF5
            result = await _load_hdf5(file_path, "processed", symbol, timeframe, fold)

            if result is not None:
                logger.info(f"[{symbol}] ✅ split {split_type} بارگذاری شد")
                return result

        logger.warning(f"[{symbol}] ⚠️ نوع split ناشناخته: {split_type}")
        return None

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطا در بارگذاری split {split_type}: {e}")
        return None




async def get_active_symbol_config(fold: int = None) -> Dict[str, Any]:
    """
    دریافت پیکربندی کامل نماد فعال با پشتیبانی از fold

    Args:
        fold: شماره فولد (اختیاری) - اگر None باشد، از آخرین فولد استفاده می‌کند

    Returns:
        Dict با پیکربندی کامل
    """
    try:
        logger.info("🚀 مرحله 1: شروع دریافت پیکربندی نماد فعال")

        # مرحله 1: دریافت نماد و تایم‌فریم
        symbol, timeframe = get_active_symbol_timeframe()
        logger.info(f"✅ نماد و تایم‌فریم: {symbol} - {timeframe}")

        # مرحله 2: تعیین فولدهای صحیح
        try:
            from main import sum_folds as TOTAL_FOLDS
            from main import folds as NUM_FOLDS

            if fold is None:
                # استفاده از آخرین فولدهای آموزشی
                latest_folds = get_latest_training_folds(TOTAL_FOLDS, NUM_FOLDS)
                if latest_folds:
                    fold = latest_folds[-1]  # آخرین فولد
                    logger.info(f"✅ استفاده از آخرین فولد آموزشی: {fold}")
                else:
                    fold = TOTAL_FOLDS  # fallback به آخرین فولد
                    logger.warning(f"⚠️ استفاده از fallback فولد: {fold}")
            else:
                logger.info(f"📊 استفاده از فولد دستی: {fold}")

            # ذخیره اطلاعات فولد
            fold_info = {
                'fold_used': fold,
                'total_folds': TOTAL_FOLDS,
                'num_training_folds': NUM_FOLDS,
                'latest_training_folds': latest_folds if 'latest_folds' in locals() else []
            }

        except Exception as fold_error:
            logger.error(f"❌ خطا در تعیین فولد: {fold_error}")
            fold_info = {
                'fold_used': fold,
                'error': str(fold_error)
            }

        # مرحله 3: بارگذاری اطلاعات مختلف
        config_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "fold_info": fold_info,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "loading_phases": {}
        }

        # فاز 1: اطلاعات رژیم
        try:
            logger.info("🔄 فاز 1: بارگذاری اطلاعات رژیم")
            from core.regim import get_symbol_regime
            regime_info = await get_symbol_regime()
            config_data["regime_info"] = regime_info
            config_data["loading_phases"]["regime"] = "success"
            logger.info("✅ اطلاعات رژیم بارگیری شد")
        except Exception as regime_error:
            logger.error(f"❌ خطا در بارگذاری رژیم: {regime_error}")
            config_data["regime_info"] = None
            config_data["loading_phases"]["regime"] = f"error: {regime_error}"

        # فاز 2: اطلاعات پروژه
        try:
            logger.info("🔄 فاز 2: بارگذاری اطلاعات پروژه")
            project_info = await universal_load("project_info")
            config_data["project_info"] = project_info
            config_data["loading_phases"]["project_info"] = "success"
            logger.info("✅ اطلاعات پروژه بارگیری شد")
        except Exception as project_error:
            logger.error(f"❌ خطا در بارگذاری پروژه: {project_error}")
            config_data["project_info"] = {}
            config_data["loading_phases"]["project_info"] = f"error: {project_error}"

        # فاز 3: اطلاعات مدل برای فولد مشخص
        try:
            logger.info(f"🔄 فاز 3: بارگذاری اطلاعات مدل برای فولد {fold}")
            model_dims = await universal_load("model_dims", symbol=symbol, fold=fold)
            config_data["model_dims"] = model_dims
            config_data["loading_phases"]["model_dims"] = "success"
            logger.info(f"✅ اطلاعات مدل برای فولد {fold} بارگیری شد")
        except Exception as model_error:
            logger.error(f"❌ خطا در بارگذاری اطلاعات مدل: {model_error}")
            config_data["model_dims"] = None
            config_data["loading_phases"]["model_dims"] = f"error: {model_error}"

        # فاز 4: اطلاعات ویژگی‌ها
        try:
            logger.info(f"🔄 فاز 4: بارگذاری اطلاعات ویژگی‌ها برای فولد {fold}")
            features_info = await universal_load("features_info", symbol=symbol,
                                                 timeframe=timeframe, fold=fold)
            config_data["features_info"] = features_info
            config_data["loading_phases"]["features_info"] = "success"
            logger.info("✅ اطلاعات ویژگی‌ها بارگیری شد")
        except Exception as features_error:
            logger.error(f"❌ خطا در بارگذاری ویژگی‌ها: {features_error}")
            config_data["features_info"] = None
            config_data["loading_phases"]["features_info"] = f"error: {features_error}"

        # فاز 5: اطلاعات متادیتا
        try:
            logger.info(f"🔄 فاز 5: بارگذاری متادیتا برای فولد {fold}")
            metadata = await universal_load("metadata", symbol=symbol,
                                            timeframe=timeframe, fold=fold)
            config_data["metadata"] = metadata
            config_data["loading_phases"]["metadata"] = "success"
            logger.info("✅ متادیتا بارگیری شد")
        except Exception as meta_error:
            logger.error(f"❌ خطا در بارگذاری متادیتا: {meta_error}")
            config_data["metadata"] = None
            config_data["loading_phases"]["metadata"] = f"error: {meta_error}"

        # گزارش نهایی
        successful_phases = sum(1 for phase in config_data["loading_phases"].values()
                                if phase == "success")
        total_phases = len(config_data["loading_phases"])

        config_data["summary"] = {
            "total_phases": total_phases,
            "successful_phases": successful_phases,
            "success_rate": f"{(successful_phases / total_phases) * 100:.1f}%" if total_phases > 0 else "0%",
            "fold_used": fold,
            "symbol": symbol,
            "timeframe": timeframe
        }

        logger.info(f"📊 خلاصه پیکربندی: {successful_phases}/{total_phases} فاز موفق")

        return config_data

    except Exception as e:
        logger.error(f"❌ خطای کلی در دریافت پیکربندی: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emergency_mode": True
        }


async def check_all_model_files_complete(symbol: str, specific_fold: int = None) -> Dict[str, Any]:
    """
    🔍 بررسی کامل بودن همه فایل‌های مدل برای یک نماد با پشتیبانی از فولد خاص

    Args:
        symbol: نماد
        specific_fold: فولد خاص برای بررسی (اختیاری)

    Returns:
        Dict با وضعیت فایل‌ها
    """
    

    try:
        symbol = symbol.upper()

        # مرحله 1: تعیین فولدهای مورد بررسی
        try:
            from main import sum_folds as TOTAL_FOLDS
            from main import folds as NUM_FOLDS

            logger.info(f"[{symbol}] 🔍 مرحله 1: تعیین فولدهای بررسی")

            if specific_fold is not None:
                # بررسی فولد خاص
                folds_to_check = [specific_fold]
                logger.info(f"✅ بررسی فولد خاص: {specific_fold}")
            else:
                # استفاده از آخرین فولدهای آموزشی
                latest_folds = get_latest_training_folds(TOTAL_FOLDS, NUM_FOLDS)
                if latest_folds:
                    folds_to_check = latest_folds
                    logger.info(f"✅ بررسی آخرین فولدهای آموزشی: {latest_folds}")
                else:
                    # fallback: بررسی همه فولدها
                    folds_to_check = list(range(1, TOTAL_FOLDS + 1))
                    logger.warning(f"⚠️ بررسی همه فولدها: 1 تا {TOTAL_FOLDS}")

            fold_info = {
                'folds_to_check': folds_to_check,
                'total_folds': TOTAL_FOLDS,
                'num_training_folds': NUM_FOLDS,
                'selection_method': 'specific' if specific_fold else 'latest_training',
                'latest_training_folds': latest_folds if 'latest_folds' in locals() else []
            }

        except Exception as fold_error:
            logger.error(f"[{symbol}] ❌ خطا در تعیین فولدها: {fold_error}")
            fold_info = {
                'error': str(fold_error),
                'folds_to_check': [1]  # fallback
            }
            folds_to_check = [1]

        logger.info(f"[{symbol}] 🔍 شروع بررسی کامل بودن فایل‌های مدل")
        logger.info(f"[{symbol}] 📊 فولدهای مورد بررسی: {folds_to_check}")

        results = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fold_info": fold_info,
            "fold_status": {},
            "overall_status": "unknown",
            "missing_files": [],
            "inconsistent_folds": [],
            "recommendations": [],
            "checking_phases": {}
        }

        # فاز 1: بررسی فایل‌های model_dims
        try:
            logger.info("🔄 فاز 1: بررسی فایل‌های model_dims")
            dims_status = {}

            for fold in folds_to_check:
                fold_key = f"fold_{fold}"
                fold_result = {
                    "fold_number": fold,
                    "dims_file_exists": False,
                    "dims_file_path": None,
                    "dims_file_size": 0,
                    "dimensions": {},
                    "has_metadata": False,
                    "status": "unknown",
                    "check_phase": "model_dims"
                }

                try:
                    # بارگذاری ابعاد مدل
                    dims_data = await load_model_dims(symbol, fold)

                    if dims_data and dims_data.get("error"):
                        fold_result["status"] = "error"
                        fold_result["error"] = dims_data.get("error")
                        logger.warning(f"[{symbol}] ❌ فولد {fold}: خطا در model_dims")
                    elif dims_data:
                        fold_result["dims_file_exists"] = True
                        fold_result["dimensions"] = {
                            "lookback": dims_data.get("lookback"),
                            "features": dims_data.get("features"),
                            "hidden_size": dims_data.get("hidden_size")
                        }
                        fold_result["data_source"] = dims_data.get("data_source", "unknown")
                        fold_result["status"] = "loaded"

                        # بررسی consistency ابعاد
                        if (fold_result["dimensions"]["lookback"] and
                                fold_result["dimensions"]["features"]):
                            fold_result["status"] = "complete"
                            logger.info(f"[{symbol}] ✅ فولد {fold}: model_dims کامل")
                        else:
                            fold_result["status"] = "incomplete"
                            results["inconsistent_folds"].append(fold)
                            logger.warning(f"[{symbol}] ⚠️ فولد {fold}: model_dims ناقص")
                    else:
                        fold_result["status"] = "missing"
                        results["missing_files"].append(f"fold_{fold}_dims.json")
                        logger.warning(f"[{symbol}] ⚠️ فولد {fold}: model_dims مفقود")

                except Exception as fold_error:
                    fold_result["status"] = "error"
                    fold_result["error"] = str(fold_error)
                    logger.error(f"[{symbol}] ❌ فولد {fold}: خطای بررسی: {fold_error}")

                dims_status[fold_key] = fold_result

            results["fold_status"].update(dims_status)
            results["checking_phases"]["model_dims"] = {
                "status": "completed",
                "folds_checked": len(folds_to_check),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as phase1_error:
            logger.error(f"[{symbol}] ❌ خطا در فاز 1: {phase1_error}")
            results["checking_phases"]["model_dims"] = {
                "status": "error",
                "error": str(phase1_error)
            }

        # فاز 2: بررسی فایل‌های sequence
        try:
            logger.info("🔄 فاز 2: بررسی فایل‌های sequence")
            sequence_status = {}

            for fold in folds_to_check:
                fold_key = f"fold_{fold}"
                if fold_key not in sequence_status:
                    sequence_status[fold_key] = {
                        "fold_number": fold,
                        "sequence_files_exist": False,
                        "check_phase": "sequence"
                    }

                try:
                    # بررسی وجود فایل‌های sequence
                    sequence_data = await universal_load("sequence", symbol=symbol, fold=fold)

                    if sequence_data:
                        sequence_status[fold_key]["sequence_files_exist"] = True
                        sequence_status[fold_key]["sequence_status"] = "exists"

                        # بررسی شکل داده‌ها
                        if isinstance(sequence_data, tuple) and len(sequence_data) >= 1:
                            X = sequence_data[0]
                            if hasattr(X, 'shape'):
                                sequence_status[fold_key]["shape"] = X.shape
                                sequence_status[fold_key]["samples"] = X.shape[0]
                                logger.info(f"[{symbol}] ✅ فولد {fold}: sequence کامل - shape: {X.shape}")
                            else:
                                sequence_status[fold_key]["sequence_status"] = "incomplete"
                                logger.warning(f"[{symbol}] ⚠️ فولد {fold}: sequence بدون shape")
                        else:
                            sequence_status[fold_key]["sequence_status"] = "unknown_format"
                            logger.warning(f"[{symbol}] ⚠️ فولد {fold}: فرمت sequence نامشخص")
                    else:
                        sequence_status[fold_key]["sequence_status"] = "missing"
                        results["missing_files"].append(f"fold_{fold}_sequence")
                        logger.warning(f"[{symbol}] ⚠️ فولد {fold}: sequence مفقود")

                except Exception as seq_error:
                    sequence_status[fold_key]["sequence_status"] = "error"
                    sequence_status[fold_key]["error"] = str(seq_error)
                    logger.error(f"[{symbol}] ❌ فولد {fold}: خطای sequence: {seq_error}")

            # ادغام نتایج
            for fold_key, seq_result in sequence_status.items():
                if fold_key in results["fold_status"]:
                    results["fold_status"][fold_key].update(seq_result)
                else:
                    results["fold_status"][fold_key] = seq_result

            results["checking_phases"]["sequence"] = {
                "status": "completed",
                "folds_checked": len(folds_to_check),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as phase2_error:
            logger.error(f"[{symbol}] ❌ خطا در فاز 2: {phase2_error}")
            results["checking_phases"]["sequence"] = {
                "status": "error",
                "error": str(phase2_error)
            }

        # فاز 3: بررسی فایل‌های split
        try:
            logger.info("🔄 فاز 3: بررسی فایل‌های split")
            split_status = {}

            for fold in folds_to_check:
                fold_key = f"fold_{fold}"
                if fold_key not in split_status:
                    split_status[fold_key] = {
                        "fold_number": fold,
                        "split_files_exist": False,
                        "splits_available": [],
                        "check_phase": "split"
                    }

                try:
                    # بررسی split‌های مختلف
                    splits_to_check = ["train", "val", "test"]
                    available_splits = []

                    for split_type in splits_to_check:
                        try:
                            split_data = await universal_load("split", symbol=symbol,
                                                              fold=fold, split_type=split_type)

                            if split_data:
                                available_splits.append(split_type)

                                # ثبت اطلاعات split
                                split_key = f"{split_type}_info"
                                split_status[fold_key][split_key] = {
                                    "exists": True,
                                    "checked_at": datetime.now(timezone.utc).isoformat()
                                }

                                # اگر داده tuple است، shape آن را ثبت کن
                                if isinstance(split_data, tuple) and len(split_data) >= 1:
                                    X = split_data[0]
                                    if hasattr(X, 'shape'):
                                        split_status[fold_key][split_key]["shape"] = X.shape
                            else:
                                split_status[fold_key][f"{split_type}_info"] = {
                                    "exists": False,
                                    "error": "not_found"
                                }
                                results["missing_files"].append(f"fold_{fold}_{split_type}_split")

                        except Exception as split_error:
                            split_status[fold_key][f"{split_type}_info"] = {
                                "exists": False,
                                "error": str(split_error)
                            }
                            logger.warning(f"[{symbol}] ⚠️ فولد {fold}, split {split_type}: {split_error}")

                    split_status[fold_key]["splits_available"] = available_splits
                    split_status[fold_key]["split_files_exist"] = len(available_splits) > 0

                    if len(available_splits) >= 2:  # حداقل train و val
                        logger.info(f"[{symbol}] ✅ فولد {fold}: splits کافی - {available_splits}")
                    else:
                        logger.warning(f"[{symbol}] ⚠️ فولد {fold}: splits ناکافی - {available_splits}")

                except Exception as fold_split_error:
                    split_status[fold_key]["error"] = str(fold_split_error)
                    logger.error(f"[{symbol}] ❌ فولد {fold}: خطای بررسی split: {fold_split_error}")

            # ادغام نتایج
            for fold_key, split_result in split_status.items():
                if fold_key in results["fold_status"]:
                    results["fold_status"][fold_key].update(split_result)
                else:
                    results["fold_status"][fold_key] = split_result

            results["checking_phases"]["split"] = {
                "status": "completed",
                "folds_checked": len(folds_to_check),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as phase3_error:
            logger.error(f"[{symbol}] ❌ خطا در فاز 3: {phase3_error}")
            results["checking_phases"]["split"] = {
                "status": "error",
                "error": str(phase3_error)
            }

        # محاسبه آمار نهایی
        total_folds = len(folds_to_check)
        complete_folds = 0
        incomplete_folds = 0
        missing_folds = 0
        error_folds = 0

        for fold_key, fold_status in results["fold_status"].items():
            status = fold_status.get("status", "unknown")

            if status == "complete":
                complete_folds += 1
            elif status == "incomplete":
                incomplete_folds += 1
            elif status == "missing":
                missing_folds += 1
            elif status == "error":
                error_folds += 1

        # تعیین وضعیت کلی
        if complete_folds == total_folds:
            results["overall_status"] = "complete"
            results["recommendations"].append("✅ همه فایل‌های مدل کامل هستند")
        elif complete_folds >= len(folds_to_check) * 0.7:  # 70% کامل
            results["overall_status"] = "partial"
            results["recommendations"].append(f"⚠️ {complete_folds}/{total_folds} فولد کامل هستند")
        elif complete_folds > 0:
            results["overall_status"] = "limited"
            results["recommendations"].append(f"⚠️ تنها {complete_folds}/{total_folds} فولد کامل هستند")
        else:
            results["overall_status"] = "missing"
            results["recommendations"].append("❌ هیچ فایل مدل کاملی یافت نشد")

        # افزودن آمار
        results["statistics"] = {
            "total_folds_checked": total_folds,
            "complete_folds": complete_folds,
            "incomplete_folds": incomplete_folds,
            "missing_folds": missing_folds,
            "error_folds": error_folds,
            "completion_percentage": (complete_folds / total_folds) * 100 if total_folds > 0 else 0
        }

        # گزارش نهایی
        logger.info(f"[{symbol}] 📊 ======== گزارش کامل بودن فایل‌های مدل ========")
        logger.info(f"[{symbol}] 📋 وضعیت کلی: {results['overall_status']}")
        logger.info(f"[{symbol}] 📊 آمار: {complete_folds} کامل, {incomplete_folds} ناقص, "
                    f"{missing_folds} مفقود, {error_folds} خطا")

        for fold_key, status in results["fold_status"].items():
            logger.info(
                f"[{symbol}]   • {fold_key}: {status.get('status', 'unknown')} | "
                f"lookback={status.get('dimensions', {}).get('lookback')}, "
                f"features={status.get('dimensions', {}).get('features')}"
            )

        logger.info(f"[{symbol}] ✅ بررسی کامل بودن فایل‌ها پایان یافت")

        return results

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطا در بررسی فایل‌های مدل: {e}", exc_info=True)
        return {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "overall_status": "error",
            "emergency_mode": True
        }



async def _load_project_info() -> Optional[dict]:

    try:
        logger.debug("📁 loading project_info")
        return load_project_info()
    except Exception as e:
        logger.error(f"❌ load_project_info failed: {e}")
        return None

async def _load_regime(symbol: str, timeframe: str) -> Any:
    from core.regim import get_symbol_regime

    try:
        regime_info = await get_symbol_regime()

        if regime_info is None:
            raise ValueError("get_symbol_regime returned None")

        # normalize output
        if hasattr(regime_info, "regime"):
            regime_value = regime_info.regime
            attrs = vars(regime_info)

        elif isinstance(regime_info, dict):
            regime_value = regime_info.get("regime", "unknown")
            attrs = regime_info

        elif isinstance(regime_info, str):
            regime_value = regime_info
            attrs = {"regime": regime_value}

    except Exception as e:
        logger.error(f"[{symbol}] ❌ load_regime failed: {e}")



async def _load_expiry(
    symbol: str,
    timeframe: str,
    fold: Optional[int]
) -> Optional[dict]:

    from core.regim import ExpiryDateManager

    try:
        fold_to_use = fold if fold is not None else TOTAL_FOLDS
        manager = ExpiryDateManager()

        logger.debug(f"[{symbol}] ⏳ loading expiry from file (fold={fold_to_use})")

        expiry = await manager._load_expiry_from_file(
            symbol=symbol,
            timeframe=timeframe,
            fold_number=fold_to_use,  # استفاده از fold_to_use
            expiry_type="final"
        )

        if expiry:
            return expiry

        logger.debug(f"[{symbol}] ⏳ expiry not found → recalculating (fold={fold_to_use})")
        return await manager.calculate_expiry_date(
            symbol=symbol,
            timeframe=timeframe,
            fold_number=fold_to_use,  # استفاده از fold_to_use
            expiry_type="final"
        )

    except Exception as e:
        logger.error(f"[{symbol}] ❌ load_expiry failed: {e}")
        return None


def find_project_root(marker="data"):
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Project root not found")


def detect_project_root(start_path: Optional[Path] = None) -> Path:
    """
    🔍 تشخیص هوشمند روت پروژه از هر نقطه‌ای

    Args:
        start_path: مسیر شروع جستجو (اختیاری)

    Returns:
        Path: مسیر روت پروژه
    """
    global _project_root_cache

    # استفاده از کش اگر موجود باشد
    if _project_root_cache is not None:
        return _project_root_cache

    # تعیین مسیر شروع
    if start_path is None:
        # اولویت‌های مختلف برای شروع
        possible_starts = [
            Path(__file__).resolve().parent.parent,  # از core/project_root.py
            Path.cwd(),  # دایرکتوری جاری
            Path(sys.argv[0]).resolve().parent,  # از فایل اجرایی
        ]

        for start in possible_starts:
            if start.exists():
                start_path = start
                logger.debug(f"شروع جستجو از: {start_path}")
                break
        else:
            start_path = Path.cwd()
            logger.warning(f"⚠️ استفاده از دایرکتوری جاری به عنوان شروع: {start_path}")

    # نشانگرهای روت پروژه
    project_markers = [
        'requirements.txt',
        'pyproject.toml',
        'setup.py',
        '.git',
        'README.md',
        'LICENSE',
        'core',  # پوشه core
        'data',  # پوشه data
        'configs',  # پوشه configs
        'models',  # پوشه models
    ]

    # جستجوی سلسله‌مراتبی
    current_path = start_path.resolve()
    checked_paths = []

    while True:
        checked_paths.append(current_path)

        # بررسی نشانگرها
        found_markers = []
        for marker in project_markers:
            marker_path = current_path / marker
            if marker_path.exists():
                found_markers.append(marker)

        # اگر نشانگر کافی یافت شد
        if len(found_markers) >= 2:  # حداقل 2 نشانگر
            logger.info(f"✅ تشخیص روت پروژه: {current_path}")
            logger.info(f"   • نشانگرهای یافت شده: {', '.join(found_markers[:3])}")

            # ذخیره در کش
            _project_root_cache = current_path

            # ذخیره در متغیر محیطی برای استفاده ماژول‌های دیگر
            os.environ['PROJECT_ROOT'] = str(current_path)

            return current_path

        # اگر به ریشه فایل سیستم رسیدیم
        if current_path.parent == current_path:
            logger.warning(f"⚠️ به ریشه فایل سیستم رسیدیم - استفاده از: {start_path}")
            _project_root_cache = start_path
            return start_path

        # یک سطح بالاتر برو
        current_path = current_path.parent

    # fallback (نباید به اینجا برسد)
    fallback = Path.cwd()
    logger.warning(f"🚨 تشخیص روت شکست خورد - fallback به: {fallback}")
    _project_root_cache = fallback
    return fallback


async def collect_global_values_report(
        symbol: str,
        timeframe: str = "1H",
        num_folds: int = None,
        training_types: list = None,
        save_to_cache: bool = True
) -> Dict[str, Any]:
    """
    🔍 جمع‌آوری مقادیر گلوبال - نسخه اصلاح شده با ساختار model_dimensions
    """
    logger.info(f"[{symbol}] 🔍 COLLECT_GLOBAL_VALUES_REPORT | START")

    try:
        from main import sum_folds as TOTAL_FOLDS
        from main import folds as NUM_FOLDS

        symbol = symbol.strip().upper()

        # ساختار یکتا و نهایی
        global_values = {
            "report_type": "global_values_final",
            "version": "1.0.0-no-cache",
            "symbol": symbol,
            "timeframe": timeframe,
            "total_folds": TOTAL_FOLDS,
            "num_training_folds": NUM_FOLDS,
            "allowed_folds": get_latest_training_folds(TOTAL_FOLDS, NUM_FOLDS),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "collected_by": "collect_global_values_report_final",
            "folds": {},
            "model_dimensions": {},  # 🔥 ساختار اصلی برای ارسال به pipeline
            "regime_info": None,
            "expiry_info": None,
            "project_info": None,
            "errors": [],
            "warnings": []
        }

        # ========== خواندن اطلاعات رژیم ==========
        try:
            regime_file = Path(f"cache/{symbol}_{timeframe}_fold{global_values['allowed_folds'][-1]}.regime")
            if regime_file.exists():
                with open(regime_file, 'r', encoding='utf-8') as f:
                    global_values["regime_info"] = json.load(f)
                logger.info(f"[{symbol}] ✅ رژیم از فایل خوانده شد: {regime_file}")
        except Exception as e:
            global_values["warnings"].append(f"خطا در خواندن رژیم: {e}")

        # ========== خواندن اطلاعات انقضا ==========
        try:
            expiry_file = Path(
                f"cache/expiry_dates/{symbol}_{timeframe}_fold{global_values['allowed_folds'][-1]}_final.json")
            if expiry_file.exists():
                with open(expiry_file, 'r', encoding='utf-8') as f:
                    global_values["expiry_info"] = json.load(f)
                logger.info(f"[{symbol}] ✅ انقضا از فایل خوانده شد: {expiry_file}")
        except Exception as e:
            global_values["warnings"].append(f"خطا در خواندن انقضا: {e}")

        # ========== خواندن اطلاعات هر فولد و پر کردن model_dimensions ==========
        for fold in global_values['allowed_folds']:
            fold_data = {
                "fold": fold,
                "status": "checking",
                "model_dims": None,
                "metadata": None,
                "error": None
            }

            # خواندن ابعاد مدل از فایل اصلی
            dims_file = Path(f"cache/model_dims/{symbol}_{timeframe}_fold{fold}_dims.json")
            if dims_file.exists():
                try:
                    with open(dims_file, 'r', encoding='utf-8') as f:
                        dims_content = json.load(f)

                    lookback = dims_content.get("lookback")
                    features = dims_content.get("features")
                    hidden_size = dims_content.get("hidden_size")

                    # 🔥 ذخیره در model_dimensions به فرمت [lookback, features, hidden_size]
                    global_values["model_dimensions"][fold] = [lookback, features, hidden_size]

                    fold_data["model_dims"] = {
                        "lookback": lookback,
                        "features": features,
                        "hidden_size": hidden_size,
                        "file": str(dims_file),
                        "file_size": dims_file.stat().st_size,
                        "file_modified": datetime.fromtimestamp(dims_file.stat().st_mtime).isoformat()
                    }
                    logger.info(f"[{symbol}] ✅ ابعاد فولد {fold} خوانده شد: lookback={lookback}, features={features}")
                except Exception as e:
                    fold_data["error"] = f"model_dims_read_error: {e}"
                    global_values["errors"].append(f"فولد {fold}: {e}")
            else:
                fold_data["error"] = "model_dims_file_not_found"
                global_values["errors"].append(f"فولد {fold}: فایل ابعاد وجود ندارد")

            # خواندن metadata
            meta_file = Path(f"data/sequenced/{symbol}/{timeframe}/{symbol}_{timeframe}_fold{fold}_metadata.json")
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        fold_data["metadata"] = json.load(f)
                except Exception as e:
                    fold_data["metadata_error"] = str(e)

            fold_data["status"] = "complete" if fold_data["model_dims"] else "missing"
            global_values["folds"][f"fold_{fold}"] = fold_data

        # ========== ذخیره گزارش نهایی ==========
        if save_to_cache:
            reports_dir = Path("global_reports_final")
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"{symbol}_global_final_{timestamp}.json"

            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(global_values, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"[{symbol}] 💾 گزارش نهایی ذخیره شد: {report_file}")

        return global_values

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطا در collect_global_values_report: {e}")
        return {
            "symbol": symbol,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def load_latest_global_report(symbol: str) -> Dict[str, Any]:
    """
    📥 بارگذاری آخرین گزارش گلوبال از فایل
    """
    try:
        symbol = symbol.strip().upper()

        # ========== جستجو در پوشه‌های مختلف ==========
        search_dirs = [
            Path("global_reports_final"),
            Path("cache") / "global_reports",
            Path("global_values_reports"),
            Path("reports")
        ]

        all_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                # الگوهای مختلف
                patterns = [
                    f"{symbol}_global_final_*.json",
                    f"{symbol}_global_*.json",
                    f"{symbol}_*_global.json"
                ]

                for pattern in patterns:
                    files = list(search_dir.glob(pattern))
                    all_files.extend(files)

        if not all_files:
            logger.warning(f"[{symbol}] ❌ No global report files found")
            return None

        # مرتب‌سازی بر اساس زمان تغییر
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = all_files[0]

        logger.info(f"[{symbol}] 📄 Loading global report: {latest_file}")

        with open(latest_file, 'r', encoding='utf-8') as f:
            global_values = json.load(f)

        return global_values

    except Exception as e:
        logger.error(f"[{symbol}] ❌ Error loading global report: {e}")
        return None

def load_global_report_file(symbol: str, fold: int) -> Dict[str, Any]:
    """
    📥 بارگذاری کامل گزارش گلوبال از فایل JSON - نسخه اصلاح شده نهایی
    ✅ استخراج ابعاد از model_dimensions با اولویت بالا
    """
    try:
        if symbol is None or not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"❌ load_global_report_file: symbol نمی‌تواند None باشد")

        symbol = symbol.strip().upper()
        logger.info(f"[{symbol}] 📥 START load_global_report_file - فولد: {fold}")

        if fold is None:
            logger.error(f"[{symbol}] ❌ شماره فولد الزامی است")
            return {"success": False, "error": "شماره فولد الزامی است"}

        # پیدا کردن آخرین فایل گلوبال
        reports_dir = Path("cache") / "global_reports"
        if not reports_dir.exists():
            reports_dir = Path("global_reports_final")  # 🔥 مسیر جایگزین

        if not reports_dir.exists():
            logger.error(f"[{symbol}] ❌ پوشه reports وجود ندارد")
            return {"success": False, "error": "پوشه reports وجود ندارد"}

        # پیدا کردن آخرین فایل
        pattern = f"{symbol}_global_final_*.json"
        global_files = list(reports_dir.glob(pattern))

        if not global_files:
            # fallback به الگوی قدیمی
            global_files = list(reports_dir.glob(f"{symbol}_global_*.json"))

        if not global_files:
            logger.error(f"[{symbol}] ❌ هیچ فایل گلوبالی یافت نشد")
            return {"success": False, "error": "فایل گلوبال یافت نشد"}

        # مرتب‌سازی بر اساس timestamp
        global_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = global_files[0]
        logger.info(f"[{symbol}] ✅ آخرین فایل گلوبال: {latest_file.name}")

        # خواندن فایل
        with open(latest_file, 'r', encoding='utf-8') as f:
            global_data = json.load(f)

        # ========== استخراج ابعاد مدل ==========
        lookback = None
        features = None
        hidden_size = None

        # 🔥 اولویت ۱: استفاده از model_dimensions
        if 'model_dimensions' in global_data and fold in global_data['model_dimensions']:
            dims = global_data['model_dimensions'][fold]
            if isinstance(dims, (list, tuple)) and len(dims) >= 2:
                lookback = dims[0]
                features = dims[1]
                hidden_size = dims[2] if len(dims) >= 3 else None
                logger.info(f"[{symbol}] ✅ ابعاد از model_dimensions استخراج شد: {lookback}, {features}")

        # 🔥 اولویت ۲: استفاده از folds[fold_ X].model_dims
        if (lookback is None or features is None) and 'folds' in global_data:
            fold_key = f"fold_{fold}"
            if fold_key in global_data['folds']:
                fold_data = global_data['folds'][fold_key]
                if fold_data.get('model_dims'):
                    md = fold_data['model_dims']
                    lookback = md.get('lookback', lookback)
                    features = md.get('features', features)
                    hidden_size = md.get('hidden_size', hidden_size)
                    logger.info(f"[{symbol}] ✅ ابعاد از folds استخراج شد: {lookback}, {features}")

        # ========== ساخت نتیجه ==========
        result = {
            "success": True,
            "symbol": global_data.get('symbol', symbol),
            "timeframe": global_data.get('timeframe', '1H'),
            "fold": fold,
            "fold_number": fold,
            "lookback": lookback,
            "features": features,
            "hidden_size": hidden_size,
            "model_dims": {
                "lookback": lookback,
                "features": features,
                "hidden_size": hidden_size,
                "source": "global_report_file",
                "is_valid": lookback is not None and features is not None,
                "fold": fold
            },
            "regime_info": global_data.get('regime_info'),
            "expiry_info": global_data.get('expiry_info'),
            "project_info": global_data.get('project_info'),
            "collected_at": global_data.get('collected_at'),
            "source_file": latest_file.name,
            "file_path": str(latest_file),
            "loaded_at": datetime.now().isoformat(),
            "function": "load_global_report_file_fixed"
        }

        logger.info(f"[{symbol}] 📊 نتیجه نهایی: lookback={lookback}, features={features}")
        return result

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطا در load_global_report_file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "symbol": symbol,
            "fold": fold,
            "function": "load_global_report_file"
        }

async def load_model_dims(symbol: str, fold: int, timeframe: str = "1H") -> Dict[str, Any]:
    """
    📂 نسخه اصلاح شده - فقط از فایل اصلی می‌خواند، بدون کش

    ویژگی‌ها:
    ✅ فقط و فقط از فایل JSON در cache/model_dims/ می‌خواند
    ✅ هیچ گونه کش حافظه‌ای یا فایل‌کش استفاده نمی‌کند
    ✅ در صورت نبود فایل، خطای مشخص برمی‌گرداند
    ✅ لاگ کامل از مسیر فایل و وضعیت

    Args:
        symbol: نماد (مثال: BTCUSD)
        fold: شماره فولد
        timeframe: تایم‌فریم (پیش‌فرض 1H)

    Returns:
        Dict شامل lookback, features, hidden_size یا دیکشنری خطا
    """

    symbol = str(symbol).upper()
    fold = int(fold)

    logger.info(f"[{symbol}] 📦 START load_model_dims | نماد={symbol}, فولد={fold}, زمان={timeframe}")

    try:
        # ==============================================
        # فاز ۱: ساخت مسیر فایل اصلی
        # ==============================================
        dims_dir = Path("cache") / "model_dims"
        file_name = f"{symbol}_{timeframe}_fold{fold}_dims.json"
        file_path = dims_dir / file_name

        logger.info(f"[{symbol}] 🔍 جستجوی فایل ابعاد مدل:")
        logger.info(f"[{symbol}]   • مسیر کامل: {file_path.absolute()}")
        logger.info(f"[{symbol}]   • دایرکتوری: {dims_dir}")
        logger.info(f"[{symbol}]   • نام فایل: {file_name}")

        # ==============================================
        # فاز ۲: بررسی وجود فایل
        # ==============================================
        if not file_path.exists():
            logger.warning(f"[{symbol}] ❌ فایل ابعاد مدل وجود ندارد: {file_path}")

            # گزارش دقیق از دایرکتوری
            if dims_dir.exists():
                files = list(dims_dir.glob(f"{symbol}*fold{fold}*.json"))
                logger.info(f"[{symbol}] 📋 فایل‌های مشابه در دایرکتوری: {[f.name for f in files]}")
            else:
                logger.warning(f"[{symbol}] 📁 دایرکتوری model_dims وجود ندارد")

            return {
                "symbol": symbol,
                "fold": fold,
                "timeframe": timeframe,
                "lookback": None,
                "features": None,
                "hidden_size": None,
                "error": f"فایل وجود ندارد: {file_name}",
                "error_type": "file_not_found",
                "file_path": str(file_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cache_hit": False,
                "source": "none"
            }

        # ==============================================
        # فاز ۳: خواندن فایل
        # ==============================================
        file_size = file_path.stat().st_size
        logger.info(f"[{symbol}] 📄 فایل یافت شد - حجم: {file_size:,} بایت")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dims_data = json.load(f)

            logger.info(f"[{symbol}] ✅ فایل JSON با موفقیت خوانده شد")

        except json.JSONDecodeError as e:
            logger.error(f"[{symbol}] ❌ خطا در پردازش JSON: {e}")
            return {
                "symbol": symbol,
                "fold": fold,
                "timeframe": timeframe,
                "lookback": None,
                "features": None,
                "hidden_size": None,
                "error": f"فرمت JSON نامعتبر: {e}",
                "error_type": "json_decode_error",
                "file_path": str(file_path),
                "file_size": file_size,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cache_hit": False,
                "source": "none"
            }

        # ==============================================
        # فاز ۴: استخراج ابعاد
        # ==============================================

        # استخراج از ساختارهای مختلف (سازگاری با عقب)
        lookback = None
        features = None
        hidden_size = None

        # روش ۱: دیکشنری مستقیم
        if isinstance(dims_data, dict):
            lookback = dims_data.get('lookback')
            features = dims_data.get('features')
            hidden_size = dims_data.get('hidden_size')

            # بررسی nested structures
            if lookback is None and 'dimensions' in dims_data:
                lookback = dims_data['dimensions'].get('lookback') or \
                           dims_data['dimensions'].get('input_shape', [None, None])[0]
                features = dims_data['dimensions'].get('features') or \
                           dims_data['dimensions'].get('input_shape', [None, None])[1]

            if lookback is None and 'model_configuration' in dims_data:
                lookback = dims_data['model_configuration'].get('lookback')
                features = dims_data['model_configuration'].get('features')

        # روش ۲: لیست/تاپل (فرمت قدیمی)
        elif isinstance(dims_data, (list, tuple)):
            if len(dims_data) >= 1:
                lookback = dims_data[0]
            if len(dims_data) >= 2:
                features = dims_data[1]
            if len(dims_data) >= 3:
                hidden_size = dims_data[2]

        # ==============================================
        # فاز ۵: اعتبارسنجی
        # ==============================================

        # تبدیل به int
        try:
            lookback = int(lookback) if lookback is not None else None
        except (ValueError, TypeError):
            lookback = None

        try:
            features = int(features) if features is not None else None
        except (ValueError, TypeError):
            features = None

        try:
            hidden_size = int(hidden_size) if hidden_size is not None else None
        except (ValueError, TypeError):
            hidden_size = None

        # اعتبارسنجی نهایی
        is_valid = (
                lookback is not None and lookback > 0 and
                features is not None and features > 0
        )

        if not is_valid:
            logger.error(f"[{symbol}] ❌ ابعاد نامعتبر در فایل:")
            logger.error(f"[{symbol}]   • lookback: {lookback}")
            logger.error(f"[{symbol}]   • features: {features}")
            logger.error(f"[{symbol}]   • hidden_size: {hidden_size}")

            return {
                "symbol": symbol,
                "fold": fold,
                "timeframe": timeframe,
                "lookback": lookback,
                "features": features,
                "hidden_size": hidden_size,
                "error": "ابعاد نامعتبر (صفر یا منفی یا None)",
                "error_type": "invalid_dimensions",
                "file_path": str(file_path),
                "file_content_preview": str(dims_data)[:200] if dims_data else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cache_hit": False,
                "source": "file_invalid"
            }

        # ==============================================
        # فاز ۶: گزارش موفقیت
        # ==============================================
        logger.info(f"[{symbol}] ✅ ابعاد مدل با موفقیت از فایل خوانده شد:")
        logger.info(f"[{symbol}]   • مسیر فایل: {file_path}")
        logger.info(f"[{symbol}]   • حجم فایل: {file_size:,} بایت")
        logger.info(f"[{symbol}]   • lookback: {lookback}")
        logger.info(f"[{symbol}]   • features: {features}")
        logger.info(f"[{symbol}]   • hidden_size: {hidden_size}")

        # اطلاعات متادیتا
        metadata = {
            "symbol": symbol,
            "fold": fold,
            "timeframe": timeframe,
            "lookback": lookback,
            "features": features,
            "hidden_size": hidden_size,
            "file_path": str(file_path),
            "file_size_bytes": file_size,
            "file_modified": datetime.fromtimestamp(
                file_path.stat().st_mtime).isoformat() if file_path.exists() else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_hit": False,  # همیشه False چون از کش استفاده نمی‌کنیم
            "source": "file_direct_read",
            "is_valid": True
        }

        # اضافه کردن محتوای اصلی برای سازگاری
        if isinstance(dims_data, dict):
            for key, value in dims_data.items():
                if key not in metadata and not key.startswith('_'):
                    metadata[key] = value

        return metadata

    except Exception as e:
        logger.error(f"[{symbol}] 💥 خطای بحرانی در load_model_dims: {e}", exc_info=True)

        return {
            "symbol": symbol,
            "fold": fold,
            "timeframe": timeframe,
            "lookback": None,
            "features": None,
            "hidden_size": None,
            "error": f"خطای سیستمی: {str(e)}",
            "error_type": "system_error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_hit": False,
            "source": "error"
        }




class SmartSyncBridge:  
    """
    پل ارتباطی هوشمند سینک به آسینک - FIXED VERSION (v11)
    ✅ بدون RuntimeWarning «coroutine was never awaited»: نسخهٔ قبلی coroutine
       را قبل از تصمیم‌گیری مسیر می‌ساخت و در مسیر thread-pool آن را رها می‌کرد.
    ✅ راه‌اندازی مجدد coroutine تازه برای مسیر بازیابی (coroutine مصرف‌شده قابل
       اجرای مجدد نیست).
    """

    _thread_local = threading.local()
    _default_loop: Optional[asyncio.AbstractEventLoop] = None
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="AsyncBridge")

    @classmethod
    def _get_or_create_loop(cls) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            if not hasattr(cls._thread_local, 'loop') or cls._thread_local.loop is None or cls._thread_local.loop.is_closed():
                cls._thread_local.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(cls._thread_local.loop)
            return cls._thread_local.loop

    @classmethod
    def _run_in_existing_loop(cls, coro, loop: asyncio.AbstractEventLoop) -> Any:
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=300)
        except Exception as e:
            logger.error(f"❌ _run_in_existing_loop failed: {e}")
            raise

    @classmethod
    def _run_in_new_loop(cls, coro) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            if "cannot be run" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            raise

    @classmethod
    def _run_in_thread_pool(cls, async_func: Callable, *args, **kwargs) -> Any:
        async def _wrapped():
            return await async_func(*args, **kwargs)

        try:
            future = cls._executor.submit(asyncio.run, _wrapped())
            return future.result(timeout=300)
        except Exception as e:
            logger.error(f"❌ _run_in_thread_pool failed: {e}")
            raise

    @classmethod
    def sync(cls, async_func: Callable, *args, **kwargs) -> Any:
        """
        تبدیل خودکار تابع آسنک به سینک با تشخیص هوشمند محیط
        ✅ CRITICAL FIX v11: حذف RuntimeWarning «coroutine was never awaited»
           و ساخت coroutine تازه در مسیر بازیابی.
        """
        result_or_coro = async_func(*args, **kwargs)

        if not asyncio.iscoroutine(result_or_coro):
            func_name = getattr(async_func, "__name__", str(async_func))
            logger.warning(f"⚠️ {func_name} did not return a coroutine, returning as-is")
            return result_or_coro

        coro = result_or_coro

        try:
            running_loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            # داخل یک event loop فعال هستیم (مثل Jupyter)؛ مسیر thread-pool را
            # انتخاب می‌کنیم. coroutine از‌پیش‌ساخته در این مسیر await نمی‌شود،
            # پس آن را می‌بندیم تا RuntimeWarning ثبت نشود؛ thread خودش نمونهٔ
            # تازه‌ای از coroutine را می‌سازد.
            try:
                coro.close()
            except Exception:
                pass
            return cls._run_in_thread_pool(async_func, *args, **kwargs)

        try:
            return cls._run_in_new_loop(coro)

        except Exception as e:
            func_name = getattr(async_func, "__name__", str(async_func))
            logger.error(f"❌ sync failed for {func_name}: {e}")

            # مسیر بازیابی: coroutine قبلی (به‌هرحال مصرف‌شده) را رها کرده و
            # نمونهٔ تازه می‌سازیم؛ coroutineِ تمام‌شده/خطادیده قابل await مجدد
            # نیست.
            try:
                fresh_coro = async_func(*args, **kwargs)
            except Exception as rebuild_error:
                logger.error(f"❌ could not rebuild coroutine for fallback: {rebuild_error}")
                raise

            if not asyncio.iscoroutine(fresh_coro):
                return fresh_coro

            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(fresh_coro)
            finally:
                new_loop.close()


# ============================================================
# FIXED: universal_load_sync
# ============================================================

def universal_load_sync(data_type: str, symbol: str = None, timeframe: str = None, 
                        fold: int = None, split_type: str = None, 
                        scaler_type: str = None, **kwargs) -> Any:
    """
    نسخه همگام universal_load - FIXED VERSION
    ✅ اجرای صحیح await
    ✅ مدیریت خطاهای coroutine
    """
    try:
        # اجرای مستقیم از طریق SmartSyncBridge
        result = SmartSyncBridge.sync(
            universal_load,
            data_type=data_type,
            symbol=symbol,
            timeframe=timeframe,
            fold=fold,
            split_type=split_type,
            scaler_type=scaler_type,
            **kwargs
        )
        
        # اعتبارسنجی نتیجه
        if result is None:
            logger.warning(f"⚠️ universal_load_sync returned None for data_type={data_type}, symbol={symbol}")
        else:
            logger.debug(f"✅ universal_load_sync successful for {data_type}")
        
        return result
        
    except asyncio.TimeoutError:
        logger.error(f"❌ Timeout loading {data_type}")
        return None
    except Exception as e:
        logger.error(f"❌ universal_load_sync failed for {data_type}: {e}", exc_info=True)
        return None


# ============================================================
# FIXED: universal_load_sync_pro
# ============================================================

def universal_load_sync_pro(data_type: str, symbol: str = None, timeframe: str = None,
                            fold: int = None, split_type: str = None,
                            scaler_type: str = None,
                            use_cache: bool = True,
                            timeout: int = 300,
                            **kwargs) -> Any:
    """
    نسخه نهایی و حرفه‌ای universal_load سینک - FIXED VERSION (v12)
    ✅ اجرای صحیح await با timeout
    ✅ کش هوشمند — اما ایمن در برابر kwargs های unhashable:
       نسخهٔ قبلی به‌طور مستقیم ``frozenset(kwargs.items())`` را می‌ساخت و اگر
       هر مقدار لیست/دیکشنری/آرایه می‌بود، قبل از هر کاری با TypeError می‌افتاد.
       در این نسخه ساخت کلید داخل try محافظت شده و در صورت شکست، کش به‌سادگی
       skip می‌شود.
    """
    # ساخت کلید کش (ایمن در برابر مقادیر unhashable)
    cache_key = None
    if use_cache:
        try:
            cache_key = (
                data_type,
                symbol,
                timeframe,
                fold,
                split_type,
                scaler_type,
                frozenset(kwargs.items()),
            )
            hash(cache_key)  # اعتبارسنجی hashable بودن نهایی
        except (TypeError, ValueError):
            cache_key = None

    if use_cache and cache_key is not None:
        if not hasattr(universal_load_sync_pro, "_cache"):
            universal_load_sync_pro._cache = {}
        if cache_key in universal_load_sync_pro._cache:
            logger.debug(f"✅ Cache hit for {data_type}")
            return universal_load_sync_pro._cache[cache_key]

    try:
        func_timeout = int(kwargs.pop("func_timeout", timeout))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                universal_load_sync,
                data_type,
                symbol=symbol,
                timeframe=timeframe,
                fold=fold,
                split_type=split_type,
                scaler_type=scaler_type,
                **kwargs,
            )
            result = future.result(timeout=func_timeout)

        if use_cache and cache_key is not None and result is not None:
            if not hasattr(universal_load_sync_pro, "_cache"):
                universal_load_sync_pro._cache = {}
            universal_load_sync_pro._cache[cache_key] = result

            if len(universal_load_sync_pro._cache) > 128:
                keys_to_remove = list(universal_load_sync_pro._cache.keys())[:32]
                for key in keys_to_remove:
                    del universal_load_sync_pro._cache[key]

        if result is None:
            logger.warning(f"⚠️ universal_load_sync_pro returned None for {data_type}")

        return result

    except concurrent.futures.TimeoutError:
        logger.error(f"❌ Timeout ({timeout}s) loading {data_type}")
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load {data_type}: {e}", exc_info=True)
        return None


# ============================================================
# FIXED: syncify decorator
# ============================================================

def syncify(async_func: Callable) -> Callable:
    """
    دکوراتور برای تبدیل خودکار هر تابع آسنک به سینک - FIXED VERSION
    ✅ اجرای صحیح await
    
    Usage:
        @syncify
        async def my_async_func(x, y):
            return x + y
        
        result = my_async_func(5, 3)
    """
    @wraps(async_func)
    def wrapper(*args, **kwargs):
        return SmartSyncBridge.sync(async_func, *args, **kwargs)
    return wrapper


# ============================================================
# FIXED: load_sync_simple
# ============================================================

def load_sync_simple(data_type: str, **kwargs):
    """
    نسخه بسیار ساده - بدون event loop - FIXED VERSION
    ✅ اجرای صحیح await
    
    Examples:
        data = load_sync_simple('sequenced', symbol='BTCUSD', split_type='train')
    """
    try:
        import asyncio
        # اجرای مستقیم
        return asyncio.run(universal_load(data_type, **kwargs))
    except RuntimeError as e:
        if "cannot be run" in str(e):
            # لوپ بسته شده - ایجاد لوپ جدید
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(universal_load(data_type, **kwargs))
            finally:
                loop.close()
        else:
            logger.error(f"❌ load_sync_simple failed: {e}")
            # Fallback به universal_load_sync
            return universal_load_sync(data_type, **kwargs)
    except Exception as e:
        logger.error(f"❌ load_sync_simple failed: {e}")
        return universal_load_sync(data_type, **kwargs)


# ============================================================
# FIXED: Example usage - load_my_data
# ============================================================

@syncify
async def load_my_data(symbol: str, data_type: str = "raw", **kwargs) -> Any:
    """
    بارگذاری داده‌ها با استفاده از syncify decorator - FIXED VERSION
    ✅ حالا سینک کار می‌کند
    """
    try:
        return await universal_load(data_type=data_type, symbol=symbol, **kwargs)
    except Exception as e:
        logger.error(f"❌ load_my_data failed for {symbol}: {e}")
        return None


# ============================================================
# Cached version for production
# ============================================================

class CachedSyncBridge(SmartSyncBridge):
    """نسخه با کش حافظه برای بارگذاری‌های تکراری - FIXED VERSION"""
    
    _cache = {}
    _cache_maxsize = 128
    
    @classmethod
    def sync(cls, async_func: Callable, *args, **kwargs) -> Any:
        # ساخت کلید کش (بر اساس نام تابع و آرگومان‌ها)
        cache_key = (async_func.__name__, args, frozenset(kwargs.items()))
        
        if cache_key in cls._cache:
            logger.debug(f"✅ Cache hit for {async_func.__name__}")
            return cls._cache[cache_key]
        
        result = super().sync(async_func, *args, **kwargs)
        
        # مدیریت اندازه کش
        if len(cls._cache) >= cls._cache_maxsize:
            # حذف 25% قدیمی‌ترین‌ها
            keys_to_remove = list(cls._cache.keys())[:cls._cache_maxsize // 4]
            for key in keys_to_remove:
                del cls._cache[key]
        
        if result is not None:
            cls._cache[cache_key] = result
        
        return result


# ============================================================
# DIRECT FIX FOR LINE 8734 AND 8786
# ============================================================

def universal_load_sync_direct(data_type: str, **kwargs) -> Any:
    """
    نسخه مستقیم و بی‌آلایش - برای خطوط 8734 و 8786
    ✅ بدون وابستگی به event loop پیچیده
    """
    try:
        # ایجاد event loop جدید
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # اجرای مستقیم با await
            result = loop.run_until_complete(universal_load(data_type, **kwargs))
            return result
        finally:
            # بستن لوپ
            loop.close()
    except Exception as e:
        logger.error(f"❌ direct load failed: {e}")
        return None

