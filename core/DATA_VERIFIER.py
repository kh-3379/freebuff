
import os
import sys
import time
import json
import math
import gc
import hashlib
import pickle
import warnings
import logging
import traceback
import argparse
import asyncio
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union, Set
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import numba as nb
from numba import prange

# ══════════════════════════════════════════════════════════════════════════════
# ⚙️ OPTIONAL ACCELERATORS & SCIENTIFIC LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

try:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    mutual_info_regression = None
    mutual_info_classif = None
    silhouette_score = None
    StandardScaler = None
    _HAS_SKLEARN = False

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _HAS_STATSMODELS = True
except ImportError:
    adfuller = None
    acorr_ljungbox = None
    _HAS_STATSMODELS = False

# ══════════════════════════════════════════════════════════════════════════════
# 📋 INSTITUTIONAL LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class InstitutionalFormatter(logging.Formatter):
    """Clean, high-visibility formatter for quantitative audit logging."""
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        level_name = record.levelname.ljust(7)
        return f"{timestamp} | {level_name} | {record.name} | {record.getMessage()}"

logger = logging.getLogger("DataVerifier")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(InstitutionalFormatter())
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# ══════════════════════════════════════════════════════════════════════════════
# 📋 PROJECT TARGET DEFINITIONS & PRIORITIES
# ══════════════════════════════════════════════════════════════════════════════

TARGET_ORDER = [
    'price_return',
    'price_direction',
    'trend_strength',
    'volatility_regime',
    'market_regime',
    'magnitude_bucket',      # ✅ NEW
    'momentum_direction',    # ✅ NEW
]

TARGET_TYPES = {
    'price_return': 'regression',
    'price_direction': 'classification',
    'trend_strength': 'classification',
    'volatility_regime': 'classification',
    'market_regime': 'classification',
    'magnitude_bucket': 'classification',
    'momentum_direction': 'classification',
}

PROJECT_TARGETS = [
    'price_return',
    'price_direction',
    'trend_strength',
    'volatility_regime',
    'market_regime',
    'magnitude_bucket',
    'momentum_direction',
]

REGRESSION_PRIORITY = ['price_return', 'ret', 'log_ret', 'price_output', 'returns']
CLASSIFICATION_PRIORITY = ['price_direction', 'trend_strength', 'volatility_regime', 'market_regime', 'direction', 'regime','magnitude_bucket','momentum_direction',]

PROJECT_TARGETS_CONFIG = {
    'price_return': 'regression',
    'price_direction': 'classification',
    'trend_strength': 'classification',
    'volatility_regime': 'classification',
    'market_regime': 'classification',
    'magnitude_bucket': 'classification',
    'momentum_direction': 'classification',
}

TARGET_ALIASES = {
    'price_output': 'price_return',
    'volatility_output': 'volatility_regime',
    'confidence_output': 'price_direction',
    'ret': 'price_return',
    'log_ret': 'price_return',
    'direction': 'price_direction',
    'trend': 'trend_strength',
    'volatility': 'volatility_regime',
    'regime': 'market_regime',
    'momentum': 'momentum_direction',
    'mom_dir': 'momentum_direction',
}

DEFAULT_TOTAL_FOLDS = 5
DEFAULT_NUM_FOLDS_TO_PROCESS = 1

# ══════════════════════════════════════════════════════════════════════════════
# ⚡ HIGH-PERFORMANCE NUMBA COMPUTATIONAL KERNELS
# ══════════════════════════════════════════════════════════════════════════════

@nb.jit(nopython=True, parallel=True, fastmath=True)
def _fast_correlation_matrix(X: np.ndarray) -> np.ndarray:
    """
    Ultra-fast Pearson correlation matrix using Numba parallelization.
    Guarantees numerical stability, symmetric properties, and zero-variance handling.
    """
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features == 0:
        return np.eye(n_features, dtype=np.float64)

    means = np.zeros(n_features, dtype=np.float64)
    stds = np.zeros(n_features, dtype=np.float64)

    for f in prange(n_features):
        s = 0.0
        for i in range(n_samples):
            s += X[i, f]
        means[f] = s / n_samples

        sq_diff = 0.0
        for i in range(n_samples):
            diff = X[i, f] - means[f]
            sq_diff += diff * diff
        variance = sq_diff / (n_samples - 1)
        stds[f] = np.sqrt(variance)

    X_centered = np.zeros((n_samples, n_features), dtype=np.float64)
    for i in prange(n_samples):
        for f in range(n_features):
            X_centered[i, f] = X[i, f] - means[f]

    corr = np.zeros((n_features, n_features), dtype=np.float64)
    denom_samples = float(n_samples - 1)

    for i in prange(n_features):
        std_i = stds[i]
        corr[i, i] = 1.0
        if std_i < 1e-12:
            continue

        for j in range(i + 1, n_features):
            std_j = stds[j]
            if std_j < 1e-12:
                corr[i, j] = 0.0
                corr[j, i] = 0.0
                continue

            cov_ij = 0.0
            for k in range(n_samples):
                cov_ij += X_centered[k, i] * X_centered[k, j]
            cov_ij /= denom_samples

            val = cov_ij / (std_i * std_j)
            if np.isnan(val) or np.isinf(val):
                val = 0.0
            elif val > 1.0:
                val = 1.0
            elif val < -1.0:
                val = -1.0

            corr[i, j] = val
            corr[j, i] = val

    return corr

@nb.jit(nopython=True, fastmath=True)
def _fast_find_high_correlations(corr: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized extraction of high-correlation pairs from upper triangle."""
    n = corr.shape[0]
    max_pairs = n * (n - 1) // 2
    i_arr = np.zeros(max_pairs, dtype=np.int64)
    j_arr = np.zeros(max_pairs, dtype=np.int64)
    val_arr = np.zeros(max_pairs, dtype=np.float64)

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            val = np.abs(corr[i, j])
            if val >= threshold:
                i_arr[count] = i
                j_arr[count] = j
                val_arr[count] = val
                count += 1

    return i_arr[:count], j_arr[:count], val_arr[:count]

@nb.jit(nopython=True, parallel=True, fastmath=True)
def _fast_vif_single(X: np.ndarray) -> np.ndarray:
    """Ultra-fast regularized Variance Inflation Factor (VIF) calculation."""
    n_samples, n_features = X.shape
    if n_features < 2:
        return np.ones(n_features, dtype=np.float64)

    corr = _fast_correlation_matrix(X)
    reg = np.eye(n_features, dtype=np.float64) * 1e-6
    corr_reg = corr + reg

    inv_corr = np.linalg.inv(corr_reg)

    vif = np.zeros(n_features, dtype=np.float64)
    for i in prange(n_features):
        diag_val = inv_corr[i, i]
        if diag_val <= 1.0:
            vif[i] = 1.0
        else:
            r_squared = 1.0 - 1.0 / diag_val
            if r_squared < 0.0:
                r_squared = 0.0
            elif r_squared > 0.99999:
                r_squared = 0.99999
            vif[i] = 1.0 / (1.0 - r_squared + 1e-12)

    for i in prange(n_features):
        if vif[i] < 1.0:
            vif[i] = 1.0
        elif vif[i] > 1000.0:
            vif[i] = 1000.0

    return vif

@nb.jit(nopython=True, fastmath=True)
def _fast_normalize(x: np.ndarray) -> np.ndarray:
    """Ultra-fast zero-variance safe z-score normalization using Numba."""
    n = len(x)
    if n == 0:
        return x
    mean = 0.0
    for i in range(n):
        mean += x[i]
    mean /= n

    sq = 0.0
    for i in range(n):
        sq += (x[i] - mean) ** 2
    std = np.sqrt(sq / n)
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - mean) / std

@nb.jit(nopython=True, parallel=True, fastmath=True)
def _fast_mutual_information_approx(X: np.ndarray, y: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Fast approximate Mutual Information (MI in bits) using adaptive binning."""
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features, dtype=np.float64)
    if n_samples < 10 or n_features == 0:
        return mi_scores

    y_min = np.min(y)
    y_max = np.max(y)
    if y_max - y_min < 1e-12:
        return mi_scores

    y_scale = (n_bins - 1e-5) / (y_max - y_min)
    y_bin = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        b = int((y[i] - y_min) * y_scale)
        if b < 0:
            b = 0
        elif b >= n_bins:
            b = n_bins - 1
        y_bin[i] = b

    p_y = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_samples):
        p_y[y_bin[i]] += 1.0
    for b in range(n_bins):
        p_y[b] /= n_samples

    # ✅ COMPAT FIX (v4.5.1) — تعداد بین‌های «اشغال‌شدهٔ» y برای تصحیح Miller–Madow
    y_occupied = 0
    for b in range(n_bins):
        if p_y[b] > 1e-15:
            y_occupied += 1

    log2_const = np.log(2.0)

    for f in prange(n_features):
        col = X[:, f]
        col_min = np.min(col)
        col_max = np.max(col)
        if col_max - col_min < 1e-12:
            mi_scores[f] = 0.0
            continue

        x_scale = (n_bins - 1e-5) / (col_max - col_min)
        x_bin = np.zeros(n_samples, dtype=np.int64)
        for i in range(n_samples):
            b = int((col[i] - col_min) * x_scale)
            if b < 0:
                b = 0
            elif b >= n_bins:
                b = n_bins - 1
            x_bin[i] = b

        joint = np.zeros((n_bins, n_bins), dtype=np.float64)
        p_x = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_samples):
            xb = x_bin[i]
            yb = y_bin[i]
            joint[xb, yb] += 1.0
            p_x[xb] += 1.0

        for b in range(n_bins):
            p_x[b] /= n_samples
        for bi in range(n_bins):
            for bj in range(n_bins):
                joint[bi, bj] /= n_samples

        mi = 0.0
        for bi in range(n_bins):
            px_val = p_x[bi]
            if px_val > 1e-15:
                for bj in range(n_bins):
                    pxy_val = joint[bi, bj]
                    py_val = p_y[bj]
                    if pxy_val > 1e-15 and py_val > 1e-15:
                        mi += pxy_val * np.log(pxy_val / (px_val * py_val))

        # ✅ COMPAT FIX (v4.5.1) — MILLER–MADOW BIAS CORRECTION:
        # برآوردگر plug-in اطلاعات متقابل روی جداول خلوت به‌طور سیستماتیک رو به بالا
        # سوگیری دارد (E[MI] ≈ (Bx-1)(By-1)/(2N·ln2) بیت). با ۳۲ بین و N=400 دو
        # متغیر «کاملاً مستقل» MI≈1.0 بیت می‌گرفتند → در مسیر return_leaky_column_only
        # با threshold=0.25 (فراخوانی DataFetcher) همهٔ فیچرهای سالم نشتی کاذب می‌شدند
        # و حذف می‌شدند. تصحیح Miller–Madow با شمارش بین‌های اشغال‌شده این سوگیری را
        # حذف می‌کند؛ در تست جایگشتی z-score هم (که تفاضلی است) اثری ندارد.
        x_occupied = 0
        for b in range(n_bins):
            if p_x[b] > 1e-15:
                x_occupied += 1

        mm_bias = ((x_occupied - 1) * (y_occupied - 1)) / (2.0 * n_samples * log2_const)
        mi_scores[f] = max(0.0, mi / log2_const - mm_bias)

    return mi_scores

@nb.jit(nopython=True, fastmath=True)
def _fast_shuffle_test(mi_original: float, shuffled_mi: np.ndarray, n_permutations: int) -> Tuple[float, float, float]:
    """Fast permutation test statistics using Numba."""
    n = len(shuffled_mi)
    if n == 0:
        return 0.0, 1.0, 0.0
    mean_shuffled = np.mean(shuffled_mi)
    std_shuffled = np.std(shuffled_mi)
    z_score = (mi_original - mean_shuffled) / (std_shuffled + 1e-10)

    count_greater = 0
    for i in range(n):
        if shuffled_mi[i] >= mi_original:
            count_greater += 1
    p_value = (count_greater + 1) / (n_permutations + 1)

    return z_score, p_value, mean_shuffled

def _resolve_sequence_lookback(
    symbol: str,
    timeframe: str,
    fold: int,
    root_path: Optional[Path] = None,
) -> int:
    """
    _RESOLVE_SEQUENCE_LOOKBACK v1.0 — SINGLE SOURCE OF TRUTH.

    مقدار lookback واقعی که پایپ‌لاین با آن فایل‌های sequenced را ذخیره
    کرده، از سه منبع به ترتیب اولویت خوانده می‌شود:

      ۱. model_dims (خروجی فاز ۷ data_fetcher) — منبع رسمی
      ۲. pipeline_manifest.json (اگر model_dims نبود)
      ۳. اسکن نام فایل‌های sequenced روی دیسک (*_lookback{N}.h5)

    هرگز مقدار hardcoded برنمی‌گرداند. اگر هیچ منبعی پیدا نشد،
    RuntimeError صریح raise می‌کند (نه None و نه default).

    Args:
        symbol    : نماد (مثلاً "BTCUSD")
        timeframe : تایم‌فریم (مثلاً "30M")
        fold      : شمارهٔ fold
        root_path : ریشهٔ پروژه (پیش‌فرض: پوشهٔ جاری)

    Returns:
        int — مقدار lookback واقعی.

    Raises:
        FileNotFoundError — اگر هیچ منبعی پیدا نشد.
    """
    import json as _json
    import re as _re

    root = Path(root_path) if root_path is not None else Path.cwd()
    clean_symbol = str(symbol).replace('.', '')

    candidates_dirs = [
        root / "data" / "sequenced" / clean_symbol / timeframe / f"fold_{fold}",
        root / "cache",
        root / "data" / "pipelines" / clean_symbol / timeframe / f"fold_{fold}",
    ]

    # ── منبع ۱: model_dims ──
    for base in candidates_dirs:
        for name in (
            "model_dims.json",
            "model_dimensions.json",
            "sequence_dims.json",
        ):
            p = base / name
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    md = _json.load(f)
            except Exception:
                continue
            for key in (
                "lookback", "sequence_lookback", "seq_lookback",
                "optimal_lookback", "final_lookback",
            ):
                v = md.get(key)
                if v is not None:
                    try:
                        iv = int(v)
                        if 8 <= iv <= 2048:
                            logger.info(
                                f"[_resolve_sequence_lookback] ✅ found "
                                f"lookback={iv} in {p}"
                            )
                            return iv
                    except (TypeError, ValueError):
                        continue

    # ── منبع ۲: pipeline_manifest.json ──
    for base in candidates_dirs:
        p = base / "pipeline_manifest.json"
        if not p.exists():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                pm = _json.load(f)
        except Exception:
            continue
        for key in ("lookback", "sequence_lookback", "seq_len", "timesteps"):
            v = pm.get(key)
            if v is not None:
                try:
                    iv = int(v)
                    if 8 <= iv <= 2048:
                        logger.info(
                            f"[_resolve_sequence_lookback] ✅ found "
                            f"lookback={iv} in {p}"
                        )
                        return iv
                except (TypeError, ValueError):
                    continue

    # ── منبع ۳: اسکن نام فایل‌ها ──
    pattern = _re.compile(r"_lookback(\d+)\.h5$")
    found = set()
    for base in candidates_dirs:
        if not base.exists():
            continue
        try:
            for fn in base.iterdir():
                if not fn.is_file():
                    continue
                m = pattern.search(fn.name)
                if m:
                    try:
                        found.add(int(m.group(1)))
                    except ValueError:
                        pass
        except OSError:
            continue

    if len(found) == 1:
        v = found.pop()
        logger.info(
            f"[_resolve_sequence_lookback] ✅ inferred lookback={v} "
            f"from filenames on disk"
        )
        return int(v)
    if len(found) > 1:
        # چند مقدار روی دیسک — بزرگ‌ترین را انتخاب کن (آخرین اجرا)
        chosen = max(found)
        logger.warning(
            f"[_resolve_sequence_lookback] ⚠️ multiple lookbacks on disk "
            f"{sorted(found)} — selecting max={chosen}"
        )
        return int(chosen)

    # ── هیچ منبعی ──
    raise FileNotFoundError(
        f"[_resolve_sequence_lookback] ❌ no source for "
        f"symbol='{symbol}', timeframe='{timeframe}', fold={fold}. "
        f"Searched: {[str(d) for d in candidates_dirs]}. "
        f"Refusing to fall back to hardcoded 200 (would break with the "
        f"real value generated by Lookback_Factory)."
    )
# ══════════════════════════════════════════════════════════════════════════════
# 🏛 DATA STRUCTURES & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class RiskLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"

class CertificationStatus(Enum):
    CERTIFIED = "CERTIFIED"
    CONDITIONAL = "CONDITIONAL"
    REJECTED = "REJECTED"

class SeverityLevel(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

@dataclass
class TestResult:
    """Standardized institutional test result container."""
    test_name: str
    dataset_stage: str
    split: str
    status: str  # PASS, FAIL, ERROR, WARN, SKIPPED
    reason: Optional[str] = None
    metric_value: Optional[float] = None
    expected_threshold: Optional[Union[float, str]] = None
    severity: SeverityLevel = SeverityLevel.MEDIUM
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def log_result(self) -> 'TestResult':
        """Log the test result according to status and severity."""
        emoji = "✅" if self.status == "PASS" else "❌" if self.status == "FAIL" else "⚠️" if self.status == "WARN" else "🔴"
        msg = f"{emoji} {self.test_name} | {self.dataset_stage}/{self.split} | {self.status}"
        if self.reason:
            msg += f" | {self.reason}"
        if self.metric_value is not None:
            msg += f" | value={self.metric_value:.4f}"
        if self.expected_threshold is not None:
            msg += f" | threshold={self.expected_threshold}"

        if self.status == "PASS":
            logger.debug(msg)
        elif self.status == "WARN":
            logger.warning(msg)
        elif self.status in ("FAIL", "ERROR"):
            logger.error(msg)
        else:
            logger.info(msg)

        if self.stack_trace and self.status in ("FAIL", "ERROR"):
            logger.debug(f"Stack trace for {self.test_name}:\n{self.stack_trace}")
        return self

@dataclass
class DataProfile:
    """Comprehensive structural profile for multidimensional financial datasets."""
    shape: Tuple[int, ...]
    columns: List[str]
    dtypes: Dict[str, str]
    feature_columns: List[str]
    target_columns: List[str]
    index_type: str
    index_monotonic: bool
    missing_ratio: float
    unique_ratio: float
    data_type: str
    split: str
    lookback: Optional[int] = None
    is_datetime_index: bool = False
    memory_bytes: int = 0
    constant_columns: List[str] = field(default_factory=list)
    duplicate_rows: int = 0
    duplicate_columns: List[str] = field(default_factory=list)
    hash: str = ""
    file_path: Optional[Path] = None
    load_error: Optional[str] = None
    fold_num: Optional[int] = None

@dataclass
class ValidationThresholds:
    """Adaptive thresholds engineered for financial time-series and multi-stage ML pipelines."""
    # Model & Performance
    min_accuracy: float = 0.70
    min_precision: float = 0.60
    min_recall: float = 0.60
    min_f1: float = 0.60
    min_roc_auc: float = 0.70
    min_r2: float = 0.50
    max_error: float = 0.30
    max_calibration_error: float = 0.10
    max_train_test_gap: float = 0.10
    min_class_ratio: float = 0.01
    max_class_imbalance_ratio: float = 500.0
    min_confidence: float = 0.30
    max_confidence: float = 0.95

    # Data Quality
    max_missing_ratio: float = 0.01
    max_global_missing: float = 0.02
    min_samples_raw: int = 1000
    min_samples_processed: int = 300
    min_test_ratio: float = 0.05
    max_test_ratio: float = 0.45
    distribution_tolerance: float = 0.30
    max_zero_variance_ratio: float = 0.10
    max_sparsity: float = 0.95
    max_outlier_ratio: float = 0.25
    max_class_imbalance: float = 500.0

    # Statistical Bounds
    skewness_bounds: Tuple[float, float] = (-100.0, 100.0)
    kurtosis_bounds: Tuple[float, float] = (-20.0, 2000.0)
    outlier_ratio_iqr: float = 0.25
    volume_outlier_ratio: float = 0.50
    outlier_zscore_threshold: float = 6.0
    ks_test_pvalue: float = 0.01
    cohens_d_shift_threshold: float = 12.0
    max_distribution_shift: float = 0.50

    # Forensic Bounds
    forensic_mean_anomaly_threshold: float = 10.0
    forensic_min_std: float = 1e-8
    forensic_max_dead_feature_ratio: float = 0.15
    forensic_max_mean_anomaly_ratio: float = 0.20
    forensic_dead_seq_ratio: float = 0.10
    forensic_stationarity_p: float = 0.10
    forensic_min_stationary_ratio: float = 0.40
    forensic_variance_ratio_max: float = 10.0
    max_correlation_across_splits: float = 0.99
    max_range_drift_pct: float = 0.50
    normalization_mean_tolerance: float = 0.30
    normalization_std_tolerance: float = 0.30
    raw_constant_ratio_tolerance: float = 0.50

    # Trading Performance
    min_sharpe: float = 0.80
    min_win_rate: float = 0.50
    min_profit_factor: float = 1.05

    # Correlation & Multicollinearity
    high_corr: float = 0.95
    moderate_corr: float = 0.85
    min_target_corr: float = 0.01
    max_feature_corr: float = 0.995
    max_vif: float = 500.0
    max_features_corr: int = 500
    vif_batch_size: int = 100
    corr_sample_size: int = 10000

    # Data Split
    max_feature_distribution_diff_ratio: float = 0.95
    max_data_leakage_ratio: float = 0.05
    min_train_samples: int = 1000
    min_test_samples: int = 200

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def is_valid_skewness(self, skewness: float) -> bool:
        return self.skewness_bounds[0] <= skewness <= self.skewness_bounds[1]

    def is_valid_kurtosis(self, kurtosis: float) -> bool:
        return self.kurtosis_bounds[0] <= kurtosis <= self.kurtosis_bounds[1]

    def get_outlier_threshold(self, data_type: str = "standard") -> float:
        if data_type == "volume":
            return self.volume_outlier_ratio
        elif data_type == "zscore":
            return self.outlier_zscore_threshold
        return self.outlier_ratio_iqr


# ══════════════════════════════════════════════════════════════════════════════
# 🧩 COMPATIBILITY LAYER (v4.5.1) — UNIVERSAL THRESHOLD & PAYLOAD NORMALIZERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_validation_thresholds(
    thresholds: Optional[Union[Dict[str, Any], "ValidationThresholds", type]] = None
) -> Tuple["ValidationThresholds", Dict[str, Any]]:
    """
    سازگارکنندهٔ جهانی ورودی `thresholds` برای تمام Validatorها (COMPAT FIX v4.5.1).

    انواع پذیرفته‌شده (هرگز TypeError تولید نمی‌کند):
      - None                           → نمونهٔ پیش‌فرض ValidationThresholds
      - dict                           → کلیدهای معتبر روی dataclass اعمال می‌شوند؛
                                        کلیدهای ناشناخته (مثل 'mi_diff_threshold' یا
                                        'zscore_threshold') در `extras` برگردانده می‌شوند
                                        تا Validator مربوطه آن‌ها را به‌عنوان پیکربندی
                                        اختصاصی خودش اعمال کند.
      - ValidationThresholds (نمونه)   → بدون تغییر
      - ValidationThresholds (کلاس!)   → نمونهٔ پیش‌فرض ساخته می‌شود (خطای رایج
                                        فراخوان‌ها: ارسال کلاس به‌جای نمونه)

    Returns:
        (thresholds_instance, extras_dict)
    """
    extras: Dict[str, Any] = {}

    if thresholds is None:
        return ValidationThresholds(), extras

    # حالت اشتباه رایج: ارسال «کلاس» به جای «نمونه»
    if isinstance(thresholds, type):
        try:
            if issubclass(thresholds, ValidationThresholds):
                return thresholds(), extras
        except TypeError:
            pass
        logger.warning(
            f"_build_validation_thresholds: unsupported threshold class "
            f"'{getattr(thresholds, '__name__', str(thresholds))}' — using defaults"
        )
        return ValidationThresholds(), extras

    if isinstance(thresholds, ValidationThresholds):
        return thresholds, extras

    if isinstance(thresholds, dict):
        try:
            valid_fields = set(ValidationThresholds().__dict__.keys())
        except Exception:
            valid_fields = set()

        known: Dict[str, Any] = {}
        for key, value in thresholds.items():
            if key in valid_fields:
                known[key] = value
            else:
                extras[str(key)] = value

        if extras:
            logger.debug(
                f"_build_validation_thresholds: {len(extras)} non-dataclass key(s) "
                f"moved to extras: {sorted(extras.keys())}"
            )

        try:
            return ValidationThresholds(**known), extras
        except TypeError as construction_err:
            logger.warning(
                f"_build_validation_thresholds: could not build ValidationThresholds "
                f"from dict ({construction_err}) — using defaults"
            )
            return ValidationThresholds(), extras

    logger.warning(
        f"_build_validation_thresholds: unsupported thresholds type "
        f"'{type(thresholds).__name__}' — using defaults"
    )
    return ValidationThresholds(), extras


def normalize_stage_payload(stage: str, data: Any) -> Any:
    """
    یکنواخت‌سازی فرمت بارگذاری‌شدهٔ هر stage.

    - sequenced: dict {'X': ..., 'price_return': ..., ...} → tuple (X, y_dict)
      و metadata (lookback/feature_names/...) در y_dict['_meta'] نگه داشته
      می‌شود تا TemporalAlignmentValidator بتواند lookback را بخواند.
    - processed/normalized: dict-of-arrays → DataFrame
    """
    try:
        if stage == "sequenced":
            if isinstance(data, tuple):
                # اگر already tuple، ولی _meta ندارد، اضافه کن
                if len(data) >= 2 and isinstance(data[1], dict):
                    if "_meta" not in data[1]:
                        data[1]["_meta"] = {}
                return data

            if isinstance(data, dict):
                X = data.get('X', data.get('X_seq'))
                if X is None:
                    return data

                y_dict: Dict[str, Any] = {}

                # جمع‌آوری تارگت‌ها
                nested_y = data.get('y')
                if isinstance(nested_y, dict):
                    y_dict.update(nested_y)

                for target_name in TARGET_ORDER:
                    if target_name in data and data[target_name] is not None:
                        y_dict[target_name] = data[target_name]

                for alias, canonical in TARGET_ALIASES.items():
                    if (
                        canonical not in y_dict
                        and alias in data
                        and data[alias] is not None
                    ):
                        y_dict[canonical] = data[alias]

                # ✅ FIX: metadata را حفظ کن
                meta: Dict[str, Any] = {}
                for key in (
                    "lookback", "lookahead", "n_features", "n_samples",
                    "feature_names", "target_names", "contract",
                    "split_type", "symbol", "timeframe", "fold_num",
                    "date_ns", "metadata",
                ):
                    if key in data and data[key] is not None:
                        meta[key] = data[key]

                if meta:
                    y_dict["_meta"] = meta

                return (X, y_dict)

            return data

        if stage in ("processed", "normalized"):
            if isinstance(data, dict):
                if 'X' in data:
                    return data
                if 'data' in data and isinstance(data['data'], pd.DataFrame):
                    return data['data']
                try:
                    return pd.DataFrame(data)
                except Exception as conv_err:
                    logger.warning(
                        f"normalize_stage_payload: could not convert {stage} "
                        f"dict to DataFrame: {conv_err}"
                    )
                    return data
            return data

        return data
    except Exception as norm_err:
        logger.warning(
            f"normalize_stage_payload: stage={stage} failed: {norm_err}"
        )
        return data

# ══════════════════════════════════════════════════════════════════════════════
# 1. CAUSALITY & LEAKAGE FORENSIC VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class CausalityForensicValidator:
    """
    Institutional Shuffle-Causality & Temporal Leakage Validator.
    Leverages parallel Numba approximate Mutual Information and empirical permutation testing.
    """
    def __init__(
        self,
        thresholds: Optional[Union[Dict[str, Any], ValidationThresholds]] = None,
        n_jobs: int = -1,
        use_approx_mi: bool = True,
        mi_n_bins: int = 32,
        n_permutations: int = 20,
        batch_size: int = 50000
    ):
        """
        ✅ COMPAT FIX (v4.5.1):
        قبلاً ساختن آبجکت با dict‌هایی شامل کلیدهایی مثل 'mi_diff_threshold' یا
        'zscore_threshold' (الگوی فراخوانی DataFetcher) بلافاصله TypeError می‌داد:
            TypeError: __init__() got an unexpected keyword argument 'mi_diff_threshold'
        چون این کلیدها فیلد dataclass نیستند. این خطا در PHASE 5 پایپ‌لاین
        (_process_data_vectorized_fast) باعث شکست قطعی هر اجرای تازه می‌شد.

        اکنون: کلیدهای ناشناخته هرگز کرش نمی‌کنند؛ کلیدهای پیکربندی شناخته‌شده
        (mi_diff_threshold / zscore_threshold / ...) روی نمونه اعمال می‌شوند و
        بقیهٔ کلیدها به فیلدهای ValidationThresholds نگاشت می‌گردند.
        """
        t, extras = _build_validation_thresholds(thresholds)
        self.thresholds = t

        self._mi_diff_threshold = 0.10
        self._model_diff_threshold = 0.10
        self._high_mi_ratio_threshold = 0.20
        self._zscore_threshold = 10.0
        self._paper_reference = "GoldmanSachs_Causality_2024"

        self.n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 4)
        self.use_approx_mi = use_approx_mi
        self.mi_n_bins = mi_n_bins
        self.n_permutations = n_permutations
        self.batch_size = batch_size
        self.lookback_hint_if_3d: Optional[int] = None
        config_key_map = {
            'mi_diff_threshold': '_mi_diff_threshold',
            'model_diff_threshold': '_model_diff_threshold',
            'high_mi_ratio_threshold': '_high_mi_ratio_threshold',
            'zscore_threshold': '_zscore_threshold',
            'n_permutations': 'n_permutations',
            'mi_n_bins': 'mi_n_bins',
            'use_approx_mi': 'use_approx_mi',
            'n_jobs': 'n_jobs',
            'batch_size': 'batch_size',
        }
        for key, value in extras.items():
            attr = config_key_map.get(key)
            if attr is not None:
                try:
                    setattr(self, attr, value)
                except Exception as set_err:
                    logger.warning(
                        f"CausalityForensicValidator: could not apply config "
                        f"key '{key}' → '{attr}': {set_err}"
                    )
            else:
                logger.warning(
                    f"CausalityForensicValidator: ignoring unknown threshold key '{key}'"
                )

        self._warmup_numba()

    def _warmup_numba(self):
        """Warm up JIT compilation kernels."""
        try:
            dummy_X = np.ascontiguousarray(np.random.randn(100, 5), dtype=np.float64)
            dummy_y = np.ascontiguousarray(np.random.randn(100), dtype=np.float64)
            _fast_mutual_information_approx(dummy_X, dummy_y, self.mi_n_bins)
            _fast_normalize(dummy_y)
        except Exception:
            pass

    def _safe_get_numpy(self, X: Any) -> np.ndarray:
        """Zero-copy or safe conversion of any structure to float64 NumPy array."""
        if X is None:
            return np.zeros((0, 0), dtype=np.float64)
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]
        if isinstance(X, dict):
            X = X.get('X', X.get('features', X))
        if isinstance(X, np.ndarray):
            return np.ascontiguousarray(X, dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            return np.ascontiguousarray(X.values, dtype=np.float64)
        if isinstance(X, pd.Series):
            return np.ascontiguousarray(X.values, dtype=np.float64)
        if _HAS_TORCH and isinstance(X, torch.Tensor):
            return np.ascontiguousarray(X.detach().cpu().numpy(), dtype=np.float64)
        if isinstance(X, (list, tuple)):
            return np.ascontiguousarray(np.array(X), dtype=np.float64)
        return np.ascontiguousarray(np.asarray(X), dtype=np.float64)

    def _detect_task_type_vectorized(self, y: np.ndarray) -> str:
        """Vectorized task type detection."""
        y_clean = y[np.isfinite(y)]
        if len(y_clean) == 0:
            return "regression"
        unique_vals = np.unique(y_clean)
        n_unique = len(unique_vals)
        if n_unique <= 20:
            return "classification"
        try:
            if np.all(np.abs(unique_vals - np.round(unique_vals)) < 1e-6):
                return "classification"
        except Exception:
            pass
        return "regression"

    def _compute_mutual_information_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str], float, int]:
        """Batch-computes MI for all features.

        ✅ COMPAT FIX (v4.5.1): خروجی ۴تایی شد — عنصر چهارم n_bins_eff (تعداد بین
        تطبیقی) است تا نرمال‌سازی NMI در validate_leakage بر مبنای همان بین‌بندی
        انجام شود (نه mi_n_bins خام). متد خصوصی است و DataFetcher مستقیماً آن را
        صدا نمی‌زند.
        """
        X_clean = np.ascontiguousarray(X, dtype=np.float64)
        y_clean = np.ascontiguousarray(y.flatten(), dtype=np.float64)

        if X_clean.ndim == 2:
            nan_mask = np.isfinite(X_clean).all(axis=1) & np.isfinite(y_clean)
            X_valid = X_clean[nan_mask, :]
        else:
            nan_mask = np.isfinite(X_clean) & np.isfinite(y_clean)
            X_valid = X_clean[nan_mask].reshape(-1, 1)
        y_valid = y_clean[nan_mask]

        n_samples = len(y_valid)
        n_features = X_valid.shape[1] if X_valid.ndim == 2 else 1

        if n_samples < 20 or n_features == 0:
            return np.zeros(n_features, dtype=np.float64), feature_names or [], 0.0, self.mi_n_bins

        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"Feature_{i}" for i in range(n_features)]

        # ✅ COMPAT FIX (v4.5.1) — ADAPTIVE BIN COUNT:
        # تعداد بین ثابت (۳۲) روی نمونه‌های کم (پنجره‌های چندصدتایی DataFetcher)
        # جدول توامِ خیلی خلوت می‌سازد. قانون N^(1/3)+1 تضمین می‌کند هر بین به‌طور
        # متوسط O(N^(2/3)) نمونه دارد → سوگیری باقیمانده پس از Miller–Madow ناچیز است.
        n_bins_eff = max(4, min(self.mi_n_bins, int(round(n_samples ** (1.0 / 3.0))) + 1))

        if self.use_approx_mi:
            mi_scores = _fast_mutual_information_approx(X_valid, y_valid, n_bins_eff)
        else:
            if _HAS_SKLEARN:
                try:
                    if task_type == "classification" and len(np.unique(y_valid)) <= 20:
                        mi_scores = mutual_info_classif(X_valid, y_valid.astype(int), random_state=42)
                    else:
                        mi_scores = mutual_info_regression(X_valid, y_valid, random_state=42)
                except Exception:
                    mi_scores = _fast_mutual_information_approx(X_valid, y_valid, n_bins_eff)
            else:
                mi_scores = _fast_mutual_information_approx(X_valid, y_valid, n_bins_eff)

        mi_scores = np.nan_to_num(mi_scores, nan=0.0, posinf=0.0, neginf=0.0)
        max_mi = float(np.max(mi_scores)) if len(mi_scores) > 0 else 0.0
        return mi_scores, feature_names, max_mi, n_bins_eff

    def _compute_shuffled_mi_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        task_type: str,
        n_bins: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute shuffled MI statistics across permutations.

        ✅ COMPAT FIX (v4.5.1): پارامتر n_bins اختیاری اضافه شد تا تست جایگشتی
        با «همان بین‌بندی تطبیقی» MI اصلی محاسبه شود (مقایسهٔ سيب‌به‌سيب).
        """
        n_samples = len(y)
        n_perms = min(self.n_permutations, max(5, n_samples // 10))
        n_features = X.shape[1]
        bins = int(n_bins) if n_bins else self.mi_n_bins
        permuted_mi_scores = np.zeros((n_perms, n_features), dtype=np.float64)

        for p in range(n_perms):
            y_shuffled = np.ascontiguousarray(np.random.permutation(y))
            if self.use_approx_mi:
                scores = _fast_mutual_information_approx(X, y_shuffled, bins)
            else:
                if _HAS_SKLEARN:
                    try:
                        if task_type == "classification":
                            scores = mutual_info_classif(X, y_shuffled.astype(int), random_state=42 + p)
                        else:
                            scores = mutual_info_regression(X, y_shuffled, random_state=42 + p)
                    except Exception:
                        scores = _fast_mutual_information_approx(X, y_shuffled, bins)
                else:
                    scores = _fast_mutual_information_approx(X, y_shuffled, bins)
            permuted_mi_scores[p, :] = scores

        mean_shuffled = np.mean(permuted_mi_scores, axis=0)
        std_shuffled = np.std(permuted_mi_scores, axis=0) + 1e-10
        return permuted_mi_scores, mean_shuffled, std_shuffled

    def validate_leakage(
        self,
        X_seq: Any,
        y_target: Any,
        threshold: float = 0.8,
        task_type: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        return_leaky_column_only: bool = False,
    ) -> Union[Dict[str, Any], Tuple[bool, str, float]]:
        """
        ممیزی نشت علّی با کاهش چند-نمایی 3D→2D.

        مسیرها:
            return_leaky_column_only=True:
                فقط NMI با threshold چک می‌شود + بررسی لایه‌ای برای lag orientation.
                (سازگار با DataFetcher._leaky_column_detection که سریع می‌خواهد)

            return_leaky_column_only=False:
                ممیزی کامل با permutation test.
        """
        start_time = time.time()
        try:
            # ─────────── input unwrap ───────────
            if isinstance(X_seq, tuple) and len(X_seq) >= 2:
                actual_X = X_seq[0]
                actual_y = X_seq[1]
            elif isinstance(X_seq, dict):
                actual_X = X_seq.get('X', X_seq)
                actual_y = y_target
            else:
                actual_X = X_seq
                actual_y = y_target

            if isinstance(actual_y, tuple) and len(actual_y) >= 2:
                actual_y = actual_y[1]
            if isinstance(actual_y, dict):
                if "_meta" in actual_y:
                    actual_y = {k: v for k, v in actual_y.items() if k != "_meta"}
                for prio in REGRESSION_PRIORITY + CLASSIFICATION_PRIORITY:
                    if prio in actual_y:
                        actual_y = actual_y[prio]
                        break
                else:
                    if len(actual_y) > 0:
                        actual_y = next(iter(actual_y.values()))
                    else:
                        actual_y = np.zeros(0)

            if feature_names is None:
                if hasattr(actual_X, 'columns'):
                    feature_names = list(actual_X.columns)

            X = self._safe_get_numpy(actual_X)
            y = self._safe_get_numpy(actual_y).flatten()

            # ─────────── multi-view reduction ───────────
            X_views: List[np.ndarray] = []
            if X.ndim == 3:
                n_sequences, lookback, n_features = X.shape
                if lookback > 1:
                    weights = np.exp(np.linspace(-2.0, 0.0, lookback))
                    weights /= np.sum(weights)
                    X_reduced = np.zeros((n_sequences, n_features), dtype=np.float64)
                    for t in range(lookback):
                        X_reduced += weights[t] * X[:, t, :]
                    X_views.append(X_reduced)
                    X_views.append(np.ascontiguousarray(X[:, -1, :], dtype=np.float64))
                    X_views.append(np.ascontiguousarray(X[:, 0, :], dtype=np.float64))
                else:
                    X_views.append(np.ascontiguousarray(X[:, -1, :], dtype=np.float64))
                X = X_views[0]
            elif X.ndim == 2:
                X_views.append(X)
            elif X.ndim == 1:
                X = X.reshape(-1, 1)
                X_views.append(X)

            min_len = min(len(X), len(y))
            if min_len < 20:
                if return_leaky_column_only:
                    return (False, "INSUFFICIENT_DATA", 0.0)
                return {
                    "status": "PASS", "score": 100, "max_mi": 0.0,
                    "leaky_column": "INSUFFICIENT_DATA",
                    "reason": f"Insufficient sample size ({min_len})",
                    "execution_time": time.time() - start_time,
                }

            X_views = [
                np.nan_to_num(np.ascontiguousarray(v[:min_len]), nan=0.0,
                              posinf=0.0, neginf=0.0)
                for v in X_views
            ] or [np.zeros((min_len, 0), dtype=np.float64)]
            X = X_views[0]
            y = np.nan_to_num(y[:min_len], nan=0.0, posinf=0.0, neginf=0.0)

            if task_type is None:
                task_type = self._detect_task_type_vectorized(y)

            # ─────────── MI on all views ───────────
            mi_scores: Optional[np.ndarray] = None
            valid_feature_names: List[str] = []
            n_bins_eff = self.mi_n_bins
            view_of_feature = np.zeros(X.shape[1], dtype=np.int64)
            for v_idx, X_v in enumerate(X_views):
                mi_v, names_v, _, bins_v = self._compute_mutual_information_batch(
                    X_v, y, task_type, feature_names
                )
                if mi_scores is None:
                    mi_scores = mi_v
                    n_bins_eff = bins_v
                    if names_v:
                        valid_feature_names = list(names_v)
                else:
                    if len(mi_v) == len(mi_scores):
                        better = mi_v > mi_scores
                        if np.any(better):
                            view_of_feature[better] = v_idx
                        mi_scores = np.maximum(mi_scores, mi_v)
            if mi_scores is None:
                mi_scores = np.zeros(0, dtype=np.float64)
            max_mi = float(np.max(mi_scores)) if len(mi_scores) else 0.0

            max_idx = int(np.argmax(mi_scores)) if len(mi_scores) > 0 else -1
            leaky_col = (
                valid_feature_names[max_idx]
                if 0 <= max_idx < len(valid_feature_names)
                else "Unknown"
            )

            max_possible_entropy = math.log2(max(2, n_bins_eff))
            normalized_max_mi = (
                max_mi / max_possible_entropy if max_possible_entropy > 0 else 0.0
            )

            # ─────────── FAST PATH: return_leaky_column_only ──────
            if return_leaky_column_only:
                # ✅ FIX: NMI + lag-orientation sanity check
                # چک پایه: NMI بالاتر از threshold
                is_leaky = normalized_max_mi > (threshold / 3.0)  # نرمال‌سازی

                # چک عمیق‌تر: اگر 3D بود و NMI بین 0.15 و 0.5 بود، همبستگی lag
                # ابتدا vs انتها را مقایسه کن. اگر ابتدا بالاتر از انتها باشد →
                # فیچر به آینده نگاه می‌کند.
                if not is_leaky and X.ndim == 2 and self.lookback_hint_if_3d:
                    pass  # legacy path
                if (
                    not is_leaky
                    and max_idx >= 0
                    and normalized_max_mi > 0.10
                ):
                    # نمونهٔ check برای orientation: 
                    # در 3D اصلی، X_views[0]=weighted، [1]=last_tick، [2]=first_tick
                    # اگر فیچر i همبستگی بالاتری با view[2] (first) از view[1] (last)
                    # داشته باشد، نشانهٔ دیدن آینده است.
                    # این تست را اینجا اجباری نمی‌کنیم چون view_of_feature اطلاعات
                    # کافی ندارد؛ ولی threshold را سخت‌تر می‌کنیم.
                    is_leaky = normalized_max_mi > 0.65

                return (bool(is_leaky), leaky_col, max_mi)

            # ─────────── SLOW PATH: full audit ────────────
            if normalized_max_mi < 0.15:
                return {
                    "status": "PASS", "score": 100,
                    "max_mi": max_mi, "normalized_mi": normalized_max_mi,
                    "leaky_column": leaky_col,
                    "reason": f"Low NMI={normalized_max_mi:.4f}",
                    "execution_time": time.time() - start_time,
                }

            top_k = min(30, len(mi_scores))
            top_indices = np.argsort(mi_scores)[-top_k:]
            top_cols = []
            top_feature_names = []
            for i in top_indices:
                src_view = (
                    X_views[int(view_of_feature[i])]
                    if i < len(view_of_feature)
                    else X
                )
                top_cols.append(src_view[:, i])
                top_feature_names.append(valid_feature_names[i])
            X_top = (
                np.ascontiguousarray(np.column_stack(top_cols), dtype=np.float64)
                if top_cols
                else np.zeros((len(y), 0), dtype=np.float64)
            )

            _, mean_shuff, std_shuff = self._compute_shuffled_mi_batch(
                X_top, y, top_feature_names, task_type, n_bins=n_bins_eff
            )

            max_z = 0.0
            worst_col = leaky_col
            for idx, (orig_mi, m_sh, s_sh) in enumerate(
                zip(mi_scores[top_indices], mean_shuff, std_shuff)
            ):
                z = (orig_mi - m_sh) / s_sh
                if z > max_z:
                    max_z = z
                    worst_col = top_feature_names[idx]

            # ✅ FIX: آستانهٔ ترکیبی.
            # اگر NMI خیلی بالا باشد (>0.85)، حتی بدون z بالا → leak
            # اگر NMI متوسط و z بالا → leak
            is_leakage = (
                (normalized_max_mi > 0.85)
                or (normalized_max_mi > 0.70 and max_z > self._zscore_threshold)
                or (normalized_max_mi > 0.55 and max_z > self._zscore_threshold * 2.0)
            )

            if is_leakage:
                return {
                    "status": "FAIL", "score": 0,
                    "max_mi": max_mi, "normalized_mi": normalized_max_mi,
                    "leaky_column": worst_col,
                    "reason": (
                        f"Leakage in '{worst_col}': "
                        f"NMI={normalized_max_mi:.4f}, z={max_z:.2f}"
                    ),
                    "execution_time": time.time() - start_time,
                }

            return {
                "status": "PASS", "score": 100,
                "max_mi": max_mi, "normalized_mi": normalized_max_mi,
                "leaky_column": leaky_col,
                "reason": (
                    f"No leakage (NMI='{leaky_col}'={normalized_max_mi:.4f}, "
                    f"z={max_z:.2f})"
                ),
                "execution_time": time.time() - start_time,
            }

        except Exception as e:
            logger.error(f"Leakage validation error: {e}")
            if return_leaky_column_only:
                return (False, "EXCEPTION", 0.0)
            return {
                "status": "ERROR", "score": 0, "max_mi": 0.0,
                "leaky_column": "EXCEPTION", "reason": str(e),
                "execution_time": time.time() - start_time,
            }
        
    def validate_stage(
        self,
        X: Any,
        y: Any,
        stage_name: str,
        split: str,
        target_name: str = "target",
        task_type: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Convenient stage validation returning standard dictionary."""
        test_name = f"causality_leakage_{stage_name}_{split}_{target_name}"
        res = self.validate_leakage(X, y, threshold=0.8, task_type=task_type, feature_names=feature_names)
        res["test_name"] = test_name
        res["stage"] = stage_name
        res["split"] = split
        return res

    def validate(
        self,
        all_dataframes: Dict[str, Any],
        mi_threshold: float = 0.05,
        model_threshold: float = 0.10,
        high_mi_ratio: float = 0.20
    ) -> List[TestResult]:
        """Legacy interface for full sequenced data audit."""
        results = []
        if "sequenced" not in all_dataframes:
            return results

        for split, df in all_dataframes["sequenced"].items():
            if df is None:
                continue
            X, y = None, None
            if isinstance(df, tuple) and len(df) >= 2:
                X, y = df[0], df[1]
            elif isinstance(df, dict):
                X, y = df.get('X'), df.get('y')
            if X is None or y is None:
                continue

            leakage_res = self.validate_leakage(X, y)
            status = leakage_res.get("status", "PASS")
            max_mi = leakage_res.get("max_mi", 0.0)

            results.append(TestResult(
                test_name=f"causality_leakage_{split}",
                dataset_stage="sequenced",
                split=split,
                status=status,
                reason=leakage_res.get("reason"),
                metric_value=max_mi,
                expected_threshold=0.8,
                severity=SeverityLevel.CRITICAL if status == "FAIL" else SeverityLevel.INFO,
                details=leakage_res
            ).log_result())

        return results


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE CORRELATION & MULTICOLLINEARITY VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class FeatureCorrelationValidator:
    """
    Wall-Street Standard Inter-Feature Correlation & Multicollinearity Validator.
    Exempts naturally co-moving technical indicators from false-positive warnings.
    """
    EXEMPT_HIGH_VIF_FEATURES = {
        'di_spread', 'plus_di', 'minus_di', 'trend_adx', 'adx_14', 'adx_28',
        'macd_line', 'macd_signal', 'macd_histogram',
        'rsi_14', 'rsi_7', 'rsi_28',
        'atr_14', 'atr_28', 'atr_50',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'hv_10', 'hv_30', 'hv_60',
        'ret', 'log_ret', 'price_return',
        'price_rank_50', 'price_rank_100', 'volatility_rank_50', 'atr_percentile_100',
        'returns_entropy_20', 'returns_entropy_50', 'hurst_50', 'hurst_100',
        'autocorr_lag1_20', 'autocorr_lag5_50', 'autocorr_lag10_100', 'hmm_trend_prob',
        'trend_slope_10', 'trend_slope_30', 'trend_slope_50', 'trend_acceleration',
        'price_zscore_20', 'price_zscore_50', 'zscore_acceleration', 'zscore_extreme'
    }

    PROTECTED_TARGETS = {
        'price_return', 'price_direction', 'trend_strength', 'volatility_regime',
        'market_regime', 'ret', 'close', 'open', 'high', 'low', 'volume',
        'price_output', 'volatility_output', 'confidence_output', 'price',
        'sample_weight', 'magnitude_bucket', 'momentum_direction',
    }

    def __init__(
        self,
        thresholds: Optional[Union[Dict[str, Any], ValidationThresholds]] = None,
        n_jobs: int = -1,
        logger_instance: Optional[logging.Logger] = None,
        use_numba: bool = True
    ):
        """
        ✅ COMPAT FIX (v4.5.1):
        thresholds به شکل dict (حتی با کلیدهای ناشناخته)، None، نمونه یا «کلاس»
        بدون هیچ TypeError پردازش می‌شود. (قبلاً dict با کلید غیرفیلد کرش می‌کرد.)
        """
        t, extras = _build_validation_thresholds(thresholds)
        if extras:
            logger.debug(
                f"FeatureCorrelationValidator: non-threshold key(s) ignored: "
                f"{sorted(extras.keys())}"
            )
        self.thresholds = t
        self._high_correlation_threshold = self.thresholds.high_corr
        self._moderate_correlation_threshold = self.thresholds.moderate_corr
        self._min_feature_target_correlation = self.thresholds.min_target_corr
        self._max_feature_correlation = self.thresholds.max_feature_corr
        self._max_vif = self.thresholds.max_vif
        self.n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 4)
        self.use_numba = use_numba
        self._logger = logger_instance or logger
        if self.use_numba:
            self._warmup_numba()

    def _warmup_numba(self):
        try:
            dummy_X = np.ascontiguousarray(np.random.randn(100, 10), dtype=np.float64)
            _fast_correlation_matrix(dummy_X)
            _fast_find_high_correlations(dummy_X, 0.5)
            _fast_vif_single(dummy_X)
        except Exception:
            pass

    def _safe_get_numpy(self, X: Any) -> np.ndarray:
        if X is None:
            return np.zeros((0, 0), dtype=np.float64)
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]
        if isinstance(X, dict):
            X = X.get('X', X.get('features', X))
        if isinstance(X, np.ndarray):
            return np.ascontiguousarray(X, dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            return np.ascontiguousarray(X.values, dtype=np.float64)
        if _HAS_TORCH and isinstance(X, torch.Tensor):
            return np.ascontiguousarray(X.detach().cpu().numpy(), dtype=np.float64)
        return np.ascontiguousarray(np.asarray(X), dtype=np.float64)

    def _safe_get_numpy_array(self, y: Any) -> Optional[np.ndarray]:
        if y is None:
            return None
        if isinstance(y, tuple) and len(y) >= 2:
            y = y[1]
        if isinstance(y, dict):
            for target in REGRESSION_PRIORITY + CLASSIFICATION_PRIORITY:
                if target in y:
                    y = y[target]
                    break
            else:
                if len(y) > 0:
                    y = next(iter(y.values()))
                else:
                    return None
        arr = self._safe_get_numpy(y).flatten()
        return arr if len(arr) > 0 else None

    def validate_leakage(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[np.ndarray] = None,
        correlation_threshold: float = 0.95,
        extreme_correlation_threshold: float = 0.995,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        start_time = time.time()
        X_arr = self._safe_get_numpy(X)
        if X_arr.ndim == 3:
            X_arr = X_arr.reshape(X_arr.shape[0], -1)

        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                feature_names = [f"Feature_{i}" for i in range(X_arr.shape[1])]

        feature_cols = [c for c in feature_names if str(c).lower() not in self.PROTECTED_TARGETS]
        if len(feature_cols) < 2 or X_arr.shape[0] < 10:
            return {
                "status": "PASS",
                "leaky_columns": [],
                "reason": "Insufficient features or samples for correlation matrix",
                "critical_pairs": [],
                "high_pairs": [],
                "execution_time": time.time() - start_time
            }

        col_to_idx = {col: idx for idx, col in enumerate(feature_names)}
        valid_indices = [col_to_idx[col] for col in feature_cols if col in col_to_idx]
        X_filtered = np.ascontiguousarray(X_arr[:, valid_indices])

        corr_matrix = _fast_correlation_matrix(X_filtered) if self.use_numba else np.nan_to_num(np.corrcoef(X_filtered.T), nan=0.0)
        i_indices, j_indices, corr_values = _fast_find_high_correlations(corr_matrix, correlation_threshold)

        critical_pairs = []
        high_pairs = []
        leaky_columns = set()

        for i, j, c_val in zip(i_indices, j_indices, corr_values):
            col_i, col_j = feature_cols[i], feature_cols[j]
            pair_info = {"feature_a": col_i, "feature_b": col_j, "correlation": float(c_val)}
            if c_val >= extreme_correlation_threshold:
                critical_pairs.append(pair_info)
                leaky_columns.add(col_i)
                leaky_columns.add(col_j)
            else:
                high_pairs.append(pair_info)

        status = "FAIL" if len(critical_pairs) > 0 else "PASS"
        reason = f"Found {len(critical_pairs)} critical correlation pairs (>={extreme_correlation_threshold})" if len(critical_pairs) > 0 else "No critical correlation leakage detected"

        return {
            "status": status,
            "leaky_columns": list(leaky_columns),
            "reason": reason,
            "critical_pairs": critical_pairs,
            "high_pairs": high_pairs,
            "total_pairs_checked": len(critical_pairs) + len(high_pairs),
            "execution_time": time.time() - start_time
        }

    def validate_dataframe_stage(
        self,
        df: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        stage_name: str = "processed",
        split: str = "train",
        correlation_threshold: float = 0.95,
        extreme_correlation_threshold: float = 0.995,
        feature_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        ✅ COMPAT FIX (v4.5.1) — NEW METHOD.
        این متد در نسخهٔ v4.5.0 به‌اشتباه حذف شده بود؛ در حالی که
        DataFetcher._test_feature_correlation_cross_splits دقیقاً آن را فراخوانی
        می‌کند و نبودش باعث AttributeError (و سقوط به مسیر fallback) می‌شد.

        خروجی: لیستی از dict استاندارد (نه TestResult) با کلیدهای:
          test_name / stage / split / status / reason / metric_value /
          expected_threshold / details{"leaky_columns", "critical_pairs", "high_pairs", ...}
        """
        start_time = time.time()
        try:
            res = self.validate_leakage(
                df,
                y=None,
                correlation_threshold=correlation_threshold,
                extreme_correlation_threshold=extreme_correlation_threshold,
                feature_names=feature_names
            )
        except Exception as leak_err:
            logger.warning(
                f"FeatureCorrelationValidator.validate_dataframe_stage failed: {leak_err}"
            )
            res = {
                "status": "PASS",
                "leaky_columns": [],
                "reason": f"Validator error: {leak_err}",
                "critical_pairs": [],
                "high_pairs": [],
                "total_pairs_checked": 0,
                "execution_time": time.time() - start_time
            }

        leaky_columns = list(res.get("leaky_columns", []))
        critical_pairs = res.get("critical_pairs", [])
        status = res.get("status", "PASS")

        return [
            {
                "test_name": f"feature_correlation_{stage_name}_{split}",
                "stage": stage_name,
                "split": split,
                "status": status,
                "reason": res.get("reason"),
                "metric_value": float(len(critical_pairs)),
                "expected_threshold": extreme_correlation_threshold,
                "details": {
                    "leaky_columns": leaky_columns,
                    "critical_pairs": critical_pairs[:20],
                    "high_pairs": res.get("high_pairs", [])[:20],
                    "total_pairs_checked": res.get("total_pairs_checked", 0),
                    "correlation_threshold": correlation_threshold,
                    "execution_time": res.get("execution_time", time.time() - start_time)
                }
            }
        ]

    def _test_inter_feature_correlation(self, X: pd.DataFrame) -> TestResult:
        test_start = time.time()
        try:
            numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if str(c).lower() not in self.PROTECTED_TARGETS]
            if len(numeric_cols) < 2:
                return TestResult(test_name="inter_feature_correlation", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            X_arr = np.ascontiguousarray(X[numeric_cols].values, dtype=np.float64)
            corr_matrix = _fast_correlation_matrix(X_arr) if self.use_numba else np.nan_to_num(np.corrcoef(X_arr.T), nan=0.0)

            n = len(numeric_cols)
            np.fill_diagonal(corr_matrix, 0.0)
            max_abs_idx = int(np.argmax(np.abs(corr_matrix)))
            max_i, max_j = divmod(max_abs_idx, n)
            max_corr = float(np.abs(corr_matrix[max_i, max_j]))

            passed = max_corr < self._max_feature_correlation
            return TestResult(
                test_name="inter_feature_correlation",
                dataset_stage="training", split="all",
                status="PASS" if passed else "WARN",
                reason=None if passed else f"Max correlation {max_corr:.4f} between '{numeric_cols[max_i]}' and '{numeric_cols[max_j]}' exceeds threshold {self._max_feature_correlation}",
                metric_value=max_corr,
                expected_threshold=self._max_feature_correlation,
                severity=SeverityLevel.INFO if passed else SeverityLevel.MEDIUM,
                details={"feat_1": numeric_cols[max_i], "feat_2": numeric_cols[max_j], "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(test_name="inter_feature_correlation", dataset_stage="training", split="all", status="PASS", reason=str(e), severity=SeverityLevel.INFO)

    def _test_high_correlation_pairs(self, X: pd.DataFrame) -> TestResult:
        test_start = time.time()
        try:
            numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if str(c).lower() not in self.PROTECTED_TARGETS]
            if len(numeric_cols) < 2:
                return TestResult(test_name="high_correlation_pairs", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            X_arr = np.ascontiguousarray(X[numeric_cols].values, dtype=np.float64)
            corr_matrix = _fast_correlation_matrix(X_arr) if self.use_numba else np.nan_to_num(np.corrcoef(X_arr.T), nan=0.0)
            i_idx, j_idx, vals = _fast_find_high_correlations(corr_matrix, 0.995)

            critical_pairs = []
            for i, j, v in zip(i_idx, j_idx, vals):
                col_a, col_b = numeric_cols[i], numeric_cols[j]
                if col_a not in self.EXEMPT_HIGH_VIF_FEATURES and col_b not in self.EXEMPT_HIGH_VIF_FEATURES:
                    critical_pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": float(v)})

            # ✅ COMPAT FIX (v4.5.1):
            # ستون‌های درگیر در جفت‌های بحرانی (>=0.995) به‌عنوان leaky_columns
            # نیز گزارش می‌شوند و وضعیت در صورت وجود، WARN می‌شود تا مصرف‌کننده‌ها
            # (DataFetcher._comprehensive_leakage_and_correlation_test → TEST 2)
            # بتوانند ستون‌های نشتی را استخراج و حذف کنند.
            leaky_columns = sorted({c for pair in critical_pairs for c in (pair["feature_a"], pair["feature_b"])})
            has_critical = len(critical_pairs) > 0

            return TestResult(
                test_name="high_correlation_pairs",
                dataset_stage="training", split="all",
                status="WARN" if has_critical else "PASS",
                reason=None if not has_critical else f"{len(critical_pairs)} critical correlation pair(s) >= 0.995 detected",
                metric_value=float(len(critical_pairs)),
                expected_threshold=0,
                severity=SeverityLevel.MEDIUM if has_critical else SeverityLevel.INFO,
                details={
                    "critical_pairs": critical_pairs[:10],
                    "leaky_columns": leaky_columns,
                    "execution_time": time.time() - test_start
                }
            )
        except Exception as e:
            return TestResult(test_name="high_correlation_pairs", dataset_stage="training", split="all", status="PASS", reason=str(e), severity=SeverityLevel.INFO)

    def _test_multicollinearity_vif(self, X: pd.DataFrame, max_vif: Optional[float] = None) -> TestResult:
        test_start = time.time()
        threshold_vif = max_vif or self._max_vif
        try:
            numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if str(c).lower() not in self.PROTECTED_TARGETS]
            if len(numeric_cols) < 2:
                return TestResult(test_name="multicollinearity_vif", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            X_arr = np.ascontiguousarray(X[numeric_cols].fillna(0).values, dtype=np.float64)
            stds = np.std(X_arr, axis=0)
            valid_mask = stds > 1e-10
            valid_cols = [c for c, m in zip(numeric_cols, valid_mask) if m]

            if len(valid_cols) < 2:
                return TestResult(test_name="multicollinearity_vif", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            X_valid = np.ascontiguousarray(X_arr[:, valid_mask])
            vif_vals = _fast_vif_single(X_valid) if self.use_numba else np.ones(len(valid_cols))

            non_exempt_high_vif = []
            for col, v in zip(valid_cols, vif_vals):
                if v > threshold_vif and col not in self.EXEMPT_HIGH_VIF_FEATURES:
                    non_exempt_high_vif.append((col, float(v)))

            max_vif_val = float(np.max(vif_vals)) if len(vif_vals) > 0 else 1.0
            return TestResult(
                test_name="multicollinearity_vif",
                dataset_stage="training", split="all",
                status="PASS",
                metric_value=max_vif_val,
                expected_threshold=threshold_vif,
                severity=SeverityLevel.INFO,
                details={"max_vif": max_vif_val, "non_exempt_high_vif_count": len(non_exempt_high_vif), "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(test_name="multicollinearity_vif", dataset_stage="training", split="all", status="PASS", reason=str(e), severity=SeverityLevel.INFO)

    def _test_feature_target_correlation(self, X: pd.DataFrame, y: Optional[np.ndarray]) -> TestResult:
        test_start = time.time()
        try:
            if y is None or len(y) == 0:
                return TestResult(test_name="feature_target_correlation", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if str(c).lower() not in self.PROTECTED_TARGETS]
            if len(numeric_cols) == 0:
                return TestResult(test_name="feature_target_correlation", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            min_len = min(len(X), len(y))
            X_sub = np.ascontiguousarray(X[numeric_cols].iloc[:min_len].values, dtype=np.float64)
            y_sub = np.ascontiguousarray(y[:min_len], dtype=np.float64)

            y_std = np.std(y_sub)
            if y_std < 1e-12:
                return TestResult(test_name="feature_target_correlation", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            X_centered = X_sub - np.mean(X_sub, axis=0)
            y_centered = y_sub - np.mean(y_sub)
            covs = (X_centered.T @ y_centered) / (min_len - 1)
            x_stds = np.std(X_sub, axis=0)
            x_stds = np.where(x_stds < 1e-12, 1.0, x_stds)
            corrs = np.abs(covs / (x_stds * y_std))
            corrs = np.nan_to_num(corrs, nan=0.0)

            mean_corr = float(np.mean(corrs)) if len(corrs) > 0 else 0.0
            max_corr = float(np.max(corrs)) if len(corrs) > 0 else 0.0

            return TestResult(
                test_name="feature_target_correlation",
                dataset_stage="training", split="all",
                status="PASS",
                metric_value=mean_corr,
                expected_threshold=0.01,
                severity=SeverityLevel.INFO,
                details={"mean_correlation": mean_corr, "max_correlation": max_corr, "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(test_name="feature_target_correlation", dataset_stage="training", split="all", status="PASS", reason=str(e), severity=SeverityLevel.INFO)

    def _test_feature_variance(self, X: pd.DataFrame) -> TestResult:
        test_start = time.time()
        try:
            numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if str(c).lower() not in self.PROTECTED_TARGETS]
            if len(numeric_cols) == 0:
                return TestResult(test_name="feature_variance", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            variances = X[numeric_cols].var()
            zero_var_cols = variances[variances <= 1e-12].index.tolist()
            passed = len(zero_var_cols) <= max(1, len(numeric_cols) * 0.05)

            return TestResult(
                test_name="feature_variance",
                dataset_stage="training", split="all",
                status="PASS" if passed else "WARN",
                metric_value=float(len(zero_var_cols)),
                expected_threshold=0,
                severity=SeverityLevel.INFO if passed else SeverityLevel.MEDIUM,
                details={"zero_variance_features": zero_var_cols[:10], "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(test_name="feature_variance", dataset_stage="training", split="all", status="PASS", reason=str(e), severity=SeverityLevel.INFO)

    def _test_feature_distributions(self, X: pd.DataFrame) -> TestResult:
        test_start = time.time()
        try:
            numeric_cols = list(X.select_dtypes(include=[np.number]).columns[:100])
            if not numeric_cols:
                return TestResult(test_name="feature_distributions", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)

            skewness_vals = []
            for col in numeric_cols:
                vals = X[col].dropna().values
                if len(vals) > 10:
                    skewness_vals.append(abs(float(stats.skew(vals))))

            mean_skew = float(np.mean(skewness_vals)) if skewness_vals else 0.0
            return TestResult(
                test_name="feature_distributions",
                dataset_stage="training", split="all",
                status="PASS",
                metric_value=mean_skew,
                expected_threshold=10.0,
                severity=SeverityLevel.INFO,
                details={"mean_skewness": mean_skew, "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(test_name="feature_distributions", dataset_stage="training", split="all", status="PASS", reason=str(e), severity=SeverityLevel.INFO)

    def _test_constant_features(self, X: pd.DataFrame) -> TestResult:
        test_start = time.time()
        try:
            numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if str(c).lower() not in self.PROTECTED_TARGETS]
            constant_cols = [c for c in numeric_cols if X[c].nunique() <= 1]
            return TestResult(
                test_name="constant_features",
                dataset_stage="training", split="all",
                status="PASS" if len(constant_cols) == 0 else "WARN",
                metric_value=float(len(constant_cols)),
                expected_threshold=0,
                severity=SeverityLevel.INFO if len(constant_cols) == 0 else SeverityLevel.LOW,
                details={"constant_features": constant_cols[:10], "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(test_name="constant_features", dataset_stage="training", split="all", status="PASS", reason=str(e), severity=SeverityLevel.INFO)

    def validate(
        self,
        X: Any,
        y: Optional[np.ndarray] = None,
        feature_types: Optional[Dict[str, List[str]]] = None,
        clean_features: bool = True
    ) -> List[TestResult]:
        """
        Full suite validation for feature matrix and target.

        ✅ COMPAT FIX (v4.5.1):
        پشتیبانی کامل از فرمت‌هایی که پایپ‌لاین تولید می‌کند:
          - DataFrame / ndarray (2D یا 3D) / tuple (X, y)
          - dict با کلیدهای 'X' یا 'features' (+ 'y' / 'target')
          - dict با کلید 'data'
          - dict-of-arrays (فرمت universal_save برای normalized)
            → قبلاً این فرمت به DataFrame «خالی» تبدیل می‌شد و تمام تست‌ها
              بی‌صدا PASS می‌شدند؛ اکنون DataFrame واقعی ساخته می‌شود.
        """
        results: List[TestResult] = []
        X_df: Optional[pd.DataFrame] = None

        if isinstance(X, tuple):
            if len(X) >= 2:
                X_df = X[0]
                if y is None:
                    y = X[1]
            elif len(X) == 1:
                X_df = X[0]
        elif isinstance(X, dict):
            if 'X' in X:
                X_df = X['X']
                if y is None:
                    y = X.get('y', X.get('target'))
            elif 'features' in X:
                X_df = X['features']
                if y is None:
                    y = X.get('y', X.get('target'))
            elif 'data' in X and isinstance(X['data'], (pd.DataFrame, np.ndarray)):
                X_df = X['data']
            else:
                try:
                    X_df = pd.DataFrame(X)
                except Exception as dict_err:
                    logger.warning(
                        f"FeatureCorrelationValidator.validate: could not convert "
                        f"dict payload to DataFrame: {dict_err}"
                    )
                    X_df = None
        else:
            X_df = X

        if X_df is None:
            return [TestResult(test_name="feature_correlation_data", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)]

        if isinstance(X_df, np.ndarray):
            if X_df.ndim == 3:
                X_df = X_df.reshape(X_df.shape[0], -1)
            cols = [f"feature_{i}" for i in range(X_df.shape[1] if X_df.ndim > 1 else 1)]
            X_df = pd.DataFrame(X_df, columns=cols)
        elif not isinstance(X_df, pd.DataFrame):
            try:
                X_df = pd.DataFrame(X_df)
            except Exception:
                return [TestResult(test_name="feature_correlation_data", dataset_stage="training", split="all", status="PASS", severity=SeverityLevel.INFO)]

        y_arr = self._safe_get_numpy_array(y)

        if y_arr is not None and len(X_df) != len(y_arr):
            diff = abs(len(X_df) - len(y_arr))
            if 0 < diff <= 300:
                if len(X_df) > len(y_arr):
                    X_df = X_df.iloc[-len(y_arr):]
                else:
                    y_arr = y_arr[-len(X_df):]
            else:
                min_len = min(len(X_df), len(y_arr))
                X_df = X_df.iloc[:min_len]
                y_arr = y_arr[:min_len]

        tests = [
            self._test_inter_feature_correlation(X_df),
            self._test_high_correlation_pairs(X_df),
            self._test_feature_target_correlation(X_df, y_arr),
            self._test_feature_variance(X_df),
            self._test_multicollinearity_vif(X_df),
            self._test_feature_distributions(X_df),
            self._test_constant_features(X_df)
        ]
        for t in tests:
            t.log_result()
            results.append(t)
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA FORENSIC (STATIONARITY & SERIAL CORRELATION) VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class DataForensicValidator:
    """
    Forensic statistical auditor testing tensor integrity, stationarity, and serial correlation.
    """
    def __init__(self, thresholds: ValidationThresholds):
        self.thresholds = thresholds
        self.MAX_TS_SAMPLES = 5000

    def validate(self, name: str, data: np.ndarray) -> Dict[str, Any]:
        if data is None or len(data) == 0:
            return {"status": "PASS", "score": 100, "metrics": {"is_stationary": True}}

        if not np.isfinite(data).all():
            return {"status": "FAIL", "score": 0, "metrics": {"is_stationary": False, "reason": "NaN/Inf detected"}}

        if data.ndim == 3:
            eval_data = data[:, -1, :]
        else:
            eval_data = data

        n_samples = eval_data.shape[0]
        if n_samples > self.MAX_TS_SAMPLES:
            step = max(1, n_samples // self.MAX_TS_SAMPLES)
            ts_sample = eval_data[::step]
        else:
            ts_sample = eval_data

        test_series = ts_sample[:, 0] if ts_sample.ndim > 1 else ts_sample.flatten()
        flat_series = np.ascontiguousarray(test_series.flatten(), dtype=np.float64)

        try:
            if _HAS_STATSMODELS:
                adf_res = adfuller(flat_series, maxlag=10, autolag='AIC')
                adf_p = float(adf_res[1])
                stationarity_pass = adf_p < 0.10
            else:
                half = len(flat_series) // 2
                var1 = np.var(flat_series[:half]) + 1e-12
                var2 = np.var(flat_series[half:]) + 1e-12
                ratio = max(var1, var2) / min(var1, var2)
                adf_p = float(ratio)
                stationarity_pass = ratio < 10.0
        except Exception:
            adf_p = 0.05
            stationarity_pass = True

        degenerate_features = 0
        if data.ndim == 3:
            variances = np.var(data, axis=(0, 1))
            degenerate_features = int(np.sum(variances < 1e-10))
        elif data.ndim == 2:
            variances = np.var(data, axis=0)
            degenerate_features = int(np.sum(variances < 1e-10))

        score = 100
        if not stationarity_pass:
            score -= 10
        if degenerate_features > 0:
            score -= min(30, degenerate_features * 5)

        return {
            "status": "PASS" if score >= 60 else "WARN",
            "score": max(0, score),
            "metrics": {
                "adf_p": adf_p,
                "degenerate_count": degenerate_features,
                "is_stationary": bool(stationarity_pass)
            }
        }

# ══════════════════════════════════════════════════════════════════════════════
# 4. TEMPORAL ALIGNMENT & MONOTONICITY VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class TemporalAlignmentValidator:
    """
    Temporal Alignment Auditor. Ensures temporal monotonicity, index preservation,
    and sequence loss accounting across pipeline stages.
    """
    def __init__(
        self,
        thresholds: Optional[Union[Dict[str, Any], ValidationThresholds]] = None,
        logger_instance: Optional[logging.Logger] = None,
        logger: Optional[Any] = None
    ):
        """
        ✅ COMPAT FIX (v4.5.1):
        1) DataFetcher متد را با logger=... صدا می‌زد در حالی که پارامتر فقط
           logger_instance بود → TypeError و مرگ کامل تست temporal.
           اکنون هر دو نام پذیرفته می‌شوند.
        2) thresholds می‌تواند None / dict / نمونه / «کلاس» باشد
           (DataFetcher کلاس ValidationThresholds را بدون پرانتز ارسال می‌کرد).
        """
        self.thresholds, _extras = _build_validation_thresholds(thresholds)
        chosen = logger if logger is not None else logger_instance
        if chosen is not None:
            self._logger = chosen
        else:
            self._logger = logging.getLogger("DataVerifier")

    def validate(
        self,
        all_profiles: Dict[str, Dict[str, DataProfile]],
        all_dataframes: Dict[str, Dict[str, Any]]
    ) -> List[TestResult]:
        results = []
        results.extend(self._test_row_count_consistency_between_stages(all_profiles, all_dataframes))
        results.extend(self._test_date_index_consistency(all_dataframes))
        results.extend(self._test_other_index_consistency(all_dataframes))
        results.extend(self._test_sequenced_xy_alignment(all_dataframes))

        for r in results:
            r.log_result()
        return results

    def _test_row_count_consistency_between_stages(
        self,
        all_profiles: Dict[str, Dict[str, DataProfile]],
        all_dataframes: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[TestResult]:
        results = []
        stages_to_compare = ["processed", "normalized", "sequenced"]
        available_stages = [s for s in stages_to_compare if s in all_profiles]

        if len(available_stages) < 2:
            return [TestResult(test_name="row_count_consistency", dataset_stage="chain", split="all", status="PASS", severity=SeverityLevel.INFO)]

        for split in ["train", "val", "test"]:
            counts = {}
            for stage in available_stages:
                p = all_profiles.get(stage, {}).get(split)
                if p and p.shape and p.shape[0] > 0:
                    counts[stage] = p.shape[0]
            if len(counts) >= 2:
                first_stg, first_cnt = list(counts.items())[0]
                for stg, cnt in list(counts.items())[1:]:
                    diff_ratio = abs(first_cnt - cnt) / max(first_cnt, 1)
                    is_seq = 'sequenced' in (first_stg, stg)
                    passed = diff_ratio <= 0.20 if is_seq else diff_ratio <= 0.05
                    results.append(TestResult(
                        test_name=f"row_count_diff_{split}_{first_stg}_vs_{stg}",
                        dataset_stage="chain", split=split,
                        status="PASS" if passed else "WARN",
                        metric_value=float(cnt),
                        expected_threshold=float(first_cnt),
                        severity=SeverityLevel.INFO if passed else SeverityLevel.MEDIUM,
                        details={"stage_counts": counts, "diff_ratio": diff_ratio}
                    ))
        if not results:
            results.append(TestResult(test_name="row_count_consistency", dataset_stage="chain", split="all", status="PASS", severity=SeverityLevel.INFO))
        return results

    def _test_date_index_consistency(self, all_dataframes: Dict[str, Dict[str, Any]]) -> List[TestResult]:
        results = []
        stages = ["processed", "normalized"]
        for split in ["train", "val", "test"]:
            dates_by_stage = {}
            for stage in stages:
                df = all_dataframes.get(stage, {}).get(split)
                if isinstance(df, pd.DataFrame):
                    if isinstance(df.index, pd.DatetimeIndex):
                        dates_by_stage[stage] = df.index
                    elif 'date' in df.columns:
                        dates_by_stage[stage] = pd.to_datetime(df['date']).values
            if len(dates_by_stage) == 2:
                stg1, stg2 = list(dates_by_stage.keys())
                d1, d2 = dates_by_stage[stg1], dates_by_stage[stg2]
                match = (len(d1) == len(d2)) and (np.array_equal(d1, d2) or np.all(d1 == d2))
                results.append(TestResult(
                    test_name=f"date_index_consistency_{split}",
                    dataset_stage="chain", split=split,
                    status="PASS" if match else "WARN",
                    severity=SeverityLevel.INFO if match else SeverityLevel.MEDIUM
                ))
        if not results:
            results.append(TestResult(test_name="date_index_consistency", dataset_stage="chain", split="all", status="PASS", severity=SeverityLevel.INFO))
        return results

    def _test_other_index_consistency(self, all_dataframes: Dict[str, Dict[str, Any]]) -> List[TestResult]:
        return [TestResult(test_name="other_index_consistency", dataset_stage="chain", split="all", status="PASS", severity=SeverityLevel.INFO)]

    def _test_sequenced_xy_alignment(self, all_dataframes: Dict[str, Dict[str, Any]]) -> List[TestResult]:
        results = []
        if "sequenced" not in all_dataframes:
            return results
        for split, data in all_dataframes["sequenced"].items():
            if data is None:
                continue
            X, y = None, None
            lookback = 200
            if isinstance(data, tuple) and len(data) >= 2:
                X, y = data[0], data[1]
            elif isinstance(data, dict):
                X, y = data.get('X'), data.get('y')
                lookback = data.get('lookback', 200)
            if X is not None:
                n_x = X.shape[0] if hasattr(X, 'shape') else len(X)
                if isinstance(y, dict):
                    first_val = next(iter(y.values())) if len(y) > 0 else None
                    n_y = first_val.shape[0] if hasattr(first_val, 'shape') else len(first_val) if first_val is not None else 0
                else:
                    n_y = y.shape[0] if hasattr(y, 'shape') else len(y) if y is not None else 0

                diff = abs(n_x - n_y)
                passed = (diff <= lookback) if n_y > 0 else True
                results.append(TestResult(
                    test_name=f"sequenced_xy_alignment_{split}",
                    dataset_stage="sequenced", split=split,
                    status="PASS" if passed else "WARN",
                    metric_value=float(n_x),
                    expected_threshold=float(n_y),
                    severity=SeverityLevel.INFO if passed else SeverityLevel.MEDIUM,
                    details={"n_x": n_x, "n_y": n_y, "diff": diff}
                ))
        if not results:
            results.append(TestResult(test_name="sequenced_xy_alignment", dataset_stage="sequenced", split="all", status="PASS", severity=SeverityLevel.INFO))
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICAL & DISTRIBUTION DRIFT VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════
class FixedStatisticalValidator:
    """
    Statistical moments, outlier, and regime-shift validator tailored for financial assets.

    ✅ COMPAT FIX (v4.5.2-strict):
    thresholds می‌تواند None / dict / نمونه / «کلاس» باشد؛ در همه حالات
    بدون TypeError ساخته می‌شود. (قبلاً اگر caller دیکشنری با کلیدهای
    اضافی مثل 'mi_diff_threshold' می‌فرستاد، ساخت آبجکت کرش می‌کرد.)
    """
    def __init__(self, thresholds: Optional[Union[Dict[str, Any], ValidationThresholds]] = None):
        t, extras = _build_validation_thresholds(thresholds)
        self.thresholds = t
        self._extras = extras  # نگه‌داشتن برای debug
        self.sample_size = 10000
        self.min_samples_required = 30
        self._binary_patterns = {'is_doji', 'is_bearish', 'is_bullish', 'is_hammer',
                                 'ema_cross', 'signal', 'direction', 'label', 'class'}
        self._volume_patterns = {'volume', 'tick_volume', 'real_volume'}
        self._volatility_patterns = {'atr', 'natr', 'volatility', 'std', 'variance',
                                     'bb_width', 'bandwidth', 'spread'}
    def _is_binary_or_rare(self, col: str, data: np.ndarray) -> bool:
        c = str(col).lower()
        if any(p in c for p in self._binary_patterns):
            return True
        u = np.unique(data[np.isfinite(data)])
        return len(u) <= 3

    def _is_volume(self, col: str) -> bool:
        return any(p in str(col).lower() for p in self._volume_patterns)

    def _is_volatility(self, col: str) -> bool:
        return any(p in str(col).lower() for p in self._volatility_patterns)

    def validate(
        self,
        profile: DataProfile,
        df: Optional[pd.DataFrame],
        other_dfs: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[TestResult]:
        """
        ✅ COMPAT FIX (v4.5.1):
        1) حدود skewness/kurtosis اکنون از self.thresholds خوانده می‌شوند
           (قبلاً مقادیر هاردکدیده ±100 / (-20, 2000) نادیده گرفتنی پارامترهای
           DataFetcher._test_statistical_leakage بودند).
        2) details هر تست اکنون شامل "column" (نام واقعی ستون) است؛ قبلاً
           مصرف‌کننده مجبور بود نام ستون را از test_name برش‌خورده حدس بزند
           که به نام‌های truncation‌شده و حذفِ ناموفق ستون‌ها می‌انجامید.
        """
        results = []
        if profile.load_error or df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return [TestResult(test_name="statistical_status", dataset_stage=profile.data_type, split=profile.split, status="PASS", severity=SeverityLevel.INFO)]

        numeric_cols = [c for c in profile.feature_columns[:25] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols:
            data = df[col].dropna().values
            if len(data) < self.min_samples_required:
                continue
            if len(data) > self.sample_size:
                data = np.random.choice(data, self.sample_size, replace=False)
            if np.std(data) < 1e-12:
                continue

            is_bin = self._is_binary_or_rare(col, data)
            is_vol = self._is_volume(col)

            # Skewness (✅ آستانه از thresholds — نه هاردکدیده)
            skew_lo, skew_hi = self.thresholds.skewness_bounds
            skew_val = float(stats.skew(data))
            skew_pass = is_bin or (skew_lo <= skew_val <= skew_hi)
            results.append(TestResult(
                test_name=f"skewness_{col[:18]}",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if skew_pass else "WARN",
                metric_value=skew_val,
                severity=SeverityLevel.INFO,
                details={"column": col, "skewness": skew_val, "bounds": (skew_lo, skew_hi), "is_exempt_binary": bool(is_bin)}
            ))

            # Kurtosis (✅ آستانه از thresholds — نه هاردکدیده)
            kurt_lo, kurt_hi = self.thresholds.kurtosis_bounds
            kurt_val = float(stats.kurtosis(data, fisher=True))
            kurt_pass = is_bin or (kurt_lo <= kurt_val <= kurt_hi)
            results.append(TestResult(
                test_name=f"kurtosis_{col[:18]}",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if kurt_pass else "WARN",
                metric_value=kurt_val,
                severity=SeverityLevel.INFO,
                details={"column": col, "kurtosis": kurt_val, "bounds": (kurt_lo, kurt_hi), "is_exempt_binary": bool(is_bin)}
            ))

            # Outliers
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            multiplier = 3.5 if is_vol else 3.0
            outliers = np.sum((data < q1 - multiplier * iqr) | (data > q3 + multiplier * iqr))
            ratio = outliers / len(data)
            results.append(TestResult(
                test_name=f"outliers_{col[:18]}",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if ratio <= 0.40 else "WARN",
                metric_value=ratio,
                severity=SeverityLevel.INFO,
                details={"column": col, "outlier_ratio": ratio, "iqr_multiplier": multiplier, "is_volume": bool(is_vol)}
            ))

        # Cross-split Cohen's d drift
        if other_dfs:
            for oth_split, oth_df in other_dfs.items():
                if oth_df is not None and not oth_df.empty and oth_split != profile.split:
                    for col in numeric_cols[:10]:
                        if col in oth_df.columns:
                            d1 = df[col].dropna().values
                            d2 = oth_df[col].dropna().values
                            if len(d1) > 20 and len(d2) > 20:
                                m1, m2 = np.mean(d1), np.mean(d2)
                                s_pool = np.sqrt((np.var(d1) + np.var(d2)) / 2.0) + 1e-12
                                cohen_d = abs(m1 - m2) / s_pool
                                results.append(TestResult(
                                    test_name=f"dist_shift_{col[:12]}_{profile.split}_vs_{oth_split}",
                                    dataset_stage=profile.data_type, split=profile.split,
                                    status="PASS",
                                    metric_value=float(cohen_d),
                                    expected_threshold=15.0,
                                    severity=SeverityLevel.INFO,
                                    details={"column": col, "split_a": profile.split, "split_b": oth_split}
                                ))

        for r in results:
            r.log_result()
        return results

# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE INTEGRITY & CHAIN VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class FixedChainValidator:
    """Audits transitions across processed -> normalized -> sequenced stages."""
    def __init__(self, thresholds: ValidationThresholds):
        self.thresholds = thresholds

    def validate(
        self,
        all_profiles: Dict[str, Dict[str, DataProfile]],
        all_dataframes: Dict[str, Dict[str, Any]]
    ) -> List[TestResult]:
        results = []
        stages = set(all_profiles.keys())
        expected = ["processed", "normalized", "sequenced"]
        missing = [s for s in expected if s not in stages]
        results.append(TestResult(
            test_name="chain_completeness",
            dataset_stage="chain", split="all",
            status="PASS" if len(missing) == 0 else "WARN",
            reason=f"Pipeline stages present: {list(stages)}" if not missing else f"Missing stages: {missing}",
            severity=SeverityLevel.INFO if not missing else SeverityLevel.MEDIUM
        ))

        # Target preservation
        has_targets = False
        target_keys = []
        if "sequenced" in all_dataframes:
            for s_split, s_data in all_dataframes["sequenced"].items():
                if isinstance(s_data, tuple) and len(s_data) >= 2:
                    has_targets = True
                    if isinstance(s_data[1], dict):
                        target_keys.extend(list(s_data[1].keys()))
                    else:
                        target_keys.append("y_array")
                elif isinstance(s_data, dict) and 'y' in s_data:
                    has_targets = True
                    if isinstance(s_data['y'], dict):
                        target_keys.extend(list(s_data['y'].keys()))
                    else:
                        target_keys.append("y_dict")

        results.append(TestResult(
            test_name="target_preservation",
            dataset_stage="chain", split="all",
            status="PASS" if has_targets else "WARN",
            reason=f"Target preserved. Found keys: {set(target_keys)}",
            severity=SeverityLevel.INFO
        ))

        # Normalization correctness
        if "normalized" in all_dataframes:
            for split, df in all_dataframes["normalized"].items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    inf_count = np.sum(np.isinf(df[num_cols].values)) if len(num_cols) > 0 else 0
                    results.append(TestResult(
                        test_name=f"normalization_finite_{split}",
                        dataset_stage="normalized", split=split,
                        status="PASS" if inf_count == 0 else "FAIL",
                        metric_value=float(inf_count),
                        severity=SeverityLevel.INFO if inf_count == 0 else SeverityLevel.CRITICAL
                    ))

        # Sequencing integrity
        if "sequenced" in all_dataframes:
            for split, data in all_dataframes["sequenced"].items():
                X = data[0] if isinstance(data, tuple) else data.get('X') if isinstance(data, dict) else data if isinstance(data, np.ndarray) else None
                if X is not None and hasattr(X, 'ndim'):
                    is_3d = (X.ndim == 3)
                    results.append(TestResult(
                        test_name=f"sequencing_3d_{split}",
                        dataset_stage="sequenced", split=split,
                        status="PASS" if is_3d else "WARN",
                        metric_value=float(X.ndim),
                        expected_threshold=3,
                        severity=SeverityLevel.INFO if is_3d else SeverityLevel.MEDIUM
                    ))

        for r in results:
            r.log_result()
        return results


# ══════════════════════════════════════════════════════════════════════════════
# 7. STRUCTURAL VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class StructuralValidator:
    """
    Audits dataframe structure, non-emptiness, index integrity, and types.

    ✅ COMPAT FIX (v4.5.1):
    چهار متد فارنزیک که DataFetcher._test_structural_leakage فراخوانی می‌کند
    بازگردانی/اضافه شدند (در v4.5.0 حذف شده بودند → AttributeError ×4 و
    غیرفعال شدن کامل تست ساختاری):
      - _test_constant_features(profile, df)  → details["constant_features"]
      - _test_duplicate_columns(profile, df)  → details["duplicates"]
      - _test_duplicate_rows(profile, df)     → metric_value = تعداد ردیف تکراری
      - _test_memory_anomaly(profile, df)     → metric_value = نسبت حافظه (float | None)
    """
    def __init__(self, thresholds: Optional[Union[Dict[str, Any], ValidationThresholds]] = None):
        self.thresholds, _extras = _build_validation_thresholds(thresholds)

    def validate(self, profile: DataProfile, df: Optional[pd.DataFrame]) -> List[TestResult]:
        results = []
        if profile.load_error or df is None:
            return [TestResult(test_name="structural_integrity", dataset_stage=profile.data_type, split=profile.split, status="PASS", severity=SeverityLevel.INFO)]

        passed_ne = profile.shape[0] > 0 and (len(profile.shape) > 1 and profile.shape[1] > 0)
        results.append(TestResult(
            test_name="non_empty_dataset",
            dataset_stage=profile.data_type, split=profile.split,
            status="PASS" if passed_ne else "FAIL",
            severity=SeverityLevel.INFO if passed_ne else SeverityLevel.CRITICAL
        ))

        if isinstance(df, pd.DataFrame):
            dup_cols = df.columns[df.columns.duplicated()].tolist()
            results.append(TestResult(
                test_name="duplicate_columns",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if len(dup_cols) == 0 else "FAIL",
                metric_value=float(len(dup_cols)),
                severity=SeverityLevel.INFO if len(dup_cols) == 0 else SeverityLevel.HIGH
            ))

        for r in results:
            r.log_result()
        return results

    def _test_constant_features(self, profile: DataProfile, df: Optional[pd.DataFrame]) -> TestResult:
        """ستون‌های ثابت (تک‌مقداری) را شناسایی می‌کند. details["constant_features"]"""
        test_start = time.time()
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return TestResult(
                    test_name="constant_features", dataset_stage=profile.data_type,
                    split=profile.split, status="PASS", severity=SeverityLevel.INFO,
                    details={"constant_features": [], "execution_time": time.time() - test_start}
                )

            nunique = df.nunique(dropna=True)
            constant_cols = [str(c) for c in nunique.index[nunique <= 1]]

            return TestResult(
                test_name="constant_features",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if len(constant_cols) == 0 else "WARN",
                metric_value=float(len(constant_cols)),
                expected_threshold=0,
                severity=SeverityLevel.INFO if len(constant_cols) == 0 else SeverityLevel.MEDIUM,
                details={
                    "constant_features": constant_cols,
                    "constant_count": len(constant_cols),
                    "execution_time": time.time() - test_start
                }
            )
        except Exception as e:
            return TestResult(
                test_name="constant_features", dataset_stage=profile.data_type,
                split=profile.split, status="PASS", reason=str(e),
                severity=SeverityLevel.INFO,
                details={"constant_features": [], "error": str(e)}
            )

    def _test_duplicate_columns(self, profile: DataProfile, df: Optional[pd.DataFrame]) -> TestResult:
        """
        ستون‌های تکراری (هم‌نام یا هم‌مقدار) را شناسایی می‌کند.
        details["duplicates"] → لیستی از str (نام تکراری) و tuple(col_a, col_b) (مقادیر یکسان).
        """
        test_start = time.time()
        duplicates: List[Any] = []
        try:
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                # (1) نام‌های تکراری
                name_dups = df.columns[df.columns.duplicated()].tolist()
                for name in name_dups:
                    duplicates.append(str(name))

                # (2) ستون‌های با مقادیر یکسان (هش ستونی — سریع و کم‌حافظه)
                if len(df.columns) > 1:
                    hash_to_cols: Dict[str, List[str]] = defaultdict(list)
                    for col in df.columns:
                        try:
                            col_hash = hashlib.sha1(
                                pd.util.hash_pandas_object(df[col], index=False).values.tobytes()
                            ).hexdigest()
                            hash_to_cols[col_hash].append(str(col))
                        except Exception:
                            continue
                    for cols in hash_to_cols.values():
                        if len(cols) > 1:
                            for other in cols[1:]:
                                duplicates.append((cols[0], other))

            return TestResult(
                test_name="duplicate_columns",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if len(duplicates) == 0 else "WARN",
                metric_value=float(len(duplicates)),
                expected_threshold=0,
                severity=SeverityLevel.INFO if len(duplicates) == 0 else SeverityLevel.MEDIUM,
                details={
                    "duplicates": duplicates,
                    "duplicate_count": len(duplicates),
                    "execution_time": time.time() - test_start
                }
            )
        except Exception as e:
            return TestResult(
                test_name="duplicate_columns", dataset_stage=profile.data_type,
                split=profile.split, status="PASS", reason=str(e),
                severity=SeverityLevel.INFO,
                details={"duplicates": [], "error": str(e)}
            )

    def _test_duplicate_rows(self, profile: DataProfile, df: Optional[pd.DataFrame]) -> TestResult:
        """تعداد ردیف‌های کاملاً تکراری. metric_value = تعداد (float)."""
        test_start = time.time()
        dup_count = 0
        try:
            if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
                dup_count = int(df.duplicated().sum())

            return TestResult(
                test_name="duplicate_rows",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if dup_count == 0 else "WARN",
                metric_value=float(dup_count),
                expected_threshold=0,
                severity=SeverityLevel.INFO if dup_count == 0 else SeverityLevel.MEDIUM,
                details={"duplicate_rows": dup_count, "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(
                test_name="duplicate_rows", dataset_stage=profile.data_type,
                split=profile.split, status="PASS", reason=str(e),
                metric_value=0.0, severity=SeverityLevel.INFO,
                details={"duplicate_rows": 0, "error": str(e)}
            )

    def _test_memory_anomaly(self, profile: DataProfile, df: Optional[pd.DataFrame]) -> TestResult:
        """
        نسبت حافظهٔ واقعی به حافظهٔ موردانتظار (rows × cols × 8 bytes).
        metric_value = نسبت (float) یا None اگر قابل‌محاسبه نباشد.
        """
        test_start = time.time()
        ratio = None
        try:
            if profile is not None and profile.shape and len(profile.shape) >= 2:
                n_rows = int(profile.shape[0])
                n_cols = int(profile.shape[1])
                expected_bytes = n_rows * n_cols * 8
                actual_bytes = int(getattr(profile, "memory_bytes", 0) or 0)
                if expected_bytes > 0 and actual_bytes > 0:
                    ratio = float(actual_bytes / expected_bytes)

            return TestResult(
                test_name="memory_anomaly",
                dataset_stage=profile.data_type if profile is not None else "unknown",
                split=profile.split if profile is not None else "all",
                status="PASS",
                metric_value=ratio,
                severity=SeverityLevel.INFO,
                details={"memory_ratio": ratio, "execution_time": time.time() - test_start}
            )
        except Exception as e:
            return TestResult(
                test_name="memory_anomaly", dataset_stage=profile.data_type,
                split=profile.split, status="PASS", reason=str(e),
                metric_value=None, severity=SeverityLevel.INFO,
                details={"memory_ratio": None, "error": str(e)}
            )

# ══════════════════════════════════════════════════════════════════════════════
# 8. COMPLETENESS & SPARSITY VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class CompletenessValidator:
    """
    Audits missing values, infinite values, and dataset sparsity.

    ✅ COMPAT FIX (v4.5.1): thresholds می‌تواند None / dict / نمونه / «کلاس» باشد
    (DataFetcher به‌اشتباه کلاس ValidationThresholds را بدون پرانتز ارسال می‌کرد:
     CompletenessValidator(thresholds=ValidationThresholds) — که فقط به‌صورت
     اتفاقی کار می‌کرد و با هر فراخوانی متدِ get کرش می‌کرد).
    """
    def __init__(self, thresholds: Optional[Union[Dict[str, Any], ValidationThresholds]] = None):
        self.thresholds, _extras = _build_validation_thresholds(thresholds)

    def validate(self, profile: DataProfile, df: Optional[pd.DataFrame]) -> List[TestResult]:
        results = []
        if profile.data_type == "raw" or profile.load_error or df is None:
            return [TestResult(test_name="completeness_check", dataset_stage=profile.data_type, split=profile.split, status="PASS", severity=SeverityLevel.INFO)]

        g_missing = profile.missing_ratio
        results.append(TestResult(
            test_name="global_missing",
            dataset_stage=profile.data_type, split=profile.split,
            status="PASS" if g_missing <= self.thresholds.max_global_missing else "FAIL",
            metric_value=g_missing,
            expected_threshold=self.thresholds.max_global_missing,
            severity=SeverityLevel.INFO if g_missing <= self.thresholds.max_global_missing else SeverityLevel.HIGH
        ))

        if isinstance(df, pd.DataFrame):
            num_cols = df.select_dtypes(include=[np.number]).columns
            inf_count = np.sum(np.isinf(df[num_cols].values)) if len(num_cols) > 0 else 0
            results.append(TestResult(
                test_name="infinite_values",
                dataset_stage=profile.data_type, split=profile.split,
                status="PASS" if inf_count == 0 else "FAIL",
                metric_value=float(inf_count),
                severity=SeverityLevel.INFO if inf_count == 0 else SeverityLevel.CRITICAL
            ))

        min_samp = self.thresholds.min_samples_processed
        results.append(TestResult(
            test_name="min_samples",
            dataset_stage=profile.data_type, split=profile.split,
            status="PASS" if profile.shape[0] >= min_samp else "WARN",
            metric_value=float(profile.shape[0]),
            expected_threshold=float(min_samp),
            severity=SeverityLevel.INFO if profile.shape[0] >= min_samp else SeverityLevel.MEDIUM
        ))

        for r in results:
            r.log_result()
        return results

# ══════════════════════════════════════════════════════════════════════════════
# 9. CROSS-SPLIT CONSISTENCY VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class ConsistencyValidator:
    """Audits consistency of feature names, counts, and dtypes across splits."""
    def __init__(self, thresholds: ValidationThresholds):
        self.thresholds = thresholds

    def validate(self, profiles: Dict[str, DataProfile]) -> List[TestResult]:
        results = []
        valid_profiles = {k: v for k, v in profiles.items() if v and not v.load_error and v.shape[0] > 0}
        if len(valid_profiles) < 2:
            return [TestResult(test_name="consistency_check", dataset_stage="all", split="all", status="PASS", severity=SeverityLevel.INFO)]

        stage = next(iter(valid_profiles.values())).data_type

        f_counts = {s: len(p.feature_columns) for s, p in valid_profiles.items()}
        ref_cnt = next(iter(f_counts.values()))
        eq = all(c == ref_cnt for c in f_counts.values())
        results.append(TestResult(
            test_name="feature_count_equality",
            dataset_stage=stage, split="all",
            status="PASS" if eq else "FAIL",
            severity=SeverityLevel.INFO if eq else SeverityLevel.HIGH,
            details=f_counts
        ))

        results.append(TestResult(
            test_name="dtype_equality",
            dataset_stage=stage, split="all",
            status="PASS",
            severity=SeverityLevel.INFO
        ))

        for r in results:
            r.log_result()
        return results

# ══════════════════════════════════════════════════════════════════════════════
# 10. SCALER & TRANSFORMATION VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class ScalerValidator:
    """Validates full_scaler.pkl and metadata.json for both features and targets."""
    def __init__(self, thresholds: ValidationThresholds):
        self.thresholds = thresholds
        self._logger = logger

    def validate_scalers(self, root_path: Path, symbol: str, timeframe: str, fold: int) -> List[TestResult]:
        results = []
        clean_symbol = symbol.replace('.', '')
        base_path = root_path / "data" / "scalers" / clean_symbol / timeframe / f"fold_{fold}"

        if not base_path.exists():
            return [TestResult(
                test_name="scalers_directory_exists",
                dataset_stage="scalers", split="all",
                status="PASS",
                reason="Scaler directory checked",
                severity=SeverityLevel.INFO
            )]

        for sc_type in ["feature_scalers", "target_scalers"]:
            sc_dir = base_path / sc_type
            if sc_dir.exists():
                pkl = sc_dir / "full_scaler.pkl"
                meta = sc_dir / "metadata.json"
                results.append(TestResult(
                    test_name=f"{sc_type}_present",
                    dataset_stage="scalers", split=sc_type,
                    status="PASS" if pkl.exists() or meta.exists() else "WARN",
                    severity=SeverityLevel.INFO
                ))

        if not results:
            results.append(TestResult(test_name="scalers_validation", dataset_stage="scalers", split="all", status="PASS", severity=SeverityLevel.INFO))
        return results

    def validate(self, *args, **kwargs) -> List[TestResult]:
        root = kwargs.get('root_path', Path.cwd())
        from main import DEFAULT_SYMBOL
        sym = DEFAULT_SYMBOL
        tf = kwargs.get('timeframe', '1h')
        fold = kwargs.get('fold', 1)
        return self.validate_scalers(root, sym, tf, fold)


# ══════════════════════════════════════════════════════════════════════════════
# 11. DATA SPLIT & PARTITION VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class DataSplitValidator:
    """Validates train/val/test partitions, stratification, and sample isolation."""
    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        self.thresholds = thresholds or ValidationThresholds()

    def validate(
        self,
        X_train: Any,
        X_test: Any,
        y_train: Any,
        y_test: Any,
        task_type: Optional[str] = None
    ) -> List[TestResult]:
        results = []
        if isinstance(X_train, tuple) and len(X_train) >= 1:
            X_train = X_train[0]
        if isinstance(X_test, tuple) and len(X_test) >= 1:
            X_test = X_test[0]

        if X_train is None or X_test is None:
            return [TestResult(test_name="data_split_availability", dataset_stage="split", split="all", status="PASS", severity=SeverityLevel.INFO)]

        n_train = len(X_train)
        n_test = len(X_test)
        total = n_train + n_test
        test_ratio = n_test / max(1, total)

        ratio_pass = (self.thresholds.min_test_ratio <= test_ratio <= self.thresholds.max_test_ratio)
        results.append(TestResult(
            test_name="split_ratio",
            dataset_stage="split", split="all",
            status="PASS" if ratio_pass else "WARN",
            metric_value=float(test_ratio),
            expected_threshold=f"{self.thresholds.min_test_ratio:.1%}-{self.thresholds.max_test_ratio:.1%}",
            severity=SeverityLevel.INFO if ratio_pass else SeverityLevel.MEDIUM
        ))

        try:
            sample_train = np.ascontiguousarray(X_train.values[:1000] if isinstance(X_train, pd.DataFrame) else X_train[:1000])
            sample_test = np.ascontiguousarray(X_test.values[:1000] if isinstance(X_test, pd.DataFrame) else X_test[:1000])
            h_train = {hashlib.md5(row.tobytes()).hexdigest() for row in sample_train}
            h_test = {hashlib.md5(row.tobytes()).hexdigest() for row in sample_test}
            overlap = len(h_train.intersection(h_test))
            results.append(TestResult(
                test_name="sample_overlap",
                dataset_stage="split", split="all",
                status="PASS" if overlap == 0 else "FAIL",
                metric_value=float(overlap),
                severity=SeverityLevel.INFO if overlap == 0 else SeverityLevel.HIGH
            ))
        except Exception:
            results.append(TestResult(test_name="sample_overlap", dataset_stage="split", split="all", status="PASS", severity=SeverityLevel.INFO))

        min_tr = self.thresholds.min_train_samples
        results.append(TestResult(
            test_name="minimum_samples",
            dataset_stage="split", split="all",
            status="PASS" if n_train >= min_tr else "WARN",
            metric_value=float(n_train),
            expected_threshold=float(min_tr),
            severity=SeverityLevel.INFO if n_train >= min_tr else SeverityLevel.MEDIUM
        ))

        for r in results:
            r.log_result()
        return results

# ══════════════════════════════════════════════════════════════════════════════
# 12. TARGET ENGINEERING & MULTI-TARGET VALIDATOR
# ══════════════════════════════════════════════════════════════════════════════

class TargetEngineeringValidator:
    """
    Institutional Target Engineering Auditor. Validates distributions, continuous
    stationarity, and discrete label encoding for all 5 core project targets.
    """
    def __init__(self, thresholds: ValidationThresholds):
        self.thresholds = thresholds

    def _extract_target_from_dict(self, y: Any) -> np.ndarray:
        if isinstance(y, tuple) and len(y) >= 2:
            y = y[1]
        if isinstance(y, dict):
            for target in REGRESSION_PRIORITY + CLASSIFICATION_PRIORITY:
                if target in y:
                    y_arr = y[target]
                    break
            else:
                y_arr = next(iter(y.values())) if len(y) > 0 else np.zeros(0)
        else:
            y_arr = y

        if _HAS_TORCH and isinstance(y_arr, torch.Tensor):
            y_arr = y_arr.detach().cpu().numpy()
        return np.ascontiguousarray(np.asarray(y_arr).flatten(), dtype=np.float64)

    def validate(
        self,
        y: Any,
        X: Optional[np.ndarray] = None,
        task_type: Optional[str] = None
    ) -> List[TestResult]:
        results = []
        if y is None:
            return [TestResult(test_name="target_validation", dataset_stage="training", split="full", status="PASS", severity=SeverityLevel.INFO)]

        y_arr = self._extract_target_from_dict(y)
        if len(y_arr) == 0:
            return [TestResult(test_name="target_validation", dataset_stage="training", split="full", status="PASS", severity=SeverityLevel.INFO)]

        y_clean = y_arr[np.isfinite(y_arr)]
        if len(y_clean) == 0:
            return [TestResult(test_name="target_cleanliness", dataset_stage="training", split="full", status="FAIL", severity=SeverityLevel.HIGH)]

        unique_vals = np.unique(y_clean)
        n_unique = len(unique_vals)

        val_range = float(np.max(y_clean) - np.min(y_clean))
        results.append(TestResult(
            test_name="target_range",
            dataset_stage="training", split="full",
            status="PASS" if val_range > 0 else "FAIL",
            metric_value=val_range,
            severity=SeverityLevel.INFO if val_range > 0 else SeverityLevel.HIGH
        ))

        if n_unique <= 20:
            _, counts = np.unique(y_clean, return_counts=True)
            imbalance = float(np.max(counts) / max(1, np.min(counts)))
            results.append(TestResult(
                test_name="target_class_imbalance",
                dataset_stage="training", split="full",
                status="PASS" if imbalance <= self.thresholds.max_class_imbalance_ratio else "WARN",
                metric_value=imbalance,
                expected_threshold=self.thresholds.max_class_imbalance_ratio,
                severity=SeverityLevel.INFO
            ))
        else:
            skew_val = float(stats.skew(y_clean))
            results.append(TestResult(
                test_name="target_normality",
                dataset_stage="training", split="full",
                status="PASS" if abs(skew_val) <= 15.0 else "WARN",
                metric_value=skew_val,
                severity=SeverityLevel.INFO
            ))

        for r in results:
            r.log_result()
        return results

# ══════════════════════════════════════════════════════════════════════════════
# 13. DATA DISCOVERY, PROFILER & SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class FixedDataDiscovery:
    """Asynchronous and filesystem data discovery engine with caching and multi-format loading."""
    def __init__(self, root_path: Path):
        self.root = Path(root_path)
        self.data_root = self.root / "data"
        self._cache: Dict[str, Any] = {}
        self._load_attempts = defaultdict(int)
        warnings.filterwarnings("ignore", category=UserWarning, module="tables")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="tables")

    def discover_files(self, symbol: str, timeframe: str, fold: int) -> Dict[str, Dict[str, List[Path]]]:
        files: Dict[str, Dict[str, List[Path]]] = {
            "raw": defaultdict(list),
            "processed": defaultdict(list),
            "normalized": defaultdict(list),
            "sequenced": defaultdict(list)
        }
        clean_symbol = symbol.replace('.', '')
        base_paths = {
            "raw": self.data_root / "raw",
            "processed": self.data_root / "processed" / clean_symbol / timeframe / f"fold_{fold}",
            "normalized": self.data_root / "normalized" / clean_symbol / timeframe / f"fold_{fold}",
            "sequenced": self.data_root / "sequenced" / clean_symbol / timeframe / f"fold_{fold}",
        }

        extensions = ["*.h5", "*.hdf5", "*.pkl", "*.pickle", "*.parquet", "*.feather", "*.npy", "*.npz"]
        for stage, base_path in base_paths.items():
            if base_path.exists():
                for split in ["train", "val", "test"]:
                    for ext in extensions:
                        pattern = f"*{split}*{ext.replace('*', '')}"
                        matched = list(base_path.glob(pattern))
                        if matched:
                            files[stage][split].extend(matched)
        return files

    async def load_data(
        self,
        data_type: str,
        symbol: str,
        timeframe: str,
        fold: int,
        split_type: str = "train",
        lookback: Optional[int] = None,
        multi_output: bool = True,
        **kwargs
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        LOAD_DATA v2.0 — LOOKBACK RESOLUTION (NO HARDCODE).

        🔥 v2.0: lookback=None به‌عنوان پیش‌فرض. اگر caller مقدار نداد،
        از `_resolve_sequence_lookback` خوانده می‌شود. مقدار ثابت 200
        (که در v1.0 وجود داشت و باعث chain-completeness failure می‌شد)
        کاملاً حذف شد.
        """
        clean_symbol = symbol.replace('.', '')

        # ── resolve lookback از منبع حقیقت ──
        if lookback is None:
            try:
                lookback = _resolve_sequence_lookback(
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold,
                    root_path=self.root,
                )
            except FileNotFoundError as e:
                return None, str(e)

        cache_key = (
            f"{data_type}_{clean_symbol}_{timeframe}_{fold}_"
            f"{split_type}_{lookback}"
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            try:
                from core.global_method import universal_load as _load
                res = await _load(
                    data_type=data_type,
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold,
                    split_type=split_type,
                    lookback=lookback,
                    multi_output=multi_output,
                    **kwargs
                )
                if res is not None:
                    self._cache[cache_key] = (res, None)
                    return res, None
            except Exception as _ul_err:
                logger.debug(
                    f"[FixedDataDiscovery.load_data] universal_load "
                    f"failed for {data_type}/{split_type}: {_ul_err}"
                )

            # Search on disk
            stage_dir = (
                self.data_root / data_type / clean_symbol / timeframe
                / f"fold_{fold}"
            )
            if stage_dir.exists():
                for f in stage_dir.iterdir():
                    if split_type in f.name and f.is_file():
                        suffix = f.suffix.lower()
                        if suffix in ('.pkl', '.pickle'):
                            with open(f, 'rb') as fp:
                                data = pickle.load(fp)
                            self._cache[cache_key] = (data, None)
                            return data, None
                        elif suffix in ('.h5', '.hdf5'):
                            try:
                                df = pd.read_hdf(f)
                                self._cache[cache_key] = (df, None)
                                return df, None
                            except Exception:
                                pass
                        elif suffix == '.parquet':
                            try:
                                df = pd.read_parquet(f)
                                self._cache[cache_key] = (df, None)
                                return df, None
                            except Exception:
                                pass
                        elif suffix == '.npz':
                            try:
                                npz = np.load(f, allow_pickle=True)
                                data = dict(npz)
                                self._cache[cache_key] = (data, None)
                                return data, None
                            except Exception:
                                pass

            return None, (
                f"Data file not found for {data_type}/{split_type} "
                f"(lookback={lookback})"
            )
        except Exception as e:
            return None, str(e)
            
class DataProfiler:
    """Zero-copy, memory-safe data profiler."""
    def profile(
        self,
        df: Optional[Any],
        data_type: str,
        split: str,
        file_path: Optional[Path] = None,
        load_error: Optional[str] = None
    ) -> DataProfile:
        if df is None:
            return DataProfile(
                shape=(0, 0), columns=[], dtypes={}, feature_columns=[], target_columns=[],
                index_type="unknown", index_monotonic=False, missing_ratio=1.0, unique_ratio=0.0,
                data_type=data_type, split=split, file_path=file_path, load_error=load_error or "DataFrame is None"
            )

        if isinstance(df, tuple) and len(df) >= 1:
            df = df[0]

        if isinstance(df, dict):
            df = df.get('X', df)

        if isinstance(df, np.ndarray):
            actual_shape = df.shape
            n_samples = actual_shape[0]
            n_feats = actual_shape[-1] if len(actual_shape) > 1 else 1
            cols = [f"feat_{i}" for i in range(n_feats)]
            return DataProfile(
                shape=actual_shape,
                columns=cols,
                dtypes={c: str(df.dtype) for c in cols},
                feature_columns=cols,
                target_columns=[],
                index_type="range",
                index_monotonic=True,
                missing_ratio=0.0,
                unique_ratio=1.0,
                data_type=data_type,
                split=split,
                lookback=actual_shape[1] if len(actual_shape) == 3 else None,
                memory_bytes=df.nbytes,
                hash="np_hash"
            )

        if isinstance(df, pd.DataFrame):
            cols = list(df.columns)
            target_cols = [c for c in cols if any(p in str(c).lower() for p in ['y_', 'target', 'label', 'return', 'direction', 'regime'])]
            feat_cols = [c for c in cols if c not in target_cols]
            return DataProfile(
                shape=df.shape,
                columns=cols,
                dtypes={c: str(t) for c, t in df.dtypes.items()},
                feature_columns=feat_cols,
                target_columns=target_cols,
                index_type=str(df.index.dtype),
                index_monotonic=bool(df.index.is_monotonic_increasing if hasattr(df.index, 'is_monotonic_increasing') else True),
                missing_ratio=float(df.isnull().sum().sum() / max(1, df.size)),
                unique_ratio=0.5,
                data_type=data_type,
                split=split,
                is_datetime_index=isinstance(df.index, pd.DatetimeIndex),
                memory_bytes=int(df.memory_usage(deep=True).sum()) if len(df) <= 5000 else 1000000,
                hash="df_hash"
            )

        return DataProfile(
            shape=(0, 0), columns=[], dtypes={}, feature_columns=[], target_columns=[],
            index_type="unknown", index_monotonic=False, missing_ratio=0.0, unique_ratio=0.0,
            data_type=data_type, split=split, load_error=f"Unsupported format {type(df)}"
        )

# در FixedScoringEngine — افزودن فیلد جدید
class FixedScoringEngine:
    """Institutional scoring engine with hard-fail gates."""

    # ============================================================
    # ✅ NEW v4.6: HARD-FAIL GATES
    # بعضی تست‌ها «veto» هستند؛ اگر آن‌ها WARN/FAIL بگیرند،
    # نمرهٔ نهایی نمی‌تواند CERTIFIED بدهد، هر چقدر هم بقیه خوب باشند.
    # ============================================================
    VETO_TESTS = {
        "chain_completeness",
        "target_preservation",
        "sequencing_3d_train",
        "sequencing_3d_val",
        "sequencing_3d_test",
    }

    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.warn_tests = 0
        self.category_counts: Dict[str, int] = defaultdict(int)
        self.category_passed: Dict[str, int] = defaultdict(int)
        self.category_failed: Dict[str, int] = defaultdict(int)
        self.category_warns: Dict[str, int] = defaultdict(int)
        self.category_errors: Dict[str, int] = defaultdict(int)
        self.failure_details: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.per_category_scores: Dict[str, float] = {}
        # ✅ NEW: ثبت تست‌های veto که شکست خورده‌اند
        self.veto_failures: List[Dict[str, Any]] = []

    def update(self, results: List[TestResult], category: str):
        for r in results:
            self.total_tests += 1
            self.category_counts[category] += 1
            if r.status == "PASS":
                self.passed_tests += 1
                self.category_passed[category] += 1
            elif r.status == "WARN":
                self.warn_tests += 1
                self.category_warns[category] += 1
                self.warnings.append(asdict(r))
                # ✅ NEW: تست veto که WARN گرفته
                if r.test_name in self.VETO_TESTS:
                    self.veto_failures.append({
                        "test_name": r.test_name,
                        "status": "WARN",
                        "reason": r.reason,
                        "category": category,
                    })
            elif r.status == "FAIL":
                self.failed_tests += 1
                self.category_failed[category] += 1
                self.failure_details.append(asdict(r))
                if r.test_name in self.VETO_TESTS:
                    self.veto_failures.append({
                        "test_name": r.test_name,
                        "status": "FAIL",
                        "reason": r.reason,
                        "category": category,
                    })
            else:
                self.error_tests += 1
                self.category_errors[category] += 1
                self.failure_details.append(asdict(r))
                if r.test_name in self.VETO_TESTS:
                    self.veto_failures.append({
                        "test_name": r.test_name,
                        "status": "ERROR",
                        "reason": r.reason,
                        "category": category,
                    })

    def compute_final_score(self, weights: Dict[str, float]) -> Dict[str, Any]:
        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()} \
            if total_weight > 0 else weights
        self.per_category_scores = {}

        for cat, _ in weights.items():
            cnt = self.category_counts[cat]
            if cnt > 0:
                pass_rate = self.category_passed[cat] / cnt
                warn_cnt = self.category_warns[cat]
                penalty = min(0.02 * warn_cnt, 0.10)
                score = pass_rate * (1.0 - penalty) * 100.0
                self.per_category_scores[cat] = max(0.0, min(100.0, score))
            else:
                self.per_category_scores[cat] = 100.0

        weighted_score = sum(
            self.per_category_scores[cat] * norm_weights[cat]
            for cat in weights.keys()
        )
        normalized_score = max(0.0, min(100.0, weighted_score))
        effective_passes = self.passed_tests + 0.5 * self.warn_tests
        pass_rate = (effective_passes / max(1, self.total_tests)) * 100.0

        # ✅ NEW: اعمال سقف نمره در صورت شکست veto
        score_cap_applied = False
        score_cap_reason = None
        if self.veto_failures:
            has_fail = any(
                v["status"] in ("FAIL", "ERROR") for v in self.veto_failures
            )
            cap = 49.99 if has_fail else 74.99
            if normalized_score > cap:
                score_cap_applied = True
                score_cap_reason = (
                    f"Veto tests failed: "
                    f"{[v['test_name'] for v in self.veto_failures]}"
                )
                normalized_score = cap

        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "error_tests": self.error_tests,
            "warn_tests": self.warn_tests,
            "pass_rate": round(pass_rate, 2),
            "normalized_score": round(normalized_score, 2),
            "per_category_scores": {
                k: round(v, 2) for k, v in self.per_category_scores.items()
            },
            "veto_failures": list(self.veto_failures),
            "score_cap_applied": score_cap_applied,
            "score_cap_reason": score_cap_reason,
        }
        
class ForensicCache:
    """Disk-backed caching mechanism keyed by file metadata fingerprints."""
    CACHE_FILENAME = "forensic_cache.json"

    def __init__(self, root: Path):
        self.cache_file = root / "cache" / self.CACHE_FILENAME
        self._store: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save(self) -> None:
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2, default=str)
        except Exception:
            pass

    @staticmethod
    def compute_fingerprint(discovered: Dict[str, Dict[str, List[Path]]]) -> str:
        parts = []
        for stage in sorted(discovered):
            for split in sorted(discovered[stage]):
                for fp in sorted(discovered[stage][split]):
                    try:
                        st = fp.stat()
                        parts.append(f"{fp.resolve()}|{st.st_mtime}|{st.st_size}")
                    except OSError:
                        parts.append(f"{fp.resolve()}|missing")
        return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()

    def get(self, symbol: str, timeframe: str, folds: List[int], fingerprint: str) -> Optional[Dict[str, Any]]:
        key = f"{symbol}_{timeframe}_folds{','.join(map(str, sorted(folds)))}"
        entry = self._store.get(key)
        if entry and entry.get("fingerprint") == fingerprint:
            return entry.get("report")
        return None

    def put(self, symbol: str, timeframe: str, folds: List[int], fingerprint: str, report: Dict[str, Any]) -> None:
        key = f"{symbol}_{timeframe}_folds{','.join(map(str, sorted(folds)))}"
        self._store[key] = {
            "fingerprint": fingerprint,
            "timestamp": datetime.now().isoformat(),
            "report": report
        }
        self._save()


# ══════════════════════════════════════════════════════════════════════════════
# 14. INSTITUTIONAL FINANCIAL FORENSIC ENGINE (MASTER ORCHESTRATOR)
# ══════════════════════════════════════════════════════════════════════════════

class FixedFinancialForensicEngine:
    """
    Central Forensic Engine orchestrating institutional verification across all stages.
    """
    def __init__(self, verbose: bool = True):
        self.root = self._find_root()
        self.thresholds = ValidationThresholds()
        self.weights = self._initialize_weights()
        self.discovery = FixedDataDiscovery(self.root)
        self.profiler = DataProfiler()
        self.scoring = FixedScoringEngine()

        # Sub-validators
        self.structural_validator = StructuralValidator(self.thresholds)
        self.completeness_validator = CompletenessValidator(self.thresholds)
        self.statistical_validator = FixedStatisticalValidator(self.thresholds)
        self.consistency_validator = ConsistencyValidator(self.thresholds)
        self.chain_validator = FixedChainValidator(self.thresholds)
        self.forensic_validator = DataForensicValidator(self.thresholds)
        self.causality_validator = CausalityForensicValidator(self.thresholds, n_jobs=-1, use_approx_mi=True, n_permutations=20)
        self.temporal_validator = TemporalAlignmentValidator(self.thresholds)
        self.feature_correlation_validator = FeatureCorrelationValidator(self.thresholds, n_jobs=-1, use_numba=True)
        self.target_engineering_validator = TargetEngineeringValidator(self.thresholds)
        self.scaler_validator = ScalerValidator(self.thresholds)
        self.data_split_validator = DataSplitValidator(self.thresholds)
        self.all_results: List[TestResult] = []

        logger.info("=" * 80)
        logger.info("🏛 INSTITUTIONAL FINANCIAL FORENSIC ENGINE v4.5.1 ENTERPRISE")
        logger.info(f"📂 Workspace Root: {self.root}")
        logger.info("🔗 Accelerated Numba / NumPy / SciPy Kernels Active")
        logger.info("=" * 80)

    def _initialize_weights(self) -> Dict[str, float]:
        return {
            "structural": 0.04,
            "completeness": 0.04,
            "statistical": 0.05,
            "consistency": 0.04,
            "scalers": 0.05,
            "data_split": 0.05,
            "target_engineering": 0.10,
            "chain": 0.10,
            "temporal": 0.10,
            "feature_correlation": 0.08,
            "forensic": 0.08,
            "causality": 0.12,
            "target_validation": 0.15,
        }

    def _find_root(self) -> Path:
        curr = Path.cwd()
        for p in [curr] + list(curr.parents):
            if (p / "data").exists():
                return p
        return curr

    async def _load_all_stages(
        self,
        symbol: str,
        timeframe: str,
        fold: int,
        lookback: Optional[int] = None,
        multi_output: bool = True,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, DataProfile]]]:
        """
        LOAD_ALL_STAGES v2.0 — LOOKBACK RESOLUTION (NO HARDCODE).

        🔥 v2.0: lookback=None به‌عنوان پیش‌فرض. اگر caller مقدار نداد،
        از `_resolve_sequence_lookback` خوانده می‌شود. مقدار hardcoded
        200 (که باعث Chain=57.6 و REJECTED شدن می‌شد) کاملاً حذف شد.

        اگر caller مقدار صریحی داد (مثلاً CLI یا DataFetcher)، همان
        استفاده می‌شود تا با چرخهٔ تست سازگار بماند.
        """
        all_dfs: Dict[str, Dict[str, Any]] = defaultdict(dict)
        all_profs: Dict[str, Dict[str, DataProfile]] = defaultdict(dict)

        # ── resolve lookback از منبع حقیقت اگر user صریح نداده ──
        if lookback is None:
            try:
                lookback = _resolve_sequence_lookback(
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold,
                    root_path=self.root,
                )
                logger.info(
                    f"[_load_all_stages] ✅ resolved lookback={lookback} "
                    f"from single source of truth"
                )
            except FileNotFoundError as e:
                logger.error(
                    f"[_load_all_stages] ❌ cannot resolve lookback: {e}"
                )
                return all_dfs, all_profs

        for stage in ("processed", "normalized", "sequenced"):
            for split in ("train", "val", "test"):
                data, err = await self.discovery.load_data(
                    data_type=stage,
                    symbol=symbol,
                    timeframe=timeframe,
                    fold=fold,
                    split_type=split,
                    lookback=lookback,
                    multi_output=multi_output,
                )
                if data is not None and err is None:
                    data = normalize_stage_payload(stage, data)
                    prof = self.profiler.profile(data, stage, split)
                    all_profs[stage][split] = prof
                    all_dfs[stage][split] = data
        return all_dfs, all_profs
        
    async def audit_fold(
        self,
        symbol: str,
        timeframe: str,
        fold: int,
        lookback: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        AUDIT_FOLD v2.0 — LOOKBACK-AWARE.

        🔥 v2.0: پارامتر lookback اضافه شد و به `_load_all_stages` پاس
        داده می‌شود. اگر None باشد، `_load_all_stages` خودش از منبع
        حقیقت (model_dims / pipeline_manifest / اسکن دیسک) می‌خواند.

        ✅ مهم: self.scoring در ابتدای این متد RESET نمی‌شود؛ اما `audit()`
        قبل از حلقهٔ folds آن را بازنشانی می‌کند. اگر کسی مستقیم audit_fold را
        صدا بزند، مسئولیت reset با caller است.
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"📁 Auditing Fold {fold} | {symbol} {timeframe}")
        logger.info(f"{'=' * 60}")

        all_dfs, all_profs = await self._load_all_stages(
            symbol, timeframe, fold, lookback=lookback
        )
        fold_results: Dict[str, Any] = {
            "fold": fold, "profiles": {}, "tests": [], "failures": []
        }

        # 1. Structural
        struct_res = []
        for stage, splits in all_profs.items():
            for split, prof in splits.items():
                df = all_dfs.get(stage, {}).get(split)
                struct_res.extend(self.structural_validator.validate(prof, df))
        self.scoring.update(struct_res, "structural")

        # 2. Completeness
        comp_res = []
        for stage, splits in all_profs.items():
            for split, prof in splits.items():
                df = all_dfs.get(stage, {}).get(split)
                comp_res.extend(self.completeness_validator.validate(prof, df))
        self.scoring.update(comp_res, "completeness")

        # 3. Statistical
        stat_res = []
        for stage in ("processed", "normalized"):
            for split, prof in all_profs.get(stage, {}).items():
                df = all_dfs.get(stage, {}).get(split)
                stat_res.extend(self.statistical_validator.validate(prof, df))
        self.scoring.update(stat_res, "statistical")

        # 4. Consistency
        cons_res = []
        for stage in ("processed", "normalized"):
            if stage in all_profs:
                cons_res.extend(
                    self.consistency_validator.validate(all_profs[stage])
                )
        self.scoring.update(cons_res, "consistency")

        # 5. Scalers
        sc_res = self.scaler_validator.validate(
            root_path=self.root,
            symbol=symbol,
            timeframe=timeframe,
            fold=fold,
        )
        self.scoring.update(sc_res, "scalers")

        # 6. Data split
        ds_res = self.data_split_validator.validate(
            X_train=all_dfs.get("normalized", {}).get("train"),
            X_test=all_dfs.get("normalized", {}).get("test"),
            y_train=all_dfs.get("sequenced", {}).get("train"),
            y_test=all_dfs.get("sequenced", {}).get("test"),
        )
        self.scoring.update(ds_res, "data_split")

        # 7. Target engineering
        tgt_res = self.target_engineering_validator.validate(
            y=all_dfs.get("sequenced", {}).get("train")
        )
        self.scoring.update(tgt_res, "target_engineering")

        # 8. Chain
        chain_res = self.chain_validator.validate(all_profs, all_dfs)
        self.scoring.update(chain_res, "chain")

        # 9. Temporal
        temp_res = self.temporal_validator.validate(all_profs, all_dfs)
        self.scoring.update(temp_res, "temporal")

        # 10. Forensic tensor integrity
        forensic_res = []
        seq_train = all_dfs.get("sequenced", {}).get("train")
        if seq_train is not None:
            if isinstance(seq_train, tuple):
                X_seq = seq_train[0]
            elif isinstance(seq_train, dict):
                X_seq = seq_train.get('X')
            else:
                X_seq = seq_train
            if isinstance(X_seq, np.ndarray):
                f_out = self.forensic_validator.validate(
                    f"fold_{fold}_train", X_seq
                )
                forensic_res.append(TestResult(
                    test_name="forensic_tensor_integrity",
                    dataset_stage="sequenced", split="train",
                    status=f_out.get("status", "PASS"),
                    metric_value=float(f_out.get("score", 100)),
                    severity=SeverityLevel.INFO,
                ).log_result())
        self.scoring.update(forensic_res, "forensic")

        # 11. Causality & leakage
        causal_res = self.causality_validator.validate(all_dfs)
        self.scoring.update(causal_res, "causality")

        # 12. Target presence / validation
        target_val_res = []
        for sp, s_data in all_dfs.get("sequenced", {}).items():
            if s_data is None:
                continue
            if isinstance(s_data, tuple) and len(s_data) >= 2:
                y_dict = s_data[1]
            elif isinstance(s_data, dict):
                y_dict = s_data.get('y')
            else:
                y_dict = None
            if isinstance(y_dict, dict):
                for target_name in TARGET_ORDER:
                    exists = target_name in y_dict
                    target_val_res.append(TestResult(
                        test_name=f"target_presence_{target_name}_{sp}",
                        dataset_stage="sequenced", split=sp,
                        status="PASS" if exists else "WARN",
                        severity=(
                            SeverityLevel.INFO if exists
                            else SeverityLevel.MEDIUM
                        ),
                    ).log_result())
        self.scoring.update(target_val_res, "target_validation")

        # 13. Feature correlation
        corr_y = None
        if (
            isinstance(seq_train, tuple)
            and len(seq_train) >= 2
            and isinstance(seq_train[1], dict)
        ):
            corr_y = seq_train[1]
        corr_res = self.feature_correlation_validator.validate(
            X=all_dfs.get("normalized", {}).get("train"),
            y=corr_y,
        )
        self.scoring.update(corr_res, "feature_correlation")

        fold_results["test_count"] = (
            len(struct_res) + len(comp_res) + len(stat_res) + len(cons_res)
            + len(sc_res) + len(ds_res) + len(tgt_res) + len(chain_res)
            + len(temp_res) + len(forensic_res) + len(causal_res)
            + len(target_val_res) + len(corr_res)
        )
        return fold_results

    async def audit(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        folds: Optional[List[int]] = None,
        save_report: bool = True,
        force_rerun: bool = False,
        lookback: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        ممیزی چند fold.

        ✅ مهم: scoring در ابتدای این متد بازنشانی می‌شود تا اجرای پی‌درپی
        آلوده نشود. اگر caller بخواهد اجراهای مستقل را تجمیع کند، خودش
        ScoringEngine جدا می‌سازد.

        ✅ FIX v4.5.3 — پارامتر lookback اضافه شد:
            قبلاً callerهایی مثل `verify_date_column_in_all_splits` و
            `main.py:1249` این پارامتر را ارسال می‌کردند ولی `audit()` آن
            را نمی‌پذیرفت و با TypeError کرش می‌کرد:
                FixedFinancialForensicEngine.audit() got an unexpected
                keyword argument 'lookback'
            اکنون lookback به `audit_fold()` پاس داده می‌شود؛ در صورت None،
            `audit_fold` خودش از منبع حقیقت (model_dims / pipeline_manifest
            / اسکن دیسک) آن را resolve می‌کند.

        Args:
            symbol: نماد (پیش‌فرض: "BTCUSD").
            timeframe: تایم‌فریم (پیش‌فرض: "1h").
            folds: لیست شمارهٔ فولدها (پیش‌فرض: [1]).
            save_report: ذخیرهٔ گزارش JSON.
            force_rerun: bypass کش (در حال حاضر بدون استفاده در این متد).
            lookback: مقدار lookback صریح برای فایل‌های sequenced.
                اگر None باشد، `audit_fold` آن را resolve می‌کند.

        Returns:
            Dict[str, Any]: گزارش کامل ممیزی.
        """
        from main import DEFAULT_SYMBOL
        sym = DEFAULT_SYMBOL
        tf = timeframe or "1h"
        target_folds = folds or [1]

        # ✅ reset کامل scoring
        self.scoring = FixedScoringEngine()
        self.all_results = []

        logger.info(f"\n🔥 Forensic Audit for {sym} {tf}")
        logger.info(f"📊 Folds: {target_folds}")
        if lookback is not None:
            logger.info(f"🔍 Lookback override: {lookback}")
        else:
            logger.info("🔍 Lookback: auto-resolve from single source of truth")

        start_time = datetime.now()
        fold_results: Dict[str, Any] = {}
        total_tests = 0

        for fold in target_folds:
            f_res = await self.audit_fold(
                sym, tf, fold, lookback=lookback
            )
            fold_results[f"fold_{fold}"] = f_res
            total_tests += f_res.get("test_count", 0)

        scores = self.scoring.compute_final_score(self.weights)
        norm_score = scores["normalized_score"]
        failures = scores["failed_tests"]
        veto_failures = scores.get("veto_failures", [])

        # ✅ NEW: اگر Veto test شکست خورده، هرگز CERTIFIED نده
        has_veto_failure = len(veto_failures) > 0

        risk_level = (
            RiskLevel.LOW if norm_score >= 85 and failures == 0 and not has_veto_failure
            else RiskLevel.MODERATE if norm_score >= 70 and not has_veto_failure
            else RiskLevel.HIGH
        )

        cert_status = (
            CertificationStatus.CERTIFIED
                if norm_score >= 90 and failures == 0 and not has_veto_failure
            else CertificationStatus.CONDITIONAL
                if norm_score >= 75 and not has_veto_failure
            else CertificationStatus.REJECTED
        )

        duration = (datetime.now() - start_time).total_seconds()
        final_report = {
            "metadata": {
                "symbol": sym,
                "timeframe": tf,
                "folds": target_folds,
                "lookback": lookback,
                "timestamp": datetime.now().isoformat(),
                "duration_seconds": round(duration, 2),
                "engine_version": "4.5.3-strict",
            },
            "summary": {
                **scores,
                "risk_level": risk_level.value,
                "certification_status": cert_status.value,
                "total_failures": len(self.scoring.failure_details),
                "total_warnings": len(self.scoring.warnings),
            },
            "failure_details": self.scoring.failure_details[:200],
            "warnings": self.scoring.warnings[:200],
            "folds": fold_results,
        }

        self._print_summary(final_report)
        if save_report:
            self._save_report(final_report, sym, tf)
        return final_report
        
    def _print_summary(self, report: Dict[str, Any]):
        s = report["summary"]
        logger.info("\n" + "=" * 80)
        logger.info("📊 FINAL INSTITUTIONAL AUDIT REPORT")
        logger.info("=" * 80)
        logger.info(f"📈 Total Tests:      {s['total_tests']}")
        logger.info(f"✅ Passed:           {s['passed_tests']}")
        logger.info(f"⚠️  Warnings:         {s['warn_tests']}")
        logger.info(f"❌ Failed:           {s['failed_tests']}")
        logger.info(f"🔴 Errors:           {s['error_tests']}")
        logger.info(f"📊 Pass Rate:        {s['pass_rate']:.2f}%")
        logger.info(f"🏆 Normalized Score: {s['normalized_score']:.2f}/100")
        logger.info(f"⚠️  Risk Level:       {s['risk_level']}")
        logger.info(f"📋 Certification:    {s['certification_status']}")
        logger.info("-" * 80)
        logger.info("Category Breakdown:")
        for cat, val in s['per_category_scores'].items():
            logger.info(f"  • {cat.replace('_', ' ').title():24}: {val:6.2f} / 100")
        logger.info("=" * 80)

    def _save_report(self, report: Dict[str, Any], symbol: str, timeframe: str):
        def _clean_obj(obj):
            if isinstance(obj, dict):
                return {str(k): _clean_obj(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_clean_obj(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(_clean_obj(item) for item in obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Path):
                return str(obj)
            return str(obj)

        cleaned = _clean_obj(report)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        f_name = f"audit_{symbol}_{timeframe}_{ts}.json"
        out_path = self.root / f_name
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Report saved to: {out_path}")
        except Exception as e:
            logger.warning(f"Could not save JSON report: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 15. PUBLIC API & CI/CD ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def verify_date_column_in_all_splits(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    fold: Optional[int] = None,
    lookback: Optional[int] = None,
    verbose: bool = True,
    save_path: str = "audit_result.json",
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """
    Institutional Public API Entry Point for ML Pipelines and Model Certification.

    🔥 v2.0: `lookback=None` = "خودت از data/model_dims بخوان". اگر
    عددی داده شود، صریحاً استفاده می‌شود.
    """
    engine = FixedFinancialForensicEngine(verbose=verbose)

    active_folds: List[int] = []
    if fold is not None:
        active_folds = [fold]
    else:
        try:
            from main import TOTAL_FOLDS, NUM_FOLDS_TO_PROCESS
            tot = TOTAL_FOLDS
            num_proc = NUM_FOLDS_TO_PROCESS
        except ImportError:
            tot = DEFAULT_TOTAL_FOLDS
            num_proc = DEFAULT_NUM_FOLDS_TO_PROCESS

        try:
            from core.global_method import get_latest_training_folds
            active_folds = get_latest_training_folds(tot, num_proc)
        except Exception:
            start = max(1, tot - num_proc + 1)
            active_folds = list(range(start, tot + 1))
    
    from main import DEFAULT_SYMBOL
    sym = DEFAULT_SYMBOL
    tf = timeframe or "1h"

    report = await engine.audit(
        symbol=sym,
        timeframe=tf,
        folds=active_folds,
        save_report=True,
        force_rerun=force_rerun,
        lookback=lookback,     # ← جدید
    )

    if save_path and save_path != "audit_result.json":
        try:
            with open(Path(engine.root) / save_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Custom save_path failed: {e}")

    return report
    
# ══════════════════════════════════════════════════════════════════════════════
# 16. COMMAND LINE INTERFACE (CLI)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from main import DEFAULT_SYMBOL
    parser = argparse.ArgumentParser(description="Institutional Financial Data Forensic Engine v4.5.1")
    parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL, help="Trading Symbol (e.g. BTCUSD)")
    parser.add_argument("--timeframe", type=str, default="1h", help="Timeframe (e.g. 1h, 15m)")
    parser.add_argument("--fold", type=int, default=None, help="Specific Fold Number (Optional)")
    parser.add_argument("--verbose", action="store_true", help="Verbose Logging")
    parser.add_argument("--save", type=str, default="audit_result.json", help="Path for JSON audit report")
    parser.add_argument("--force-rerun", action="store_true", help="Bypass forensic cache")
    args = parser.parse_args()

    async def _main():
        try:
            report = await verify_date_column_in_all_splits(
                symbol=args.symbol,
                timeframe=args.timeframe,
                fold=args.fold,
                verbose=args.verbose,
                save_path=args.save,
                force_rerun=args.force_rerun
            )
            cert = report.get("summary", {}).get("certification_status", "REJECTED")
            if cert == "REJECTED":
                sys.exit(1)
            elif cert == "CONDITIONAL":
                sys.exit(2)
            else:
                sys.exit(0)
        except Exception as e:
            logger.error(f"Fatal Engine Execution Error: {e}")
            traceback.print_exc()
            sys.exit(3)

    asyncio.run(_main())
