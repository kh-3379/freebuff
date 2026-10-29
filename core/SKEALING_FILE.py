#SKEALING_FILE.py
from core.imports import *
from core.global_method import *
from core.global_method import _canonical_column_name, _shift_future_state
import numba
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import scipy.stats as stats
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA
# ═══════════════════════════════════════════════════════════════════════════
# SKEALING_FILE.py — Constants v13.0 (AUXILIARY-AWARE)
# ═══════════════════════════════════════════════════════════════════════════

MAIN_TARGETS: List[str] = [
    'price_return',
    'price_direction',
    'trend_strength',
    'volatility_regime',
    'market_regime',
]

AUXILIARY_TARGETS: List[str] = [
    'magnitude_bucket',
    'momentum_direction',   # ✅ NEW
]

AUXILIARY_WEIGHTS: List[str] = [
    'sample_weight',
]

# فهرست کامل تارگت‌ها
PROJECT_TARGETS: List[str] = MAIN_TARGETS + AUXILIARY_TARGETS

PROJECT_TARGETS_TYPE: Dict[str, str] = {
    'price_return': 'regression',
    'price_direction': 'classification',
    'trend_strength': 'classification',
    'volatility_regime': 'classification',
    'market_regime': 'classification',
    'magnitude_bucket': 'classification',
    'momentum_direction': 'classification',
}

# ✅ AUXILIARY_COLUMNS = هر چیزی که در X نباید باشد ولی در y_dict/H5 ذخیره می‌شود
AUXILIARY_COLUMNS: List[str] = AUXILIARY_TARGETS + AUXILIARY_WEIGHTS

AUXILIARY_COLUMNS_LOWER: frozenset = frozenset(
    str(c).strip().lower() for c in AUXILIARY_COLUMNS
)

AUXILIARY_TYPES: Dict[str, str] = {
    'sample_weight': 'weight',
    'magnitude_bucket': 'classification',
    'momentum_direction': 'classification',
}

AUXILIARY_DTYPES: Dict[str, np.dtype] = {
    'sample_weight': np.dtype(np.float32),
    'magnitude_bucket': np.dtype(np.int32),
    'momentum_direction': np.dtype(np.int32),
}


from core.logger import AdvancedTradeLogger
logger = AdvancedTradeLogger(__name__)



def log_target_distribution(
    target_name: str,
    target_data: Union[np.ndarray, pd.Series, pd.DataFrame, list],
    symbol: str = "📊",
    prefix: str = "       ",
    is_regression: Optional[bool] = None,
    show_class_names: bool = True,
) -> Dict[str, Any]:
    """
    ✅ WALL STREET V7.1 — ZERO-LEAKAGE + INVALID-AWARE LOGGING

    تغییر نسبت به V7.0:
        - INVALID_LABEL (-1) دیگر در شمارش n_classes لحاظ نمی‌شود.
        - کلاس‌های معتبر جدا از INVALID گزارش می‌شوند.
        - تعداد کلاس‌های واقعی (مثلاً ۵ برای price_direction) نمایش داده
          می‌شود، نه ۶.
    """
    import scipy.stats as stats

    INVALID_LABEL_LOCAL = -1
    result: Dict[str, Any] = {}

    try:
        # ── استخراج آرایه ────────────────────────────────────────
        if isinstance(target_data, pd.DataFrame):
            if target_name in target_data.columns:
                target_arr = target_data[target_name].values
            else:
                logger.warning(f"{symbol} ⚠️ ستون '{target_name}' یافت نشد")
                return result
        elif isinstance(target_data, pd.Series):
            target_arr = target_data.values
        elif isinstance(target_data, np.ndarray):
            target_arr = target_data
        elif isinstance(target_data, list):
            target_arr = np.array(target_data)
        else:
            target_arr = np.array(target_data)

        if target_arr is None or len(target_arr) == 0:
            logger.warning(f"{symbol} ⚠️ {target_name}: داده‌ای وجود ندارد")
            return result

        # ── تمیزکاری ─────────────────────────────────────────────
        try:
            clean_data = target_arr.astype(float)
            clean_data = clean_data[~np.isnan(clean_data)]
        except (ValueError, TypeError):
            clean_data = target_arr

        if len(clean_data) == 0:
            logger.warning(f"{symbol} ⚠️ {target_name}: داده معتبری وجود ندارد")
            return result

        # ── تشخیص نوع ────────────────────────────────────────────
        if is_regression is not None:
            _is_regression = is_regression
        else:
            n_unique = len(np.unique(clean_data))
            _is_regression = n_unique > 20

        total_samples = len(clean_data)

        # ── رگرسیون ──────────────────────────────────────────────
        if _is_regression:
            mean_val = float(np.mean(clean_data))
            std_val = float(np.std(clean_data))
            min_val = float(np.min(clean_data))
            max_val = float(np.max(clean_data))
            q1 = float(np.percentile(clean_data, 25))
            median_val = float(np.percentile(clean_data, 50))
            q3 = float(np.percentile(clean_data, 75))
            skewness = float(stats.skew(clean_data))
            kurtosis = float(stats.kurtosis(clean_data))
            iqr = q3 - q1

            result = {
                "type": "regression",
                "count": int(total_samples),
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "q1": q1,
                "median": median_val,
                "q3": q3,
                "iqr": iqr,
                "skewness": skewness,
                "kurtosis": kurtosis,
            }

            logger.info(f"{symbol} 📈 رگرسیون '{target_name}' (n={total_samples:,}):")
            logger.info(f"{prefix}┌─ میانگین (Mean):           {mean_val:>15.6f}")
            logger.info(f"{prefix}├─ انحراف معیار (Std):       {std_val:>15.6f}")
            logger.info(f"{prefix}├─ مینیمم (Min):             {min_val:>15.6f}")
            logger.info(f"{prefix}├─ ماکزیمم (Max):            {max_val:>15.6f}")
            logger.info(f"{prefix}├─ چارك اول (Q1):            {q1:>15.6f}")
            logger.info(f"{prefix}├─ میانه (Median):           {median_val:>15.6f}")
            logger.info(f"{prefix}├─ چارك سوم (Q3):            {q3:>15.6f}")
            logger.info(f"{prefix}├─ دامنه بین‌چارکی (IQR):    {iqr:>15.6f}")
            logger.info(f"{prefix}├─ چولگی (Skew):             {skewness:>15.3f}")
            logger.info(f"{prefix}└─ كشیدگی (Kurt):            {kurtosis:>15.3f}")

            return result

        # ── طبقه‌بندی ────────────────────────────────────────────
        try:
            unique_all, counts_all = np.unique(
                clean_data.astype(int), return_counts=True
            )
        except (ValueError, TypeError):
            unique_all, counts_all = np.unique(clean_data, return_counts=True)

        sorted_pairs_all = sorted(zip(unique_all, counts_all), key=lambda x: x[0])

        # ✅ جداسازی INVALID از کلاس‌های معتبر
        invalid_count = 0
        valid_pairs: List[Tuple[int, int]] = []
        for cls, cnt in sorted_pairs_all:
            if int(cls) == INVALID_LABEL_LOCAL:
                invalid_count = int(cnt)
            else:
                valid_pairs.append((int(cls), int(cnt)))

        n_valid_classes = len(valid_pairs)

        result = {
            "type": "classification",
            "count": int(total_samples),
            "n_classes": int(n_valid_classes),
            "invalid_count": int(invalid_count),
            "classes": {int(cls): int(cnt) for cls, cnt in valid_pairs},
        }

        # ── نام‌گذاری کلاس‌ها ────────────────────────────────────
        class_names_map: Dict[int, str] = {}
        if show_class_names:
            if target_name == "price_direction":
                name_map = PriceDirectionNames.NAMES
            elif target_name == "trend_strength":
                name_map = TrendStrengthNames.NAMES
            elif target_name == "volatility_regime":
                name_map = VolatilityRegimeNames.NAMES
            elif target_name == "market_regime":
                name_map = MarketRegimeNames.NAMES
            else:
                name_map = {}

            for cls, _cnt in valid_pairs:
                class_names_map[int(cls)] = name_map.get(
                    int(cls), f"CLASS_{int(cls)}"
                )
            result["classes_with_names"] = class_names_map

        logger.info(
            f"{symbol} 🎯 توزیع کلاسی '{target_name}' "
            f"(n={total_samples:,}, کلاس‌های معتبر={n_valid_classes}, "
            f"INVALID={invalid_count}):"
        )

        for cls, cnt in valid_pairs:
            percentage = (cnt / total_samples) * 100.0
            bar_len = min(int(percentage / 2), 30)
            bar = "█" * bar_len

            if show_class_names:
                if target_name == "price_direction":
                    class_desc = PriceDirectionNames.get_name(int(cls))
                elif target_name == "trend_strength":
                    class_desc = TrendStrengthNames.get_name(int(cls))
                elif target_name == "volatility_regime":
                    class_desc = VolatilityRegimeNames.get_name(int(cls))
                elif target_name == "market_regime":
                    class_desc = MarketRegimeNames.get_name(int(cls))
                else:
                    class_desc = f"Class {cls}"

                logger.info(
                    f"{prefix}├─ کلاس {int(cls):>3d}: {cnt:>8,} نمونه "
                    f"({percentage:>5.2f}%) {bar}  →  {class_desc}"
                )
            else:
                logger.info(
                    f"{prefix}├─ کلاس {int(cls):>3d}: {cnt:>8,} نمونه "
                    f"({percentage:>5.2f}%) {bar}"
                )

        # ✅ نمایش INVALID به‌صورت جدا
        if invalid_count > 0:
            invalid_pct = (invalid_count / total_samples) * 100.0
            logger.info(
                f"{prefix}├─ INVALID(-1): {invalid_count:>8,} نمونه "
                f"({invalid_pct:>5.2f}%)  →  UNKNOWN / SKIP"
            )

        # ── سیگنال‌های معاملاتی ──────────────────────────────────
        if show_class_names and target_name == "market_regime":
            logger.info(f"{prefix}│")
            logger.info(f"{prefix}├─ 📊 سیگنال معاملاتی بر اساس رژیم بازار:")
            for cls, _cnt in valid_pairs:
                action = MarketRegimeNames.get_trading_action(int(cls))
                logger.info(f"{prefix}│     • کلاس {int(cls)} → {action}")

        if show_class_names and target_name == "price_direction":
            logger.info(f"{prefix}│")
            logger.info(f"{prefix}├─ 📊 سیگنال معاملاتی بر اساس جهت:")
            for cls, _cnt in valid_pairs:
                signal = PriceDirectionNames.get_trading_signal(int(cls))
                logger.info(f"{prefix}│     • کلاس {int(cls)} → {signal}")

        logger.info(
            f"{prefix}└─ تعداد کلاس‌های معتبر: {n_valid_classes}"
        )

        return result

    except Exception as e:
        logger.error(
            f"{symbol} ❌ خطا در تحلیل توزیع '{target_name}': {str(e)}"
        )
        return result
        
def transform_with_history(
    engine: "AntiCorrelationFeatureEngine",
    history_df: pd.DataFrame,
    split_df: pd.DataFrame,
    max_lookback: int,
    mode: str = "transform",
) -> pd.DataFrame:
    """
    استخراج features برای val/test با context تاریخی.

    ✅ v2.0 — رفع schema mismatch:
        نسخهٔ قبلی `hist_cols != split_cols` را چک می‌کرد و اگر
        ستون `timestamp` در split_df نبود (چون در فرآیند
        `_process_single_split_with_enhancement` حذف شده بود)،
        ValueError می‌داد.

        راه‌حل: ستون‌های datetime / timestamp از هر دو طرف
        حذف می‌شوند قبل از مقایسه. این ستون‌ها فقط برای
        context historical استفاده می‌شوند، نه برای feature.

    قرارداد:
        context = concat(history_df.tail(max_lookback), split_df)
        features_full = engine.transform(context)
        features_split = features_full.iloc[-len(split_df):]

    Args:
        engine: AntiCorrelationFeatureEngine fitted.
        history_df: داده‌های واقعیِ قبل از split_df (شامل OHLCV).
        split_df: داده split جاری.
        max_lookback: بیشترین پنجره‌ای که فیچرها دارند (مثلاً 200).
        mode: 'transform' یا 'fit'.

    Returns:
        DataFrame features فقط برای split_df.

    Raises:
        ValueError: در کمبود history یا عدم تطابق schema واقعی.
        RuntimeError: در خطای engine.
    """
    # ─── اعتبارسنجی ورودی ───
    if not isinstance(history_df, pd.DataFrame):
        raise TypeError("history_df must be a DataFrame")
    if not isinstance(split_df, pd.DataFrame):
        raise TypeError("split_df must be a DataFrame")
    if history_df.empty or split_df.empty:
        raise ValueError("history_df and split_df must be non-empty")
    if len(history_df) < max_lookback:
        raise ValueError(
            f"history_df length {len(history_df)} < max_lookback "
            f"{max_lookback}; rolling features at split start cannot be "
            f"correctly computed."
        )

    # ─── ✅ FIX: حذف ستون‌های datetime از هر دو طرف قبل از مقایسه ───
    def _drop_datetime_cols(frame: pd.DataFrame) -> pd.DataFrame:
        """حذف ستون‌هایی که datetime هستند یا نام‌شان timestamp/date/time دارد."""
        drop_cols: List[str] = []
        for col in frame.columns:
            col_lower = str(col).lower()
            # نام‌های رایج datetime
            if col_lower in ("timestamp", "datetime", "date", "time",
                             "open_time", "close_time"):
                drop_cols.append(col)
                continue
            # dtype datetime
            try:
                if pd.api.types.is_datetime64_any_dtype(frame[col]):
                    drop_cols.append(col)
            except Exception:
                pass
        if drop_cols:
            return frame.drop(columns=drop_cols, errors="ignore")
        return frame

    history_aligned = _drop_datetime_cols(history_df)
    split_aligned = _drop_datetime_cols(split_df)

    # ─── schema consistency (بعد از حذف datetime) ───
    hist_cols = set(history_aligned.columns)
    split_cols = set(split_aligned.columns)
    if hist_cols != split_cols:
        missing_in_hist = sorted(split_cols - hist_cols)
        missing_in_split = sorted(hist_cols - split_cols)
        raise ValueError(
            f"schema mismatch (after removing datetime columns): "
            f"missing_in_history={missing_in_hist[:10]}, "
            f"missing_in_split={missing_in_split[:10]}"
        )

    # ─── ساخت context ───
    history_tail = history_aligned.tail(max_lookback).reset_index(drop=True)
    split_reset = split_aligned.reset_index(drop=True)

    context = pd.concat(
        [history_tail, split_reset],
        axis=0,
        ignore_index=True,
    )

    expected_len = len(history_tail) + len(split_reset)
    if len(context) != expected_len:
        raise RuntimeError(
            f"context length mismatch after concat: "
            f"got {len(context)}, expected {expected_len}"
        )

    # ─── اجرای engine ───
    if mode == "transform":
        features_full = engine.extract_all_features(context, mode="transform")
    elif mode == "fit":
        features_full = engine.extract_all_features(context, mode="fit")
    else:
        raise ValueError(f"mode must be 'transform' or 'fit', got {mode!r}")

    if features_full is None or features_full.empty:
        raise RuntimeError("engine returned empty features for context")

    if len(features_full) != len(context):
        raise RuntimeError(
            f"engine returned {len(features_full)} rows for context of "
            f"{len(context)}"
        )

    # ─── برش features برای split_df ───
    features_split = features_full.iloc[-len(split_df):].reset_index(drop=True)
    if len(features_split) != len(split_df):
        raise RuntimeError(
            f"post-slicing length mismatch: got {len(features_split)}, "
            f"expected {len(split_df)}"
        )

    return features_split
    
# ============================================================================
# SECTION: ADVANCED VOLATILITY FEATURES (RV, BV, Jump)
# ============================================================================

@njit(fastmath=True, cache=True)
def _compute_rv_bv_numba(returns: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    محاسبه Realized Variance (RV)، Bipower Variation (BV) و Jump Component
    INPUT: returns (daily/hourly returns)
    OUTPUT: rv, bv, jump (arrays of same length as returns)
    """
    n = len(returns)
    rv = np.full(n, np.nan, dtype=np.float64)
    bv = np.full(n, np.nan, dtype=np.float64)
    jump = np.full(n, np.nan, dtype=np.float64)
    
    if n < window + 2:
        return rv, bv, jump
    
    mu1 = np.sqrt(2.0 / np.pi)  # E(|Z|) for normal distribution ~ 0.7979
    
    for i in range(window, n):
        # RV: sum of squared returns
        sum_rv = 0.0
        # BV: sum of |r_t| * |r_{t-1}|
        sum_bv = 0.0
        for j in range(i - window + 1, i + 1):
            r_j = returns[j]
            if not np.isnan(r_j) and not np.isinf(r_j):
                sum_rv += r_j * r_j
                if j > i - window + 1:
                    r_j_prev = returns[j-1]
                    if not np.isnan(r_j_prev) and not np.isinf(r_j_prev):
                        sum_bv += np.abs(r_j) * np.abs(r_j_prev)
        
        rv[i] = sum_rv
        
        if window > 1:
            bv[i] = (np.pi / 2.0) * (sum_bv / (window - 1)) / (mu1 * mu1)
        else:
            bv[i] = rv[i]
        
        # Jump component: max(0, RV - BV)
        jump[i] = max(0.0, rv[i] - bv[i])
    
    return rv, bv, jump


def _safe_compute_rv_bv(returns: np.ndarray, window: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper for _compute_rv_bv_numba with error handling"""
    try:
        return _compute_rv_bv_numba(returns.astype(np.float64), int(window))
    except Exception as e:
        logger.error(f"❌ _compute_rv_bv failed: {e}")
        n = len(returns)
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)


# ============================================================================
# SECTION: GARCH(1,1) SIMULATION (Past-Only)
# ============================================================================

def _compute_garch_features_numba(
    returns: np.ndarray,
    window: int = 100,
    omega: float = 1e-6,
    alpha: float = 0.1,
    beta: float = 0.85,
) -> np.ndarray:
    """
    GARCH(1,1) کاملاً past-only.

    قرارداد:
        cond_var[t] فقط از {returns[0..t-1]} ساخته می‌شود.
        ردیف‌های t < init_window به‌عنوان warm-up نامعتبر (NaN) برمی‌گردند؛
        هیچ مقداردهی اولیهٔ مبتنی بر آیندهٔ همان ردیف انجام نمی‌شود.

    Args:
        returns: بازده‌های 1D.
        window: حداقل تعداد نمونهٔ گذشته برای شروع.
        omega, alpha, beta: پارامترهای GARCH(1,1).

    Returns:
        آرایهٔ هم‌طول با مقدار شرطی واریانس؛ ابتدای سری NaN.
    """
    r = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = int(r.shape[0])
    out = np.full(n, np.nan, dtype=np.float64)

    w = int(window)
    if w < 2:
        raise ValueError(f"window must be >=2, got {w}")

    try:
        alpha_f = float(alpha)
        beta_f = float(beta)
        omega_f = float(omega)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha/beta/omega must be float-convertible") from exc

    if not (0.0 <= alpha_f < 1.0):
        raise ValueError(f"alpha must be in [0,1), got {alpha_f}")
    if not (0.0 <= beta_f < 1.0):
        raise ValueError(f"beta must be in [0,1), got {beta_f}")
    if alpha_f + beta_f >= 1.0:
        raise ValueError(
            f"alpha+beta must be <1 for stationarity, got {alpha_f + beta_f}"
        )
    if omega_f <= 0.0:
        raise ValueError(f"omega must be >0, got {omega_f}")

    if n <= w:
        return out

    # ─────────────────────────────────────────────────────────────
    # مقداردهی اولیه: فقط از [0, w) که کاملاً گذشتهٔ ردیف w است.
    # ─────────────────────────────────────────────────────────────
    init_slice = r[:w]
    init_finite = init_slice[np.isfinite(init_slice)]
    if init_finite.size < max(2, w // 2):
        # دادهٔ ابتدایی ناکافی؛ کل ستون NaN می‌ماند (سیگنال صریح)
        logger.warning(
            "GARCH init: only %d finite values in first %d rows; column stays NaN",
            int(init_finite.size), w,
        )
        return out

    long_term_var = float(np.var(init_finite))
    if not np.isfinite(long_term_var) or long_term_var < 1e-12:
        long_term_var = 1e-4

    # شروع معتبر از ردیف w (نه از 0) — مقدار اولیهٔ cond_var[w-1] = long_term_var
    # که فقط از [0, w) ساخته شده و هیچ اطلاعی از [w, n) ندارد.
    prev_var = long_term_var
    out[w - 1] = prev_var

    for t in range(w, n):
        prev_ret = r[t - 1]
        if np.isfinite(prev_ret):
            prev_var = omega_f + alpha_f * (prev_ret * prev_ret) + beta_f * prev_var
            if prev_var < 1e-8:
                prev_var = long_term_var
        else:
            # carry-forward: تابع صرفاً state را نگه می‌دارد
            pass
        out[t] = prev_var

    return out
    

def _safe_compute_garch_features(returns: np.ndarray, window: int = 100) -> np.ndarray:
    """Wrapper for GARCH calculation with fallback to rolling variance"""
    try:
        garch_var = _compute_garch_features_numba(returns.astype(np.float64), int(window))
        # Replace any remaining NaN/Inf with rolling variance
        rolling_var = rolling_var_numba(returns, window)
        garch_var = np.where(np.isfinite(garch_var), garch_var, rolling_var)
        return garch_var
    except Exception as e:
        logger.error(f"❌ GARCH calculation failed: {e}")
        return rolling_var_numba(returns.astype(np.float64), window)


@njit(cache=True)
def rolling_var_numba(data: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling variance کاملاً past-only.

    مقدار خروجی در index=i فقط از داده‌های بازه [i-window, i)
    استفاده می‌کند. پنجره ناقص یا دارای NaN/Inf خروجی NaN می‌دهد.
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)

    if window < 2 or n <= window:
        return result

    for i in range(window, n):
        start = i - window
        sum_value = 0.0
        sum_square = 0.0
        valid_count = 0

        for j in range(start, i):
            value = data[j]

            if np.isfinite(value):
                sum_value += value
                sum_square += value * value
                valid_count += 1

        if valid_count != window:
            continue

        mean_value = sum_value / valid_count
        variance = (sum_square / valid_count) - (mean_value * mean_value)

        if variance < 0.0 and variance > -1e-15:
            variance = 0.0

        if variance >= 0.0 and np.isfinite(variance):
            result[i] = variance

    return result

# ============================================================================
# SECTION: HIDDEN MARKOV MODEL (HMM) - Simplified 2-state
# ============================================================================

def _fit_hmm_past_only(
    returns: np.ndarray,
    volatility: np.ndarray,
    window: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ✅ WALL STREET: Simplified 2-state HMM (Trending / Ranging)
    کاملاً past-only با استفاده از EM الگوریتم ساده شده.
    
    INPUT:
        returns: بازده‌ها
        volatility: نوسان (مثلاً از ATR یا RV)
        window: پنجره برای fit
    OUTPUT:
        state_prob: احتمال حالت روندی (0 = Ranging, 1 = Trending)
        regime_label: برچسب قطعی (0 یا 1)
    """
    n = len(returns)
    state_prob = np.full(n, np.nan, dtype=np.float64)
    regime_label = np.full(n, -1, dtype=np.int32)
    
    if n < window + 50:
        return state_prob, regime_label
    
    # Use only past data to fit HMM at each point
    for i in range(window, n):
        # Get past window data
        past_returns = returns[i - window:i]
        past_vol = volatility[i - window:i]
        
        # Clean data
        valid_mask = ~(np.isnan(past_returns) | np.isinf(past_returns) | 
                      np.isnan(past_vol) | np.isinf(past_vol))
        
        if np.sum(valid_mask) < 50:
            state_prob[i] = 0.5
            continue
        
        ret_clean = past_returns[valid_mask]
        vol_clean = past_vol[valid_mask]
        
        # Feature for HMM: normalized return * (1 + normalized volatility)
        # High absolute return + high volatility → Trending
        # Low absolute return + low volatility → Ranging
        ret_abs = np.abs(ret_clean)
        vol_norm = vol_clean / (np.median(vol_clean) + 1e-8)
        feature = ret_abs * (1.0 + vol_norm)
        
        # Simple threshold-based HMM (2 states)
        # State 0: Ranging (low feature)
        # State 1: Trending (high feature)
        threshold = np.percentile(feature, 60)  # 60th percentile as threshold
        
        # Simple EM-like: estimate probability of being in trending state
        # Using logistic function based on recent feature
        recent_feature = feature[-20:] if len(feature) >= 20 else feature
        recent_mean = np.mean(recent_feature)
        
        # Logistic probability
        prob = 1.0 / (1.0 + np.exp(-(recent_mean - threshold) * 10.0))
        state_prob[i] = prob
        
        # Hard label
        regime_label[i] = 1 if prob > 0.6 else 0
    
    return state_prob, regime_label


def _safe_fit_hmm(returns: np.ndarray, volatility: np.ndarray, window: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Wrapper for HMM with fallback"""
    try:
        return _fit_hmm_past_only(returns.astype(np.float64), volatility.astype(np.float64), int(window))
    except Exception as e:
        logger.error(f"❌ HMM fitting failed: {e}")
        n = len(returns)
        return np.full(n, 0.5, dtype=np.float64), np.full(n, 0, dtype=np.int32)


class FeatureOrthogonalizer:
    """
    Orthogonalizer تکراری با merging خودکار clusterها.

    الگوریتم:
        1. clustering اولیه با corr_threshold روی correlation ماتریس.
        2. whitening هر cluster با ZCA.
        3. محاسبه residual correlation.
        4. اگر max residual > target_residual، بدترین جفت را پیدا کن
           و clusterهای حاوی آن دو را merge کن.
        5. بازگشت به گام ۲ تا convergence یا max_merge_iterations.

    این روش تضمین می‌کند که cross-cluster residual correlation نیز
    کاهش یابد، برخلاف نسخه تک‌مرحله‌ای قبلی که فقط intra-cluster را
    مدیریت می‌کرد.
    """

    def __init__(
        self,
        corr_threshold: float = 0.75,
        min_group_size: int = 2,
        variance_floor: float = 1e-10,
        preserve_scale: bool = True,
        symbol: str = "UNKNOWN",
        target_residual: float = 0.85,
        max_merge_iterations: int = 15,
    ) -> None:
        self.corr_threshold = float(corr_threshold)
        self.min_group_size = int(min_group_size)
        self.variance_floor = float(variance_floor)
        self.preserve_scale = bool(preserve_scale)
        self.symbol = symbol
        self.target_residual = float(target_residual)
        self.max_merge_iterations = int(max_merge_iterations)

        self._fitted: bool = False
        self._all_columns: Optional[List[str]] = None
        self._groups: List[Tuple[str, ...]] = []
        self._transforms: Dict[
            Tuple[str, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._final_residual: float = 0.0

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame) -> "FeatureOrthogonalizer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"[{self.symbol}] X must be DataFrame")
        if X.empty:
            raise ValueError(f"[{self.symbol}] X is empty")
        if X.columns.duplicated().any():
            raise ValueError(f"[{self.symbol}] duplicate columns not allowed")

        arr = X.to_numpy(dtype=np.float64, copy=False)
        finite_rows = np.isfinite(arr).all(axis=1)
        arr_f = arr[finite_rows]

        if arr_f.shape[0] < 20:
            logger.warning(
                f"[{self.symbol}] FeatureOrthogonalizer: only "
                f"{arr_f.shape[0]} finite rows; skipping orthogonalization"
            )
            self._all_columns = list(X.columns)
            self._groups = []
            self._transforms = {}
            self._fitted = True
            return self

        cols = list(X.columns)

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(arr_f, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)

        groups_idx: List[set] = self._cluster_correlated(np.abs(corr))

        if not groups_idx:
            self._all_columns = cols
            self._groups = []
            self._transforms = {}
            self._fitted = True
            logger.info(
                f"[{self.symbol}] FeatureOrthogonalizer: no clusters at "
                f"threshold {self.corr_threshold}; nothing to orthogonalize"
            )
            return self

        final_transforms: Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        for iteration in range(self.max_merge_iterations):
            transforms = self._fit_group_transforms(arr_f, groups_idx)
            arr_trans = self._apply_group_transforms(arr_f, transforms)

            with np.errstate(invalid="ignore"):
                res_corr = np.abs(np.corrcoef(arr_trans, rowvar=False))
            res_corr = np.nan_to_num(res_corr, nan=0.0, posinf=0.0, neginf=0.0)
            np.fill_diagonal(res_corr, 0.0)

            max_res = float(res_corr.max()) if res_corr.size > 0 else 0.0
            self._final_residual = max_res
            final_transforms = transforms

            if max_res <= self.target_residual:
                if self.max_merge_iterations > 0:
                    logger.debug(
                        f"[{self.symbol}] Orthogonalizer converged at "
                        f"iter {iteration + 1}: max_res={max_res:.4f}"
                    )
                break

            if iteration == self.max_merge_iterations - 1:
                logger.warning(
                    f"[{self.symbol}] Orthogonalizer reached max iterations "
                    f"({self.max_merge_iterations}); final max_res={max_res:.4f} "
                    f"(target={self.target_residual:.4f})"
                )
                break

            flat_idx = int(np.argmax(res_corr))
            i, j = divmod(flat_idx, res_corr.shape[1])
            if i == j:
                break

            merged = False
            gi_set: Optional[set] = None
            gj_set: Optional[set] = None
            for g in groups_idx:
                if i in g:
                    gi_set = g
                if j in g:
                    gj_set = g

            if gi_set is None and gj_set is None:
                new_group = {i, j}
                groups_idx.append(new_group)
                merged = True
            elif gi_set is not None and gj_set is not None and gi_set is gj_set:
                merged = False
            elif gi_set is not None and gj_set is not None:
                combined = gi_set | gj_set
                groups_idx = [
                    g for g in groups_idx if g is not gi_set and g is not gj_set
                ]
                groups_idx.append(combined)
                merged = True
            else:
                target_group = gi_set if gi_set is not None else gj_set
                other_element = j if gi_set is not None else i
                target_group.add(other_element)
                merged = True

            if not merged:
                break

            logger.debug(
                f"[{self.symbol}] Orthogonalizer iter {iteration + 1}: "
                f"max_res={max_res:.4f} > target={self.target_residual:.4f}; "
                f"merged clusters → sizes={[len(g) for g in groups_idx]}"
            )

        self._all_columns = cols
        self._groups = []
        self._transforms = {}

        for g_idx_tuple, transform_tuple in final_transforms.items():
            g_cols = tuple(cols[i] for i in g_idx_tuple)
            self._groups.append(g_cols)
            self._transforms[g_cols] = transform_tuple

        self._fitted = True

        logger.info(
            f"[{self.symbol}] FeatureOrthogonalizer fitted | "
            f"groups={len(self._groups)} | "
            f"largest_group={max((len(g) for g in self._groups), default=0)} | "
            f"final_residual={self._final_residual:.4f} "
            f"(target={self.target_residual:.4f})"
        )
        return self

    # ------------------------------------------------------------------ #
    def _fit_group_transforms(
        self,
        arr_f: np.ndarray,
        groups_idx: List[set],
    ) -> Dict[Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        transforms: Dict[
            Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}

        for g_idx in groups_idx:
            if len(g_idx) < self.min_group_size:
                continue

            g_list = tuple(sorted(g_idx))
            sub = arr_f[:, list(g_list)]
            mean = sub.mean(axis=0)
            sub_c = sub - mean
            n = sub_c.shape[0]

            if n < 2:
                continue

            C = (sub_c.T @ sub_c) / max(n - 1, 1)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, self.variance_floor)

            W = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
            target_std = np.sqrt(np.maximum(np.diag(C), self.variance_floor))

            transforms[g_list] = (mean, W, target_std)

        return transforms

    # ------------------------------------------------------------------ #
    def _apply_group_transforms(
        self,
        arr_f: np.ndarray,
        transforms: Dict[
            Tuple[int, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> np.ndarray:
        arr_out = arr_f.copy()

        for g_list, (mean, W, target_std) in transforms.items():
            g_idx = list(g_list)
            sub = arr_out[:, g_idx]
            sub_c = sub - mean
            sub_o = sub_c @ W
            if self.preserve_scale:
                sub_o = sub_o * target_std
            arr_out[:, g_idx] = sub_o

        return arr_out

    # ------------------------------------------------------------------ #
    def _cluster_correlated(self, abs_corr: np.ndarray) -> List[set]:
        n = abs_corr.shape[0]
        adj = abs_corr > self.corr_threshold
        np.fill_diagonal(adj, False)

        visited = np.zeros(n, dtype=bool)
        groups: List[set] = []

        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            comp: set = set()
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                comp.add(node)
                nbrs = np.where(adj[node])[0]
                for nb in nbrs:
                    if not visited[nb]:
                        stack.append(nb)
            if len(comp) >= self.min_group_size:
                groups.append(comp)

        return groups

    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError(
                f"[{self.symbol}] FeatureOrthogonalizer must be fitted first"
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"[{self.symbol}] X must be DataFrame")

        cols = list(X.columns)
        if cols != list(self._all_columns):
            missing = [c for c in self._all_columns if c not in cols]
            extra = [c for c in cols if c not in self._all_columns]
            raise ValueError(
                f"[{self.symbol}] FeatureOrthogonalizer schema mismatch: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        arr = X.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(arr).all():
            raise ValueError(
                f"[{self.symbol}] FeatureOrthogonalizer input contains NaN/Inf"
            )

        if not self._transforms:
            return pd.DataFrame(arr, columns=cols, index=X.index)

        col_index = {c: i for i, c in enumerate(cols)}

        for g_cols, (mean, W, target_std) in self._transforms.items():
            g_idx = [col_index[c] for c in g_cols]
            sub = arr[:, g_idx]
            sub_c = sub - mean
            sub_o = sub_c @ W
            if self.preserve_scale:
                sub_o = sub_o * target_std
            arr[:, g_idx] = sub_o

        return pd.DataFrame(arr, columns=cols, index=X.index)

    # ------------------------------------------------------------------ #
    def residual_max_corr(self, X: pd.DataFrame) -> float:
        out = self.transform(X)
        arr = out.to_numpy(dtype=np.float64)
        if arr.shape[1] < 2:
            return 0.0
        with np.errstate(invalid="ignore"):
            c = np.abs(np.corrcoef(arr, rowvar=False))
        c = np.nan_to_num(c, nan=0.0)
        np.fill_diagonal(c, 0.0)
        return float(c.max())

    # ------------------------------------------------------------------ #
    def get_groups(self) -> List[Tuple[str, ...]]:
        return list(self._groups)
        
@dataclass
class LowCorrelationFeatureConfig:
    """
    ✅ WALL STREET OPTIMIZED - ORDER FIXED
    پیکربندی فیچرهای ضدهمبستگی - کاهش سخت‌گیری برای حفظ فیچرهای باکیفیت
    """
    # ⬇️ فیلد بدون default باید اول بیاید
    FEATURE_CATEGORIES: Dict[str, List[str]] = field(default_factory=lambda: {
        # ═══════════════════════════════════════════════════════════
        # دسته 1: مومنتوم خام - نگه‌داری همه
        # ═══════════════════════════════════════════════════════════
        'MOMENTUM_RAW': [
            'roc_momentum',
            'momentum_acceleration',
            'momentum_roc_ma',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 2: مومنتوم نرمال‌شده - نگه‌داری همه
        # ═══════════════════════════════════════════════════════════
        'MOMENTUM_NORMALIZED': [
            'rsi_14',
            'stoch_k',
            'stoch_d',
            'williams_r',
            'cci_20',
            'ultimate_osc',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 3: مومنتوم هیبرید (MACD)
        # ═══════════════════════════════════════════════════════════
        'MOMENTUM_HYBRID': [
            'macd_line',
            'macd_signal',
            'macd_histogram',
            'macd_zscore',
            'macd_hist_roc',
            'price_macd_ratio',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 4: نوسان مطلق
        # ═══════════════════════════════════════════════════════════
        'VOLATILITY_ABSOLUTE': [
            'atr_14',
            'atr_raw_14',
            'atr_raw_28',
            'garman_klass',
            'rogers_satchell',
            'parkinson_vol',
            'yang_zhang',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 5: نوسان نرمال‌شده
        # ═══════════════════════════════════════════════════════════
        'VOLATILITY_NORMALIZED': [
            'atr_percentile',
            'atr_ratio',
            'atr_relative',
            'hv_30',
            'vol_regime',
            'vol_skew',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 6: نوسان STD
        # ═══════════════════════════════════════════════════════════
        'VOLATILITY_STD': [
            'return_volatility',
            'vol_of_vol',
            'vol_trend',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 7: حجم خام
        # ═══════════════════════════════════════════════════════════
        'VOLUME_RAW': [
            'volume_ratio',
            'volume_zscore',
            'volume_percentile',
            'volume_acceleration',
            'volume_roc',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 8: جریان حجم
        # ═══════════════════════════════════════════════════════════
        'VOLUME_FLOW': [
            'vwap_distance',
            'obv_ma_cross',
            'vpin',
            'ad_line',
            'ad_line_ema5',
            'ad_line_ema20',
            'vwap_upper',
            'vwap_lower',
            'vwap_band_width',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 9: حجم مومنتوم
        # ═══════════════════════════════════════════════════════════
        'VOLUME_MOMENTUM': [
            'volume_roc',
            'volume_acceleration',
            'volume_impulse',
            'volume_trend_strength',
            'smart_money_index',
            'smi_zscore',
            'volume_absorption',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 10: روند EMA
        # ═══════════════════════════════════════════════════════════
        'TREND_EMA': [
            'ema20_dist',
            'ema50_dist',
            'ema200_dist',
            'ema_cross',
            'ema_cross_long',
            'ema_angle',
            'ema_ribbon',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 11: شیب قیمت
        # ═══════════════════════════════════════════════════════════
        'TREND_SLOPE': [
            'trend_slope',
            'trend_acceleration',
            'trend_quality',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 12: قدرت روند
        # ═══════════════════════════════════════════════════════════
        'TREND_STRENGTH': [
            'trend_adx',
            'plus_di',
            'minus_di',
            'di_spread',
            'aroon_up',
            'aroon_down',
            'aroon_osc',
            'trend_intensity',
            'price_efficiency',
            'kst',
            'kst_signal',
            'trix',
            'trix_signal',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 13: ساختار قیمت
        # ═══════════════════════════════════════════════════════════
        'PRICE_STRUCTURE': [
            'body_ratio',
            'body_size',
            'upper_shadow',
            'lower_shadow',
            'hl_range',
            'oc_strength',
            'candle_type',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 14: Bollinger Bands
        # ═══════════════════════════════════════════════════════════
        'BB_STRUCTURE': [
            'bb_width',
            'bb_position',
            'bb_zscore',
            'bb_squeeze',
            'bb_width_50',
            'bb_percentile',
            'bb_relative',
            'bb_expansion_rate',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 15: پیوت
        # ═══════════════════════════════════════════════════════════
        'PIVOT_STRUCTURE': [
            'pivot_distance',
            'resistance_gap',
            'support_gap',
            'support_strength',
            'resistance_strength',
            'sr_balance',
            'fib_236',
            'fib_382',
            'fib_618',
            'fib_786',
            'range_position',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 16: توزیع آماری
        # ═══════════════════════════════════════════════════════════
        'DISTRIBUTION': [
            'skewness',
            'kurtosis',
            'normality_test',
            'price_entropy',
            'dist_shift',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 17: رتبه‌ای
        # ═══════════════════════════════════════════════════════════
        'RANK_BASED': [
            'price_rank',
            'volume_rank',
            'volatility_rank',
            'price_percentile',
            'volume_percentile',
            'vol_percentile',
            'price_volume_combined_rank',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 18: Z-Score
        # ═══════════════════════════════════════════════════════════
        'ZSCORE_BASED': [
            'price_zscore',
            'zscore_acceleration',
            'zscore_extreme',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 19: زمانی
        # ═══════════════════════════════════════════════════════════
        'TIME_CYCLE': [
            'hour_sin',
            'hour_cos',
            'dow_sin',
            'dow_cos',
            'month_sin',
            'month_cos',
            'trend_duration',
            'consolidation',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 20: Order Flow
        # ═══════════════════════════════════════════════════════════
        'ORDER_FLOW_RAW': [
            'ofi_normalized',
            'ofi_cumsum',
            'ofi_ma5',
            'ofi_ma20',
            'cvd_zscore',
            'cvd_slope',
            'delta_ratio',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 21: فشار خرید/فروش
        # ═══════════════════════════════════════════════════════════
        'BUY_SELL_PRESSURE': [
            'buy_pressure',
            'sell_pressure',
            'pressure_ratio',
            'absorption',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 22: ترکیبی مومنتوم-نوسان
        # ═══════════════════════════════════════════════════════════
        'HYBRID_MOMENTUM_VOL': [
            'sharpe_ratio',
            'sortino_ratio',
            'calmar_ratio',
            'info_ratio',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 23: ترکیبی قیمت-حجم
        # ═══════════════════════════════════════════════════════════
        'HYBRID_PRICE_VOLUME': [
            'price_volume_correlation',
            'volume_confirmed_trend',
            'trend_confirmation',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 24: الیوت ویو
        # ═══════════════════════════════════════════════════════════
        'ELLIOTT_WAVE': [
            'zigzag_signal',
            'wave_count',
            'fib_time_zone',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 25: آنتروپی
        # ═══════════════════════════════════════════════════════════
        'ENTROPY_FEATURES': [
            'price_entropy',
            'volume_entropy',
            'tsallis_entropy',
            'conditional_entropy',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 26: پیچیدگی
        # ═══════════════════════════════════════════════════════════
        'COMPLEXITY': [
            'hurst_exponent',
            'fractal_dimension',
            'lyapunov_approx',
            'sample_entropy',
        ],

        # ═══════════════════════════════════════════════════════════
        # دسته 27: autocorrelation
        # ═══════════════════════════════════════════════════════════
        'AUTOCORRELATION': [
            'autocorr_lag1',
            'autocorr_lag5',
            'autocorr_lag10',
            'pacf_1',
        ],
        
        # ⬇️ دسته محافظت شده - هرگز حذف نشوند
        'PROTECTED_FEATURES': [
            'trend_adx',
            'atr_14',
            'macd_line',
            'macd_histogram',
            'bb_width',
            'volume_zscore',
            'price_return',
            'price_direction',
            'trend_strength',
            'volatility_regime',
            'market_regime',
        ],
    })
    
    # ⬇️ فیلدهای با default ساده بعد از field می‌آیند
    CORRELATION_THRESHOLD: float = 0.95
    VARIANCE_THRESHOLD: float = 1e-8
    MAX_PER_CATEGORY: int = 5
    CROSS_CATEGORY_CORRELATION: bool = False
    SMART_REDUNDANCY_REMOVAL: bool = True

    def get_all_feature_names(self) -> List[str]:
        """همه نام فیچرها را برگردان"""
        all_features = []
        for category, features in self.FEATURE_CATEGORIES.items():
            all_features.extend(features)
        return all_features

    def get_protected_features(self) -> List[str]:
        """فیچرهای محافظت شده که هرگز نباید حذف شوند"""
        return self.FEATURE_CATEGORIES.get('PROTECTED_FEATURES', [])

    def get_category(self, feature_name: str) -> str:
        """دسته یک فیچر را برگردان"""
        for category, features in self.FEATURE_CATEGORIES.items():
            if feature_name in features:
                return category
        return 'UNKNOWN'

class AntiCorrelationFeatureEngine:
    """
    ✅ VERSION 6.3 - WALL STREET GOLD FINAL (COMPLETE FIX)
    ✅ FIXED: extract_all_features now properly handles transform mode without fitting
    ✅ FIXED: Uses loaded scaler metadata instead of refitting
    """

    def __init__(self, symbol: str = None, config=None, target_n_features: int = None):
        self.config = config
        self.feature_names = []
        self.feature_categories = {}
        self.symbol = symbol
        self.target_n_features = target_n_features
        
        # ✅ FIT STATE - FROZEN AFTER fit()
        self._is_fitted = False
        self._fitted_stats = {
            'means': {},
            'stds': {},
            'medians': {},
            'mins': {},
            'maxs': {},
            'feature_names': []
        }
        self._fitted_feature_names = []
        self._fitted_feature_columns = None
        self._protected_columns = PROJECT_TARGETS + ['open', 'high', 'low', 'close', 'volume', 'ret']
        
        # ✅ NEW: Loaded scaler from metadata (for live data)
        self._loaded_scaler = None
        self._loaded_feature_names = None
        self._loaded_feature_profiles = None
        
        logger.info("=" * 80)
        logger.info("✅ AntiCorrelationFeatureEngine v6.3 (WALL STREET GOLD FINAL - COMPLETE)")
        logger.info("   • NO look-ahead: fit() on TRAIN only")
        logger.info("   • transform() uses FROZEN statistics")
        logger.info("   • ALL 150+ features PRESERVED (no compression)")
        logger.info("   • SMART missing value handling (no blind zero-fill)")
        logger.info("   • COMPLETE logging with try-except per phase")
        if target_n_features:
            logger.info(f"   • TARGET FEATURES: {target_n_features} (padded/trimmed to match model)")
        logger.info("=" * 80)

    def load_scaler_from_metadata(self, scaler_data: Dict) -> bool:
        """
        ✅ NEW: Load scaler from saved metadata (for live data evaluation)
        
        Args:
            scaler_data: Dictionary from universal_load with features data
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            if scaler_data is None:
                logger.error(f"[{self.symbol}] ❌ scaler_data is None")
                return False
            
            features_data = scaler_data.get('features', {})
            if not features_data:
                logger.error(f"[{self.symbol}] ❌ No features data found")
                return False
            
            # Extract scaler object
            self._loaded_scaler = features_data.get('_scaler')
            if self._loaded_scaler is None:
                logger.error(f"[{self.symbol}] ❌ No scaler object found")
                return False
            
            # Extract feature names from metadata
            metadata = scaler_data.get('_metadata', {})
            feature_profiles = metadata.get('feature_profiles', {})
            
            if feature_profiles:
                self._loaded_feature_names = list(feature_profiles.keys())
                self._loaded_feature_profiles = feature_profiles
                logger.info(f"[{self.symbol}] ✅ Loaded {len(self._loaded_feature_names)} feature names from metadata")
            else:
                # Fallback: try to get from scaler object
                if hasattr(self._loaded_scaler, 'feature_names'):
                    self._loaded_feature_names = self._loaded_scaler.feature_names
                    logger.info(f"[{self.symbol}] ✅ Loaded {len(self._loaded_feature_names)} feature names from scaler")
                else:
                    logger.warning(f"[{self.symbol}] ⚠️ No feature names found")
                    return False
            
            # Set as fitted for transform mode
            self._is_fitted = True
            self._fitted_feature_names = self._loaded_feature_names
            
            logger.info(f"[{self.symbol}] ✅ Scaler loaded successfully from metadata")
            return True
            
        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ Failed to load scaler from metadata: {e}")
            return False

    def extract_all_features(
        self,
        df: pd.DataFrame,
        mode: str = "auto",
    ) -> pd.DataFrame:
        """
        استخراج Featureهای past-only با قرارداد FIT/TRANSFORM سخت‌گیرانه.

        قرارداد:

            mode='fit':
                - Feature engineering روی TRAIN انجام می‌شود.
                - آمار imputation و scaling ذخیره می‌شود.
                - Feature names و ترتیب آن‌ها freeze می‌شود.

            mode='transform':
                - فقط از Feature engineering ثابت استفاده می‌شود.
                - فقط از آمار ذخیره‌شدهٔ TRAIN استفاده می‌شود.
                - Feature جدید یا Feature گمشدهٔ بدون آمار باعث خطا می‌شود.
                - هیچ fit مجددی انجام نمی‌شود.

            mode='auto':
                - اگر قبلاً fitted باشد، transform
                - در غیر این صورت، fit

        نکته:
            این تابع Target تولید نمی‌کند و باید فقط روی DataFrame ویژگی‌ها
            یا OHLCV خام اجرا شود. Targetها باید قبل از ورود به مدل جداگانه
            نگهداری شوند.

        Args:
            df: دادهٔ خام یا DataFrame ویژگی‌ها
            mode: fit، transform یا auto

        Returns:
            DataFrame عددی با Featureهای ثابت و هم‌تراز

        Raises:
            TypeError: برای نوع نادرست
            ValueError: برای دادهٔ خالی، ستون‌های ضروری یا خروجی نامعتبر
            RuntimeError: برای Transform بدون FIT
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"df must be pandas.DataFrame, got {type(df).__name__}"
            )

        if df.empty:
            raise ValueError("Cannot extract features from an empty DataFrame")

        normalized_mode = str(mode).strip().lower()

        if normalized_mode not in {"fit", "transform", "auto"}:
            raise ValueError(
                f"mode must be 'fit', 'transform' or 'auto', got {mode!r}"
            )

        if normalized_mode == "auto":
            final_mode = "transform" if self._is_fitted else "fit"
        else:
            final_mode = normalized_mode

        if final_mode == "transform" and not self._is_fitted:
            raise RuntimeError(
                "AntiCorrelationFeatureEngine must be fitted on TRAIN "
                "before transform mode"
            )

        required_columns = ["open", "high", "low", "close"]

        input_columns = {
            _canonical_column_name(column): str(column)
            for column in df.columns
        }

        missing_required = [
            column
            for column in required_columns
            if _canonical_column_name(column) not in input_columns
        ]

        if missing_required:
            raise ValueError(
                f"Missing required OHLC columns: {missing_required}"
            )

        duplicate_canonical_columns = {}

        for column in df.columns:
            canonical = _canonical_column_name(column)
            duplicate_canonical_columns.setdefault(canonical, []).append(
                str(column)
            )

        ambiguous_columns = {
            canonical: names
            for canonical, names in duplicate_canonical_columns.items()
            if len(names) > 1
        }

        if ambiguous_columns:
            raise ValueError(
                f"Ambiguous columns after canonicalization: "
                f"{ambiguous_columns}"
            )

        try:
            numeric_input = df.copy(deep=True)

            for column in numeric_input.columns:
                if not pd.api.types.is_numeric_dtype(numeric_input[column]):
                    numeric_input[column] = pd.to_numeric(
                        numeric_input[column],
                        errors="raise",
                    )

            raw_values = numeric_input[
                [
                    input_columns[_canonical_column_name(column)]
                    for column in required_columns
                ]
            ].to_numpy(dtype=np.float64, copy=False)

            if not np.isfinite(raw_values).all():
                bad_columns = [
                    column
                    for column in required_columns
                    if not np.isfinite(
                        numeric_input[
                            input_columns[_canonical_column_name(column)]
                        ].to_numpy(dtype=np.float64, copy=False)
                    ).all()
                ]

                raise ValueError(
                    f"OHLC input contains NaN or Inf in columns: {bad_columns}"
                )

            features_df = self._compute_all_150_features_past_only(
                close=numeric_input[
                    input_columns[_canonical_column_name("close")]
                ].to_numpy(dtype=np.float64, copy=False),
                high=numeric_input[
                    input_columns[_canonical_column_name("high")]
                ].to_numpy(dtype=np.float64, copy=False),
                low=numeric_input[
                    input_columns[_canonical_column_name("low")]
                ].to_numpy(dtype=np.float64, copy=False),
                open_=numeric_input[
                    input_columns[_canonical_column_name("open")]
                ].to_numpy(dtype=np.float64, copy=False),
                volume=(
                    numeric_input[
                        input_columns[_canonical_column_name("volume")]
                    ].to_numpy(dtype=np.float64, copy=False)
                    if _canonical_column_name("volume") in input_columns
                    else np.ones(len(numeric_input), dtype=np.float64)
                ),
            )

        except Exception as exc:
            raise RuntimeError(
                "Past-only feature generation failed"
            ) from exc

        if not isinstance(features_df, pd.DataFrame):
            raise RuntimeError(
                "Feature generator did not return pandas.DataFrame"
            )

        if features_df.empty:
            raise ValueError(
                "Past-only feature generator returned an empty DataFrame"
            )

        if len(features_df) != len(df):
            raise ValueError(
                "Feature row count mismatch: "
                f"input={len(df)}, output={len(features_df)}"
            )

        if features_df.columns.duplicated().any():
            duplicated = features_df.columns[
                features_df.columns.duplicated()
            ].tolist()

            raise ValueError(
                f"Feature generator returned duplicate columns: {duplicated}"
            )

        features_df = features_df.reset_index(drop=True)

        for column in features_df.columns:
            if not pd.api.types.is_numeric_dtype(features_df[column]):
                features_df[column] = pd.to_numeric(
                    features_df[column],
                    errors="raise",
                )

        if final_mode == "fit":
            self._fitted_feature_names = [
                str(column)
                for column in features_df.columns
            ]

            self._total_features_generated = len(
                self._fitted_feature_names
            )

            self._fitted_stats = {
                "feature_names": list(self._fitted_feature_names),
                "feature_count": len(self._fitted_feature_names),
                "symbol": self.symbol,
                "fitted_at": datetime.now(timezone.utc).isoformat(),
                "means": {},
                "stds": {},
                "medians": {},
                "mins": {},
                "maxs": {},
                "q1": {},
                "q3": {},
            }

            raw_features = features_df.copy(deep=True)

            for column in self._fitted_feature_names:
                values = raw_features[column].to_numpy(
                    dtype=np.float64,
                    copy=False,
                )

                values[~np.isfinite(values)] = np.nan
                finite_values = values[np.isfinite(values)]

                if finite_values.size == 0:
                    raise ValueError(
                        f"Feature {column!r} has no finite TRAIN values"
                    )

                mean_value = float(np.mean(finite_values))
                std_value = float(np.std(finite_values))
                median_value = float(np.median(finite_values))
                min_value = float(np.min(finite_values))
                max_value = float(np.max(finite_values))
                q1_value = float(np.percentile(finite_values, 25))
                q3_value = float(np.percentile(finite_values, 75))

                self._fitted_stats["means"][column] = mean_value
                self._fitted_stats["stds"][column] = (
                    std_value if std_value > 1e-12 else 1.0
                )
                self._fitted_stats["medians"][column] = median_value
                self._fitted_stats["mins"][column] = min_value
                self._fitted_stats["maxs"][column] = max_value
                self._fitted_stats["q1"][column] = q1_value
                self._fitted_stats["q3"][column] = q3_value

            features_df = self._handle_missing_values_smart(
                df=features_df,
                column_stats=self._fitted_stats,
                is_train=True,
                strict=True,
            )

            self._fitted_feature_columns = list(
                self._fitted_feature_names
            )
            self.feature_names = list(self._fitted_feature_names)
            self._is_fitted = True

        else:
            features_df = self._align_columns_smart(
                features_df=features_df,
                fitted_feature_names=list(self._fitted_feature_names),
                column_stats=self._fitted_stats,
                strict=True,
            )

            features_df = self._handle_missing_values_smart(
                df=features_df,
                column_stats=self._fitted_stats,
                is_train=False,
                strict=True,
            )

        if list(features_df.columns) != list(self._fitted_feature_names):
            raise RuntimeError(
                "Final feature order differs from frozen TRAIN feature order"
            )

        if len(features_df.columns) != len(self._fitted_feature_names):
            raise RuntimeError(
                "Final feature count differs from frozen TRAIN feature count"
            )

        final_values = features_df.to_numpy(
            dtype=np.float64,
            copy=False,
        )

        if not np.isfinite(final_values).all():
            raise ValueError(
                "Final extracted feature matrix contains NaN or Inf"
            )

        if self.target_n_features is not None:
            expected_features = int(self.target_n_features)

            if expected_features <= 0:
                raise ValueError(
                    f"target_n_features must be positive, got {expected_features}"
                )

            actual_features = features_df.shape[1]

            if actual_features != expected_features:
                raise RuntimeError(
                    "Feature dimension does not match model contract: "
                    f"expected={expected_features}, actual={actual_features}. "
                    "Zero-padding and silent trimming are disabled."
                )

        features_df = features_df.astype(np.float32, copy=False)

        return features_df
    
    def _apply_fallback_transform(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Transform فقط با آمار frozen شده TRAIN.

        محاسبه mean/std از VAL/TEST/LIVE ممنوع است. اگر آمار یک Feature
        وجود نداشته باشد، پایپ‌لاین باید متوقف شود.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"[{self.symbol}] df must be pandas.DataFrame, "
                f"got {type(df).__name__}"
            )

        if df.empty:
            raise ValueError(
                f"[{self.symbol}] Cannot transform an empty DataFrame"
            )

        means = self._fitted_stats.get("means", {})
        stds = self._fitted_stats.get("stds", {})

        missing_statistics = [
            column
            for column in df.columns
            if column not in means or column not in stds
        ]

        if missing_statistics:
            raise RuntimeError(
                f"[{self.symbol}] Frozen TRAIN scaling statistics are missing "
                f"for columns: {missing_statistics}"
            )

        for column in df.columns:
            mean_value = float(means[column])
            std_value = float(stds[column])

            if not np.isfinite(mean_value):
                raise ValueError(
                    f"[{self.symbol}] TRAIN mean for {column!r} is not finite"
                )

            if not np.isfinite(std_value) or std_value <= 1e-12:
                raise ValueError(
                    f"[{self.symbol}] TRAIN std for {column!r} is invalid: "
                    f"{std_value}"
                )

            values = df[column].to_numpy(dtype=np.float64, copy=True)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"[{self.symbol}] Transform input contains NaN/Inf "
                    f"in feature {column!r}"
                )

            transformed = (values - mean_value) / std_value

            if not np.isfinite(transformed).all():
                raise ValueError(
                    f"[{self.symbol}] Transform produced NaN/Inf "
                    f"for feature {column!r}"
                )

            df[column] = transformed
            
    
    def _ema_past_only(self, arr: np.ndarray, alpha: float) -> np.ndarray:
        """
        Exponential Moving Average (past-only) with NaN-safe initialization.

        ✅ FIX v2.0:
            - دیگر NaN از ابتدای آرایه به کل خروجی منتشر نمی‌شود.
            - EMA از نخستین مقدار finite موجود مقداردهی می‌شود.
            - مقادیر غیر-finite بعدی با carry-forward مدیریت می‌شوند
              (بدون نشت اطلاعات آینده).

        Args:
            arr: آرایهٔ ورودی (ممکن است ابتدایش NaN باشد).
            alpha: ضریب EMA در بازهٔ (0, 1].

        Returns:
            آرایهٔ هم‌طول با EMA محاسبه‌شده. مقادیر پیش از نخستین
            مقدار finite همچنان NaN خواهند بود.
        """
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
                    if np.isfinite(prev):
                        out[i] = alpha * curr + one_minus_alpha * prev
                    else:
                        out[i] = curr
                else:
                    out[i] = prev

            return out

        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _ema_past_only failed: {e}")
            return np.full(len(arr), np.nan, dtype=np.float64)
            
    def _rolling_mean_past_only(
        self,
        arr: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        Rolling mean (past-only) با مدیریت کامل NaN.

        ✅ FIX v2.0:
            - cumsum دیگر با NaN آلوده نمی‌شود.
            - از prefix-sum و prefix-count برای محاسبهٔ میانگین استفاده
              می‌شود؛ تنها زمانی مقدار finite برمی‌گردد که تمام
              window مقدار پنجره finite باشند.
        """
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

        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _rolling_mean_past_only failed: {e}")
            return np.full(len(arr), np.nan, dtype=np.float64)
            
    def _rolling_std_past_only(
        self,
        arr: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        Rolling standard deviation با پنجره کامل و مدیریت صحیح NaN/Inf.

        خروجی index=i بر اساس پنجره شامل index=i محاسبه می‌شود. از آنجا
        که ورودی Feature Engine قبلاً یک کندل shift شده، این مقدار همچنان
        نسبت به کندل اصلی کاملاً past-only است.
        """
        values = np.asarray(arr, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"[{self.symbol}] Rolling std input must be one-dimensional, "
                f"got shape={values.shape}"
            )

        n = values.shape[0]
        window_size = int(window)
        output = np.full(n, np.nan, dtype=np.float64)

        if window_size < 2 or n < window_size:
            return output

        finite = np.isfinite(values)
        safe_values = np.where(finite, values, 0.0)

        prefix_sum = np.empty(n + 1, dtype=np.float64)
        prefix_square = np.empty(n + 1, dtype=np.float64)
        prefix_count = np.empty(n + 1, dtype=np.int64)

        prefix_sum[0] = 0.0
        prefix_square[0] = 0.0
        prefix_count[0] = 0

        np.cumsum(safe_values, out=prefix_sum[1:])
        np.cumsum(safe_values * safe_values, out=prefix_square[1:])
        np.cumsum(finite.astype(np.int64), out=prefix_count[1:])

        for end in range(window_size - 1, n):
            start = end - window_size + 1
            valid_count = int(prefix_count[end + 1] - prefix_count[start])

            if valid_count != window_size:
                continue

            total = prefix_sum[end + 1] - prefix_sum[start]
            total_square = prefix_square[end + 1] - prefix_square[start]

            mean_value = total / window_size
            variance = total_square / window_size - mean_value * mean_value

            if variance < 0.0 and variance > -1e-12:
                variance = 0.0

            if variance >= 0.0 and np.isfinite(variance):
                output[end] = np.sqrt(variance)

        return output
        
    def _rsi_past_only(self, c: np.ndarray, period: int) -> np.ndarray:
        """
        RSI با مدیریت کامل NaN در ابتدای آرایه.

        ✅ FIX v2.0:
            - delta[1] دیگر NaN تولید نمی‌کند (چون c[0]=NaN را رد می‌کند).
            - نخستین پنجرهٔ معتبر period-تایی از delta پیدا می‌شود.
            - مقادیر پیش از پنجرهٔ اول به‌صورت NaN بازگردانده می‌شوند
              تا توسط _handle_missing_values_smart پر شوند.
        """
        try:
            n = int(len(c))
            period_i = int(period)
            out = np.full(n, np.nan, dtype=np.float64)

            if period_i < 1 or n < period_i + 2:
                return out

            c_f = np.asarray(c, dtype=np.float64)

            delta = np.full(n, np.nan, dtype=np.float64)
            valid_pair = np.isfinite(c_f[:-1]) & np.isfinite(c_f[1:])
            delta[1:][valid_pair] = (
                c_f[1:][valid_pair] - c_f[:-1][valid_pair]
            )

            gain = np.where(
                np.isfinite(delta),
                np.where(delta > 0.0, delta, 0.0),
                np.nan,
            )
            loss = np.where(
                np.isfinite(delta),
                np.where(delta < 0.0, -delta, 0.0),
                np.nan,
            )

            start_idx = -1
            for i in range(1, n - period_i + 1):
                if np.all(np.isfinite(delta[i:i + period_i])):
                    start_idx = i
                    break

            if start_idx < 0:
                return out

            init_idx = start_idx + period_i - 1

            sum_gain = float(np.sum(gain[start_idx:start_idx + period_i]))
            sum_loss = float(np.sum(loss[start_idx:start_idx + period_i]))

            avg_gain = np.full(n, np.nan, dtype=np.float64)
            avg_loss = np.full(n, np.nan, dtype=np.float64)

            avg_gain[init_idx] = sum_gain / period_i
            avg_loss[init_idx] = sum_loss / period_i

            alpha = 1.0 / float(period_i)
            one_minus_alpha = 1.0 - alpha

            for i in range(init_idx + 1, n):
                if np.isfinite(delta[i]):
                    avg_gain[i] = alpha * gain[i] + one_minus_alpha * avg_gain[i - 1]
                    avg_loss[i] = alpha * loss[i] + one_minus_alpha * avg_loss[i - 1]
                else:
                    avg_gain[i] = avg_gain[i - 1]
                    avg_loss[i] = avg_loss[i - 1]

            for i in range(init_idx, n):
                ag = avg_gain[i]
                al = avg_loss[i]

                if not (np.isfinite(ag) and np.isfinite(al)):
                    continue

                if al > 1e-10:
                    rs = ag / al
                    out[i] = 100.0 - (100.0 / (1.0 + rs))
                else:
                    out[i] = 100.0

            return out

        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _rsi_past_only failed: {e}")
            return np.full(len(c), np.nan, dtype=np.float64)
            
    def _atr_past_only(
        self,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        period: int,
    ) -> np.ndarray:
        """
        Average True Range (past-only) با مدیریت کامل NaN.

        ✅ FIX v2.0:
            - TR[1] دیگر NaN نمی‌شود چون c[0]=NaN در محاسبهٔ hc/lc استفاده نمی‌شود.
            - نخستین پنجرهٔ معتبر period-تایی از TR پیدا می‌شود.
            - مقادیر پیش از پنجرهٔ اول NaN می‌مانند تا در imputation پر شوند.
        """
        try:
            n = int(len(c))
            period_i = int(period)
            out = np.full(n, np.nan, dtype=np.float64)

            if period_i < 1 or n < period_i + 2:
                return out

            h_f = np.asarray(h, dtype=np.float64)
            l_f = np.asarray(l, dtype=np.float64)
            c_f = np.asarray(c, dtype=np.float64)

            tr = np.full(n, np.nan, dtype=np.float64)

            for i in range(n):
                hh = h_f[i]
                ll = l_f[i]

                if not (np.isfinite(hh) and np.isfinite(ll)):
                    continue

                hl = hh - ll

                if i == 0:
                    tr[i] = hl
                    continue

                prev_c = c_f[i - 1]

                if np.isfinite(prev_c):
                    hc = abs(hh - prev_c)
                    lc = abs(ll - prev_c)

                    if hl >= hc and hl >= lc:
                        tr[i] = hl
                    elif hc >= hl and hc >= lc:
                        tr[i] = hc
                    else:
                        tr[i] = lc
                else:
                    tr[i] = hl

            start_idx = -1
            for i in range(1, n - period_i + 1):
                if np.all(np.isfinite(tr[i:i + period_i])):
                    start_idx = i
                    break

            if start_idx < 0:
                return out

            init_idx = start_idx + period_i - 1
            alpha = 1.0 / float(period_i)
            one_minus_alpha = 1.0 - alpha

            out[init_idx] = float(
                np.mean(tr[start_idx:start_idx + period_i])
            )

            for i in range(init_idx + 1, n):
                curr_tr = tr[i]

                if np.isfinite(curr_tr):
                    if np.isfinite(out[i - 1]):
                        out[i] = alpha * curr_tr + one_minus_alpha * out[i - 1]
                    else:
                        out[i] = curr_tr
                else:
                    out[i] = out[i - 1]

            return out

        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _atr_past_only failed: {e}")
            return np.full(len(c), np.nan, dtype=np.float64)
            
    def _adx_past_only(
        self,
        h: np.ndarray,
        l: np.ndarray,
        c: np.ndarray,
        period: int,
    ):
        """
        ADX (past-only) با مدیریت کامل NaN.

        ✅ FIX v2.0:
            - تمام محاسبات با آگاهی از NaN انجام می‌شوند.
            - +DM/-DM/TR فقط زمانی محاسبه می‌شوند که هر دو
              کندل جاری و قبلی مقادیر finite داشته باشند.
            - خروجی‌ها به‌صورت tuple (adx, plus_di, minus_di) بازگردانده
              می‌شوند و عناصر نامعتبر NaN می‌مانند.
        """
        try:
            n = int(len(c))
            period_i = int(period)

            plus_di = np.full(n, np.nan, dtype=np.float64)
            minus_di = np.full(n, np.nan, dtype=np.float64)
            adx_out = np.full(n, np.nan, dtype=np.float64)

            if period_i < 1 or n < period_i * 2 + 2:
                return adx_out, plus_di, minus_di

            h_f = np.asarray(h, dtype=np.float64)
            l_f = np.asarray(l, dtype=np.float64)
            c_f = np.asarray(c, dtype=np.float64)

            plus_dm = np.full(n, np.nan, dtype=np.float64)
            minus_dm = np.full(n, np.nan, dtype=np.float64)
            tr = np.full(n, np.nan, dtype=np.float64)

            for i in range(1, n):
                prev_h = h_f[i - 1]
                prev_l = l_f[i - 1]
                prev_c = c_f[i - 1]
                curr_h = h_f[i]
                curr_l = l_f[i]

                if not (
                    np.isfinite(prev_h)
                    and np.isfinite(prev_l)
                    and np.isfinite(prev_c)
                    and np.isfinite(curr_h)
                    and np.isfinite(curr_l)
                ):
                    continue

                up = curr_h - prev_h
                dn = prev_l - curr_l

                pdm = up if (up > dn and up > 0.0) else 0.0
                mdm = dn if (dn > up and dn > 0.0) else 0.0

                plus_dm[i] = pdm
                minus_dm[i] = mdm

                hl = curr_h - curr_l
                hc = abs(curr_h - prev_c)
                lc = abs(curr_l - prev_c)

                if hl >= hc and hl >= lc:
                    tr[i] = hl
                elif hc >= hl and hc >= lc:
                    tr[i] = hc
                else:
                    tr[i] = lc

            alpha = 1.0 / float(period_i)
            one_minus_alpha = 1.0 - alpha

            def _wilder_smooth(series: np.ndarray) -> np.ndarray:
                smoothed = np.full(len(series), np.nan, dtype=np.float64)
                start = -1
                for k in range(1, len(series) - period_i + 1):
                    if np.all(np.isfinite(series[k:k + period_i])):
                        start = k
                        break

                if start < 0:
                    return smoothed

                init = start + period_i - 1
                smoothed[init] = float(
                    np.sum(series[start:start + period_i])
                )

                for k in range(init + 1, len(series)):
                    curr_v = series[k]
                    if np.isfinite(curr_v):
                        if np.isfinite(smoothed[k - 1]):
                            smoothed[k] = (
                                smoothed[k - 1] * (1.0 - alpha) + curr_v
                            )
                        else:
                            smoothed[k] = curr_v
                    else:
                        smoothed[k] = smoothed[k - 1]

                return smoothed

            smooth_tr = _wilder_smooth(tr)
            smooth_plus = _wilder_smooth(plus_dm)
            smooth_minus = _wilder_smooth(minus_dm)

            for i in range(n):
                a = smooth_tr[i]
                if not np.isfinite(a) or a <= 1e-10:
                    continue

                p = smooth_plus[i]
                m = smooth_minus[i]

                if np.isfinite(p):
                    plus_di[i] = p / a * 100.0
                if np.isfinite(m):
                    minus_di[i] = m / a * 100.0

            dx = np.full(n, np.nan, dtype=np.float64)
            for i in range(n):
                p = plus_di[i]
                m = minus_di[i]
                if not (np.isfinite(p) and np.isfinite(m)):
                    continue
                s = p + m
                if s > 1e-10:
                    dx[i] = 100.0 * abs(p - m) / s

            adx_out = _wilder_smooth(dx) / float(period_i)

            return adx_out, plus_di, minus_di

        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _adx_past_only failed: {e}")
            n_fallback = len(c)
            return (
                np.full(n_fallback, np.nan, dtype=np.float64),
                np.full(n_fallback, np.nan, dtype=np.float64),
                np.full(n_fallback, np.nan, dtype=np.float64),
            )
            
    def _zscore_past_only(
        self,
        arr: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        Rolling Z-score (past-only) با مدیریت کامل NaN.

        ✅ FIX v2.0:
            - count دیگر به‌اشتباه برابر window نیست؛ تعداد
              مقادیر finite شمرده می‌شود.
            - اگر تعداد finite کافی نباشد، مقدار NaN بازگردانده می‌شود.
        """
        try:
            n = int(len(arr))
            w = int(window)
            out = np.full(n, np.nan, dtype=np.float64)

            if n == 0 or w < 2:
                return out

            arr_f = np.asarray(arr, dtype=np.float64)

            for i in range(w, n):
                window_vals = arr_f[i - w:i]
                finite_vals = window_vals[np.isfinite(window_vals)]

                if finite_vals.size < 2:
                    continue

                mean = float(np.mean(finite_vals))
                var = float(np.var(finite_vals))
                std = float(np.sqrt(max(var, 1e-10)))

                curr = arr_f[i]
                if np.isfinite(curr):
                    out[i] = (curr - mean) / std

            return out

        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _zscore_past_only failed: {e}")
            return np.full(len(arr), np.nan, dtype=np.float64)
            
    def _linear_slope_past_only(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Linear slope (past-only) - پیاده‌سازی مستقیم"""
        try:
            n = len(arr)
            out = np.full(n, np.nan, dtype=np.float64)
            if n < window:
                return out
            
            for i in range(window, n):
                sum_x = 0.0
                sum_y = 0.0
                for j in range(window):
                    sum_x += j
                    sum_y += arr[i - window + j]
                mx = sum_x / window
                my = sum_y / window
                sxy = 0.0
                sxx = 0.0
                for j in range(window):
                    idx = i - window + j
                    sxy += (j - mx) * (arr[idx] - my)
                    sxx += (j - mx) * (j - mx)
                if sxx > 1e-10:
                    out[i] = sxy / sxx
                else:
                    out[i] = 0.0
            
            return out
        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _linear_slope_past_only failed: {e}")
            return np.full(len(arr), np.nan)

    def _rolling_rank_past_only(
        self,
        arr: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """
        Percentile rank مقدار جاری نسبت به window مقدار قبلی.

        خروجی در بازه [0, 1] است. Tieها با mid-rank مدیریت می‌شوند.
        Warm-up و پنجره‌های ناقص NaN باقی می‌مانند.
        """
        values = np.asarray(arr, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"[{self.symbol}] Rolling rank input must be one-dimensional, "
                f"got shape={values.shape}"
            )

        n = values.shape[0]
        window_size = int(window)
        output = np.full(n, np.nan, dtype=np.float64)

        if window_size < 1 or n <= window_size:
            return output

        for i in range(window_size, n):
            current = values[i]

            if not np.isfinite(current):
                continue

            history = values[i - window_size:i]
            finite_history = history[np.isfinite(history)]

            if finite_history.size != window_size:
                continue

            less_count = int(np.count_nonzero(finite_history < current))
            equal_count = int(np.count_nonzero(finite_history == current))

            output[i] = (
                less_count + 0.5 * equal_count
            ) / float(window_size)

        return output
        
    def _rolling_autocorr_past_only(
        self,
        arr: np.ndarray,
        lag: int,
        window: int,
    ) -> np.ndarray:
        """
        Rolling Pearson autocorrelation روی داده‌های گذشته.

        خروجی index=i فقط از آرایه [i-window, i) استفاده می‌کند.
        """
        values = np.asarray(arr, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"[{self.symbol}] Autocorrelation input must be one-dimensional, "
                f"got shape={values.shape}"
            )

        lag_size = int(lag)
        window_size = int(window)
        n = values.shape[0]
        output = np.full(n, np.nan, dtype=np.float64)

        if lag_size < 1:
            raise ValueError(
                f"[{self.symbol}] lag must be positive, got {lag_size}"
            )

        if window_size <= lag_size or n <= window_size:
            return output

        for i in range(window_size, n):
            sample = values[i - window_size:i]
            left = sample[:-lag_size]
            right = sample[lag_size:]

            valid = np.isfinite(left) & np.isfinite(right)

            if int(valid.sum()) < max(3, window_size - lag_size):
                continue

            x = left[valid]
            y = right[valid]

            x_centered = x - np.mean(x)
            y_centered = y - np.mean(y)

            denominator = float(
                np.sqrt(
                    np.dot(x_centered, x_centered)
                    * np.dot(y_centered, y_centered)
                )
            )

            if denominator <= 1e-15:
                output[i] = 0.0
                continue

            correlation = float(
                np.dot(x_centered, y_centered) / denominator
            )

            output[i] = np.clip(correlation, -1.0, 1.0)

        return output
        
    def _rolling_hurst_past_only(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling Hurst exponent (past-only) - پیاده‌سازی مستقیم"""
        try:
            n = len(arr)
            out = np.full(n, np.nan, dtype=np.float64)
            if n < window:
                return out
            
            max_lags = min(15, window // 2)
            if max_lags < 3:
                return out
            
            for i in range(window, n):
                window_data = arr[i - window:i]
                lags = np.zeros(max_lags - 1, dtype=np.float64)
                tau = np.zeros(max_lags - 1, dtype=np.float64)
                
                for li in range(1, max_lags):
                    lag = li + 1
                    count = window - lag
                    if count <= 0:
                        continue
                    
                    diff_sum = 0.0
                    diff_sq_sum = 0.0
                    for j in range(count):
                        diff = window_data[j + lag] - window_data[j]
                        diff_sum += diff
                        diff_sq_sum += diff * diff
                    
                    mean_diff = diff_sum / count
                    variance = (diff_sq_sum / count) - (mean_diff * mean_diff)
                    tau[li - 1] = np.sqrt(max(variance, 1e-10))
                    lags[li - 1] = float(lag)
                
                sum_x = 0.0
                sum_y = 0.0
                sum_xy = 0.0
                sum_x2 = 0.0
                valid_count = 0
                
                for j in range(max_lags - 1):
                    if tau[j] > 1e-10 and lags[j] > 0:
                        lx = np.log(lags[j])
                        ly = np.log(tau[j])
                        sum_x += lx
                        sum_y += ly
                        sum_xy += lx * ly
                        sum_x2 += lx * lx
                        valid_count += 1
                
                if valid_count > 2:
                    denom = valid_count * sum_x2 - sum_x * sum_x
                    if abs(denom) > 1e-10:
                        H = (valid_count * sum_xy - sum_x * sum_y) / denom
                        out[i] = np.clip(H, 0.0, 1.0)
                    else:
                        out[i] = 0.5
                else:
                    out[i] = 0.5
            
            return out
        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _rolling_hurst_past_only failed: {e}")
            return np.full(len(arr), np.nan)

    def _rolling_entropy_past_only(self, arr: np.ndarray, window: int, n_bins: int = 10) -> np.ndarray:
        """Rolling entropy (past-only) - پیاده‌سازی مستقیم"""
        try:
            n = len(arr)
            out = np.full(n, np.nan, dtype=np.float64)
            if n < window:
                return out
            
            for i in range(window, n):
                window_data = arr[i - window:i]
                clean_vals = window_data[~np.isnan(window_data) & ~np.isinf(window_data)]
                
                if len(clean_vals) < 5:
                    out[i] = 0.0
                    continue
                
                min_val = np.min(clean_vals)
                max_val = np.max(clean_vals)
                range_val = max_val - min_val
                
                if range_val < 1e-10:
                    out[i] = 0.0
                    continue
                
                bin_width = range_val / n_bins
                bins = np.zeros(n_bins, dtype=np.float64)
                
                for val in clean_vals:
                    bin_idx = int((val - min_val) / bin_width)
                    if bin_idx >= n_bins:
                        bin_idx = n_bins - 1
                    if bin_idx < 0:
                        bin_idx = 0
                    bins[bin_idx] += 1
                
                total = float(len(clean_vals))
                entropy_val = 0.0
                for b in bins:
                    if b > 0:
                        p = b / total
                        entropy_val -= p * np.log(p + 1e-10)
                
                out[i] = entropy_val
            
            return out
        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _rolling_entropy_past_only failed: {e}")
            return np.full(len(arr), np.nan)

    def _pct_change_past_only(
        self,
        arr: np.ndarray,
        lag: int,
    ) -> np.ndarray:
        """
        Percent change با حفظ NaN در warm-up و داده نامعتبر.

        مقدار index=i از arr[i] و arr[i-lag] استفاده می‌کند. در Feature
        Engine هر دو آرایه قبلاً یک کندل shift شده‌اند و بنابراین آینده
        در محاسبه حضور ندارد.
        """
        values = np.asarray(arr, dtype=np.float64)

        if values.ndim != 1:
            raise ValueError(
                f"[{self.symbol}] Percent-change input must be one-dimensional, "
                f"got shape={values.shape}"
            )

        lag_size = int(lag)

        if lag_size < 1:
            raise ValueError(
                f"[{self.symbol}] lag must be positive, got {lag_size}"
            )

        n = values.shape[0]
        output = np.full(n, np.nan, dtype=np.float64)

        if n <= lag_size:
            return output

        current = values[lag_size:]
        previous = values[:-lag_size]

        valid = (
            np.isfinite(current)
            & np.isfinite(previous)
            & (np.abs(previous) > 1e-12)
        )

        destination = output[lag_size:]
        destination[valid] = (
            current[valid] / previous[valid]
        ) - 1.0

        return output
        
    # ───────────────────────────────────────────────────────────────────────
    # SMART DEFAULT VALUE
    # ───────────────────────────────────────────────────────────────────────

    def _get_smart_default_for_column(self, column_name: str) -> float:
        """مقدار پیش‌فرض هوشمند بر اساس نام ستون"""
        try:
            col_lower = column_name.lower()
            
            if 'rsi' in col_lower:
                return 50.0
            if 'zscore' in col_lower or 'z_score' in col_lower:
                return 0.0
            if any(x in col_lower for x in ['atr', 'hv', 'volatility', 'vol_of_vol', 'garch_var', 'garch_vol', 'rv_', 'bv_', 'jump_']):
                return 0.0001
            if col_lower in ['ret', 'log_ret']:
                return 0.0
            if 'volume' in col_lower:
                return 1.0
            if 'macd' in col_lower:
                return 0.0
            if 'corr' in col_lower or 'autocorr' in col_lower:
                return 0.0
            if 'bb_' in col_lower:
                return 0.0
            if any(x in col_lower for x in ['adx', 'plus_di', 'minus_di', 'di_spread']):
                return 25.0
            if 'entropy' in col_lower:
                return 0.5
            if 'hurst' in col_lower:
                return 0.5
            if any(x in col_lower for x in ['rank', 'percentile']):
                return 0.5
            if 'slope' in col_lower or 'trend_' in col_lower:
                return 0.0
            if 'hmm_' in col_lower:
                if 'prob' in col_lower:
                    return 0.5
                elif 'regime' in col_lower:
                    return 0.0
            
            return 0.0
        except Exception:
            return 0.0

    def _handle_missing_values_smart(
        self,
        df: pd.DataFrame,
        column_stats: Optional[Dict[str, Any]] = None,
        is_train: bool = False,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        مدیریت قطعی مقادیر گمشده بدون استفاده از اطلاعات آینده.

        قوانین:

            TRAIN:
                آمار هر ستون فقط از همان دادهٔ TRAIN استخراج می‌شود.
                مقدارهای گمشده با median همان TRAIN پر می‌شوند.

            TRANSFORM:
                هیچ آماری از دادهٔ فعلی محاسبه نمی‌شود.
                فقط median یا mean ذخیره‌شده از TRAIN استفاده می‌شود.

            ممنوع:
                - bfill
                - استفاده از median مربوط به VAL/TEST/LIVE
                - fillna(0) بدون قرارداد مشخص
                - پنهان‌کردن Inf
                - ادامه‌دادن در صورت نبود آمار Train

        Args:
            df: DataFrame ویژگی‌ها
            column_stats: آمار frozen شده از TRAIN
            is_train: آیا داده مربوط به TRAIN است؟
            strict: حالت سخت‌گیرانه

        Returns:
            DataFrame عددی و بدون NaN/Inf

        Raises:
            TypeError: اگر df DataFrame نباشد.
            ValueError: برای ستون تکراری، دادهٔ غیرعددی، نبود آمار،
                        یا باقی‌ماندن NaN/Inf.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"features must be pandas.DataFrame, got {type(df).__name__}"
            )

        if df.empty:
            raise ValueError("Cannot process an empty feature DataFrame")

        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].tolist()
            raise ValueError(
                f"Duplicate feature columns are not allowed: {duplicated}"
            )

        result = df.copy(deep=True)

        non_numeric = [
            column
            for column in result.columns
            if not pd.api.types.is_numeric_dtype(result[column])
        ]

        if non_numeric:
            raise TypeError(
                f"All feature columns must be numeric. "
                f"Non-numeric columns: {non_numeric}"
            )

        if not is_train and column_stats is None:
            raise RuntimeError(
                "Transform mode requires frozen TRAIN statistics. "
                "No column_stats were provided."
            )

        statistics = column_stats or {}

        means = statistics.get("means", {})
        medians = statistics.get("medians", {})
        stds = statistics.get("stds", {})

        for column in result.columns:
            values = pd.to_numeric(
                result[column],
                errors="coerce",
            ).to_numpy(dtype=np.float64, copy=False)

            values[~np.isfinite(values)] = np.nan

            missing_mask = np.isnan(values)

            if not missing_mask.any():
                result[column] = values
                continue

            if is_train:
                finite_values = values[np.isfinite(values)]

                if finite_values.size == 0:
                    raise ValueError(
                        f"Feature {column!r} contains no finite TRAIN values"
                    )

                replacement = float(np.median(finite_values))

                if not np.isfinite(replacement):
                    raise ValueError(
                        f"Could not calculate finite TRAIN median for {column!r}"
                    )

                means[column] = float(np.mean(finite_values))
                medians[column] = replacement

                std_value = float(np.std(finite_values))
                stds[column] = std_value if std_value > 1e-12 else 1.0

                values[missing_mask] = replacement

            else:
                replacement = medians.get(column)

                if replacement is None:
                    replacement = means.get(column)

                if replacement is None:
                    if strict:
                        raise RuntimeError(
                            f"Missing TRAIN imputation statistic for feature "
                            f"{column!r}"
                        )
                    replacement = 0.0

                replacement = float(replacement)

                if not np.isfinite(replacement):
                    raise ValueError(
                        f"Frozen TRAIN replacement for feature {column!r} "
                        "is not finite"
                    )

                values[missing_mask] = replacement

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Feature {column!r} still contains NaN or Inf after "
                    "strict missing-value handling"
                )

            result[column] = values

        if is_train:
            statistics["means"] = means
            statistics["medians"] = medians
            statistics["stds"] = stds

        final_values = result.to_numpy(dtype=np.float64, copy=False)

        if not np.isfinite(final_values).all():
            raise ValueError(
                "Feature matrix contains NaN or Inf after missing-value handling"
            )

        return result
        
    def _align_columns_smart(
        self,
        features_df: pd.DataFrame,
        fitted_feature_names: List[str],
        column_stats: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> pd.DataFrame:
        """
        هم‌ترازسازی دقیق ستون‌ها با Featureهای TRAIN.

        قوانین:

            - ترتیب ستون‌ها دقیقاً همان ترتیب TRAIN است.
            - Feature جدید در Transform حذف و نادیده گرفته نمی‌شود؛
              در حالت strict باعث خطا می‌شود.
            - Feature گمشده فقط با median/mean ذخیره‌شده از TRAIN پر می‌شود.
            - هیچ Feature جدیدی به خروجی اضافه نمی‌شود.
            - تعداد ستون‌های خروجی دقیقاً برابر fitted_feature_names است.

        ⚠️ نکتهٔ مهم:
            این متد مقادیر NaN/Inf موجود در Featureهای محاسبه‌شده را
            پاک نمی‌کند. این کار به عهده `_handle_missing_values_smart`
            است که بلافاصله پس از این تابع در `extract_all_features`
            فراخوانی می‌شود.

            علت: در حالت TRANSFORM، Featureهای مبتنی بر پنجره (rolling)
            در ابتدای سری به‌طور طبیعی NaN تولید می‌کنند. اگر اینجا
            بررسی سخت‌گیرانه انجام شود، فرصت پرکردن آن‌ها با آمار TRAIN
            از دست می‌رود.

        Args:
            features_df: ویژگی‌های محاسبه‌شده
            fitted_feature_names: ترتیب Featureهای TRAIN
            column_stats: آمار frozen شده از TRAIN
            strict: حالت سخت‌گیرانه

        Returns:
            DataFrame با ستون‌ها و ترتیب دقیق TRAIN (ممکن است هنوز NaN/Inf
            داشته باشد؛ پاک‌سازی در مرحلهٔ بعد انجام می‌شود).

        Raises:
            TypeError: برای ورودی نامعتبر
            ValueError: برای ستون تکراری یا Feature جدید در حالت strict
            RuntimeError: برای نبود آمار Train یا عدم امکان بازسازی ستون‌ها
        """
        if not isinstance(features_df, pd.DataFrame):
            raise TypeError(
                "features_df must be pandas.DataFrame, "
                f"got {type(features_df).__name__}"
            )

        if not fitted_feature_names:
            raise ValueError(
                "fitted_feature_names cannot be empty during transform"
            )

        if len(set(fitted_feature_names)) != len(fitted_feature_names):
            raise ValueError(
                "fitted_feature_names contains duplicate names"
            )

        if features_df.columns.duplicated().any():
            duplicated = features_df.columns[
                features_df.columns.duplicated()
            ].tolist()

            raise ValueError(
                f"Input features contain duplicate columns: {duplicated}"
            )

        expected = [str(column) for column in fitted_feature_names]
        current = [str(column) for column in features_df.columns]

        current_set = set(current)
        expected_set = set(expected)

        missing_columns = [
            column
            for column in expected
            if column not in current_set
        ]

        new_columns = [
            column
            for column in current
            if column not in expected_set
        ]

        if new_columns and strict:
            raise ValueError(
                "Transform input contains Feature columns not present in TRAIN: "
                f"{new_columns}"
            )

        statistics = column_stats or {}
        medians = statistics.get("medians", {})
        means = statistics.get("means", {})

        result = features_df.copy(deep=True)

        for column in missing_columns:
            replacement = medians.get(column)

            if replacement is None:
                replacement = means.get(column)

            if replacement is None:
                if strict:
                    raise RuntimeError(
                        f"Feature {column!r} is missing from Transform input "
                        "and has no frozen TRAIN replacement statistic"
                    )
                replacement = 0.0

            replacement = float(replacement)

            if not np.isfinite(replacement):
                raise ValueError(
                    f"Frozen replacement for missing feature {column!r} "
                    "is not finite"
                )

            result[column] = replacement

        missing_after_repair = [
            column
            for column in expected
            if column not in result.columns
        ]

        if missing_after_repair:
            raise RuntimeError(
                "Could not reconstruct all TRAIN feature columns: "
                f"{missing_after_repair}"
            )

        result = result.loc[:, expected].copy()

        if list(result.columns) != expected:
            raise RuntimeError(
                "Feature column order does not match TRAIN feature order"
            )

        if result.shape[1] != len(expected):
            raise RuntimeError(
                f"Feature dimension mismatch after alignment: "
                f"expected={len(expected)}, actual={result.shape[1]}"
            )

        return result
        
    def _resize_array(
        self,
        arr: np.ndarray,
        target_len: int,
        name: str,
    ) -> np.ndarray:
        """
        اعتبارسنجی طول آرایه بدون truncate یا padding مخفی.

        Featureهای زمانی نباید silently resize شوند؛ زیرا این کار می‌تواند
        alignment زمانی را از بین ببرد.
        """
        if arr is None:
            raise ValueError(
                f"[{self.symbol}] Feature array {name!r} is None"
            )

        values = np.asarray(arr)

        if values.ndim != 1:
            raise ValueError(
                f"[{self.symbol}] Feature {name!r} must be one-dimensional, "
                f"got shape={values.shape}"
            )

        expected = int(target_len)

        if expected < 0:
            raise ValueError(
                f"[{self.symbol}] target_len cannot be negative: {expected}"
            )

        actual = values.shape[0]

        if actual != expected:
            raise ValueError(
                f"[{self.symbol}] Feature {name!r} length mismatch: "
                f"expected={expected}, actual={actual}. "
                "Silent truncation and padding are disabled."
            )

        return values.astype(np.float64, copy=False)
            
    def _compute_all_150_features_past_only(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        open_: np.ndarray,
        volume: np.ndarray,
    ) -> pd.DataFrame:
        """
        محاسبهٔ ۱۶۰ فیچر past-only با طراحی کاملاً ضد-نشتی و ضد-همبستگی.

        قرارداد:
            - ورودی‌ها آرایه‌های خام OHLCV هستند.
            - خروجی DataFrame با ستون‌های numeric.
            - ردیف 0 همیشه NaN (warm-up ساختاری).
            - هیچ فیچر سطح (level) خام؛ همه rank/ratio/z-score/contrast/bounded.
            - هیچ فیچری دو نسخه‌ی همبسته ندارد.

        ✅ FIX v11.1:
            - تابع کمکی `_to_n` برای تبدیل ایمن اسکالر → آرایه (n,)
              اضافه شد. علت: `_sd(rsi14, 50.0)` و مشابه‌های آن باعث
              خطای "too many indices for array: array is 0-dimensional"
              می‌شدند، چون bb[valid] روی آرایهٔ 0-d اجرا می‌شد.
            - `_th` چندریختی شد: ورودی اسکالر → خروجی اسکالر،
              ورودی آرایه → خروجی آرایه. علت: `streak_up[i] = _th(cu, 3.0)`
              باعث خطای "setting an array element with a sequence" می‌شد.
            - `_sr` و `_log_ratio` نیز برای امنیت آینده از `_to_n` استفاده می‌کنند.

        Raises:
            TypeError/ValueError در ورودی نامعتبر.
        """
        try:
            logger.info(f"[{self.symbol}] 🚀 _compute_all_150_features_past_only v11.1")

            # ══════════════════════════════════════════════════════════════
            # PHASE 0: اعتبارسنجی + shift
            # ══════════════════════════════════════════════════════════════
            n = int(len(close))
            if n == 0:
                raise ValueError("close array is empty")

            for arr_name, arr in (
                ("close", close), ("high", high), ("low", low),
                ("open_", open_), ("volume", volume),
            ):
                if not isinstance(arr, np.ndarray):
                    raise TypeError(f"{arr_name} must be np.ndarray")
                if arr.shape[0] != n:
                    raise ValueError(
                        f"{arr_name} length mismatch: {arr.shape[0]} vs close {n}"
                    )

            result = pd.DataFrame(index=range(n))
            self._total_features_generated = 0
            self._feature_errors: List[str] = []

            # shift دقیقاً یک کندل → past-only تضمینی
            c_sh = np.full(n, np.nan, dtype=np.float64)
            h_sh = np.full(n, np.nan, dtype=np.float64)
            l_sh = np.full(n, np.nan, dtype=np.float64)
            o_sh = np.full(n, np.nan, dtype=np.float64)
            v_sh = np.full(n, np.nan, dtype=np.float64)
            if n >= 2:
                c_sh[1:] = close[:-1]
                h_sh[1:] = high[:-1]
                l_sh[1:] = low[:-1]
                o_sh[1:] = open_[:-1]
                v_sh[1:] = volume[:-1]

            # حجم degeneracy
            v_fin = v_sh[np.isfinite(v_sh)]
            volume_degenerate = bool(
                v_fin.size > 0
                and (np.all(v_fin == 0.0) or float(np.std(v_fin)) < 1e-12)
            )

            # ══════════════════════════════════════════════════════════════
            # helpers (همه vectorized / bounded)
            # ══════════════════════════════════════════════════════════════
            def _to_n(x) -> np.ndarray:
                """
                تبدیل ایمن ورودی به آرایهٔ یک‌بعدی با طول n.

                - اگر x اسکالر باشد → np.full(n, float(x))
                - اگر x آرایهٔ یک‌بعدی باشد → مستقیماً برگردانده می‌شود
                - اگر x آرایهٔ چندبعدی باشد → flatten و سپس طول آن بررسی می‌شود

                Raises:
                    ValueError: اگر طول آرایه با n مطابقت نداشته باشد.
                """
                arr = np.asarray(x, dtype=np.float64)
                if arr.ndim == 0:
                    return np.full(n, float(arr), dtype=np.float64)
                if arr.ndim > 1:
                    arr = arr.reshape(-1)
                if arr.shape[0] != n:
                    raise ValueError(
                        f"_to_n: array length {arr.shape[0]} != n {n}"
                    )
                return arr

            def _sr(num, den, eps: float = 1e-8) -> np.ndarray:
                """safe ratio — NaN اگر den~0 یا non-finite"""
                out = np.full(n, np.nan, dtype=np.float64)
                a = _to_n(num)
                b = _to_n(den)
                valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > eps)
                out[valid] = a[valid] / b[valid]
                return out

            def _sd(a, b) -> np.ndarray:
                """safe difference — پشتیبانی از اسکالر و آرایه"""
                out = np.full(n, np.nan, dtype=np.float64)
                aa = _to_n(a)
                bb = _to_n(b)
                valid = np.isfinite(aa) & np.isfinite(bb)
                out[valid] = aa[valid] - bb[valid]
                return out

            def _th(x, scale: float = 1.0):
                """
                tanh-bounded.

                - اگر x اسکالر باشد → یک float برمی‌گرداند (NaN اگر غیر-finite).
                - اگر x آرایه باشد → آرایهٔ (n,) برمی‌گرداند؛ مقادیر غیر-finite = NaN.

                این چندریختی برای پشتیبانی از فراخوانی‌هایی مثل
                `streak_up[i] = _th(cu, scale=3.0)` که cu اسکالر است لازم است.
                """
                arr = np.asarray(x, dtype=np.float64)
                denom = max(float(scale), 1e-12)

                if arr.ndim == 0:
                    val = float(arr) / denom
                    return float(np.tanh(val)) if np.isfinite(val) else np.nan

                if arr.ndim > 1:
                    arr = arr.reshape(-1)

                if arr.shape[0] != n:
                    raise ValueError(
                        f"_th: array length {arr.shape[0]} != n {n}"
                    )

                out = np.full(n, np.nan, dtype=np.float64)
                xx = arr / denom
                valid = np.isfinite(xx)
                out[valid] = np.tanh(xx[valid])
                return out

            def _log_ratio(a, b, eps: float = 1e-10) -> np.ndarray:
                """log(a/b) — symmetric, decorrelated. پشتیبانی از اسکالر."""
                out = np.full(n, np.nan, dtype=np.float64)
                aa = _to_n(a)
                bb = _to_n(b)
                valid = (np.isfinite(aa) & np.isfinite(bb)
                         & (aa > eps) & (bb > eps))
                out[valid] = np.log(aa[valid] / bb[valid])
                return out

            def _add(name: str, arr) -> None:
                """پذیرش فیچر + شمارنده"""
                result[name] = np.asarray(arr, dtype=np.float64)
                self._total_features_generated += 1

            # ══════════════════════════════════════════════════════════════
            # PHASE 1: Returns & Momentum (12)
            # ══════════════════════════════════════════════════════════════
            try:
                ret_raw = np.full(n, np.nan, dtype=np.float64)
                for i in range(1, n):
                    a, b = c_sh[i - 1], c_sh[i]
                    if (np.isfinite(a) and np.isfinite(b)
                            and abs(a) > 1e-8 and abs(b) > 1e-8):
                        ret_raw[i] = (b - a) / a

                _add("ret_rank_100", self._rolling_rank_past_only(ret_raw, 100))
                _add("ret_z20", self._zscore_past_only(ret_raw, 20))

                ret_ema10 = self._ema_past_only(ret_raw, 2.0 / 11.0)
                _add("ret_ema10_z", self._zscore_past_only(ret_ema10, 50))

                log_ret = np.full(n, np.nan, dtype=np.float64)
                for i in range(1, n):
                    a, b = c_sh[i - 1], c_sh[i]
                    if (np.isfinite(a) and np.isfinite(b)
                            and a > 1e-8 and b > 1e-8):
                        log_ret[i] = float(np.log(b / a))
                _add("log_ret_z20", self._zscore_past_only(log_ret, 20))

                def _cumret(window: int) -> np.ndarray:
                    out = np.full(n, np.nan, dtype=np.float64)
                    for i in range(window, n):
                        a, b = c_sh[i - window], c_sh[i]
                        if (np.isfinite(a) and np.isfinite(b)
                                and abs(a) > 1e-8):
                            out[i] = (b - a) / a
                    return out

                r5 = _cumret(5)
                r20 = _cumret(20)
                r60 = _cumret(60)
                r120 = _cumret(120)

                _add("ret_5_20_contrast", _th(_sd(r5, r20), scale=0.05))
                _add("ret_20_60_contrast", _th(_sd(r20, r60), scale=0.10))
                _add("ret_60_120_contrast", _th(_sd(r60, r120), scale=0.15))

                sign = np.where(np.isfinite(ret_raw), np.sign(ret_raw), np.nan)
                streak = np.full(n, np.nan, dtype=np.float64)
                last_sign = 0.0
                count = 0
                for i in range(n):
                    s = sign[i]
                    if not np.isfinite(s) or s == 0.0:
                        streak[i] = 0.0
                        last_sign = 0.0
                        count = 0
                        continue
                    if s == last_sign:
                        count += 1
                    else:
                        count = 1
                        last_sign = s
                    streak[i] = float(count) * s
                _add("ret_sign_streak", _th(streak, scale=5.0))

                pos = np.where(np.isfinite(ret_raw), (ret_raw > 0).astype(np.float64), np.nan)
                _add("ret_up_ratio_20", self._rolling_mean_past_only(pos, 20))

                abs_ret = np.abs(ret_raw)
                _add("ret_abs_z20", self._zscore_past_only(abs_ret, 20))

                ret_diff = pd.Series(ret_raw).diff(3).to_numpy(dtype=np.float64)
                _add("ret_accel", _th(ret_diff, scale=0.01))

                std5 = self._rolling_std_past_only(ret_raw, 5)
                std20 = self._rolling_std_past_only(ret_raw, 20)
                _add("ret_vol_ratio_5_20", _log_ratio(std5, std20))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 1: {e}")
                self._feature_errors.append(f"Returns: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 2: RSI & Oscillators (10)
            # ══════════════════════════════════════════════════════════════
            try:
                rsi7 = self._rsi_past_only(c_sh, 7)
                rsi14 = self._rsi_past_only(c_sh, 14)
                rsi28 = self._rsi_past_only(c_sh, 28)

                _add("rsi_14_dev", _th(_sd(rsi14, 50.0), scale=50.0))
                _add("rsi_7_rank", self._rolling_rank_past_only(rsi7, 100))
                _add("rsi_28_rank", self._rolling_rank_past_only(rsi28, 100))
                _add("rsi_14_minus_28", _th(_sd(rsi14, rsi28), scale=30.0))

                ll_14 = np.full(n, np.nan, dtype=np.float64)
                hh_14 = np.full(n, np.nan, dtype=np.float64)
                for i in range(14, n):
                    w_l = l_sh[i - 14:i]
                    w_h = h_sh[i - 14:i]
                    if np.isfinite(w_l).all() and np.isfinite(w_h).all():
                        ll_14[i] = float(np.min(w_l))
                        hh_14[i] = float(np.max(w_h))
                stoch_k = _sr(_sd(c_sh, ll_14), _sd(hh_14, ll_14))
                _add("stoch_k_dev", _th(_sd(stoch_k, 0.5), scale=0.5))

                stoch_d = self._rolling_mean_past_only(stoch_k, 3)
                _add("stoch_k_minus_d", _th(_sd(stoch_k, stoch_d), scale=0.20))

                willr = -100.0 * _sr(_sd(hh_14, c_sh), _sd(hh_14, ll_14))
                _add("williams_r_dev", _th(_sd(willr, -50.0), scale=50.0))

                tp = (h_sh + l_sh + c_sh) / 3.0
                tp_ma = self._rolling_mean_past_only(tp, 20)
                tp_mad = np.full(n, np.nan, dtype=np.float64)
                for i in range(20, n):
                    w = tp[i - 20:i]
                    if np.isfinite(w).all():
                        m = np.mean(w)
                        tp_mad[i] = float(np.mean(np.abs(w - m)))
                cci20 = _sr(_sd(tp, tp_ma), 0.015 * tp_mad)
                _add("cci_20_rank", self._rolling_rank_past_only(cci20, 100))

                bp = _sd(c_sh, np.minimum(l_sh, o_sh))
                tr_uo = _sd(np.maximum(h_sh, o_sh), np.minimum(l_sh, o_sh))
                bp7 = self._rolling_mean_past_only(bp, 7)
                bp14 = self._rolling_mean_past_only(bp, 14)
                bp28 = self._rolling_mean_past_only(bp, 28)
                tr7 = self._rolling_mean_past_only(tr_uo, 7)
                tr14 = self._rolling_mean_past_only(tr_uo, 14)
                tr28 = self._rolling_mean_past_only(tr_uo, 28)
                uo = 100.0 * (4.0 * _sr(bp7, tr7)
                              + 2.0 * _sr(bp14, tr14)
                              + _sr(bp28, tr28)) / 7.0
                _add("ultimate_osc_dev", _th(_sd(uo, 50.0), scale=50.0))

                price_chg_5 = _sd(c_sh, pd.Series(c_sh).shift(5).to_numpy(dtype=np.float64))
                rsi_chg_5 = _sd(rsi14, pd.Series(rsi14).shift(5).to_numpy(dtype=np.float64))
                div = np.sign(price_chg_5) * np.sign(rsi_chg_5)
                _add("rsi_divergence_20", div)

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 2: {e}")
                self._feature_errors.append(f"Oscillators: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 3: MACD (4)
            # ══════════════════════════════════════════════════════════════
            try:
                ema12 = self._ema_past_only(c_sh, 2.0 / 13.0)
                ema26 = self._ema_past_only(c_sh, 2.0 / 27.0)
                macd_line = _sd(ema12, ema26)
                macd_signal = self._ema_past_only(macd_line, 2.0 / 10.0)
                macd_hist = _sd(macd_line, macd_signal)

                _add("macd_line_z", self._zscore_past_only(macd_line, 50))
                _add("macd_hist_z", self._zscore_past_only(macd_hist, 50))
                _add("macd_hist_rank", self._rolling_rank_past_only(macd_hist, 100))

                macd_slope = pd.Series(macd_line).diff(3).to_numpy(dtype=np.float64)
                _add("macd_slope_norm", _th(macd_slope, scale=1e-4))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 3: {e}")
                self._feature_errors.append(f"MACD: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 4: ATR & Term Structure (8)
            # ══════════════════════════════════════════════════════════════
            try:
                atr14 = self._atr_past_only(h_sh, l_sh, c_sh, 14)
                atr28 = self._atr_past_only(h_sh, l_sh, c_sh, 28)
                atr50 = self._atr_past_only(h_sh, l_sh, c_sh, 50)

                _add("atr_14_over_50", _sr(atr14, atr50, eps=1e-10))
                _add("atr_28_over_50", _sr(atr28, atr50, eps=1e-10))
                _add("atr_14_over_28", _sr(atr14, atr28, eps=1e-10))
                _add("atr_50_rank", self._rolling_rank_past_only(atr50, 100))

                atr_pct = _sr(atr14, np.abs(c_sh))
                _add("atr_pct_rank", self._rolling_rank_past_only(atr_pct, 100))

                atr_roc = self._pct_change_past_only(atr14, 5)
                _add("atr_roc_5_z", self._zscore_past_only(atr_roc, 50))

                atr_accel = pd.Series(atr14).diff(3).to_numpy(dtype=np.float64)
                _add("atr_accel_rank", self._rolling_rank_past_only(atr_accel, 100))

                _add("atr_short_long_spread", _log_ratio(atr14, atr50))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 4: {e}")
                self._feature_errors.append(f"ATR: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 5: Bollinger (8)
            # ══════════════════════════════════════════════════════════════
            try:
                bb_mid = self._rolling_mean_past_only(c_sh, 20)
                bb_std = self._rolling_std_past_only(c_sh, 20)

                bb_middle_z = _sr(_sd(c_sh, bb_mid), bb_std)
                _add("bb_middle_z", self._zscore_past_only(bb_middle_z, 50))

                bb_width = _sr(2.0 * bb_std, np.abs(bb_mid))
                _add("bb_width_rank", self._rolling_rank_past_only(bb_width, 100))

                bb_position = _sr(_sd(c_sh, bb_mid), 2.0 * bb_std)
                bb_position = np.clip(bb_position, -1.0, 1.0)
                _add("bb_position", bb_position)
                _add("bb_position_rank", self._rolling_rank_past_only(bb_position, 100))
                _add("bb_squeeze_rank", self._rolling_rank_past_only(bb_std, 50))
                _add("bb_zscore_rank",
                     self._rolling_rank_past_only(_sr(_sd(c_sh, bb_mid), bb_std), 50))
                _add("bb_width_over_atr",
                     _sr(2.0 * bb_std, atr14, eps=1e-10))

                bb_width_diff = pd.Series(bb_width).diff(5).to_numpy(dtype=np.float64)
                _add("bb_expansion_rate", _th(bb_width_diff, scale=0.05))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 5: {e}")
                self._feature_errors.append(f"Bollinger: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 6: ADX & DI (8)
            # ══════════════════════════════════════════════════════════════
            try:
                adx_v, plus_di, minus_di = self._adx_past_only(h_sh, l_sh, c_sh, 14)

                _add("adx_rank", self._rolling_rank_past_only(adx_v, 100))
                _add("adx_z50", self._zscore_past_only(adx_v, 50))

                di_sum = np.where(np.isfinite(plus_di + minus_di) & (plus_di + minus_di > 1e-6),
                                  plus_di + minus_di, np.nan)
                _add("plus_di_minus_di_norm",
                     _sr(_sd(plus_di, minus_di), di_sum))
                _add("plus_over_minus_log_ratio", _log_ratio(plus_di, minus_di))
                _add("di_spread_rank",
                     self._rolling_rank_past_only(_sd(plus_di, minus_di), 100))

                aroon_up = np.full(n, np.nan, dtype=np.float64)
                aroon_down = np.full(n, np.nan, dtype=np.float64)
                for i in range(25, n):
                    w_h = h_sh[i - 25:i]
                    w_l = l_sh[i - 25:i]
                    if np.isfinite(w_h).all() and np.isfinite(w_l).all():
                        aroon_up[i] = 100.0 * float(np.argmax(w_h)) / 25.0
                        aroon_down[i] = 100.0 * float(np.argmin(w_l)) / 25.0

                _add("aroon_up_rank", self._rolling_rank_past_only(aroon_up, 100))
                _add("aroon_down_rank", self._rolling_rank_past_only(aroon_down, 100))
                _add("aroon_osc", _th(_sd(aroon_up, aroon_down), scale=100.0))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 6: {e}")
                self._feature_errors.append(f"ADX: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 7: Volatility Term Structure (12)
            # ══════════════════════════════════════════════════════════════
            try:
                hv10 = self._rolling_std_past_only(ret_raw, 10)
                hv30 = self._rolling_std_past_only(ret_raw, 30)
                hv60 = self._rolling_std_past_only(ret_raw, 60)

                _add("hv_10_over_60", _log_ratio(hv10, hv60))
                _add("hv_30_over_60", _log_ratio(hv30, hv60))
                _add("hv_10_over_30", _log_ratio(hv10, hv30))
                _add("hv_60_rank", self._rolling_rank_past_only(hv60, 100))

                vov_raw = self._rolling_std_past_only(hv30, 20)
                _add("vol_of_vol_norm", _sr(vov_raw, hv30, eps=1e-10))
                _add("vol_of_vol_rank", self._rolling_rank_past_only(vov_raw, 100))

                hl_ratio = _sr(h_sh, l_sh, eps=1e-10)
                co_ratio = _sr(c_sh, o_sh, eps=1e-10)
                gk = np.full(n, np.nan, dtype=np.float64)
                valid_gk = (np.isfinite(hl_ratio) & (hl_ratio > 0)
                            & np.isfinite(co_ratio) & (co_ratio > 0))
                gk[valid_gk] = (
                    0.5 * np.log(hl_ratio[valid_gk]) ** 2
                    - (2.0 * np.log(2.0) - 1.0) * np.log(co_ratio[valid_gk]) ** 2
                )
                gk_vol = np.sqrt(np.maximum(gk, 0.0))
                _add("gk_vol_norm", _sr(gk_vol, hv30, eps=1e-10))

                pk = (np.log(np.maximum(hl_ratio, 1.0)) ** 2) / (4.0 * np.log(2.0))
                pk_vol = np.sqrt(np.maximum(pk, 0.0))
                _add("parkinson_vol_norm", _sr(pk_vol, hv30, eps=1e-10))

                rs = (np.log(np.maximum(co_ratio, 1e-10))
                      * np.log(np.maximum(hl_ratio, 1e-10) / np.maximum(co_ratio, 1e-10)))
                rs_vol = np.sqrt(np.maximum(np.abs(rs), 0.0))
                _add("rs_vol_norm", _sr(rs_vol, hv30, eps=1e-10))

                yz = np.sqrt(np.maximum(hv30 ** 2 + gk_vol ** 2, 0.0))
                _add("yz_vol_norm", _sr(yz, hv30, eps=1e-10))

                _add("vol_ratio_10_30", _sr(hv10, hv30, eps=1e-10))

                hv60_rank = self._rolling_rank_past_only(hv60, 100)
                regime_flag = np.where(
                    ~np.isfinite(hv60_rank), np.nan,
                    np.where(hv60_rank > 0.7, 1.0, np.where(hv60_rank < 0.3, -1.0, 0.0))
                )
                _add("vol_regime_flag", regime_flag)

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 7: {e}")
                self._feature_errors.append(f"Volatility: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 8: Volume (10)
            # ══════════════════════════════════════════════════════════════
            try:
                vol_ma20 = self._rolling_mean_past_only(v_sh, 20)
                vol_ratio = _sr(v_sh, vol_ma20, eps=1e-10)

                _add("volume_ratio_z", self._zscore_past_only(vol_ratio, 50))
                _add("volume_rank_100", self._rolling_rank_past_only(v_sh, 100))

                v_roc = self._pct_change_past_only(v_sh, 5)
                _add("volume_roc_z", self._zscore_past_only(v_roc, 50))

                v_accel = pd.Series(vol_ratio).diff(3).to_numpy(dtype=np.float64)
                _add("volume_accel", _th(v_accel, scale=1.0))

                v_slope = self._linear_slope_past_only(v_sh, 20)
                _add("volume_trend_strength",
                     _th(v_slope, scale=float(np.nanstd(v_slope)) + 1e-10))

                cv = c_sh * v_sh
                cv_sum = np.full(n, np.nan, dtype=np.float64)
                v_sum = np.full(n, np.nan, dtype=np.float64)
                for i in range(20, n):
                    w_cv = cv[i - 20:i]
                    w_v = v_sh[i - 20:i]
                    if np.isfinite(w_cv).all() and np.isfinite(w_v).all() and np.sum(w_v) > 1e-10:
                        cv_sum[i] = float(np.sum(w_cv))
                        v_sum[i] = float(np.sum(w_v))
                vwap = _sr(cv_sum, v_sum, eps=1e-10)
                vwap_dist = _sr(_sd(c_sh, vwap), c_sh)
                _add("vwap_distance_z", self._zscore_past_only(vwap_dist, 50))

                pv_corr = np.full(n, np.nan, dtype=np.float64)
                for i in range(20, n):
                    wp = c_sh[i - 20:i]
                    wv = v_sh[i - 20:i]
                    if np.isfinite(wp).all() and np.isfinite(wv).all():
                        sp = np.std(wp); sv = np.std(wv)
                        if sp > 1e-12 and sv > 1e-12:
                            pv_corr[i] = float(np.corrcoef(wp, wv)[0, 1])
                _add("volume_price_corr_20", pv_corr)

                obv_dir = np.where(np.isfinite(ret_raw), np.sign(ret_raw), 0.0)
                obv = np.cumsum(obv_dir * np.where(np.isfinite(v_sh), v_sh, 0.0))
                obv_slope = self._linear_slope_past_only(obv, 20)
                _add("obv_slope_norm",
                     _th(obv_slope, scale=float(np.nanstd(obv_slope)) + 1e-10))

                vol_rank = self._rolling_rank_past_only(v_sh, 50)
                breakout = np.where(
                    ~np.isfinite(vol_rank), np.nan,
                    (vol_rank > 0.9).astype(np.float64)
                )
                _add("volume_breakout_flag", breakout)

                smi = _sr(_sd(c_sh, o_sh), _sd(h_sh, l_sh) + 1e-10)
                _add("smart_money_index_z", self._zscore_past_only(smi, 50))

                if volume_degenerate:
                    for name in (
                        "volume_ratio_z", "volume_rank_100", "volume_roc_z",
                        "volume_accel", "volume_trend_strength",
                        "obv_slope_norm", "volume_breakout_flag",
                        "smart_money_index_z",
                    ):
                        result[name] = np.where(
                            np.isfinite(v_sh), 0.0, np.nan
                        )

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 8: {e}")
                self._feature_errors.append(f"Volume: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 9: Trend Slope (6)
            # ══════════════════════════════════════════════════════════════
            try:
                ts10 = self._linear_slope_past_only(c_sh, 10)
                ts30 = self._linear_slope_past_only(c_sh, 30)
                ts50 = self._linear_slope_past_only(c_sh, 50)

                _add("trend_slope_10_z", self._zscore_past_only(ts10, 50))
                _add("trend_slope_30_z", self._zscore_past_only(ts30, 50))
                _add("trend_slope_50_z", self._zscore_past_only(ts50, 50))

                slope_contrast = _sr(ts10, ts50, eps=1e-10)
                _add("slope_contrast", _th(_sd(slope_contrast, 1.0), scale=1.0))

                slope_accel = pd.Series(ts10).diff(5).to_numpy(dtype=np.float64)
                _add("slope_accel_z", self._zscore_past_only(slope_accel, 50))

                trend_r2 = np.full(n, np.nan, dtype=np.float64)
                for i in range(20, n):
                    y_w = c_sh[i - 20:i]
                    if np.isfinite(y_w).all():
                        x_w = np.arange(20, dtype=np.float64)
                        xm = x_w - x_w.mean()
                        ym = y_w - y_w.mean()
                        sxx = float(np.dot(xm, xm))
                        if sxx > 1e-12:
                            slope = float(np.dot(xm, ym) / sxx)
                            pred = slope * xm
                            ss_res = float(np.dot(ym - pred, ym - pred))
                            ss_tot = float(np.dot(ym, ym))
                            if ss_tot > 1e-12:
                                trend_r2[i] = 1.0 - ss_res / ss_tot
                _add("trend_r2", np.clip(trend_r2, 0.0, 1.0))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 9: {e}")
                self._feature_errors.append(f"TrendSlope: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 10: Z-score & Deviation (8)
            # ══════════════════════════════════════════════════════════════
            try:
                _add("price_z20", self._zscore_past_only(c_sh, 20))
                _add("price_z50", self._zscore_past_only(c_sh, 50))
                _add("price_z200", self._zscore_past_only(c_sh, 200))

                z20 = self._zscore_past_only(c_sh, 20)
                z_accel = pd.Series(z20).diff(3).to_numpy(dtype=np.float64)
                _add("zscore_accel", _th(z_accel, scale=1.0))

                z_extreme = np.where(
                    ~np.isfinite(z20), np.nan,
                    (np.abs(z20) > 2.0).astype(np.float64)
                )
                _add("zscore_extreme_flag", z_extreme)

                _add("zscore_abs", np.abs(z20))

                ema20 = self._ema_past_only(c_sh, 2.0 / 21.0)
                ema50 = self._ema_past_only(c_sh, 2.0 / 51.0)
                _add("price_dev_ema20", _sr(_sd(c_sh, ema20), c_sh))
                _add("price_dev_ema50", _sr(_sd(c_sh, ema50), c_sh))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 10: {e}")
                self._feature_errors.append(f"ZScore: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 11: Entropy & Hurst (8)
            # ══════════════════════════════════════════════════════════════
            try:
                ent20 = self._rolling_entropy_past_only(ret_raw, 20)
                ent50 = self._rolling_entropy_past_only(ret_raw, 50)

                _add("entropy_20", ent20)
                _add("entropy_50", ent50)
                _add("entropy_contrast", _sd(ent20, ent50))

                hurst50 = self._rolling_hurst_past_only(c_sh, 50)
                hurst100 = self._rolling_hurst_past_only(c_sh, 100)

                _add("hurst_50", hurst50)
                _add("hurst_100", hurst100)
                _add("hurst_contrast", _sd(hurst50, hurst100))

                _add("sample_entropy", self._rolling_entropy_past_only(ret_raw, 15))

                ts = np.full(n, np.nan, dtype=np.float64)
                for i in range(20, n):
                    w = ret_raw[i - 20:i]
                    if np.isfinite(w).all():
                        hist, _ = np.histogram(w, bins=8)
                        p = hist / max(1, hist.sum())
                        ts[i] = float(1.0 - np.sum(p ** 2))
                _add("tsallis_entropy", ts)

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 11: {e}")
                self._feature_errors.append(f"Entropy: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 12: Autocorrelation (5)
            # ══════════════════════════════════════════════════════════════
            try:
                ac1 = self._rolling_autocorr_past_only(ret_raw, 1, 50)
                ac5 = self._rolling_autocorr_past_only(ret_raw, 5, 50)
                ac10 = self._rolling_autocorr_past_only(ret_raw, 10, 50)

                _add("autocorr_lag1", ac1)
                _add("autocorr_lag5", ac5)
                _add("autocorr_lag10", ac10)
                _add("autocorr_contrast", _sd(ac1, ac10))
                _add("pacf_1", ac1)

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 12: {e}")
                self._feature_errors.append(f"Autocorr: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 13: Rank Features (10)
            # ══════════════════════════════════════════════════════════════
            try:
                pr50 = self._rolling_rank_past_only(c_sh, 50)
                pr100 = self._rolling_rank_past_only(c_sh, 100)
                pr200 = self._rolling_rank_past_only(c_sh, 200)

                _add("price_rank_50", pr50)
                _add("price_rank_100", pr100)
                _add("price_rank_200", pr200)
                _add("price_rank_contrast", _sd(pr50, pr200))

                vr50 = self._rolling_rank_past_only(v_sh, 50)
                vr100 = self._rolling_rank_past_only(v_sh, 100)
                _add("vol_rank_50", vr50)
                _add("vol_rank_100", vr100)
                _add("vol_rank_contrast", _sd(vr50, vr100))

                _add("high_rank_100", self._rolling_rank_past_only(h_sh, 100))
                _add("low_rank_100", self._rolling_rank_past_only(l_sh, 100))

                _add("atr_percentile_100",
                     self._rolling_rank_past_only(atr14, 100) * 100.0)

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 13: {e}")
                self._feature_errors.append(f"Rank: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 14: Candle Structure (10)
            # ══════════════════════════════════════════════════════════════
            try:
                body = np.abs(_sd(c_sh, o_sh))
                rng_bar = _sd(h_sh, l_sh)
                safe_rng = np.where(
                    np.isfinite(rng_bar) & (np.abs(rng_bar) > 1e-10),
                    rng_bar, np.nan
                )

                _add("body_ratio", _sr(body, safe_rng))

                upper = _sd(h_sh, np.maximum(c_sh, o_sh))
                lower = _sd(np.minimum(c_sh, o_sh), l_sh)

                _add("upper_shadow_ratio", _sr(upper, safe_rng))
                _add("lower_shadow_ratio", _sr(lower, safe_rng))

                close_pos = _sr(_sd(c_sh, l_sh), safe_rng)
                _add("close_position_in_bar", np.clip(close_pos, 0.0, 1.0))

                _add("body_to_range_rank", self._rolling_rank_past_only(body, 50))

                candle_dir = np.where(
                    np.isfinite(c_sh) & np.isfinite(o_sh),
                    np.sign(_sd(c_sh, o_sh)), np.nan
                )
                _add("candle_direction", candle_dir)

                up_flag = np.where(np.isfinite(candle_dir), (candle_dir > 0).astype(np.float64), np.nan)
                dn_flag = np.where(np.isfinite(candle_dir), (candle_dir < 0).astype(np.float64), np.nan)

                streak_up = np.full(n, np.nan, dtype=np.float64)
                streak_dn = np.full(n, np.nan, dtype=np.float64)
                cu, cd = 0, 0
                for i in range(n):
                    if not np.isfinite(up_flag[i]):
                        streak_up[i] = 0.0
                        streak_dn[i] = 0.0
                        cu = cd = 0
                        continue
                    cu = cu + 1 if up_flag[i] > 0 else 0
                    cd = cd + 1 if dn_flag[i] > 0 else 0
                    streak_up[i] = _th(cu, scale=3.0)
                    streak_dn[i] = _th(cd, scale=3.0)
                _add("consecutive_up_flag", streak_up)
                _add("consecutive_down_flag", streak_dn)

                doji = np.where(
                    np.isfinite(body) & np.isfinite(safe_rng),
                    (body < 0.1 * safe_rng).astype(np.float64), np.nan
                )
                _add("doji_flag", doji)

                marubozu = np.where(
                    np.isfinite(body) & np.isfinite(safe_rng),
                    (body > 0.9 * safe_rng).astype(np.float64), np.nan
                )
                _add("marubozu_flag", marubozu)

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 14: {e}")
                self._feature_errors.append(f"Candle: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 15: Statistical Moments (8)
            # ══════════════════════════════════════════════════════════════
            try:
                skew20 = np.full(n, np.nan, dtype=np.float64)
                skew50 = np.full(n, np.nan, dtype=np.float64)
                kurt20 = np.full(n, np.nan, dtype=np.float64)
                kurt50 = np.full(n, np.nan, dtype=np.float64)
                for i in range(50, n):
                    w20 = ret_raw[i - 20:i]
                    w50 = ret_raw[i - 50:i]
                    if np.isfinite(w20).all() and np.std(w20) > 1e-12:
                        skew20[i] = float(stats.skew(w20))
                        kurt20[i] = float(stats.kurtosis(w20))
                    if np.isfinite(w50).all() and np.std(w50) > 1e-12:
                        skew50[i] = float(stats.skew(w50))
                        kurt50[i] = float(stats.kurtosis(w50))

                _add("skew_20", _th(skew20, scale=3.0))
                _add("skew_50", _th(skew50, scale=3.0))
                _add("kurt_20", _th(kurt20, scale=10.0))
                _add("kurt_50", _th(kurt50, scale=10.0))
                _add("skew_contrast", _th(_sd(skew20, skew50), scale=2.0))
                _add("kurt_contrast", _th(_sd(kurt20, kurt50), scale=5.0))

                neg_mask = np.where(np.isfinite(ret_raw), (ret_raw < 0).astype(np.float64), np.nan)
                neg_sq = np.where(np.isfinite(ret_raw), np.where(ret_raw < 0, ret_raw ** 2, 0.0), np.nan)
                dvol = np.sqrt(self._rolling_mean_past_only(neg_sq, 20))
                _add("downside_vol_ratio", _sr(dvol, hv30, eps=1e-10))

                _add("sharpe_like_20", _sr(self._rolling_mean_past_only(ret_raw, 20),
                                          hv30, eps=1e-10))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 15: {e}")
                self._feature_errors.append(f"Moments: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 16: Advanced Volatility (10)
            # ══════════════════════════════════════════════════════════════
            try:
                rv5, bv5, jump5 = _safe_compute_rv_bv(ret_raw, window=5)
                _add("rv_5_rank", self._rolling_rank_past_only(rv5, 100))
                _add("bv_5_rank", self._rolling_rank_past_only(bv5, 100))
                _add("jump_5_ratio", _sr(jump5, rv5, eps=1e-10))

                garch_var = _safe_compute_garch_features(ret_raw, window=50)
                _add("garch_var_rank", self._rolling_rank_past_only(garch_var, 100))
                _add("garch_vol_ratio",
                     _sr(np.sqrt(np.maximum(garch_var, 0.0)), hv30, eps=1e-10))

                pos_sq = np.where(np.isfinite(ret_raw), np.where(ret_raw > 0, ret_raw ** 2, 0.0), np.nan)
                uvol = np.sqrt(self._rolling_mean_past_only(pos_sq, 20))
                dvol2 = np.sqrt(self._rolling_mean_past_only(neg_sq, 20))
                _add("semivariance_ratio",
                     _sr(uvol - dvol2, uvol + dvol2 + 1e-10))
                _add("downside_vol_20", _sr(dvol2, hv30, eps=1e-10))
                _add("upside_vol_20", _sr(uvol, hv30, eps=1e-10))

                hv_trend = pd.Series(hv30).diff(10).to_numpy(dtype=np.float64)
                _add("volatility_trend", _th(hv_trend, scale=float(np.nanstd(hv_trend)) + 1e-10))

                _add("vol_breakout_flag", np.where(
                    ~np.isfinite(hv60_rank), np.nan,
                    (hv60_rank > 0.9).astype(np.float64)
                ))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 16: {e}")
                self._feature_errors.append(f"AdvVol: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 17: Volume Flow (8)
            # ══════════════════════════════════════════════════════════════
            try:
                ofi = _sr(_sd(c_sh, o_sh), safe_rng)
                ofi = np.clip(ofi, -1.0, 1.0)
                _add("ofi_normalized", ofi)
                _add("ofi_z", self._zscore_past_only(ofi, 50))

                cvd_dir = np.where(np.isfinite(c_sh) & np.isfinite(o_sh),
                                   np.sign(_sd(c_sh, o_sh)), 0.0)
                cvd = np.cumsum(cvd_dir * np.where(np.isfinite(v_sh), v_sh, 0.0))
                cvd_slope = self._linear_slope_past_only(cvd, 20)
                _add("cvd_slope", _th(cvd_slope, scale=float(np.nanstd(cvd_slope)) + 1e-10))

                _add("delta_ratio", _th(ofi * 2.0, scale=1.0))

                buy_v = np.where(np.isfinite(c_sh) & np.isfinite(o_sh) & (c_sh > o_sh),
                                 v_sh, 0.0)
                sell_v = np.where(np.isfinite(c_sh) & np.isfinite(o_sh) & (c_sh < o_sh),
                                  v_sh, 0.0)

                bp20 = self._rolling_mean_past_only(buy_v, 20)
                sp20 = self._rolling_mean_past_only(sell_v, 20)

                _add("buy_pressure", _sr(bp20, bp20 + sp20 + 1e-10))
                _add("sell_pressure", _sr(sp20, bp20 + sp20 + 1e-10))
                _add("pressure_ratio",
                     _th(_sr(bp20 - sp20, bp20 + sp20 + 1e-10), scale=1.0))

                v_high = np.where(np.isfinite(v_sh),
                                  (self._rolling_rank_past_only(v_sh, 20) > 0.8).astype(np.float64),
                                  np.nan)
                price_low_move = np.where(np.isfinite(safe_rng) & np.isfinite(c_sh),
                                          (safe_rng < 0.005 * np.abs(c_sh)).astype(np.float64),
                                          np.nan)
                _add("absorption_flag", v_high * price_low_move)

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 17: {e}")
                self._feature_errors.append(f"VolumeFlow: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 18: Pattern / Structure (10)
            # ══════════════════════════════════════════════════════════════
            try:
                low50 = np.full(n, np.nan, dtype=np.float64)
                high50 = np.full(n, np.nan, dtype=np.float64)
                for i in range(50, n):
                    w_l = l_sh[i - 50:i]
                    w_h = h_sh[i - 50:i]
                    if np.isfinite(w_l).all() and np.isfinite(w_h).all():
                        low50[i] = float(np.min(w_l))
                        high50[i] = float(np.max(w_h))

                fib_pos = _sr(_sd(c_sh, low50), _sd(high50, low50))
                _add("fibonacci_position", np.clip(fib_pos, 0.0, 1.0))

                _add("support_strength",
                     _sr(_sd(c_sh, low50), np.abs(c_sh)))
                _add("resistance_strength",
                     _sr(_sd(high50, c_sh), np.abs(c_sh)))

                pivot = (high50 + low50 + c_sh) / 3.0
                _add("pivot_distance_z",
                     self._zscore_past_only(_sr(_sd(c_sh, pivot), np.abs(pivot)), 50))

                _add("range_position", np.clip(fib_pos, 0.0, 1.0))

                _add("breakout_up_flag", np.where(
                    np.isfinite(high50) & np.isfinite(c_sh),
                    (c_sh > high50).astype(np.float64), np.nan
                ))
                _add("breakout_down_flag", np.where(
                    np.isfinite(low50) & np.isfinite(c_sh),
                    (c_sh < low50).astype(np.float64), np.nan
                ))

                _add("price_gap_ratio",
                     _sr(_sd(o_sh, pd.Series(c_sh).shift(1).to_numpy(dtype=np.float64)),
                         c_sh))

                rng5 = self._rolling_mean_past_only(safe_rng, 5)
                rng20 = self._rolling_mean_past_only(safe_rng, 20)
                _add("range_expansion", _log_ratio(rng5, rng20))

                ma_fast = self._rolling_mean_past_only(c_sh, 10)
                ma_slow = self._rolling_mean_past_only(c_sh, 50)
                _add("ma_convergence", _sr(_sd(ma_fast, ma_slow), np.abs(ma_slow) + 1e-10))

            except Exception as e:
                logger.error(f"[{self.symbol}]    ❌ PHASE 18: {e}")
                self._feature_errors.append(f"Pattern: {e}")

            # ══════════════════════════════════════════════════════════════
            # PHASE 19: Sanitization + invalidation ردیف 0
            # ══════════════════════════════════════════════════════════════
            result = result.replace([np.inf, -np.inf], np.nan)
            result = result.loc[:, ~result.columns.duplicated()]
            if n >= 1:
                result.iloc[0, :] = np.nan

            if volume_degenerate:
                logger.warning(
                    f"[{self.symbol}] ⚠️ volume degenerate — volume features set to 0"
                )

            if self._feature_errors:
                logger.warning(
                    f"[{self.symbol}] ⚠️ errors in {len(self._feature_errors)} phases: "
                    f"{self._feature_errors}"
                )

            logger.info(
                f"[{self.symbol}] ✅ features generated: "
                f"{self._total_features_generated} | shape={result.shape}"
            )
            return result

        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ FATAL: {e}")
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
    def fit(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> 'AntiCorrelationFeatureEngine':
        self.symbol = symbol
        self.extract_all_features(df, mode='fit')
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.extract_all_features(df, mode='transform')

    def fit_transform(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        self.symbol = symbol
        return self.extract_all_features(df, mode='fit')

    def get_feature_names(self) -> List[str]:
        return self._fitted_feature_names if hasattr(self, '_fitted_feature_names') else []

    def get_feature_categories(self) -> Dict[str, str]:
        return {}
        
class WallStreetQuantFeatureEngine:
    """
    Feature Engine سطح بالا با قواعد ZERO-LEAKAGE.

    قواعد:
        - fit() فقط روی TRAIN فراخوانی می‌شود.
        - transform() برای val/test/live_data فقط از state فریز استفاده می‌کند.
        - در صورت ارائهٔ history_df، ابتدای split با context واقعی warm-up می‌شود.
    """

    PROTECTED_COLUMNS = list(PROJECT_TARGETS) + list(AUXILIARY_COLUMNS) + [
        "open", "high", "low", "close", "volume", "ret", "log_ret"
    ]

    DEFAULT_SOURCE_WEIGHTS = {
        "classification": 0.30,
        "adaptive_ta": 0.45,
        "traditional": 0.25,
    }

    SOURCE_CONFIDENCE_THRESHOLDS = {
        "classification": 0.65,
        "adaptive_ta": 0.60,
        "traditional": 0.55,
    }

    HISTORY_MAX_LOOKBACK: int = 200
    from main import DEFAULT_SYMBOL
    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        target_n_features: Optional[int] = None,
        history_max_lookback: int = 200,
        **kwargs,
    ) -> None:
        self.symbol = str(symbol)
        self.target_n_features = target_n_features

        try:
            hml = int(history_max_lookback)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"history_max_lookback must be int, got {history_max_lookback!r}"
            ) from exc

        if hml < 5:
            raise ValueError(f"history_max_lookback must be >=5, got {hml}")
        self.HISTORY_MAX_LOOKBACK = hml

        self._is_fitted: bool = False
        self._feature_columns: Optional[List[str]] = None
        self._target_columns: List[str] = list(PROJECT_TARGETS)
        self._fitted_stats: Dict[str, Any] = {}

        self._source_weights = dict(self.DEFAULT_SOURCE_WEIGHTS)
        self._source_performance = {
            "classification": {"correct": 0, "total": 0, "accuracy": 0.5},
            "adaptive_ta": {"correct": 0, "total": 0, "accuracy": 0.5},
            "traditional": {"correct": 0, "total": 0, "accuracy": 0.5},
        }
        self._last_predictions: Dict[str, Any] = {}

        # ✅ state اضافه‌شده — مورد نیاز پایپ‌لاین پایین‌دستی
        self._last_fit_mask: Optional[np.ndarray] = None
        self._last_train_df: Optional[pd.DataFrame] = None

        self.reducer = AntiCorrelationFeatureEngine(
            self.symbol, target_n_features=self.target_n_features
        )

        logger.info("=" * 80)
        logger.info(f"🔧 WallStreetQuantFeatureEngine v6.0 (strict-history)")
        logger.info(f"   • symbol: {self.symbol}")
        logger.info(f"   • history_max_lookback: {self.HISTORY_MAX_LOOKBACK}")
        logger.info(f"   • target_n_features: {self.target_n_features}")
        logger.info("=" * 80)

    @staticmethod
    def _detect_fundamental_columns(df: pd.DataFrame) -> List[str]:
        """
        تشخیص ستون‌های بنیادی در DataFrame.

        ✅ v2.0 EXPANDED:
            لیست پیشوندها گسترش یافت تا تمام ستون‌هایی که
            FundamentalFeatureBuilder تولید می‌کند را پوشش دهد:
              - family derivatives: funding_, open_interest, basis
              - sentiment: fear_greed
              - interactions: funding_x_, oi_x_, oi_change_x_, vol_ratio_x_
              - snapshot aggregates: price_median, spread_median, n_markets
              - on-chain / microstructure: long_short, taker_, bid_ask, ...

        Args:
            df: DataFrame شامل ستون‌های OHLCV و شاید بنیادی.

        Returns:
            لیست نام ستون‌های بنیادی موجود در df (بدون تکرار، به ترتیب ظهور).
        """
        prefixes = (
            # ── Derivatives / Funding / Open Interest ──
            "funding_",
            "open_interest",
            "long_short",
            "long_account",
            "short_account",
            "top_trader",
            "taker_",
            "basis",
            "liquidations_",
            "spot_volume",
            "net_taker",
            # ── Sentiment ──
            "fear_greed",
            "google_trends",
            "twitter_sentiment",
            "reddit_sentiment",
            # ── On-chain ──
            "whale_trades",
            "exchange_inflow",
            "exchange_outflow",
            "active_addresses",
            "miner_reserve",
            "sopr",
            "nupl",
            "mvrv",
            "stablecoin",
            "puell",
            # ── Microstructure ──
            "bid_ask",
            "order_book",
            "book_depth",
            "microprice",
            # ── Interactions (FundamentalFeatureBuilder._add_interaction_features) ──
            "funding_x_",
            "oi_x_",
            "oi_change_x_",
            "vol_ratio_x_",
            "vol_ratio_",
            # ── Snapshot aggregates (FundamentalFeatureBuilder._aggregate_market_snapshot) ──
            "price_median",
            "spread_median",
            "n_markets",
        )

        exact_names = {
            "funding_rate_median",
            "funding_rate_mean",
            "funding_rate_std",
            "open_interest_usd_total",
            "volume_24h_usd_total",
            "basis_median",
        }

        result: List[str] = []
        seen: set = set()

        for column in df.columns:
            col_str = str(column)
            if col_str in seen:
                continue

            if col_str in exact_names:
                result.append(col_str)
                seen.add(col_str)
                continue

            for prefix in prefixes:
                if col_str.startswith(prefix):
                    result.append(col_str)
                    seen.add(col_str)
                    break

        return result
        
    def _create_empty_targets_df(self, n: int) -> pd.DataFrame:
        """
        فریم تارگت تمام-INVALID برای حالت‌های دفاعی.
        price_return=NaN و کلاس‌ها=-1 (INVALID_LABEL)؛ یعنی «بدون لیبل»،
        نه لیبل جعلی. هیچ مسیر اصلی نباید عمداً به این وضعیت برسد.
        """
        try:
            nn = int(n)
        except Exception:
            nn = 0
        if nn < 0:
            nn = 0
        return pd.DataFrame({
            "price_return": np.full(nn, np.nan, dtype=np.float64),
            "price_direction": np.full(nn, -1, dtype=np.int32),
            "trend_strength": np.full(nn, -1, dtype=np.int32),
            "volatility_regime": np.full(nn, -1, dtype=np.int32),
            "market_regime": np.full(nn, -1, dtype=np.int32),
        })
            
    def _generate_targets(
        self,
        close: np.ndarray,
        n: int,
        split_name: str = "unknown",
        sample_weight_mode: str = "direction_aware",
    ) -> pd.DataFrame:
        """
        تولید 8 ستون تارگت + sample_weight (v14.2 — Audit-Compliant).

        ─────────────────────────────────────────────────────────────────
        تغییرات v14.2 (نسبت به v14.0):
        ─────────────────────────────────────────────────────────────────

        ✅ اضافه شدن `magnitude_bucket` به `class_specs` برای defensive
           validation (پیش از این، حتی اگر label generation باگ داشت،
           validation این باگ را نمی‌گرفت).

        ✅ حفظ کلیه fixes v14.0:
           - horizon=3
           - sign_with_epsilon + epsilon=0.0015
           - direction-aware sample_weight
           - frozen magnitude thresholds
        """
        split_name = str(split_name)

        close_arr = np.asarray(close, dtype=np.float64).flatten()
        n = int(close_arr.shape[0])

        # ✅ v14.0: پارامترهای قابل تنظیم ممیزی
        AUDIT_PREDICTION_HORIZON = 3
        AUDIT_LABELING_MODE = "sign_with_epsilon"
        AUDIT_EPSILON = 0.0015

        logger.info(
            f"[{self.symbol}] 🎯 Generating targets for "
            f"{split_name.upper()} (n={n}, mode={sample_weight_mode}, "
            f"horizon={AUDIT_PREDICTION_HORIZON}, "
            f"labeling={AUDIT_LABELING_MODE}, eps={AUDIT_EPSILON})"
        )

        if n == 0:
            raise ValueError(
                f"[{self.symbol}] close array is empty for split={split_name}"
            )
        if n < AUDIT_PREDICTION_HORIZON + 10:
            raise ValueError(
                f"[{self.symbol}] need >={AUDIT_PREDICTION_HORIZON+10} rows, "
                f"got {n}"
            )

        # ──────────────────────────────────────────────────────────
        # PHASE 1: LabelGenerator lifecycle (با horizon=3 و epsilon)
        # ──────────────────────────────────────────────────────────
        if not hasattr(self, "_label_generator") or self._label_generator is None:
            self._label_generator = LabelGenerator(
                symbol=self.symbol,
                prediction_horizon=AUDIT_PREDICTION_HORIZON,
                balance_classes=True,
                dynamic_threshold=False,
                regime_aware=True,
                strong_min_samples_pct=0.03,
                labeling_mode=AUDIT_LABELING_MODE,
                epsilon=AUDIT_EPSILON,
            )
            logger.info(
                f"[{self.symbol}]    ✅ LabelGenerator created | "
                f"horizon={AUDIT_PREDICTION_HORIZON} | "
                f"mode={AUDIT_LABELING_MODE} | eps={AUDIT_EPSILON}"
            )
        else:
            existing_h = int(self._label_generator.prediction_horizon)
            if existing_h != AUDIT_PREDICTION_HORIZON:
                logger.warning(
                    f"[{self.symbol}] ⚠️ Existing LabelGenerator horizon "
                    f"({existing_h}) != requested ({AUDIT_PREDICTION_HORIZON}). "
                    f"Using existing. Delete cache to force rebuild."
                )

        generator = self._label_generator
        generator.set_caller(split_name)

        h = int(generator.prediction_horizon)

        # ──────────────────────────────────────────────────────────
        # PHASE 2: رژیم نوسان (past-only)
        # ──────────────────────────────────────────────────────────
        volatility_regime_for_fit = generator.generate_volatility_regime(close_arr)
        if volatility_regime_for_fit is None or len(volatility_regime_for_fit) != n:
            raise RuntimeError(
                f"[{self.symbol}] volatility_regime generation invalid"
            )

        valid_vol = volatility_regime_for_fit[
            volatility_regime_for_fit != generator.INVALID
        ]
        regime_str_map = {0: "LOW_VOL", 1: "NORMAL_VOL", 2: "HIGH_VOL"}
        if valid_vol.size == 0:
            dominant_regime = "NORMAL_VOL"
        else:
            u_vol, c_vol = np.unique(valid_vol, return_counts=True)
            dominant_regime = regime_str_map.get(
                int(u_vol[np.argmax(c_vol)]), "NORMAL_VOL"
            )

        # ──────────────────────────────────────────────────────────
        # PHASE 3: fit یا transform
        # ──────────────────────────────────────────────────────────
        if split_name == "train":
            if not generator._is_fitted:
                logger.info(
                    f"[{self.symbol}]    📊 FITTING LabelGenerator on TRAIN "
                    f"(regime={dominant_regime}, h={h})..."
                )
                generator.fit(close_arr, dominant_regime)
            else:
                logger.info(
                    f"[{self.symbol}]    ℹ️ LabelGenerator already fitted"
                )
        elif split_name in ("val", "test"):
            if not generator._is_fitted:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"🔴 CRITICAL: LabelGenerator NOT FITTED on TRAIN before "
                    f"processing {split_name.upper()}.\n"
                    f"🔴 Pipeline order is wrong.\n"
                    f"{'='*80}"
                )
            logger.info(
                f"[{self.symbol}]    📊 TRANSFORM with frozen TRAIN thresholds"
            )
        elif split_name == "live_data":
            raise RuntimeError(
                f"[{self.symbol}] _generate_targets() refuses to run for "
                f"live_data (inference only produces features)"
            )
        else:
            raise ValueError(
                f"[{self.symbol}] Unknown split_name: {split_name}"
            )

        if not generator._is_fitted:
            raise RuntimeError(
                f"[{self.symbol}] LabelGenerator must be fitted before "
                f"generating labels"
            )

        if (
            generator._mag_q33_frozen is None
            or generator._mag_q66_frozen is None
        ):
            raise RuntimeError(
                f"[{self.symbol}] magnitude_bucket thresholds missing — "
                f"LabelGenerator.fit() must set them"
            )

        transform_regime = str(
            getattr(generator, "_fitted_volatility_regime", "NORMAL_VOL")
        )

        # ──────────────────────────────────────────────────────────
        # PHASE 4: تولید 7 تارگت
        # ──────────────────────────────────────────────────────────
        price_return = generator.generate_price_return(close_arr)
        price_direction = generator.generate_direction_2class(
            close_arr, transform_regime
        )

        trend_state = generator.generate_trend_strength(close_arr)
        vol_state = generator.generate_volatility_regime(close_arr)
        market_state = generator.generate_market_regime(close_arr, vol_state)

        trend_target = _shift_future_state(trend_state, h, generator.INVALID)
        vol_target = _shift_future_state(vol_state, h, generator.INVALID)
        market_target = _shift_future_state(market_state, h, generator.INVALID)

        mom_state = generator.generate_momentum_direction(close_arr)
        mom_target = _shift_future_state(mom_state, h, generator.INVALID)

        q33 = float(generator._mag_q33_frozen)
        q66 = float(generator._mag_q66_frozen)

        # ──────────────────────────────────────────────────────────
        # PHASE 5: magnitude_bucket (FROZEN)
        # ──────────────────────────────────────────────────────────
        abs_return = np.abs(price_return)
        finite_abs = np.isfinite(abs_return)
        magnitude_bucket = np.full(n, -1, dtype=np.int32)
        magnitude_bucket[finite_abs & (abs_return <= q33)] = 0
        magnitude_bucket[
            finite_abs & (abs_return > q33) & (abs_return <= q66)
        ] = 1
        magnitude_bucket[finite_abs & (abs_return > q66)] = 2

        # ──────────────────────────────────────────────────────────
        # PHASE 5.5: اعتبارسنجی سخت (شامل magnitude_bucket — v14.2 FIX)
        # ──────────────────────────────────────────────────────────
        class_specs = {
            "price_direction":    (price_direction, {-1, 0, 1}),
            "trend_strength":     (trend_target,    {-1, 0, 1, 2}),
            "volatility_regime":  (vol_target,      {-1, 0, 1, 2}),
            "market_regime":      (market_target,   {-1, 0, 1, 2, 3}),
            "magnitude_bucket":   (magnitude_bucket, {-1, 0, 1, 2}),
            "momentum_direction": (mom_target,      {-1, 0, 1}),
        }
        for name, (arr, allowed) in class_specs.items():
            if arr is None or len(arr) == 0:
                raise RuntimeError(f"[{self.symbol}] {name} is empty")
            if len(arr) != n:
                raise RuntimeError(f"[{self.symbol}] {name} length mismatch")
            a = np.asarray(arr).flatten()
            finite_mask = np.isfinite(a.astype(np.float64, copy=False))
            a_int = np.where(finite_mask, a, 0).astype(np.int64)
            allowed_arr = np.asarray(sorted(allowed), dtype=np.int64)
            bad = ~np.isin(a_int, allowed_arr) & finite_mask
            if bool(np.any(bad)):
                raise RuntimeError(
                    f"[{self.symbol}] {name} invalid classes: "
                    f"{np.unique(a[bad])[:5].tolist()}"
                )

        if price_return is None or len(price_return) != n:
            raise RuntimeError(f"[{self.symbol}] price_return length mismatch")
        pr_arr = np.asarray(price_return, dtype=np.float64).flatten()
        if bool(np.any(np.isinf(pr_arr))):
            raise RuntimeError(f"[{self.symbol}] price_return contains Inf")

        # ──────────────────────────────────────────────────────────
        # PHASE 6: sample_weight جهت‌محور
        # ──────────────────────────────────────────────────────────
        if sample_weight_mode == "direction_aware":
            sample_weight = generator._compute_direction_aware_weight(
                close_arr, pr_arr
            )
        elif sample_weight_mode == "magnitude":
            logger.warning(
                f"[{self.symbol}] ⚠️ sample_weight_mode='magnitude' is "
                f"DEPRECATED — amplify direction noise. Use 'direction_aware'."
            )
            finite_abs_clean = abs_return[finite_abs & (abs_return > 1e-12)]
            if finite_abs_clean.size >= 10:
                q_mag33 = float(np.nanpercentile(finite_abs_clean, 33.0))
                magnitude_scale = np.where(
                    finite_abs,
                    np.log1p(abs_return / (q_mag33 + 1e-10)),
                    0.0,
                )
                magnitude_scale = np.clip(magnitude_scale, 0.0, 5.0)
                weight_raw = 1.0 + magnitude_scale
                mean_weight = float(np.mean(weight_raw[finite_abs]))
                if not np.isfinite(mean_weight) or mean_weight < 1e-8:
                    mean_weight = 1.0
                sample_weight = np.where(
                    finite_abs, weight_raw / mean_weight, 0.0,
                ).astype(np.float32)
            else:
                sample_weight = np.where(
                    finite_abs, 1.0, 0.0,
                ).astype(np.float32)
        else:
            raise ValueError(
                f"[{self.symbol}] sample_weight_mode must be "
                f"'direction_aware' or 'magnitude', got {sample_weight_mode!r}"
            )

        # ──────────────────────────────────────────────────────────
        # PHASE 7: DataFrame
        # ──────────────────────────────────────────────────────────
        targets_df = pd.DataFrame(index=range(n))
        targets_df["price_return"] = pr_arr[:n]
        targets_df["price_direction"] = np.asarray(
            price_direction, dtype=np.int32
        ).flatten()[:n]
        targets_df["trend_strength"] = np.asarray(
            trend_target, dtype=np.int32
        ).flatten()[:n]
        targets_df["volatility_regime"] = np.asarray(
            vol_target, dtype=np.int32
        ).flatten()[:n]
        targets_df["market_regime"] = np.asarray(
            market_target, dtype=np.int32
        ).flatten()[:n]
        targets_df["magnitude_bucket"] = magnitude_bucket[:n]
        targets_df["momentum_direction"] = np.asarray(
            mom_target, dtype=np.int32
        ).flatten()[:n]
        targets_df["sample_weight"] = sample_weight[:n]

        # ──────────────────────────────────────────────────────────
        # PHASE 8: لاگ
        # ──────────────────────────────────────────────────────────
        logger.info(
            f"[{self.symbol}] 📊 TARGET DISTRIBUTIONS FOR "
            f"{split_name.upper()}:"
        )
        for target_name in PROJECT_TARGETS:
            if target_name in targets_df.columns:
                log_target_distribution(
                    target_name=target_name,
                    target_data=targets_df[target_name],
                    symbol=f"[{self.symbol}]",
                    prefix="       ",
                    is_regression=(target_name == "price_return"),
                    show_class_names=True,
                )

        valid_mom = mom_target[mom_target != generator.INVALID]
        if valid_mom.size > 0:
            u, c = np.unique(valid_mom, return_counts=True)
            tot = float(c.sum())
            parts = [
                f"  {int(k)}:{int(v)} ({100.0 * v / tot:.1f}%)"
                for k, v in zip(u, c)
            ]
            logger.info(
                f"[{self.symbol}]    📊 momentum_direction (DOWN/UP): "
                + ", ".join(parts)
            )

        valid_w = sample_weight[sample_weight > 0]
        if valid_w.size > 0:
            logger.info(
                f"[{self.symbol}]    ⚖️ sample_weight: "
                f"mean={float(valid_w.mean()):.4f}, "
                f"std={float(valid_w.std()):.4f}, "
                f"min={float(valid_w.min()):.4f}, "
                f"max={float(valid_w.max()):.4f}"
            )

        logger.info(
            f"[{self.symbol}] ✅ Targets generated and validated for "
            f"{split_name.upper()} | horizon={h}"
        )
        return targets_df
        
    def get_source_weights(self, market_context: Optional[Dict] = None) -> Dict[str, float]:
        """
        🎯 دریافت وزن‌های پویا برای سه منبع تشخیص رژیم
        
        وزن هر منبع بر اساس:
        1. دقت تاریخی آن منبع (60%)
        2. عملکرد اخیر (20%)
        3. تناسب با رژیم بازار فعلی (20%)
        
        Args:
            market_context: دیکشنری شامل {'current_regime': str, 'volatility': float}
        
        Returns:
            Dictionary با کلیدهای 'classification', 'adaptive_ta', 'traditional'
        """
        try:
            weights = {}
            
            for source_name, perf in self._source_performance.items():
                # معیار 1: دقت تاریخی (60%)
                historical_accuracy = perf.get('accuracy', 0.5)
                historical_score = historical_accuracy * 0.60
                
                # معیار 2: عملکرد اخیر - آخرین 10 پیش‌بینی (20%)
                recent_correct = perf.get('recent_correct', 0)
                recent_total = perf.get('recent_total', 0)
                recent_accuracy = recent_correct / recent_total if recent_total > 0 else 0.5
                recent_score = recent_accuracy * 0.20
                
                # معیار 3: تناسب با رژیم بازار (20%)
                regime_fit_score = 0.70  # مقدار پیش‌فرض
                if market_context:
                    current_regime = market_context.get('current_regime', 'unknown')
                    source_strength = self._get_source_regime_strength(source_name, current_regime)
                    regime_fit_score = source_strength
                
                regime_score = regime_fit_score * 0.20
                
                # وزن نهایی
                total_weight = historical_score + recent_score + regime_score
                weights[source_name] = max(0.15, min(0.85, total_weight))
            
            # نرمالیزه کردن
            total = sum(weights.values())
            if total > 0:
                for name in weights:
                    weights[name] /= total
            else:
                weights = self.DEFAULT_SOURCE_WEIGHTS.copy()
            
            # ذخیره در کش برای استفاده بعدی
            self._source_weights = weights
            
            logger.debug(f"[{self.symbol}] 📊 Dynamic weights: "
                        f"classification={weights['classification']:.2f}, "
                        f"adaptive_ta={weights['adaptive_ta']:.2f}, "
                        f"traditional={weights['traditional']:.2f}")
            
            return weights
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error calculating source weights: {e}")
            return self.DEFAULT_SOURCE_WEIGHTS.copy()
    
    def _get_source_regime_strength(self, source_name: str, regime: str) -> float:
        """
        محاسبه قدرت هر منبع در رژیم خاص بازار
        
        Args:
            source_name: نام منبع (classification, adaptive_ta, traditional)
            regime: رژیم بازار (bull_trending, bear_trending, ranging, high_volatility)
        
        Returns:
            امتیاز تناسب بین 0.3 تا 0.95
        """
        # قدرت هر منبع در رژیم‌های مختلف
        source_strength_map = {
            'classification': {
                'bull_trending': 0.85,
                'bear_trending': 0.85,
                'ranging': 0.70,
                'high_volatility': 0.75,
                'default': 0.70
            },
            'adaptive_ta': {
                'bull_trending': 0.90,
                'bear_trending': 0.90,
                'ranging': 0.65,
                'high_volatility': 0.80,
                'default': 0.75
            },
            'traditional': {
                'bull_trending': 0.80,
                'bear_trending': 0.80,
                'ranging': 0.75,
                'high_volatility': 0.60,
                'default': 0.65
            }
        }
        
        regime_lower = regime.lower().replace('_', '')
        
        # تطابق با کلیدهای موجود
        for key in source_strength_map[source_name]:
            if key in regime_lower or regime_lower in key:
                return source_strength_map[source_name][key]
        
        return source_strength_map[source_name]['default']
    
    def update_source_performance(self, source_name: str, was_correct: bool, regime: str = None):
        """
        🎯 بروزرسانی عملکرد هر منبع پس از هر پیش‌بینی
        
        Args:
            source_name: نام منبع (classification, adaptive_ta, traditional)
            was_correct: آیا پیش‌بینی منبع صحیح بود؟
            regime: رژیم بازار در زمان پیش‌بینی (برای آمار دقیق‌تر)
        """
        try:
            if source_name not in self._source_performance:
                return
            
            perf = self._source_performance[source_name]
            
            # بروزرسانی آمار کلی
            perf['total'] = perf.get('total', 0) + 1
            if was_correct:
                perf['correct'] = perf.get('correct', 0) + 1
            
            # محاسبه دقت جدید
            if perf['total'] > 0:
                perf['accuracy'] = perf['correct'] / perf['total']
            
            # بروزرسانی آمار اخیر (آخرین 50 پیش‌بینی)
            if 'recent_predictions' not in perf:
                perf['recent_predictions'] = []
                perf['recent_correct'] = 0
                perf['recent_total'] = 0
            
            perf['recent_predictions'].append(1 if was_correct else 0)
            if len(perf['recent_predictions']) > 50:
                perf['recent_predictions'].pop(0)
            
            perf['recent_total'] = len(perf['recent_predictions'])
            perf['recent_correct'] = sum(perf['recent_predictions'])
            
            # لاگ تغییر وزن
            old_weight = self._source_weights.get(source_name, 0.33)
            new_weights = self.get_source_weights({'current_regime': regime or 'unknown'})
            new_weight = new_weights.get(source_name, old_weight)
            
            if abs(new_weight - old_weight) > 0.03:
                logger.info(f"[{self.symbol}] 📊 Source '{source_name}' weight updated: "
                           f"{old_weight:.2f} → {new_weight:.2f} (accuracy={perf['accuracy']:.1%})")
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error updating source performance: {e}")
    
    def record_prediction_outcome(self, source_name: str, predicted_regime: str, 
                                   actual_regime: str, confidence: float = 0.5):
        """
        ثبت نتیجه پیش‌بینی و بروزرسانی وزن‌ها
        
        Args:
            source_name: نام منبع
            predicted_regime: رژیم پیش‌بینی شده
            actual_regime: رژیم واقعی بعد از وقوع
            confidence: اعتماد منبع در زمان پیش‌بینی
        """
        try:
            # تعیین صحت پیش‌بینی
            was_correct = self._is_regime_match(predicted_regime, actual_regime)
            
            # بروزرسانی با وزن اعتماد
            effective_correct = was_correct
            if confidence < 0.4:
                effective_correct = False  # اعتماد پایین = پیش‌بینی نادیده گرفته شود
            
            self.update_source_performance(source_name, effective_correct, actual_regime)
            
            logger.debug(f"[{self.symbol}] 📝 Recorded {source_name}: "
                        f"predicted={predicted_regime}, actual={actual_regime}, "
                        f"correct={was_correct}, confidence={confidence:.1%}")
            
        except Exception as e:
            logger.error(f"[{self.symbol}] Error recording prediction outcome: {e}")
    
    def _is_regime_match(self, regime1: str, regime2: str) -> bool:
        """
        بررسی تطابق دو رژیم با در نظر گرفتن مترادف‌ها
        
        مثلاً 'bull_trending' و 'trending_up' هر دو صعودی محسوب می‌شوند
        """
        regime1_lower = regime1.lower().replace('_', '')
        regime2_lower = regime2.lower().replace('_', '')
        
        # تطابق دقیق
        if regime1_lower == regime2_lower:
            return True
        
        # گروه‌بندی مترادف‌ها
        bullish_group = ['bulltrending', 'trendingup', 'uptrend', 'bullish']
        bearish_group = ['beartrending', 'trendingdown', 'downtrend', 'bearish']
        ranging_group = ['ranging', 'sideways', 'sidewaysranging', 'range']
        volatile_group = ['highvolatility', 'volatile', 'highvol']
        
        def get_group(regime: str) -> str:
            if any(g in regime for g in bullish_group):
                return 'bullish'
            if any(g in regime for g in bearish_group):
                return 'bearish'
            if any(g in regime for g in ranging_group):
                return 'ranging'
            if any(g in regime for g in volatile_group):
                return 'volatile'
            return 'unknown'
        
        return get_group(regime1_lower) == get_group(regime2_lower)
    
    def get_source_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        دریافت خلاصه عملکرد سه منبع
        
        Returns:
            Dictionary با جزئیات عملکرد هر منبع
        """
        summary = {}
        
        for source_name, perf in self._source_performance.items():
            summary[source_name] = {
                'accuracy': perf.get('accuracy', 0.5),
                'total_predictions': perf.get('total', 0),
                'correct_predictions': perf.get('correct', 0),
                'recent_accuracy': perf.get('recent_correct', 0) / max(1, perf.get('recent_total', 0)),
                'current_weight': self._source_weights.get(source_name, 0.33)
            }
        
        return summary
    
    def _enhance_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance raw data with calculated columns"""
        if df is None or df.empty:
            return df
        
        try:
            enhanced_df = df.reset_index(drop=True).copy(deep=True)
            enhanced_df.columns = enhanced_df.columns.str.lower()
            
            required = ['open', 'high', 'low', 'close']
            missing = [c for c in required if c not in enhanced_df.columns]
            if missing:
                logger.error(f"[{self.symbol}] ❌ Missing columns: {missing}")
                return df
            
            o = enhanced_df['open']
            h = enhanced_df['high']
            l = enhanced_df['low']
            c = enhanced_df['close']
            
            base_price = o.abs() + 1e-10
            body_top = pd.concat([c, o], axis=1).max(axis=1)
            body_bot = pd.concat([c, o], axis=1).min(axis=1)
            
            enhanced_df['calc_body_size_pct'] = (c - o).abs() / base_price
            enhanced_df['calc_range_pct'] = (h - l).abs() / base_price
            enhanced_df['calc_upper_shadow_pct'] = (h - body_top).abs() / base_price
            enhanced_df['calc_lower_shadow_pct'] = (body_bot - l).abs() / base_price
            enhanced_df['calc_is_bullish'] = (c > o).astype(float)
            enhanced_df['calc_is_bearish'] = (c < o).astype(float)
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"[{self.symbol}] ❌ _enhance_raw_data failed: {e}")
            return df
    
    def _extract_arrays(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract required arrays"""
        n = len(df)
        result = {
            'close': np.asarray(df["close"], dtype=np.float64).flatten(),
            'high': np.asarray(df["high"], dtype=np.float64).flatten(),
            'low': np.asarray(df["low"], dtype=np.float64).flatten(),
            'open': np.asarray(df["open"], dtype=np.float64).flatten(),
            'volume': np.asarray(df["volume"], dtype=np.float64).flatten() if 'volume' in df.columns else np.ones(n, dtype=np.float64),
        }
        return result

    def fit(self, train_df: pd.DataFrame) -> "WallStreetQuantFeatureEngine":
        """
        fit فقط روی TRAIN. state فریز می‌شود.

        ✅ v2.0: قبل از feature engineering، فیچرهای بنیادی الحاق می‌شوند.
        """
        if train_df is None or train_df.empty:
            raise ValueError(f"[{self.symbol}] train_df is empty")

        logger.info(f"[{self.symbol}] 🔧 FIT | shape={train_df.shape}")

        # ── الحاق فیچرهای بنیادی (در صورت وجود timestamp) ──
        from core.global_method import attach_fundamental_features

        if "timestamp" in train_df.columns:
            train_df = attach_fundamental_features(
                train_df, symbol=self.symbol, timestamp_col="timestamp",
                require_success=False,
            )
        else:
            logger.warning(
                f"[{self.symbol}] ⚠️ no timestamp column — "
                "fundamental features cannot be attached"
            )

        df_enhanced = self._enhance_raw_data(train_df)
        if df_enhanced is None or df_enhanced.empty:
            raise RuntimeError(f"[{self.symbol}] enhance returned empty")

        features_df = self.reducer.extract_all_features(df_enhanced, mode="fit")
        if features_df is None or features_df.empty:
            raise RuntimeError(f"[{self.symbol}] feature extraction returned empty")

        arr = self._extract_arrays(train_df)
        targets_df = self._generate_targets(
            close=arr["close"], n=len(train_df), split_name="train"
        )
        if targets_df is None or targets_df.empty:
            raise RuntimeError(f"[{self.symbol}] target generation returned empty")

        # ✅ حفظ ستون‌های بنیادی که reducer نادیده می‌گیرد
        fundamental_cols = self._detect_fundamental_columns(train_df)
        if fundamental_cols:
            fundamental_block = df_enhanced[fundamental_cols].reset_index(drop=True)
            features_df = pd.concat(
                [features_df.reset_index(drop=True), fundamental_block], axis=1
            )
            logger.info(
                f"[{self.symbol}] ✅ attached {len(fundamental_cols)} "
                f"fundamental columns to feature block"
            )

        full_df = pd.concat([features_df, targets_df], axis=1)

        self._feature_columns = [
            c for c in full_df.columns if c not in PROJECT_TARGETS
        ]
        self._is_fitted = True
        self._last_train_df = full_df.copy()

        logger.info(
            f"[{self.symbol}] ✅ FIT complete | features={len(self._feature_columns)}"
        )
        return self
        

    def transform(
        self,
        df: pd.DataFrame,
        split_name: str = "unknown",
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        transform برای val/test/live_data با history context اختیاری.

        Args:
            df: split جاری.
            split_name: 'val' | 'test' | 'live_data'.
            history_df: tail دادهٔ خام قبلی برای warm-up rolling.

        Returns:
            DataFrame features (+ targets اگر split != live_data).

        Raises:
            RuntimeError: اگر fit نشده باشد.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"[{self.symbol}] must call fit() before transform()"
            )

        # _process_single_split_with_enhancement همگام است اما async تعریف
        # شده است — از SmartSyncBridge.sync استفاده می‌کنیم.
        from core.global_method import SmartSyncBridge

        result = SmartSyncBridge.sync(
            self._process_single_split_with_enhancement,
            df_split=df,
            split_name=split_name,
            symbol=self.symbol,
            timeframe="unknown",
            fold_num=0,
            train_columns=self._feature_columns,
            loaded_scaler_data=None,
            history_df=history_df,
        )
        return result


    def fit_transform(
        self, train_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        fit و transform روی TRAIN در یک مرحله.
        این تابع فقط برای راحتی است — معادل fit(train_df) سپس transform(train_df)
        با split_name='train' (بدون history).
        """
        self.fit(train_df)
        return self.transform(train_df, split_name="train", history_df=None)
            
    async def _process_single_split_with_enhancement(
        self,
        df_split: pd.DataFrame,
        split_name: str,
        symbol: str,
        timeframe: str,
        fold_num: int,
        train_columns: Optional[List[str]] = None,
        loaded_scaler_data: Optional[Dict] = None,
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        پردازش یک split با قرارداد fit/transform + history context.

        ✅ v10.1 FIX (B5, B6):
            - B5: `sample_weight_mode="direction_aware"` (نه magnitude)
              در فراخوانی _generate_targets.
            - B6: `auxiliary_cols = list(AUXILIARY_COLUMNS)` به‌جای
              لیست هاردکد `["sample_weight", "magnitude_bucket"]`
              که momentum_direction را از دست می‌داد.
        """
        from datetime import datetime as _dt

        _DATETIME_COL_NAMES = (
            "timestamp", "datetime", "date", "time",
            "open_time", "close_time",
        )
        _FUNDAMENTAL_PATTERNS = (
            "fear_greed", "funding_rate", "funding_x_",
            "open_interest", "oi_change_x_", "oi_x_",
            "basis_", "spread_", "price_median", "n_markets",
            "vol_ratio_", "volume_24h_",
        )

        def _has_fundamentals(frame):
            if frame is None or frame.empty:
                return False
            return any(
                any(p in str(c).lower() for p in _FUNDAMENTAL_PATTERNS)
                for c in frame.columns
            )

        def _is_fundamental_col(cn):
            cl = str(cn).lower()
            return any(p in cl for p in _FUNDAMENTAL_PATTERNS)

        def _find_datetime_col(frame):
            if frame is None or frame.empty:
                return None
            for col in frame.columns:
                if str(col).lower() in _DATETIME_COL_NAMES:
                    return str(col)
            for col in frame.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(frame[col]):
                        return str(col)
                except Exception:
                    continue
            return None

        async def _apply_fundamental_attach(frame, frame_label):
            if frame is None or frame.empty:
                return frame
            if _has_fundamentals(frame):
                logger.info(
                    f"[{symbol}] ⏭️ PHASE 0 ({frame_label}): "
                    f"fundamentals already attached — skipping"
                )
                return frame
            ts_col = _find_datetime_col(frame)
            if ts_col is None:
                logger.debug(
                    f"[{symbol}] PHASE 0 ({frame_label}): no datetime col — skip"
                )
                return frame
            try:
                from core.global_method import attach_fundamental_features
                logger.info(
                    f"[{symbol}] 📡 PHASE 0 ({frame_label}): "
                    f"inline fundamental attach (ts='{ts_col}')"
                )
                attached = attach_fundamental_features(
                    frame, symbol=self.symbol, timestamp_col=ts_col,
                    require_success=False,
                )
                return attached if attached is not None else frame
            except Exception as fund_err:
                logger.warning(
                    f"[{symbol}] ⚠️ PHASE 0 ({frame_label}): "
                    f"fundamental attach failed: {fund_err}"
                )
                return frame

        # VALIDATION
        if df_split is None or df_split.empty:
            raise ValueError(f"[{symbol}] df_split empty for {split_name}")
        missing_ohlc = [
            c for c in ("open", "high", "low", "close")
            if c not in df_split.columns
        ]
        if missing_ohlc:
            raise ValueError(f"[{symbol}] missing OHLC: {missing_ohlc}")

        start = _dt.now()
        logger.info(
            f"[{symbol}] 🚀 process split={split_name} | "
            f"rows={len(df_split)} | "
            f"history={'YES' if history_df is not None else 'NO'}"
        )

        # PHASE 0: fundamental attach
        df_split = await _apply_fundamental_attach(
            df_split, f"{split_name}_split"
        )
        if history_df is not None and not history_df.empty:
            history_df = await _apply_fundamental_attach(
                history_df, f"{split_name}_history"
            )

        # PHASE 0.5: schema alignment
        if history_df is not None and not history_df.empty:
            split_cols = set(map(str, df_split.columns))
            hist_cols = set(map(str, history_df.columns))
            extra_split = sorted(split_cols - hist_cols)
            extra_hist = sorted(hist_cols - split_cols)
            non_fund_split = [
                c for c in extra_split if not _is_fundamental_col(c)
            ]
            non_fund_hist = [
                c for c in extra_hist if not _is_fundamental_col(c)
            ]
            if non_fund_split or non_fund_hist:
                raise RuntimeError(
                    f"[{symbol}] {split_name}: NON-FUNDAMENTAL schema "
                    f"mismatch | split-only={non_fund_split[:10]} | "
                    f"history-only={non_fund_hist[:10]}"
                )
            fund_split = [
                c for c in extra_split if _is_fundamental_col(c)
            ]
            fund_hist = [
                c for c in extra_hist if _is_fundamental_col(c)
            ]
            if fund_split:
                logger.warning(
                    f"[{symbol}] ⚠️ PHASE 0.5 ({split_name}): dropping from "
                    f"split: {fund_split[:10]}"
                )
                df_split = df_split.drop(columns=fund_split, errors="ignore")
            if fund_hist:
                logger.warning(
                    f"[{symbol}] ⚠️ PHASE 0.5 ({split_name}): dropping from "
                    f"history: {fund_hist[:10]}"
                )
                history_df = history_df.drop(
                    columns=fund_hist, errors="ignore"
                )

        # Store timestamp, drop from split
        _ts_for_features = None
        if "timestamp" in df_split.columns:
            _ts_for_features = (
                df_split["timestamp"].copy().reset_index(drop=True)
            )
            df_split = df_split.drop(columns=["timestamp"])

        # PHASE 1.5: drop datetime from history
        if history_df is not None and not history_df.empty:
            hist_drop = []
            for col in history_df.columns:
                cl = str(col).lower()
                if cl in _DATETIME_COL_NAMES:
                    hist_drop.append(col)
                    continue
                try:
                    if pd.api.types.is_datetime64_any_dtype(history_df[col]):
                        hist_drop.append(col)
                except Exception:
                    pass
            if hist_drop:
                logger.info(
                    f"[{symbol}] 🧹 PHASE 1.5: dropping from history: "
                    f"{hist_drop}"
                )
                history_df = history_df.drop(
                    columns=hist_drop, errors="ignore"
                ).reset_index(drop=True)

        # PHASE 1: enhance split
        df_enhanced = self._enhance_raw_data(df_split)
        if df_enhanced is None or df_enhanced.empty:
            raise RuntimeError(f"[{symbol}] enhance failed for {split_name}")

        # PHASE 2: features (fit or transform)
        if split_name == "train":
            features_df = self.reducer.extract_all_features(
                df_enhanced, mode="fit"
            )
            if features_df is None or features_df.empty:
                raise RuntimeError(f"[{symbol}] fit returned empty for train")
            self._is_fitted = True
            self._feature_columns = [
                c for c in features_df.columns if c not in PROJECT_TARGETS
            ]
            if not hasattr(self, "_sanitizer") or self._sanitizer is None:
                self._sanitizer = SmartFeatureSanitizer(
                    corr_threshold=0.80, var_threshold=1e-8,
                    max_features=500,
                    essential_features=list(self.PROTECTED_COLUMNS),
                    verbose=True, orthogonalize=True,
                    orth_corr_threshold=0.75, symbol=self.symbol,
                )
            logger.info(
                f"[{symbol}] ✅ FIT: {len(self._feature_columns)} "
                f"OHLCV features (pre-fundamental)"
            )
        elif split_name in ("val", "test"):
            if not self._is_fitted:
                raise RuntimeError(
                    f"[{symbol}] reducer not fitted — call fit first"
                )
            used_history = False
            features_df = None
            if history_df is not None and not history_df.empty:
                hist_enhanced = self._enhance_raw_data(history_df)
                if hist_enhanced is not None and not hist_enhanced.empty:
                    try:
                        features_df = transform_with_history(
                            engine=self.reducer,
                            history_df=hist_enhanced,
                            split_df=df_enhanced,
                            max_lookback=self.HISTORY_MAX_LOOKBACK,
                            mode="transform",
                        )
                        used_history = True
                        logger.info(
                            f"[{symbol}] ✅ TRANSFORM-WITH-HISTORY "
                            f"({split_name}): {features_df.shape} | "
                            f"history_rows={len(hist_enhanced)}"
                        )
                    except Exception as twh_err:
                        logger.warning(
                            f"[{symbol}] ⚠️ {split_name}: "
                            f"transform_with_history failed: {twh_err}"
                        )
                        features_df = None
            if features_df is None:
                if not used_history:
                    logger.warning(
                        f"[{symbol}] ⚠️ {split_name}: no history — "
                        f"rolling NaNs possible at split start"
                    )
                features_df = self.reducer.extract_all_features(
                    df_enhanced, mode="transform"
                )
            if features_df is None or features_df.empty:
                raise RuntimeError(
                    f"[{symbol}] transform returned empty for {split_name}"
                )
        elif split_name == "live_data":
            if loaded_scaler_data is not None:
                if not self.reducer.load_scaler_from_metadata(
                    loaded_scaler_data
                ):
                    raise RuntimeError(
                        f"[{symbol}] scaler load failed"
                    )
            elif not self._is_fitted:
                raise RuntimeError(
                    f"[{symbol}] live_data requires fitted reducer"
                )
            if history_df is not None and not history_df.empty:
                hist_enhanced = self._enhance_raw_data(history_df)
                if hist_enhanced is not None and not hist_enhanced.empty:
                    try:
                        features_df = transform_with_history(
                            engine=self.reducer,
                            history_df=hist_enhanced,
                            split_df=df_enhanced,
                            max_lookback=self.HISTORY_MAX_LOOKBACK,
                            mode="transform",
                        )
                    except Exception as twh_err:
                        logger.warning(
                            f"[{symbol}] live: transform_with_history "
                            f"failed: {twh_err}"
                        )
                        features_df = self.reducer.extract_all_features(
                            df_enhanced, mode="transform"
                        )
                else:
                    features_df = self.reducer.extract_all_features(
                        df_enhanced, mode="transform"
                    )
            else:
                features_df = self.reducer.extract_all_features(
                    df_enhanced, mode="transform"
                )
        else:
            raise ValueError(
                f"[{symbol}] unknown split_name={split_name!r}"
            )

        # PHASE 2.5: fundamentals attach + frozen schema
        fundamental_cols_detected = self._detect_fundamental_columns(
            df_enhanced
        )
        fundamental_cols = [
            c for c in fundamental_cols_detected
            if c not in features_df.columns
        ]

        if split_name == "train":
            self._fundamental_columns_frozen = list(fundamental_cols)
            if fundamental_cols:
                if len(df_enhanced) != len(features_df):
                    raise RuntimeError(
                        f"[{symbol}] train: df_enhanced rows "
                        f"({len(df_enhanced)}) != features_df rows "
                        f"({len(features_df)})"
                    )
                fundamental_block = df_enhanced[
                    fundamental_cols
                ].reset_index(drop=True)
                fv = fundamental_block.to_numpy(
                    dtype=np.float64, copy=False
                )
                if not np.isfinite(fv).all():
                    for col in fundamental_cols:
                        cv = fundamental_block[col].to_numpy(
                            dtype=np.float64, copy=False
                        )
                        fm = np.isfinite(cv)
                        if fm.any() and not fm.all():
                            med = float(np.median(cv[fm]))
                            fundamental_block[col] = np.where(
                                fm, cv, med
                            )
                            logger.info(
                                f"[{symbol}] 🩹 train: filled {col} "
                                f"NaN with median={med:.4f}"
                            )
                features_df = pd.concat(
                    [features_df.reset_index(drop=True), fundamental_block],
                    axis=1,
                )
                logger.info(
                    f"[{symbol}] ✅ PHASE 2.5 (train): attached "
                    f"{len(fundamental_cols)} fundamentals: "
                    f"{fundamental_cols}"
                )
            else:
                logger.info(
                    f"[{symbol}] ℹ️ PHASE 2.5 (train): no fundamentals"
                )
        else:
            frozen_set = set(
                getattr(self, "_fundamental_columns_frozen", [])
            )
            allowed_fundamentals = [
                c for c in fundamental_cols if c in frozen_set
            ]
            filtered_out = [
                c for c in fundamental_cols if c not in frozen_set
            ]
            if filtered_out:
                logger.warning(
                    f"[{symbol}] ⚠️ PHASE 2.5 ({split_name}): filtering "
                    f"out non-frozen fundamentals: {filtered_out}"
                )
            present_in_split = set(features_df.columns)
            present_in_split.update(allowed_fundamentals)
            missing_frozen = [
                c for c in frozen_set if c not in present_in_split
            ]
            if missing_frozen:
                logger.warning(
                    f"[{symbol}] 🩹 PHASE 2.5 ({split_name}): "
                    f"{len(missing_frozen)} frozen fundamentals missing — "
                    f"fill 0.0: {missing_frozen}"
                )
            if allowed_fundamentals:
                if len(df_enhanced) != len(features_df):
                    raise RuntimeError(
                        f"[{symbol}] {split_name}: row mismatch"
                    )
                fb = df_enhanced[
                    allowed_fundamentals
                ].reset_index(drop=True)
                for col in allowed_fundamentals:
                    cv = fb[col].to_numpy(dtype=np.float64, copy=False)
                    fm = np.isfinite(cv)
                    if fm.all():
                        continue
                    if fm.any():
                        rep = float(np.median(cv[fm]))
                    else:
                        logger.warning(
                            f"[{symbol}] ⚠️ {split_name}: "
                            f"'{col}' entirely NaN — dropping"
                        )
                        fb = fb.drop(columns=[col], errors="ignore")
                        continue
                    fb[col] = np.where(fm, cv, rep)
                    logger.info(
                        f"[{symbol}] 🩹 {split_name}: filled {col} "
                        f"NaN with median={rep:.4f}"
                    )
                remaining_fund = [
                    c for c in allowed_fundamentals
                    if c in fb.columns
                ]
                if remaining_fund:
                    features_df = pd.concat(
                        [features_df.reset_index(drop=True),
                         fb[remaining_fund]], axis=1,
                    )
                    logger.info(
                        f"[{symbol}] ✅ PHASE 2.5 ({split_name}): attached "
                        f"{len(remaining_fund)} fundamentals"
                    )
            if missing_frozen:
                if len(features_df) == 0:
                    raise RuntimeError(
                        f"[{symbol}] {split_name}: features_df empty"
                    )
                for missing_col in missing_frozen:
                    if missing_col in features_df.columns:
                        continue
                    features_df[missing_col] = np.zeros(
                        len(features_df), dtype=np.float32
                    )
                logger.info(
                    f"[{symbol}] ✅ PHASE 2.5 ({split_name}): filled "
                    f"{len(missing_frozen)} missing frozen with 0.0"
                )

        # PHASE 2.5.5: reorder columns per frozen TRAIN order
        frozen_order = list(
            getattr(self, "_fundamental_columns_frozen", [])
        )
        if frozen_order:
            frozen_set_final = set(frozen_order)
            truly_missing = [
                c for c in frozen_order if c not in features_df.columns
            ]
            if truly_missing:
                logger.warning(
                    f"[{symbol}] ⚠️ PHASE 2.5 ({split_name}): "
                    f"final safety fill: {truly_missing}"
                )
                for c in truly_missing:
                    features_df[c] = np.zeros(
                        len(features_df), dtype=np.float32
                    )
            non_fund_cols = [
                c for c in features_df.columns
                if c not in frozen_set_final
            ]
            ordered_fund_cols = [
                c for c in frozen_order if c in features_df.columns
            ]
            expected_total = len(non_fund_cols) + len(ordered_fund_cols)
            if expected_total == len(features_df.columns):
                features_df = features_df[
                    non_fund_cols + ordered_fund_cols
                ]
                if split_name != "train":
                    logger.info(
                        f"[{symbol}] ✅ PHASE 2.5 ({split_name}): column "
                        f"order restored to TRAIN frozen order"
                    )
            else:
                raise RuntimeError(
                    f"[{symbol}] {split_name}: reorder failed | "
                    f"non_fund={len(non_fund_cols)}, "
                    f"fund={len(ordered_fund_cols)}"
                )

        # PHASE 2.7: sanitizer fit (train only)
        if split_name == "train":
            if not hasattr(self, "_sanitizer") or self._sanitizer is None:
                raise RuntimeError(
                    f"[{symbol}] PHASE 2.7: sanitizer not instantiated"
                )
            if features_df is None or features_df.empty:
                raise RuntimeError(f"[{symbol}] PHASE 2.7: features_df empty")
            target_leak = [
                c for c in features_df.columns if c in PROJECT_TARGETS
            ]
            if target_leak:
                raise RuntimeError(
                    f"[{symbol}] PHASE 2.7: target columns in features: "
                    f"{target_leak}"
                )
            self._sanitizer.fit(features_df)
            if not self._sanitizer.fitted:
                raise RuntimeError(
                    f"[{symbol}] PHASE 2.7: sanitizer.fit() failed"
                )
            self._feature_columns = [
                c for c in features_df.columns if c not in PROJECT_TARGETS
            ]
            logger.info(
                f"[{symbol}] ✅ PHASE 2.7: sanitizer fitted | "
                f"input={len(features_df.columns)} | "
                f"keep={len(self._sanitizer.keep_cols)}"
            )

        # PHASE 3: targets
        if split_name == "live_data":
            feature_cols = [
                c for c in features_df.columns if c not in PROJECT_TARGETS
            ]
            vals = features_df[feature_cols].to_numpy(
                dtype=np.float64, copy=False,
            )
            if not np.isfinite(vals).all():
                raise ValueError(
                    f"[{symbol}] live_data features contain NaN/Inf"
                )
            logger.info(
                f"[{symbol}] ✅ LIVE features | shape={features_df.shape}"
            )
            return features_df

        arr = self._extract_arrays(df_split)

        # ✅ FIX B5: direction_aware mode
        targets_df = self._generate_targets(
            close=arr["close"],
            n=len(df_split),
            split_name=split_name,
            sample_weight_mode="direction_aware",
        )
        if targets_df is None or targets_df.empty:
            raise RuntimeError(
                f"[{symbol}] targets empty for {split_name}"
            )

        # ✅ FIX B6: auxiliary_cols از constant به‌جای هاردکد
        auxiliary_cols = list(AUXILIARY_COLUMNS)
        auxiliary_present = [
            c for c in auxiliary_cols if c in targets_df.columns
        ]
        if auxiliary_present:
            auxiliary_block = (
                targets_df[auxiliary_present]
                .copy()
                .reset_index(drop=True)
            )
        else:
            auxiliary_block = pd.DataFrame()
            logger.warning(
                f"[{symbol}] ⚠️ {split_name}: no auxiliary in targets"
            )

        targets_core = targets_df.drop(
            columns=auxiliary_cols, errors="ignore",
        ).reset_index(drop=True)

        result = pd.concat(
            [features_df.reset_index(drop=True), targets_core],
            axis=1,
        )

        # PHASE 4: sanitization
        if not hasattr(self, "_sanitizer") or self._sanitizer is None:
            raise RuntimeError(
                f"[{symbol}] sanitizer not initialized"
            )
        if split_name in ("val", "test", "live_data"):
            if not getattr(self._sanitizer, "fitted", False):
                raise RuntimeError(
                    f"[{symbol}] PHASE 4 ({split_name}): sanitizer not fitted"
                )
        sanitized = self._sanitizer.transform(result)
        if sanitized is None or sanitized.empty:
            raise ValueError(f"[{symbol}] sanitization empty")

        # PHASE 5: re-attach auxiliary
        if not auxiliary_block.empty:
            n_s = len(sanitized)
            n_a = len(auxiliary_block)
            if n_s != n_a:
                min_len = min(n_s, n_a)
                logger.warning(
                    f"[{symbol}] ⚠️ {split_name}: sanitized={n_s} vs "
                    f"aux={n_a} — aligning to {min_len}"
                )
                sanitized = sanitized.iloc[:min_len].reset_index(drop=True)
                auxiliary_block = auxiliary_block.iloc[:min_len].reset_index(
                    drop=True
                )
            sanitized = pd.concat(
                [sanitized.reset_index(drop=True),
                 auxiliary_block.reset_index(drop=True)],
                axis=1,
            )
            logger.info(
                f"[{symbol}] ✅ {split_name}: auxiliary re-attached: "
                f"{auxiliary_present}"
            )

        # PHASE 6: filter to train_columns
        if train_columns:
            train_feats = [
                c for c in train_columns if c not in PROJECT_TARGETS
            ]
            missing_cols = [
                c for c in train_feats if c not in sanitized.columns
            ]
            if missing_cols:
                raise RuntimeError(
                    f"[{symbol}] {split_name}: train_columns missing: "
                    f"{missing_cols[:10]}"
                )
            keep = (
                train_feats
                + [c for c in sanitized.columns if c in PROJECT_TARGETS]
                + [c for c in auxiliary_present if c in sanitized.columns]
            )
            seen = set()
            keep_unique = []
            for c in keep:
                if c not in seen:
                    seen.add(c)
                    keep_unique.append(c)
            sanitized = sanitized[keep_unique]
            logger.info(
                f"[{symbol}] ✅ POST-SANITIZE filter({split_name}): "
                f"{sanitized.shape}"
            )

        # PHASE 7: finite check
        feature_cols_final = [
            c for c in sanitized.columns
            if c not in PROJECT_TARGETS and c not in auxiliary_present
        ]
        if feature_cols_final:
            mat = sanitized[feature_cols_final].to_numpy(
                dtype=np.float64, copy=False,
            )
            if not np.isfinite(mat).all():
                bad = [
                    feature_cols_final[i]
                    for i in range(mat.shape[1])
                    if not np.isfinite(mat[:, i]).all()
                ]
                raise ValueError(
                    f"[{symbol}] non-finite features: {bad[:10]}"
                )

        # PHASE 8: auxiliary validation
        for aux in auxiliary_present:
            if aux not in sanitized.columns:
                raise RuntimeError(
                    f"[{symbol}] ❌ auxiliary '{aux}' lost during "
                    f"{split_name}"
                )

        elapsed = (_dt.now() - start).total_seconds()
        logger.info(
            f"[{symbol}] ✅ {split_name.upper()} done | "
            f"shape={sanitized.shape} | {elapsed:.2f}s | "
            f"auxiliary={auxiliary_present}"
        )
        return sanitized
        
    def get_feature_names(self) -> List[str]:
        """Return feature names"""
        return self._feature_columns if self._feature_columns else []
    
    def get_feature_categories(self) -> Dict[str, str]:
        """Return feature categories"""
        return {}

# ============================================================================
# ============================================================================
# ============================================================================

# ============================================================================
# BASE SCALER CLASSES
# ============================================================================

class BaseFeatureScaler:
    _FORBIDDEN_CHARS = r'/\:*?"<>| .'

    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        self.params: Dict = {}
        self.fitted = False

    def fit(self, x: np.ndarray) -> "BaseFeatureScaler":
        raise NotImplementedError

    def transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _safe_filename(name: str) -> str:
        for ch in r'/\:*?"<>| .':
            name = name.replace(ch, "_")
        if len(name) > 100:
            h = hashlib.md5(name.encode()).hexdigest()[:8]
            name = f"{name[:90]}_{h}"
        return name

    def save(self, path: Path):
        fname = self._safe_filename(self.feature_name)
        file_path = path / f"{fname}.json"
        with open(file_path, "w") as f:
            json.dump(
                {
                    "feature_name": self.feature_name,
                    "params": self.params,
                    "type": self.__class__.__name__,
                },
                f,
                indent=2,
                default=str,
            )

    def load(self, path: Path):
        fname = self._safe_filename(self.feature_name)
        file_path = path / f"{fname}.json"
        if not file_path.exists():
            file_path = path / f"{self.feature_name}.json"
        with open(file_path) as f:
            data = json.load(f)
        self.params = data["params"]
        self.fitted = True


class StandardScaler(BaseFeatureScaler):
    """
    StandardScaler با clip سخت روی z-score برای جلوگیری از انفجار kurtosis.

    ✅ clip_zscore پیش‌فرض = 10.0 (ذخیره در params برای persistence).
    ✅ NaN/Inf در ورودی حفظ می‌شوند (NaNs برای targetهای mask شده).
    ✅ مقادیر finite فقط clip می‌شوند (نه جایگزینی).
    """

    DEFAULT_CLIP_ZSCORE: float = 10.0

    def __init__(self, feature_name: str):
        super().__init__(feature_name)
        self.clip_zscore: float = float(self.DEFAULT_CLIP_ZSCORE)

    def fit(self, x: np.ndarray) -> "StandardScaler":
        xc = x[~np.isnan(x)]
        if len(xc) == 0:
            xc = np.array([0.0])

        mean_val = float(np.mean(xc))
        std_val = float(np.std(xc))
        if std_val < 1e-10:
            std_val = 1.0

        self.params["mean"] = mean_val
        self.params["std"] = std_val
        self.params["clip_zscore"] = float(self.clip_zscore)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        mean_val = float(self.params["mean"])
        std_val = float(self.params["std"])
        clip_val = float(
            self.params.get("clip_zscore", self.DEFAULT_CLIP_ZSCORE)
        )

        values = np.asarray(x, dtype=np.float64)
        result = (values - mean_val) / std_val

        # فقط مقادیر finite را clip کن؛ NaN/Inf را دست‌نخورده عبور بده
        finite_mask = np.isfinite(result)
        if finite_mask.any():
            result[finite_mask] = np.clip(
                result[finite_mask], -clip_val, clip_val
            )

        return result.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        restored = values * float(self.params["std"]) + float(
            self.params["mean"]
        )
        return restored.astype(np.float32)

class RobustScaler(BaseFeatureScaler):
    def fit(self, x: np.ndarray) -> "RobustScaler":
        xc = x[~np.isnan(x)]
        if len(xc) == 0:
            xc = np.array([0.0, 1.0])
        self.params["median"] = float(np.median(xc))
        self.params["iqr"] = float(np.percentile(xc, 75) - np.percentile(xc, 25) + 1e-8)
        self.params["mean"] = float(np.mean(xc))
        self.params["std"] = float(np.std(xc) + 1e-8)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        scaled = (x - self.params["median"]) / self.params["iqr"]
        return scaled.astype(np.float32)
        
    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.params["iqr"] + self.params["median"]).astype(np.float32)


class BinaryScaler(BaseFeatureScaler):
    def fit(self, x: np.ndarray) -> "BinaryScaler":
        xc = x[~np.isnan(x)]
        self.params["mean"] = float(np.mean(xc)) if len(xc) else 0.0
        self.params["threshold"] = 0.5
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        return (x > self.params["threshold"]).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        return (x > self.params["threshold"]).astype(np.float32)


class LabelScaler(BaseFeatureScaler):
    def fit(self, x: np.ndarray) -> "LabelScaler":
        xc = x[~np.isnan(x)]
        if len(xc) == 0:
            xc = np.array([0, 1, 2])
        self.params["unique_values"] = sorted(list(np.unique(xc).astype(int)))
        self.params["n_classes"] = len(self.params["unique_values"])
        self.params["min"] = float(np.min(xc))
        self.params["max"] = float(np.max(xc))
        unique, counts = np.unique(xc, return_counts=True)
        self.params["class_distribution"] = {int(k): int(v) for k, v in zip(unique, counts)}
        self.params["mean"] = float(np.mean(xc))
        self.params["std"] = float(np.std(xc) + 1e-8)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=self.params.get("min", 0))
        return x.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32)


class PowerScaler(BaseFeatureScaler):
    """
    Box-Cox scaler برای داده‌هایی که با shift به محدوده مثبت منتقل می‌شوند.
    """

    def fit(self, x: np.ndarray) -> "PowerScaler":
        values = np.asarray(x, dtype=np.float64).reshape(-1)
        finite_values = values[np.isfinite(values)]

        if finite_values.size < 2:
            raise ValueError(
                f"PowerScaler {self.feature_name!r} requires at least "
                "two finite values"
            )

        minimum = float(np.min(finite_values))
        shift = (1e-6 - minimum) if minimum <= 0.0 else 0.0
        positive_values = finite_values + shift

        if not np.isfinite(positive_values).all():
            raise ValueError(
                f"PowerScaler {self.feature_name!r} received non-finite values"
            )

        if np.any(positive_values <= 0.0):
            raise ValueError(
                f"PowerScaler {self.feature_name!r} could not make all "
                "training values positive"
            )

        try:
            lambda_value = float(
                stats.boxcox_normmax(
                    positive_values,
                    method="mle",
                )
            )
        except (ValueError, FloatingPointError, OverflowError):
            lambda_value = 0.0

        self.params = {
            "lambda": lambda_value,
            "shift": float(shift),
        }

        transformed = self.transform(finite_values).astype(
            np.float64,
            copy=False,
        )

        if not np.isfinite(transformed).all():
            raise ValueError(
                f"PowerScaler {self.feature_name!r} produced invalid "
                "training values"
            )

        self.params["mean"] = float(np.mean(transformed))

        std_value = float(np.std(transformed))
        self.params["std"] = (
            std_value
            if std_value > 1e-12
            else 1.0
        )

        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if "lambda" not in self.params or "shift" not in self.params:
            raise RuntimeError(
                f"PowerScaler {self.feature_name!r} is not fitted"
            )

        values = np.asarray(x, dtype=np.float64)
        shifted = values + float(self.params["shift"])

        if not np.isfinite(shifted).all():
            raise ValueError(
                f"PowerScaler {self.feature_name!r} input contains NaN/Inf"
            )

        if np.any(shifted <= 0.0):
            raise ValueError(
                f"PowerScaler {self.feature_name!r} transform input is "
                "outside the fitted positive domain"
            )

        lambda_value = float(self.params["lambda"])

        if abs(lambda_value) < 1e-8:
            transformed = np.log(shifted)
        else:
            transformed = (
                np.power(shifted, lambda_value) - 1.0
            ) / lambda_value

        if not np.isfinite(transformed).all():
            raise ValueError(
                f"PowerScaler {self.feature_name!r} produced NaN/Inf"
            )

        return transformed.astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if "lambda" not in self.params or "shift" not in self.params:
            raise RuntimeError(
                f"PowerScaler {self.feature_name!r} is not fitted"
            )

        values = np.asarray(x, dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(
                f"PowerScaler {self.feature_name!r} inverse input "
                "contains NaN/Inf"
            )

        lambda_value = float(self.params["lambda"])

        if abs(lambda_value) < 1e-8:
            restored = np.exp(values)
        else:
            base = lambda_value * values + 1.0

            if np.any(base <= 0.0):
                raise ValueError(
                    f"PowerScaler {self.feature_name!r} inverse input is "
                    "outside the Box-Cox domain"
                )

            restored = np.power(base, 1.0 / lambda_value)

        restored -= float(self.params["shift"])

        if not np.isfinite(restored).all():
            raise ValueError(
                f"PowerScaler {self.feature_name!r} inverse transform "
                "produced NaN/Inf"
            )

        return restored.astype(np.float32)
        
class QuantileScaler(BaseFeatureScaler):
    def __init__(self, feature_name: str, n_quantiles: int = 1000):
        super().__init__(feature_name)
        self.n_quantiles = n_quantiles

    def fit(self, x: np.ndarray) -> "QuantileScaler":
        xc = x[~np.isnan(x)]
        if len(xc) == 0:
            xc = np.array([0.0, 1.0])
        qs = np.linspace(0, 100, self.n_quantiles)
        self.params["quantiles"] = np.percentile(xc, qs).tolist()
        self.params["quantile_points"] = qs.tolist()
        self.params["min"] = float(np.min(xc))
        self.params["max"] = float(np.max(xc))
        x_transformed = self.transform(xc)
        self.params["mean"] = float(np.mean(x_transformed))
        self.params["std"] = float(np.std(x_transformed) + 1e-8)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=self.params["min"])
        ranks = np.searchsorted(self.params["quantiles"], x) / self.n_quantiles
        ranks = np.clip(ranks, 0.001, 0.999)
        return stats.norm.ppf(ranks).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        ranks = stats.norm.cdf(x)
        indices = np.clip((ranks * self.n_quantiles).astype(int), 0, self.n_quantiles - 1)
        return np.array([self.params["quantiles"][i] for i in indices], dtype=np.float32)


class LogScaler(BaseFeatureScaler):
    def fit(self, x: np.ndarray) -> "LogScaler":
        xc = x[~np.isnan(x)]
        if len(xc) == 0:
            xc = np.array([1.0, 2.0])
        xc_abs = np.abs(xc) + 1e-12
        lx = np.log(xc_abs)
        self.params["mean"] = float(np.mean(lx))
        self.params["std"] = float(np.std(lx) + 1e-8)
        self.params["shift"] = 1e-12
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0)
        return ((np.log(np.abs(x) + self.params["shift"]) - self.params["mean"]) / self.params["std"]).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (np.exp(x * self.params["std"] + self.params["mean"]) - self.params["shift"]).astype(np.float32)

# ============================================================================
# FEATURE PROFILER
# ============================================================================

class FeatureProfiler:
    def __init__(self, skew_threshold: float = 1.0, kurtosis_threshold: float = 5.0):
        self.skew_threshold = skew_threshold
        self.kurtosis_threshold = kurtosis_threshold

    def is_categorical_target(self, x: np.ndarray) -> bool:
        x_clean = x[~np.isnan(x)]
        if len(x_clean) == 0:
            return False
        n_unique = len(np.unique(x_clean))
        if n_unique <= 20:
            return True
        x_range = np.max(x_clean) - np.min(x_clean)
        if n_unique <= 50 and x_range > 100:
            return True
        if n_unique < len(x_clean) * 0.1:
            percentiles = np.percentile(x_clean, [10, 25, 50, 75, 90])
            gaps = np.diff(percentiles)
            if np.std(gaps) / (np.mean(gaps) + 1e-10) < 0.5:
                return True
        return False

    def get_target_type(self, x: np.ndarray) -> str:
        return "classification" if self.is_categorical_target(x) else "regression"

    def profile(self, x: np.ndarray) -> Dict:
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return self._empty_profile()
        mean = float(np.mean(x))
        std = float(np.std(x))
        median = float(np.median(x))
        q1, q3 = np.percentile(x, [25, 75])
        iqr = float(q3 - q1)
        skewness = float(stats.skew(x))
        kurtosis = float(stats.kurtosis(x))
        z_scores = np.abs((x - median) / (iqr + 1e-8))
        outlier_ratio = float(np.mean(z_scores > 3))
        zero_ratio = float(np.mean(np.abs(x) < 1e-12))
        target_type = self.get_target_type(x)
        return {
            "mean": mean, "std": std, "median": median, "iqr": iqr,
            "skewness": skewness, "kurtosis": kurtosis,
            "outlier_ratio": outlier_ratio, "zero_ratio": zero_ratio,
            "target_type": target_type,
            "n_unique": len(np.unique(x)),
            "recommended_method": self._recommend_method(skewness, kurtosis, outlier_ratio, zero_ratio, target_type)
        }

    def _empty_profile(self) -> Dict:
        return {
            "mean": 0.0, "std": 0.0, "median": 0.0, "iqr": 0.0,
            "skewness": 0.0, "kurtosis": 0.0, "outlier_ratio": 0.0,
            "zero_ratio": 1.0, "target_type": "unknown", "n_unique": 0,
            "recommended_method": "standard"
        }

    def _recommend_method(self, skewness, kurtosis, outlier_ratio, zero_ratio, target_type: str) -> str:
        if target_type == "classification":
            return "label"
        if zero_ratio > 0.5:
            return "binary"
        if outlier_ratio > 0.05:
            return "quantile" if kurtosis > 10 else "robust"
        if abs(skewness) > 2:
            return "log"
        if abs(skewness) > 1:
            return "power"
        return "standard"


# ============================================================================
# SMART ADAPTIVE SCALER - با هماهنگی کامل
# ============================================================================

class SmartAdaptiveScaler:
    def __init__(self, symbol: str, timeframe: str, fold_num: int,
                 is_target: bool = False, auto_profile: bool = True):
        # ✅ FIX v7.1: نرمالایز کردن نام نماد (حذف نقطه) برای هماهنگی با _load_scalers
        def normalize_symbol(s: str) -> str:
            if not s:
                return s
            return s.replace('.', '')

        clean_symbol = normalize_symbol(symbol)
        clean_timeframe = timeframe.upper() if timeframe else timeframe

        self.symbol = clean_symbol
        self.timeframe = clean_timeframe
        self.fold_num = fold_num
        self.is_target = is_target
        self.auto_profile = auto_profile
        type_str = "target" if is_target else "feature"
        self.base_dir = Path(f"data/scalers/{clean_symbol}/{clean_timeframe}/fold_{fold_num}/{type_str}_scalers")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.feature_scalers: Dict[str, BaseFeatureScaler] = {}
        self.feature_names: List[str] = []
        self.fitted = False
        self.input_dim = None
        self.target_types: Dict[str, str] = {}
        self.classification_targets: List[str] = []
        self.regression_targets: List[str] = []
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    ) -> "SmartAdaptiveScaler":
        """
        Fit اسکیلر فقط روی TRAIN با قرارداد schema سخت‌گیرانه.

        ✅ v7.3 — FIX CRITICAL (auxiliary classification vs weight):
            نسخهٔ v7.2 تمام ستون‌های auxiliary را با target_type =
            'auxiliary_identity' ذخیره می‌کرد. نتیجه:
              • magnitude_bucket و momentum_direction در metadata
                به‌عنوان auxiliary_identity ثبت می‌شدند (نه classification).
              • load_scalers_from_path این نوع را نمی‌شناخت و آن‌ها را
                skip می‌کرد → هشدار "Unknown target_type" و در ادامه
                هشدار "Missing targets".
            راه‌حل: فقط ستون‌هایی که در AUXILIARY_TYPES وزن‌اند
            (sample_weight) به auxiliary_weight نگاشت می‌شوند.
            سایر auxiliaryها (magnitude_bucket، momentum_direction)
            به‌صورت classification استاندارد ذخیره می‌شوند.
        """
        if X is None:
            raise ValueError(
                f"[{self.symbol}] Cannot fit scaler on None"
            )

        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError(
                    f"[{self.symbol}] Cannot fit scaler on empty DataFrame"
                )

            if X.columns.duplicated().any():
                duplicated = X.columns[
                    X.columns.duplicated()
                ].tolist()
                raise ValueError(
                    f"[{self.symbol}] Duplicate scaler columns: {duplicated}"
                )

            actual_names = [str(column) for column in X.columns]

            if feature_names is None:
                resolved_names = actual_names
            else:
                resolved_names = [str(name) for name in feature_names]

                if resolved_names != actual_names:
                    raise ValueError(
                        f"[{self.symbol}] feature_names do not match "
                        "DataFrame column order"
                    )

            values = X.to_numpy(dtype=np.float64, copy=True)

        else:
            values = np.asarray(X)

            if values.ndim not in {2, 3}:
                raise ValueError(
                    f"[{self.symbol}] Scaler input must be 2D or 3D, "
                    f"got shape={values.shape}"
                )

            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(
                    f"[{self.symbol}] Scaler input must be numeric, "
                    f"got dtype={values.dtype}"
                )

            feature_count = (
                int(values.shape[1])
                if values.ndim == 2
                else int(values.shape[2])
            )

            if feature_names is None:
                resolved_names = [
                    f"f{index}"
                    for index in range(feature_count)
                ]
            else:
                resolved_names = [str(name) for name in feature_names]

                if len(resolved_names) != feature_count:
                    raise ValueError(
                        f"[{self.symbol}] feature_names length mismatch: "
                        f"expected={feature_count}, "
                        f"actual={len(resolved_names)}"
                    )

            values = values.astype(np.float64, copy=True)

        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError(
                f"[{self.symbol}] Duplicate feature names are not allowed"
            )

        if values.ndim == 3:
            feature_count = values.shape[2]
            working = values.reshape(-1, feature_count)
        else:
            working = values

        if working.shape[0] == 0 or working.shape[1] == 0:
            raise ValueError(
                f"[{self.symbol}] Scaler input has zero samples or features"
            )

        if working.shape[1] != len(resolved_names):
            raise ValueError(
                f"[{self.symbol}] Internal scaler dimension mismatch: "
                f"values={working.shape[1]}, names={len(resolved_names)}"
            )

        if not self.is_target and not np.isfinite(working).all():
            bad_columns = [
                resolved_names[index]
                for index in range(working.shape[1])
                if not np.isfinite(working[:, index]).all()
            ]
            raise ValueError(
                f"[{self.symbol}] Feature scaler input contains NaN/Inf: "
                f"{bad_columns}"
            )

        self.feature_names = list(resolved_names)
        self.input_dim = int(working.shape[1])
        self.feature_scalers = {}
        self.target_types = {}
        self.classification_targets = []
        self.regression_targets = []

        aux_lower = frozenset(
            str(c).strip().lower() for c in AUXILIARY_COLUMNS
        )

        for index, name in enumerate(self.feature_names):
            column = working[:, index]
            finite_values = column[np.isfinite(column)]

            if finite_values.size == 0:
                raise ValueError(
                    f"[{self.symbol}] Column {name!r} has no finite values"
                )

            if self.is_target:
                canonical_name = _canonical_column_name(name)

                # ═══════════════════════════════════════════════════════════════
                # ✅ v7.3: فقط وزن‌ها به auxiliary_weight نگاشت می‌شوند.
                #          سایر auxiliaryها (magnitude_bucket، momentum_direction)
                #          به شاخهٔ classification استاندارد می‌روند.
                # ═══════════════════════════════════════════════════════════════
                if canonical_name in aux_lower:
                    aux_kind = AUXILIARY_TYPES.get(
                        canonical_name, "auxiliary"
                    )

                    if aux_kind == "weight":
                        aux_scaler = LabelScaler(name)
                        aux_scaler.params = {
                            "min": float(np.min(finite_values)),
                            "max": float(np.max(finite_values)),
                            "mean": float(np.mean(finite_values)),
                            "std": float(np.std(finite_values) + 1e-8),
                            "scale_mode": "identity",
                            "auxiliary_type": "weight",
                        }
                        aux_scaler.fitted = True
                        self.feature_scalers[name] = aux_scaler
                        self.target_types[name] = "auxiliary_weight"

                        logger.info(
                            f"[{self.symbol}] ✅ auxiliary weight {name!r} → "
                            f"IDENTITY (no scaling) | "
                            f"mean={aux_scaler.params['mean']:.4f}, "
                            f"min={aux_scaler.params['min']:.4f}, "
                            f"max={aux_scaler.params['max']:.4f}"
                        )
                        continue

                    # else: classification auxiliary — falls through below

                # ─── classification / regression استاندارد ───
                configured_type = PROJECT_TARGETS_TYPE.get(canonical_name)

                if configured_type is None:
                    unique_count = np.unique(finite_values).size
                    configured_type = (
                        "classification"
                        if unique_count <= 20
                        else "regression"
                    )

                if configured_type == "classification":
                    if not np.all(
                        np.equal(finite_values, np.floor(finite_values))
                    ):
                        raise ValueError(
                            f"[{self.symbol}] Classification target "
                            f"{name!r} contains non-integer values"
                        )

                    valid_classes = finite_values[
                        finite_values != float(INVALID_LABEL)
                    ]

                    if valid_classes.size == 0:
                        raise ValueError(
                            f"[{self.symbol}] Classification target "
                            f"{name!r} contains only INVALID_LABEL"
                        )

                    unique_classes = sorted(
                        np.unique(valid_classes).astype(int).tolist()
                    )

                    scaler = LabelScaler(name)
                    scaler.params = {
                        "unique_values": unique_classes,
                        "n_classes": len(unique_classes),
                        "invalid_label": int(INVALID_LABEL),
                        "scale_mode": "identity",
                    }
                    scaler.fitted = True

                    self.classification_targets.append(name)
                    self.target_types[name] = "classification"
                    self.feature_scalers[name] = scaler
                    continue

                if configured_type == "regression":
                    if finite_values.size < 10:
                        raise ValueError(
                            f"[{self.symbol}] Regression target {name!r} "
                            f"has only {finite_values.size} finite values"
                        )

                    scaler = StandardScaler(name)
                    scaler.fit(finite_values)

                    self.regression_targets.append(name)
                    self.target_types[name] = "regression"
                    self.feature_scalers[name] = scaler
                    continue

                raise RuntimeError(
                    f"[{self.symbol}] Unknown target type "
                    f"{configured_type!r} for {name!r}"
                )

            scaler = StandardScaler(name)
            scaler.fit(finite_values)

            self.target_types[name] = (
                "feature_constant"
                if float(np.std(finite_values)) <= 1e-12
                else "feature"
            )
            self.feature_scalers[name] = scaler

        missing_scalers = [
            name
            for name in self.feature_names
            if name not in self.feature_scalers
        ]

        if missing_scalers:
            raise RuntimeError(
                f"[{self.symbol}] Scaler fit incomplete. Missing: "
                f"{missing_scalers}"
            )

        self.fitted = True
        return self
    
    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Transform با استفاده از آمار frozen شده.

        ✅ v7.3 — FIX:
            - پشتیبانی از target_type == "auxiliary_weight" (نوع جدید).
            - پشتیبانی legacy از "auxiliary_identity" (اسکیلرهای قدیمی).
            - magnitude_bucket و momentum_direction حالا از طریق
              شاخهٔ standard classification پاس می‌شوند (identity
              transform با یکتایی صحیح).
        """
        if not self.fitted:
            raise RuntimeError(f"[{self.symbol}] Scaler is not fitted")

        if X is None:
            raise ValueError(f"[{self.symbol}] Cannot transform None")

        input_is_dataframe = isinstance(X, pd.DataFrame)

        if input_is_dataframe:
            if X.columns.duplicated().any():
                duplicated = X.columns[X.columns.duplicated()].tolist()
                raise ValueError(
                    f"[{self.symbol}] Duplicate transform columns: {duplicated}"
                )

            input_names = [str(column) for column in X.columns]

            if input_names != list(self.feature_names):
                missing = [
                    name for name in self.feature_names
                    if name not in input_names
                ]
                extra = [
                    name for name in input_names
                    if name not in self.feature_names
                ]
                raise ValueError(
                    f"[{self.symbol}] Transform schema mismatch. "
                    f"missing={missing}, extra={extra}, "
                    f"expected_order={self.feature_names}, "
                    f"actual_order={input_names}"
                )

            array = X.to_numpy(dtype=np.float64, copy=True)
        else:
            array = np.asarray(X)

            if array.ndim not in {2, 3}:
                raise ValueError(
                    f"[{self.symbol}] Transform input must be 2D or 3D, "
                    f"got shape {array.shape}"
                )

            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(
                    f"[{self.symbol}] Transform input must be numeric"
                )

            array = array.astype(np.float64, copy=True)

        original_shape = array.shape

        if array.ndim == 3:
            n_samples, n_timesteps, n_features = array.shape

            if n_features != len(self.feature_names):
                raise ValueError(
                    f"[{self.symbol}] Feature dimension mismatch: "
                    f"expected={len(self.feature_names)}, "
                    f"actual={n_features}"
                )

            working = array.reshape(-1, n_features)
        else:
            if array.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"[{self.symbol}] Feature dimension mismatch: "
                    f"expected={len(self.feature_names)}, "
                    f"actual={array.shape[1]}"
                )

            working = array

        if not self.is_target:
            if not np.isfinite(working).all():
                bad_cols = [
                    self.feature_names[i]
                    for i in range(working.shape[1])
                    if not np.isfinite(working[:, i]).all()
                ]
                raise ValueError(
                    f"[{self.symbol}] Feature transform input contains "
                    f"NaN or Inf in columns: {bad_cols[:10]}"
                )

        output = np.empty(working.shape, dtype=np.float32)

        for index, name in enumerate(self.feature_names):
            if name not in self.feature_scalers:
                raise RuntimeError(
                    f"[{self.symbol}] No fitted scaler for column {name!r}"
                )

            scaler = self.feature_scalers[name]
            column = working[:, index]
            target_type = self.target_types.get(name)

            # ═══════════════════════════════════════════════════════════════════
            # ✅ v7.3: sample_weight — pure passthrough (new + legacy types)
            # ═══════════════════════════════════════════════════════════════════
            if self.is_target and target_type in (
                "auxiliary_weight",
                "auxiliary_identity",
            ):
                output[:, index] = column.astype(np.float32)
                continue

            # ─── classification target: عبور بدون scaling با گارد صحیح ───
            if self.is_target and target_type == "classification":
                finite_mask = np.isfinite(column)
                if not np.all(
                    np.equal(
                        column[finite_mask],
                        np.floor(column[finite_mask]),
                    )
                ):
                    raise ValueError(
                        f"[{self.symbol}] Classification target "
                        f"{name!r} contains non-integer values"
                    )
                output[:, index] = column.astype(np.float32)
                continue

            # ─── regression target یا feature ───
            try:
                transformed = scaler.transform(column)
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.symbol}] Scaling failed for column {name!r}"
                ) from exc

            transformed = np.asarray(
                transformed, dtype=np.float32
            ).reshape(-1)

            if transformed.shape[0] != column.shape[0]:
                raise RuntimeError(
                    f"[{self.symbol}] Scaler output length mismatch for "
                    f"{name!r}: expected={column.shape[0]}, "
                    f"actual={transformed.shape[0]}"
                )

            if self.is_target:
                if np.isinf(transformed).any():
                    raise ValueError(
                        f"[{self.symbol}] Target scaler produced Inf for "
                        f"{name!r}"
                    )
            else:
                if not np.isfinite(transformed).all():
                    raise ValueError(
                        f"[{self.symbol}] Scaler produced NaN or Inf "
                        f"for {name!r}"
                    )

            output[:, index] = transformed

        transformed_output = output.reshape(original_shape)

        if not self.is_target:
            if not np.isfinite(transformed_output).all():
                raise ValueError(
                    f"[{self.symbol}] Final feature output contains "
                    "NaN or Inf"
                )

        return transformed_output
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        original_shape = X.shape
        if X.ndim == 3:
            s, l, f = X.shape
            X_working = X.reshape(-1, f)
        else:
            X_working = X
        out = np.zeros_like(X_working, dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            if i < X_working.shape[1] and name in self.feature_scalers:
                scaler = self.feature_scalers[name]
                out[:, i] = scaler.inverse_transform(X_working[:, i])
            elif i < X_working.shape[1]:
                out[:, i] = X_working[:, i]
        return out.reshape(original_shape)

    def save(self):
        for name, scaler in self.feature_scalers.items():
            scaler.save(self.base_dir)
        feature_profiles = {}
        for name, scaler in self.feature_scalers.items():
            feature_profiles[name] = {
                "scaler_type": scaler.__class__.__name__,
                "params": scaler.params,
                "fitted": scaler.fitted,
                "target_type": self.target_types.get(name, "unknown")
            }
        meta = {
            "symbol": self.symbol, "timeframe": self.timeframe, "fold_num": self.fold_num,
            "is_target": self.is_target, "feature_names": self.feature_names,
            "feature_profiles": feature_profiles, "fitted": self.fitted, "input_dim": self.input_dim,
            "n_features": len(self.feature_names), "target_types": self.target_types,
            "classification_targets": self.classification_targets, "regression_targets": self.regression_targets,
            "saved_at": datetime.now().isoformat(), "engine_version": "4.2.0"
        }
        with open(self.base_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        pkl_path = self.base_dir / "full_scaler.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"💾 Scaler saved → {self.base_dir}")

    @classmethod
    def load(
        cls,
        symbol: str,
        timeframe: str,
        fold_num: int,
        is_target: bool = False,
    ) -> "SmartAdaptiveScaler":
        """
        بارگذاری Scaler بدون ساختن identity scaler مخفی.

        اگر metadata یا هر Scaler لازم خراب یا مفقود باشد، نمونه unfitted
        برگردانده می‌شود. هیچ ستون گمشده‌ای سالم فرض نمی‌شود.
        """
        instance = cls(
            symbol=symbol,
            timeframe=timeframe,
            fold_num=fold_num,
            is_target=is_target,
        )

        pkl_path = instance.base_dir / "full_scaler.pkl"
        metadata_path = instance.base_dir / "metadata.json"

        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as file_handle:
                    loaded = pickle.load(file_handle)

                if not isinstance(loaded, cls):
                    raise TypeError(
                        f"Unexpected pickle type: {type(loaded).__name__}"
                    )

                if bool(loaded.is_target) != bool(is_target):
                    raise ValueError(
                        "Loaded scaler target/feature mode mismatch"
                    )

                if not loaded.fitted:
                    raise ValueError("Loaded pickle is not fitted")

                if not loaded.feature_names:
                    raise ValueError(
                        "Loaded pickle has no feature names"
                    )

                if len(set(loaded.feature_names)) != len(
                    loaded.feature_names
                ):
                    raise ValueError(
                        "Loaded pickle contains duplicate feature names"
                    )

                missing_scalers = [
                    name
                    for name in loaded.feature_names
                    if name not in loaded.feature_scalers
                    or not loaded.feature_scalers[name].fitted
                ]

                if missing_scalers:
                    raise ValueError(
                        f"Loaded pickle is incomplete: {missing_scalers}"
                    )

                loaded.input_dim = len(loaded.feature_names)
                logger.info(
                    f"📂 Scaler loaded from PKL ← {instance.base_dir}"
                )
                return loaded

            except (
                OSError,
                EOFError,
                pickle.UnpicklingError,
                AttributeError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning(
                    f"⚠️ Invalid scaler pickle at {pkl_path}: {exc}"
                )

        if not metadata_path.exists():
            logger.warning(
                f"⚠️ No valid scaler metadata in {instance.base_dir}"
            )
            instance.fitted = False
            return instance

        try:
            with open(metadata_path, "r", encoding="utf-8") as file_handle:
                metadata = json.load(file_handle)

            metadata_is_target = bool(
                metadata.get("is_target", is_target)
            )

            if metadata_is_target != bool(is_target):
                raise ValueError(
                    "Scaler metadata target/feature mode mismatch"
                )

            feature_names = [
                str(name)
                for name in metadata.get("feature_names", [])
            ]

            if not feature_names:
                raise ValueError(
                    "Scaler metadata contains no feature names"
                )

            if len(set(feature_names)) != len(feature_names):
                raise ValueError(
                    "Scaler metadata contains duplicate feature names"
                )

            profiles = metadata.get("feature_profiles", {})

            if not isinstance(profiles, dict):
                raise TypeError(
                    "feature_profiles must be a dictionary"
                )

            scaler_types = {
                "StandardScaler": StandardScaler,
                "RobustScaler": RobustScaler,
                "BinaryScaler": BinaryScaler,
                "LabelScaler": LabelScaler,
                "PowerScaler": PowerScaler,
                "QuantileScaler": QuantileScaler,
                "LogScaler": LogScaler,
            }

            instance.feature_names = feature_names
            instance.input_dim = len(feature_names)
            instance.target_types = {
                str(key): str(value)
                for key, value in metadata.get(
                    "target_types",
                    {},
                ).items()
            }
            instance.classification_targets = [
                str(name)
                for name in metadata.get(
                    "classification_targets",
                    [],
                )
            ]
            instance.regression_targets = [
                str(name)
                for name in metadata.get(
                    "regression_targets",
                    [],
                )
            ]
            instance.feature_scalers = {}

            for name in feature_names:
                profile = profiles.get(name)

                if not isinstance(profile, dict):
                    raise ValueError(
                        f"Missing scaler profile for {name!r}"
                    )

                scaler_type_name = profile.get("scaler_type")

                if scaler_type_name not in scaler_types:
                    raise ValueError(
                        f"Unsupported scaler type "
                        f"{scaler_type_name!r} for {name!r}"
                    )

                scaler_class = scaler_types[scaler_type_name]
                scaler = scaler_class(name)

                params = profile.get("params")

                if not isinstance(params, dict):
                    raise ValueError(
                        f"Missing scaler parameters for {name!r}"
                    )

                if not bool(profile.get("fitted", False)):
                    raise ValueError(
                        f"Scaler profile for {name!r} is not fitted"
                    )

                scaler.params = params
                scaler.fitted = True
                instance.feature_scalers[name] = scaler

            missing_scalers = [
                name
                for name in feature_names
                if name not in instance.feature_scalers
            ]

            if missing_scalers:
                raise ValueError(
                    f"Metadata scaler is incomplete: {missing_scalers}"
                )

            instance.fitted = bool(
                metadata.get("fitted", False)
            )

            if not instance.fitted:
                raise ValueError(
                    "Scaler metadata reports fitted=False"
                )

            logger.info(
                f"📂 Scaler loaded from metadata ← {instance.base_dir}"
            )
            return instance

        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            logger.error(
                f"❌ Failed to load scaler from "
                f"{instance.base_dir}: {exc}"
            )
            instance.feature_scalers = {}
            instance.feature_names = []
            instance.input_dim = None
            instance.fitted = False
            return instance
            

# ============================================================================
# SMART SCALING PIPELINE - با هماهنگی کامل
# ============================================================================

class SmartScalingPipeline:
    def __init__(self, symbol: str, timeframe: str, fold_num: int):
        self.symbol = symbol
        self.timeframe = timeframe
        self.fold_num = fold_num
        self.feature_scaler: Optional[SmartAdaptiveScaler] = None
        self.target_scaler: Optional[SmartAdaptiveScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        self.fitted = False

    def fit_transform(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        feature_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        prices: Optional[np.ndarray] = None,
        drop_rows_with_invalid: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit + transform روی TRAIN.

        ✅ تغییر نسبت به نسخهٔ قبل:
            - پارامتر جدید `drop_rows_with_invalid`:
                * True  → رفتار قبلی (حذف ردیف‌های invalid target)
                * False → فقط ردیف‌های non-finite feature حذف می‌شوند؛
                          ردیف‌های invalid-target باقی می‌مانند و مقدار
                          target آن‌ها NaN (regression) یا -1 (classification)
                          می‌شود.
        """
        if X_train is None:
            raise ValueError(f"[{self.symbol}] X_train cannot be None")

        # ── feature names / array ──
        if isinstance(X_train, pd.DataFrame):
            if X_train.empty:
                raise ValueError(f"[{self.symbol}] X_train is empty")
            if feature_names is None:
                resolved_feature_names = [str(c) for c in X_train.columns]
            else:
                resolved_feature_names = [str(n) for n in feature_names]
                actual = [str(c) for c in X_train.columns]
                if resolved_feature_names != actual:
                    raise ValueError(
                        f"[{self.symbol}] feature_names do not match "
                        f"X_train column order"
                    )
            X_array = X_train.to_numpy(dtype=np.float64, copy=True)
        else:
            X_array = np.asarray(X_train)
            if X_array.ndim not in {2, 3}:
                raise ValueError(
                    f"[{self.symbol}] X_train must be 2D or 3D, "
                    f"got shape {X_array.shape}"
                )
            n_features = (
                X_array.shape[1]
                if X_array.ndim == 2
                else X_array.shape[2]
            )
            if feature_names is None:
                resolved_feature_names = [
                    f"f{i}" for i in range(n_features)
                ]
            else:
                resolved_feature_names = [str(n) for n in feature_names]
                if len(resolved_feature_names) != n_features:
                    raise ValueError(
                        f"[{self.symbol}] feature_names length mismatch"
                    )
            X_array = X_array.astype(np.float64, copy=True)

        # ── targets ──
        resolved_target_names: Optional[List[str]] = None
        y_array: Optional[np.ndarray] = None

        if y_train is not None:
            if isinstance(y_train, pd.DataFrame):
                if y_train.empty:
                    raise ValueError(f"[{self.symbol}] y_train is empty")
                if target_names is None:
                    resolved_target_names = [str(c) for c in y_train.columns]
                else:
                    resolved_target_names = [str(n) for n in target_names]
                    actual = [str(c) for c in y_train.columns]
                    if resolved_target_names != actual:
                        raise ValueError(
                            f"[{self.symbol}] target_names do not match "
                            f"y_train column order"
                        )
                y_array = y_train.to_numpy(dtype=np.float64, copy=True)
            else:
                y_array = np.asarray(y_train)
                if y_array.ndim == 1:
                    y_array = y_array.reshape(-1, 1)
                if y_array.ndim != 2:
                    raise ValueError(
                        f"[{self.symbol}] y_train must be 1D or 2D, "
                        f"got shape {y_array.shape}"
                    )
                if resolved_target_names is None:
                    if target_names is None:
                        resolved_target_names = [
                            f"target_{i}"
                            for i in range(y_array.shape[1])
                        ]
                    else:
                        resolved_target_names = [str(n) for n in target_names]
                if len(resolved_target_names) != y_array.shape[1]:
                    raise ValueError(
                        f"[{self.symbol}] target_names length mismatch"
                    )
                y_array = y_array.astype(np.float64, copy=True)

            if len(X_array) != len(y_array):
                raise ValueError(
                    f"[{self.symbol}] X/y length mismatch: "
                    f"X={len(X_array)}, y={len(y_array)}"
                )

        # ── shared mask ──
        n_samples = len(X_array)
        shared_mask = np.ones(n_samples, dtype=bool)

        X_for_mask = X_array
        if X_for_mask.ndim == 3:
            X_for_mask = X_for_mask.reshape(X_for_mask.shape[0], -1)

        shared_mask &= np.isfinite(X_for_mask).all(axis=1)

        if y_array is not None and drop_rows_with_invalid:
            for j, tname in enumerate(resolved_target_names):
                canonical = _canonical_column_name(tname)
                ttype = PROJECT_TARGETS_TYPE.get(canonical, "regression")
                y_col = y_array[:, j]
                y_finite = np.isfinite(y_col)
                if ttype == "classification":
                    y_valid = y_finite & (y_col != float(INVALID_LABEL))
                else:
                    y_valid = y_finite
                shared_mask &= y_valid

        n_dropped = int((~shared_mask).sum())
        if n_dropped > 0:
            logger.info(
                f"[{self.symbol}] 🧹 fit_transform: dropped {n_dropped} "
                f"invalid rows (drop_rows_with_invalid="
                f"{drop_rows_with_invalid})"
            )

        if int(shared_mask.sum()) < 10:
            raise RuntimeError(
                f"[{self.symbol}] After masking only "
                f"{int(shared_mask.sum())} valid samples remain "
                "(need >= 10)"
            )

        X_clean = X_array[shared_mask]
        y_clean = y_array[shared_mask] if y_array is not None else None

        # ── fit feature scaler ──
        self.feature_scaler = SmartAdaptiveScaler(
            symbol=self.symbol,
            timeframe=self.timeframe,
            fold_num=self.fold_num,
            is_target=False,
        )
        self.feature_scaler.fit(
            X=X_clean,
            feature_names=resolved_feature_names,
        )
        X_train_scaled = self.feature_scaler.transform(X_clean)

        # ── fit target scaler ──
        y_train_scaled: Optional[np.ndarray] = None
        if y_clean is not None:
            self.target_scaler = SmartAdaptiveScaler(
                symbol=self.symbol,
                timeframe=self.timeframe,
                fold_num=self.fold_num,
                is_target=True,
            )
            self.target_scaler.fit(
                X=y_clean,
                feature_names=resolved_target_names,
                y=y_clean,
            )
            y_train_scaled = self.target_scaler.transform(y_clean)
        else:
            self.target_scaler = None

        self.feature_names = list(resolved_feature_names)
        self.target_names = (
            list(resolved_target_names)
            if resolved_target_names is not None
            else None
        )
        self.fitted = True
        self._last_fit_mask = shared_mask

        return X_train_scaled, y_train_scaled
        
    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        drop_rows_with_invalid: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform داده‌های VAL/TEST/LIVE با آمار frozen شده.

        ✅ تغییر نسبت به نسخهٔ قبل:
            - پارامتر جدید `drop_rows_with_invalid`.
            - True : رفتار قبلی (حذف ردیف‌های invalid-target).
            - False: فقط ردیف‌های non-finite feature حذف می‌شوند؛
                     ردیف‌های invalid-target باقی می‌مانند.
        """
        if not self.fitted:
            raise RuntimeError(
                f"[{self.symbol}] SmartScalingPipeline is not fitted"
            )
        if self.feature_scaler is None:
            raise RuntimeError(
                f"[{self.symbol}] Feature scaler is missing"
            )
        if self.feature_names is None:
            raise RuntimeError(
                f"[{self.symbol}] Frozen feature names are missing"
            )

        # ── resolve X names ──
        if isinstance(X, pd.DataFrame):
            actual_feature_names = [str(c) for c in X.columns]
            expected_feature_names = [str(n) for n in self.feature_names]
            if actual_feature_names != expected_feature_names:
                missing = [
                    n for n in expected_feature_names
                    if n not in actual_feature_names
                ]
                extra = [
                    n for n in actual_feature_names
                    if n not in expected_feature_names
                ]
                raise ValueError(
                    f"[{self.symbol}] Feature schema mismatch: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            X_array = X.to_numpy(dtype=np.float64, copy=True)
            input_is_dataframe = True
        else:
            X_array = np.asarray(X)
            input_is_dataframe = False

        if X_array.ndim not in {2, 3}:
            raise ValueError(
                f"[{self.symbol}] Transform X must be 2D or 3D, "
                f"got {X_array.shape}"
            )

        # ── resolve y ──
        y_array: Optional[np.ndarray] = None
        y_is_dataframe = False
        resolved_target_names: Optional[List[str]] = None

        if y is not None:
            if isinstance(y, pd.DataFrame):
                y_is_dataframe = True
                actual_target_names = [str(c) for c in y.columns]
                if self.target_names is None:
                    raise RuntimeError(
                        f"[{self.symbol}] Frozen target names missing"
                    )
                expected_target_names = [
                    str(n) for n in self.target_names
                ]
                if actual_target_names != expected_target_names:
                    missing = [
                        n for n in expected_target_names
                        if n not in actual_target_names
                    ]
                    extra = [
                        n for n in actual_target_names
                        if n not in expected_target_names
                    ]
                    raise ValueError(
                        f"[{self.symbol}] Target schema mismatch: "
                        f"missing={missing[:5]}, extra={extra[:5]}"
                    )
                y_array = y.to_numpy(dtype=np.float64, copy=True)
                resolved_target_names = expected_target_names
            else:
                y_array = np.asarray(y)
                if y_array.ndim == 1:
                    y_array = y_array.reshape(-1, 1)
                if y_array.ndim != 2:
                    raise ValueError(
                        f"[{self.symbol}] Transform y must be 1D or 2D, "
                        f"got {y_array.shape}"
                    )
                resolved_target_names = list(
                    self.target_names
                    or [f"target_{i}" for i in range(y_array.shape[1])]
                )

            if len(X_array) != len(y_array):
                raise ValueError(
                    f"[{self.symbol}] X/y length mismatch: "
                    f"X={len(X_array)}, y={len(y_array)}"
                )

        # ── shared mask ──
        n_samples = len(X_array)
        shared_mask = np.ones(n_samples, dtype=bool)

        X_for_mask = (
            X_array.reshape(n_samples, -1)
            if X_array.ndim == 3
            else X_array
        )
        shared_mask &= np.isfinite(X_for_mask).all(axis=1)

        if (
            y_array is not None
            and resolved_target_names is not None
            and drop_rows_with_invalid
        ):
            for j, tname in enumerate(resolved_target_names):
                canonical = _canonical_column_name(tname)
                ttype = PROJECT_TARGETS_TYPE.get(
                    canonical, "regression"
                )
                y_col = y_array[:, j]
                y_finite = np.isfinite(y_col)
                if ttype == "classification":
                    shared_mask &= y_finite & (
                        y_col != float(INVALID_LABEL)
                    )
                else:
                    shared_mask &= y_finite

        n_invalid = int((~shared_mask).sum())
        if n_invalid > 0:
            logger.info(
                f"[{self.symbol}] 🧹 transform: dropped "
                f"{n_invalid}/{n_samples} invalid rows "
                f"(drop_rows_with_invalid={drop_rows_with_invalid})"
            )
            X_array = X_array[shared_mask]
            if y_array is not None:
                y_array = y_array[shared_mask]
            if len(X_array) < 1:
                raise RuntimeError(
                    f"[{self.symbol}] transform: no valid rows remain "
                    "after masking"
                )

        # ── feature scaling ──
        if input_is_dataframe:
            X_for_scale = pd.DataFrame(
                X_array, columns=self.feature_names
            )
        else:
            X_for_scale = X_array
        X_scaled = self.feature_scaler.transform(X_for_scale)

        # ── target scaling ──
        y_scaled: Optional[np.ndarray] = None
        if y_array is not None:
            if self.target_scaler is None:
                raise RuntimeError(
                    f"[{self.symbol}] y provided but target scaler "
                    "not fitted"
                )
            if y_is_dataframe:
                y_for_scale = pd.DataFrame(
                    y_array, columns=resolved_target_names
                )
            else:
                y_for_scale = y_array
            y_scaled = self.target_scaler.transform(y_for_scale)

        return X_scaled, y_scaled
        
    def inverse_transform_targets(self, y_scaled: np.ndarray) -> np.ndarray:
        if self.target_scaler is None:
            return y_scaled
        return self.target_scaler.inverse_transform(y_scaled)

    def get_target_info(self) -> Dict:
        if self.target_scaler is None:
            return {"status": "not_fitted"}
        return {
            "status": "fitted", "regression_targets": self.target_scaler.regression_targets,
            "classification_targets": self.target_scaler.classification_targets,
            "all_targets": self.target_names, "n_regression": len(self.target_scaler.regression_targets),
            "n_classification": len(self.target_scaler.classification_targets),
        }

    def save(self):
        if self.feature_scaler:
            self.feature_scaler.save()
        if self.target_scaler:
            self.target_scaler.save()
        # ✅ FIX: Save pipeline metadata for coordination
        meta_path = Path(f"data/pipelines/{self.symbol}/{self.timeframe}/fold_{self.fold_num}")
        meta_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "symbol": self.symbol, "timeframe": self.timeframe, "fold_num": self.fold_num,
            "feature_names": self.feature_names, "target_names": self.target_names,
            "fitted": self.fitted, "saved_at": datetime.now().isoformat()
        }
        with open(meta_path / "pipeline_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(
        cls,
        symbol: str,
        timeframe: str,
        fold_num: int,
    ) -> "SmartScalingPipeline":
        """
        بارگذاری کامل Feature و Target scaler با سازگاری auxiliary.

        ✅ v2.1 FIX (B4): dedupe allowed_targets برای جلوگیری از تکرار
           magnitude_bucket / momentum_direction که در هر دو
           PROJECT_TARGETS و AUXILIARY_COLUMNS ظاهر می‌شوند.

        ✅ v2.0: تنها اگر Feature scaler و Target scaler هر دو سالم و
           fitted باشند، pipeline.fitted = True می‌شود.
        """
        try:
            auxiliary_columns: List[str] = list(AUXILIARY_COLUMNS)
        except NameError:
            auxiliary_columns = [
                "sample_weight", "magnitude_bucket", "momentum_direction",
            ]

        # ✅ FIX B4: dedupe
        allowed_targets: List[str] = list(
            dict.fromkeys(list(PROJECT_TARGETS) + list(auxiliary_columns))
        )

        pipeline = cls(
            symbol=symbol,
            timeframe=timeframe,
            fold_num=fold_num,
        )

        feature_scaler = SmartAdaptiveScaler.load(
            symbol=symbol, timeframe=timeframe, fold_num=fold_num,
            is_target=False,
        )
        if not feature_scaler.fitted or not feature_scaler.feature_names:
            logger.warning(
                f"[{symbol}] ⚠️ Feature scaler missing/invalid — "
                f"pipeline not fitted"
            )
            pipeline.fitted = False
            return pipeline

        target_scaler = SmartAdaptiveScaler.load(
            symbol=symbol, timeframe=timeframe, fold_num=fold_num,
            is_target=True,
        )
        if not target_scaler.fitted or not target_scaler.feature_names:
            logger.warning(
                f"[{symbol}] ⚠️ Target scaler missing/invalid — "
                f"pipeline not fitted"
            )
            pipeline.fitted = False
            return pipeline

        actual_targets: List[str] = list(target_scaler.feature_names)

        # (1) تمام PROJECT_TARGETS باید موجود باشند
        missing_required = [
            name for name in PROJECT_TARGETS if name not in actual_targets
        ]
        if missing_required:
            logger.error(
                f"[{symbol}] ❌ Target scaler missing PROJECT_TARGETS: "
                f"{missing_required}"
            )
            pipeline.fitted = False
            return pipeline

        # (2) هیچ ستون ناشناخته‌ای نباید باشد
        unknown_targets = [
            name for name in actual_targets if name not in allowed_targets
        ]
        if unknown_targets:
            logger.error(
                f"[{symbol}] ❌ Target scaler contains unknown columns: "
                f"{unknown_targets}. Allowed = PROJECT_TARGETS ∪ AUXILIARY_COLUMNS"
            )
            pipeline.fitted = False
            return pipeline

        # (3) ✅ FIX B4: ترتیب مورد انتظار هم بدون تکرار
        expected_order = [
            name for name in allowed_targets if name in actual_targets
        ]

        if actual_targets != expected_order:
            logger.error(
                f"[{symbol}] ❌ Target scaler schema order mismatch: "
                f"expected={expected_order}, actual={actual_targets}"
            )
            pipeline.fitted = False
            return pipeline

        pipeline.feature_scaler = feature_scaler
        pipeline.target_scaler = target_scaler
        pipeline.feature_names = list(feature_scaler.feature_names)
        pipeline.target_names = list(target_scaler.feature_names)
        pipeline.fitted = True

        aux_present = [
            t for t in actual_targets if t in auxiliary_columns
        ]
        logger.info(
            f"[{symbol}] ✅ Scaling pipeline loaded: "
            f"features={len(pipeline.feature_names)}, "
            f"targets={len(pipeline.target_names)} "
            f"(auxiliary-aware: {aux_present})"
        )
        return pipeline
        
# ============================================================================
# DATA SPLIT NORMALIZER - نسخه هماهنگ شده
# ============================================================================

class DataSplitNormalizer:
    """
    ✅ WALL STREET GRADE - DataSplitNormalizer با هماهنگی کامل
    ✅ SINGLE SOURCE OF TRUTH: از PROJECT_TARGETS استفاده می‌کند
    """
    
    def __init__(self, symbol: str, timeframe: str, fold_num: int):
        self.symbol = symbol
        self.timeframe = timeframe
        self.fold_num = fold_num
        self.pipeline: Optional[SmartScalingPipeline] = None
        self._is_fitted = False
        # ✅ FIX: استفاده از PROJECT_TARGETS به عنوان single source of truth
        self.target_cols = PROJECT_TARGETS

    def get_pipeline_path(self) -> Path:
        """✅ FIX: مسیر یکسان برای ذخیره و بارگذاری pipeline"""
        return Path(f"data/pipelines/{self.symbol}/{self.timeframe}/fold_{self.fold_num}")

    def fit_transform_splits(
        self,
        X_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        y_train: pd.DataFrame,
        y_val: Optional[pd.DataFrame],
        y_test: Optional[pd.DataFrame],
        target_cols: List[str] = None,
        drop_rows_with_invalid: bool = False,
    ) -> Tuple:
        """
        Fit + Transform روی TRAIN.

        ✅ تغییر نسبت به نسخهٔ قبل:
            - پارامتر جدید `drop_rows_with_invalid` (پیش‌فرض False).
            - True : رفتار قبلی (حذف ردیف‌های invalid target).
            - False: ردیف‌های invalid target حفظ می‌شوند و صرفاً
                     مقدار آن‌ها NaN (regression) یا -1 (classification)
                     می‌شود.
        """
        if target_cols is None:
            target_cols = self.target_cols

        if X_train is None or X_train.empty:
            raise ValueError(f"[{self.symbol}] X_train is empty")
        if y_train is None or y_train.empty:
            raise ValueError(f"[{self.symbol}] y_train is empty")

        feature_cols = [
            c for c in X_train.columns if c not in target_cols
        ]
        if not feature_cols:
            raise ValueError(
                f"[{self.symbol}] No feature columns in X_train"
            )

        logger.info(
            f"[{self.symbol}] 🧠 fit_transform_splits | "
            f"X_train={X_train.shape} y_train={y_train.shape} "
            f"features={len(feature_cols)} | "
            f"drop_rows_with_invalid={drop_rows_with_invalid}"
        )

        self.pipeline = SmartScalingPipeline(
            symbol=self.symbol,
            timeframe=self.timeframe,
            fold_num=self.fold_num,
        )

        # ─── helper ───
        def _strict_align_and_mask(
            X_part: pd.DataFrame,
            y_part: pd.DataFrame,
            split_label: str,
        ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
            if X_part is None or X_part.empty:
                raise ValueError(
                    f"[{self.symbol}] {split_label} X is empty"
                )
            if y_part is None or y_part.empty:
                raise ValueError(
                    f"[{self.symbol}] {split_label} y is empty"
                )

            missing_feats = [
                c for c in feature_cols if c not in X_part.columns
            ]
            if missing_feats:
                raise RuntimeError(
                    f"[{self.symbol}] {split_label} missing "
                    f"{len(missing_feats)} feature columns that exist "
                    f"in TRAIN: {missing_feats[:10]}"
                )

            missing_targets = [
                c for c in target_cols if c not in y_part.columns
            ]
            if missing_targets:
                raise RuntimeError(
                    f"[{self.symbol}] {split_label} missing target "
                    f"columns: {missing_targets}"
                )

            X_aligned = X_part[feature_cols].copy()
            y_aligned = y_part[target_cols].copy()

            if len(X_aligned) != len(y_aligned):
                raise ValueError(
                    f"[{self.symbol}] {split_label} X/y length mismatch: "
                    f"X={len(X_aligned)}, y={len(y_aligned)}"
                )

            # ── feature mask (همیشه) ──
            mask = np.ones(len(X_aligned), dtype=bool)
            X_arr = X_aligned.to_numpy(dtype=np.float64, copy=False)
            mask &= np.isfinite(X_arr).all(axis=1)

            # ── target mask (فقط اگر drop_rows_with_invalid=True) ──
            if drop_rows_with_invalid:
                for tname in target_cols:
                    canonical = _canonical_column_name(tname)
                    ttype = PROJECT_TARGETS_TYPE.get(
                        canonical, "regression"
                    )
                    y_col = y_aligned[tname].to_numpy(
                        dtype=np.float64, copy=False
                    )
                    y_finite = np.isfinite(y_col)
                    if ttype == "classification":
                        mask &= y_finite & (
                            y_col != float(INVALID_LABEL)
                        )
                    else:
                        mask &= y_finite

            n_dropped = int((~mask).sum())
            if n_dropped > 0:
                logger.info(
                    f"[{self.symbol}] 🧹 {split_label}: dropped "
                    f"{n_dropped} invalid rows "
                    f"(drop_rows_with_invalid={drop_rows_with_invalid})"
                )

            if int(mask.sum()) < 10:
                raise RuntimeError(
                    f"[{self.symbol}] {split_label}: after masking "
                    f"only {int(mask.sum())} valid samples remain"
                )

            return (
                X_aligned.loc[mask].reset_index(drop=True),
                y_aligned.loc[mask].reset_index(drop=True),
                mask,
            )

        # ── TRAIN ──
        X_train_aligned, y_train_aligned, _ = _strict_align_and_mask(
            X_train, y_train, "TRAIN"
        )

        logger.info(
            f"[{self.symbol}] 📊 Fitting on TRAIN: "
            f"X={X_train_aligned.shape}, y={y_train_aligned.shape}"
        )

        X_train_scaled, y_train_scaled = self.pipeline.fit_transform(
            X_train=X_train_aligned,
            y_train=y_train_aligned,
            feature_names=feature_cols,
            target_names=target_cols,
            drop_rows_with_invalid=drop_rows_with_invalid,
        )

        self.pipeline.save()
        logger.info(
            f"[{self.symbol}] 💾 Pipeline saved to "
            f"{self.get_pipeline_path()}"
        )

        # ── VAL ──
        X_val_scaled = None
        y_val_scaled = None
        if (
            X_val is not None and not X_val.empty
            and y_val is not None and not y_val.empty
        ):
            X_val_aligned, y_val_aligned, _ = _strict_align_and_mask(
                X_val, y_val, "VAL"
            )
            logger.info(
                f"[{self.symbol}] 🔄 Transforming VAL: "
                f"X={X_val_aligned.shape}"
            )
            X_val_scaled, y_val_scaled = self.pipeline.transform(
                X_val_aligned,
                y_val_aligned,
                drop_rows_with_invalid=drop_rows_with_invalid,
            )

        # ── TEST ──
        X_test_scaled = None
        y_test_scaled = None
        if (
            X_test is not None and not X_test.empty
            and y_test is not None and not y_test.empty
        ):
            X_test_aligned, y_test_aligned, _ = _strict_align_and_mask(
                X_test, y_test, "TEST"
            )
            logger.info(
                f"[{self.symbol}] 🔄 Transforming TEST: "
                f"X={X_test_aligned.shape}"
            )
            X_test_scaled, y_test_scaled = self.pipeline.transform(
                X_test_aligned,
                y_test_aligned,
                drop_rows_with_invalid=drop_rows_with_invalid,
            )

        self._is_fitted = True
        return (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            y_train_scaled,
            y_val_scaled,
            y_test_scaled,
        )
        

    def transform_splits(
        self,
        X_val: Optional[pd.DataFrame],
        X_test: Optional[pd.DataFrame],
        y_val: Optional[pd.DataFrame],
        y_test: Optional[pd.DataFrame],
        target_cols: List[str] = None,
        drop_rows_with_invalid: bool = False,
    ) -> Tuple:
        """
        Transform VAL/TEST با pipeline ذخیره‌شده.

        ✅ تغییر نسبت به نسخهٔ قبل:
            - پارامتر جدید `drop_rows_with_invalid` (پیش‌فرض False).
        """
        if target_cols is None:
            target_cols = self.target_cols

        pipeline_path = self.get_pipeline_path()
        pipeline_meta_path = pipeline_path / "pipeline_meta.json"

        if not pipeline_meta_path.exists():
            raise RuntimeError(
                f"[{self.symbol}] No trained pipeline found for fold "
                f"{self.fold_num}! Run fit_transform_splits() on TRAIN first."
            )

        pipeline = SmartScalingPipeline.load(
            self.symbol, self.timeframe, self.fold_num
        )
        if not pipeline.fitted or pipeline.feature_scaler is None:
            raise RuntimeError(
                f"[{self.symbol}] CRITICAL: Pipeline not properly fitted"
            )

        trained_feature_names = (
            pipeline.feature_names
            or pipeline.feature_scaler.feature_names
        )
        trained_target_names = pipeline.target_names or target_cols

        if not trained_feature_names:
            raise RuntimeError(
                f"[{self.symbol}] Trained feature names are missing"
            )

        def _strict_align_and_mask(
            X_part: Optional[pd.DataFrame],
            y_part: Optional[pd.DataFrame],
            split_label: str,
        ):
            if X_part is None or X_part.empty:
                return None, None

            missing_feats = [
                c for c in trained_feature_names
                if c not in X_part.columns
            ]
            if missing_feats:
                raise RuntimeError(
                    f"[{self.symbol}] {split_label} missing "
                    f"{len(missing_feats)} trained feature columns: "
                    f"{missing_feats[:10]}"
                )

            X_aligned = X_part[trained_feature_names].copy()

            if y_part is not None and not y_part.empty:
                missing_targets = [
                    c for c in trained_target_names
                    if c not in y_part.columns
                ]
                if missing_targets:
                    raise RuntimeError(
                        f"[{self.symbol}] {split_label} missing "
                        f"target columns: {missing_targets}"
                    )
                y_aligned = y_part[trained_target_names].copy()

                if len(X_aligned) != len(y_aligned):
                    raise ValueError(
                        f"[{self.symbol}] {split_label} X/y length "
                        f"mismatch: X={len(X_aligned)}, "
                        f"y={len(y_aligned)}"
                    )
            else:
                y_aligned = pd.DataFrame(index=X_aligned.index)

            mask = np.ones(len(X_aligned), dtype=bool)
            X_arr = X_aligned.to_numpy(dtype=np.float64, copy=False)
            mask &= np.isfinite(X_arr).all(axis=1)

            if not y_aligned.empty and drop_rows_with_invalid:
                for tname in trained_target_names:
                    canonical = _canonical_column_name(tname)
                    ttype = PROJECT_TARGETS_TYPE.get(
                        canonical, "regression"
                    )
                    y_col = y_aligned[tname].to_numpy(
                        dtype=np.float64, copy=False
                    )
                    y_finite = np.isfinite(y_col)
                    if ttype == "classification":
                        mask &= y_finite & (
                            y_col != float(INVALID_LABEL)
                        )
                    else:
                        mask &= y_finite

            n_dropped = int((~mask).sum())
            if n_dropped > 0:
                logger.info(
                    f"[{self.symbol}] 🧹 {split_label}: dropped "
                    f"{n_dropped} invalid rows "
                    f"(drop_rows_with_invalid={drop_rows_with_invalid})"
                )

            return (
                X_aligned.loc[mask].reset_index(drop=True),
                (
                    y_aligned.loc[mask].reset_index(drop=True)
                    if not y_aligned.empty
                    else pd.DataFrame()
                ),
            )

        # ── VAL ──
        X_val_scaled = None
        y_val_scaled = None
        if X_val is not None and not X_val.empty:
            X_val_aligned, y_val_aligned = _strict_align_and_mask(
                X_val, y_val, "VAL"
            )
            if X_val_aligned is not None and len(X_val_aligned) > 0:
                X_val_np, y_val_np = pipeline.transform(
                    X_val_aligned,
                    y_val_aligned if not y_val_aligned.empty else None,
                    drop_rows_with_invalid=drop_rows_with_invalid,
                )
                X_val_scaled = pd.DataFrame(
                    X_val_np,
                    columns=trained_feature_names,
                    index=X_val_aligned.index,
                )
                if y_val_np is not None:
                    n_t = min(
                        y_val_np.shape[1], len(trained_target_names)
                    )
                    y_val_scaled = pd.DataFrame(
                        y_val_np,
                        columns=trained_target_names[:n_t],
                        index=y_val_aligned.index,
                    )

        # ── TEST ──
        X_test_scaled = None
        y_test_scaled = None
        if X_test is not None and not X_test.empty:
            X_test_aligned, y_test_aligned = _strict_align_and_mask(
                X_test, y_test, "TEST"
            )
            if X_test_aligned is not None and len(X_test_aligned) > 0:
                X_test_np, y_test_np = pipeline.transform(
                    X_test_aligned,
                    y_test_aligned if not y_test_aligned.empty else None,
                    drop_rows_with_invalid=drop_rows_with_invalid,
                )
                X_test_scaled = pd.DataFrame(
                    X_test_np,
                    columns=trained_feature_names,
                    index=X_test_aligned.index,
                )
                if y_test_np is not None:
                    n_t = min(
                        y_test_np.shape[1], len(trained_target_names)
                    )
                    y_test_scaled = pd.DataFrame(
                        y_test_np,
                        columns=trained_target_names[:n_t],
                        index=y_test_aligned.index,
                    )

        return (
            None,
            X_val_scaled,
            X_test_scaled,
            None,
            y_val_scaled,
            y_test_scaled,
        )
        

class SmartFeatureSanitizer:
    """
    Sanitizer V5 — Forensic Audit Compliant + Exact/Near-Exact Deduplication
    ────────────────────────────────────────────────────────────────────────
    تغییرات نسبت به V4:

        ✅ رفع باگ duplicate features (خطای «Orthogonalization failed ...»):
           قبل از orthogonalization، ستون‌هایی که corr >= dedup_corr_threshold
           (پیش‌فرض 0.9999) دارند، حذف می‌شوند. این ستون‌ها معمولاً duplicate
           ریاضی هستند (مثل stoch_k_dev و williams_r_dev) و orthogonalization
           به‌دلیل منفردشدن covariance matrix نمی‌تواند آن‌ها را جدا کند.

        ✅ حفظ کامل Public API نسخهٔ V4:
           - __init__(corr_threshold, var_threshold, max_features,
                      essential_features, verbose, orthogonalize,
                      orth_corr_threshold, symbol, strict_residual,
                      max_residual_corr)
           - fit(X, y=None)
           - transform(X)
           - fit_transform(X, y=None)
           - صفات عمومی: keep_cols، all_original_cols، feature_scores، fitted

        ✅ پارامتر جدید اختیاری:
           - dedup_corr_threshold (float, default=0.9999):
             آستانهٔ corr برای شناسایی duplicate ها.

        ✅ گزارش جدید:
           - self.dedup_dropped_pairs: لیست (kept, dropped, corr)
    """

    MANDATORY_MT5: List[str] = []

    def __init__(
        self,
        corr_threshold: float = 0.80,
        var_threshold: float = 1e-10,
        max_features: int = 500,
        essential_features: Optional[List[str]] = None,
        verbose: bool = True,
        orthogonalize: bool = True,
        orth_corr_threshold: float = 0.75,
        symbol: str = "UNKNOWN",
        strict_residual: bool = True,
        max_residual_corr: float = 0.90,
        dedup_corr_threshold: float = 0.9999,
    ) -> None:
        self.corr_threshold = float(corr_threshold)
        self.var_threshold = float(var_threshold)
        self.max_features = int(max_features)
        self.verbose = bool(verbose)
        self.orthogonalize = bool(orthogonalize)
        self.orth_corr_threshold = float(orth_corr_threshold)
        self.symbol = symbol
        self.strict_residual = bool(strict_residual)
        self.max_residual_corr = float(max_residual_corr)
        self.dedup_corr_threshold = float(dedup_corr_threshold)

        # ── اعتبارسنجی پارامترها ──
        if not (0.0 < self.dedup_corr_threshold <= 1.0):
            raise ValueError(
                f"dedup_corr_threshold must be in (0, 1], "
                f"got {self.dedup_corr_threshold}"
            )
        if self.dedup_corr_threshold <= self.corr_threshold:
            raise ValueError(
                f"dedup_corr_threshold ({self.dedup_corr_threshold}) must be "
                f"> corr_threshold ({self.corr_threshold}); otherwise exact "
                f"duplicates would not be removed before orthogonalization."
            )

        self.user_essentials = list(essential_features or [])
        self.essential_features = list(
            set(self.MANDATORY_MT5) | set(self.user_essentials)
        )

        self.keep_cols: Optional[List[str]] = None
        self.all_original_cols: Optional[List[str]] = None
        self.feature_scores: Optional[Dict[str, float]] = None
        self.fitted: bool = False

        self._orthogonalizer: Optional[FeatureOrthogonalizer] = None
        self._residual_corr_pairs: List[Tuple[str, str, float]] = []
        self.dedup_dropped_pairs: List[Tuple[str, str, float]] = []

    # ══════════════════════════════════════════════════════════════════
    # PHASE 0 (NEW): حذف duplicate های دقیق/نزدیک-به-دقیق
    # ══════════════════════════════════════════════════════════════════
    def _deduplicate_near_exact(
        self,
        X: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        حذف ستون‌های duplicate با corr >= dedup_corr_threshold.

        قواعد:
            - ستون‌های essential (MANDATORY_MT5 + user_essentials) هرگز
              حذف نمی‌شوند مگر خودشان duplicateِ یک essential دیگر باشند.
            - بین دو ستون duplicate، آن که واریانس بالاتری دارد حفظ می‌شود
              (اطلاعات بیشتر، نویز کمتر).
            - اگر هر دو essential باشند، هیچ‌کدام حذف نمی‌شوند (کاربر
              صریحاً خواسته آن‌ها بمانند).

        Args:
            X: DataFrame ورودی (بدون ستون‌های target).

        Returns:
            (X_dedup, dropped_pairs)
            - X_dedup: DataFrame بدون ستون‌های duplicate.
            - dropped_pairs: لیست (kept_name, dropped_name, corr_value).
        """
        cols = list(X.columns)
        n = len(cols)
        if n < 2:
            return X, []

        arr = X.to_numpy(dtype=np.float64, copy=False)

        with np.errstate(invalid="ignore", divide="ignore"):
            variances = np.nanvar(arr, axis=0)
        variances = np.where(np.isfinite(variances), variances, 0.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(arr, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        essential_mask = np.array(
            [c in self.essential_features for c in cols], dtype=bool
        )

        # ترتیب نزولی واریانس: بالاترین واریانس اول (بمانَد)
        order = np.argsort(-variances)

        keep = np.ones(n, dtype=bool)
        dropped: List[Tuple[str, str, float]] = []

        for i_pos in range(n):
            i = int(order[i_pos])
            if not keep[i]:
                continue
            for j_pos in range(i_pos + 1, n):
                j = int(order[j_pos])
                if not keep[j]:
                    continue
                if corr[i, j] < self.dedup_corr_threshold:
                    continue

                # هر دو essential → دست نزن
                if essential_mask[i] and essential_mask[j]:
                    continue

                # i essential → j حذف
                if essential_mask[i] and not essential_mask[j]:
                    keep[j] = False
                    dropped.append((cols[i], cols[j], float(corr[i, j])))
                    continue

                # j essential، i غیر-essential → i حذف، j بمانَد
                if essential_mask[j] and not essential_mask[i]:
                    keep[i] = False
                    dropped.append((cols[j], cols[i], float(corr[i, j])))
                    break

                # هیچ‌کدام essential → نگه‌داشتن i (واریانس بالاتر)
                keep[j] = False
                dropped.append((cols[i], cols[j], float(corr[i, j])))

        if not np.all(keep):
            keep_cols = [cols[i] for i in range(n) if keep[i]]
            X_out = X.loc[:, keep_cols].copy()
        else:
            X_out = X

        return X_out, dropped

    # ══════════════════════════════════════════════════════════════════
    # پیدا کردن جفت‌های residual پس از orthogonalization
    # ══════════════════════════════════════════════════════════════════
    def _find_high_residual_pairs(
        self,
        X: np.ndarray,
        cols: List[str],
        threshold: float,
    ) -> List[Tuple[str, str, float]]:
        n = X.shape[1]
        if n < 2:
            return []

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        pairs: List[Tuple[str, str, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if corr[i, j] > threshold:
                    pairs.append((cols[i], cols[j], float(corr[i, j])))
        pairs.sort(key=lambda t: -t[2])
        return pairs

    # ══════════════════════════════════════════════════════════════════
    # fit
    # ══════════════════════════════════════════════════════════════════
    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "SmartFeatureSanitizer":
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"[{self.symbol}] X must be DataFrame, got {type(X).__name__}"
            )
        if X.empty:
            raise ValueError(f"[{self.symbol}] X is empty")
        if X.columns.duplicated().any():
            raise ValueError(
                f"[{self.symbol}] duplicate columns not allowed in X"
            )

        self.all_original_cols = list(X.columns)

        # ── PHASE 0: dedup ──
        X_dedup, dedup_dropped = self._deduplicate_near_exact(X)
        self.dedup_dropped_pairs = list(dedup_dropped)

        if self.verbose and dedup_dropped:
            logger.warning(
                f"[{self.symbol}] 🧹 Deduplication: removed "
                f"{len(dedup_dropped)} duplicate column(s) "
                f"(corr ≥ {self.dedup_corr_threshold:.4f}):"
            )
            for a, b, c in dedup_dropped[:25]:
                logger.warning(
                    f"[{self.symbol}]    🗑️ '{b}' removed "
                    f"(duplicate of '{a}', corr={c:.6f})"
                )
            if len(dedup_dropped) > 25:
                logger.warning(
                    f"[{self.symbol}]    ... and "
                    f"{len(dedup_dropped) - 25} more"
                )
        elif self.verbose:
            logger.info(
                f"[{self.symbol}] 🧹 Deduplication: no exact/near-exact "
                f"duplicates found (threshold={self.dedup_corr_threshold})"
            )

        # ── PHASE 1: orthogonalization روی X_dedup ──
        # Orthogonalizer تلاش می‌کند residual را زیر max_residual_corr بیاورد،
        # ولی اگر نتوانست، اینجا raise نمی‌کند. در عوض، residual محاسبه شده و
        # در PHASE 2 (حذف جفت‌های باقی‌مانده) استفاده می‌شود.
        if self.orthogonalize and X_dedup.shape[1] >= 2:
            try:
                target_residual = min(
                    self.corr_threshold,
                    self.max_residual_corr * 0.95,
                )
                self._orthogonalizer = FeatureOrthogonalizer(
                    corr_threshold=self.orth_corr_threshold,
                    min_group_size=2,
                    symbol=self.symbol,
                    target_residual=target_residual,
                    max_merge_iterations=15,
                )
                self._orthogonalizer.fit(X_dedup)

                X_orth = self._orthogonalizer.transform(X_dedup)
                residual = self._orthogonalizer.residual_max_corr(X_dedup)

                arr_check = X_orth.to_numpy(dtype=np.float64, copy=False)
                self._residual_corr_pairs = self._find_high_residual_pairs(
                    arr_check,
                    cols=list(X_orth.columns),
                    threshold=self.corr_threshold,
                )

                if self.verbose:
                    logger.info(
                        f"[{self.symbol}] 🧬 Orthogonalization: "
                        f"max residual corr = {residual:.4f} "
                        f"(soft-threshold={target_residual:.4f}, "
                        f"hard-threshold={self.max_residual_corr:.4f})"
                    )
                    if self._residual_corr_pairs:
                        logger.warning(
                            f"[{self.symbol}] ⚠️ "
                            f"{len(self._residual_corr_pairs)} residual "
                            f"high-corr pair(s) detected after "
                            f"orthogonalization:"
                        )
                        for a, b, c in self._residual_corr_pairs[:10]:
                            logger.warning(
                                f"[{self.symbol}]    🔥 '{a}' <-> '{b}' "
                                f"= {c:.4f}"
                            )

                # ✅ تغییر رفتاری: به‌جای raise فوری، به PHASE 2
                # اجازه بده جفت‌های باقی‌مانده را drop کند.
                # اگر بعد از drop، residual همچنان بالا بماند، در انتهای
                # این تابع raise می‌شود (fail-safe).
                if (
                    self.strict_residual
                    and residual > self.max_residual_corr
                ):
                    logger.warning(
                        f"[{self.symbol}] ⚠️ Residual correlation "
                        f"({residual:.4f}) exceeds hard threshold "
                        f"({self.max_residual_corr:.4f}). Deferring to "
                        f"residual-drop phase — {len(self._residual_corr_pairs)} "
                        f"pair(s) will be removed."
                    )

            except Exception as e:
                logger.error(
                    f"[{self.symbol}] ❌ Orthogonalization failed: {e} | "
                    f"falling back to direct filter"
                )
                import traceback
                logger.debug(traceback.format_exc())
                self._orthogonalizer = None
                X_orth = X_dedup
                residual = 0.0
        else:
            X_orth = X_dedup
            residual = 0.0

        # ── PHASE 2: safety-net filters + drop باقی‌مانده ──
        arr, cols = self._to_float64(X_orth)

        missing = [f for f in self.MANDATORY_MT5 if f not in cols]
        if missing:
            raise ValueError(
                f"[{self.symbol}] Missing mandatory MT5 features: {missing}"
            )

        var_mask = self._variance_mask(arr)
        ess_mask = np.array(
            [c in self.essential_features for c in cols],
            dtype=bool,
        )
        combined = ess_mask | var_mask
        arr2 = arr[:, combined]
        cols2 = [cols[i] for i in range(len(cols)) if combined[i]]

        scores = self._relevance_scores(arr2, y)
        self.feature_scores = dict(zip(cols2, scores.tolist()))

        corr_mask = self._drop_high_corr_residual(arr2, scores, cols2)
        for i, c in enumerate(cols2):
            if c in self.essential_features:
                corr_mask[i] = True

        arr3 = arr2[:, corr_mask]
        cols3 = [cols2[i] for i in range(len(cols2)) if corr_mask[i]]
        scores3 = scores[corr_mask]

        # ✅ fail-safe نهایی: اگر strict_residual=True و بعد از drop
        # هنوز residual بالاست، raise کن.
        if self.strict_residual and arr3.shape[1] >= 2:
            with np.errstate(invalid="ignore", divide="ignore"):
                post_corr = np.abs(np.corrcoef(arr3, rowvar=False))
            post_corr = np.nan_to_num(
                post_corr, nan=0.0, posinf=0.0, neginf=0.0
            )
            np.fill_diagonal(post_corr, 0.0)
            post_residual = float(post_corr.max()) if post_corr.size else 0.0

            if post_residual > self.max_residual_corr:
                flat_idx = int(np.argmax(post_corr))
                pi, pj = divmod(flat_idx, post_corr.shape[1])
                raise RuntimeError(
                    f"[{self.symbol}] ❌ Residual correlation remains "
                    f"above {self.max_residual_corr:.4f} even after "
                    f"residual-drop phase. Max residual = "
                    f"{post_residual:.4f}. Worst pair: "
                    f"'{cols3[pi]}' <-> '{cols3[pj]}'. "
                    f"Consider lowering `dedup_corr_threshold` or "
                    f"reviewing feature engineering."
                )

            if self.verbose:
                logger.info(
                    f"[{self.symbol}] ✅ Post-drop residual corr = "
                    f"{post_residual:.4f} (≤ {self.max_residual_corr:.4f})"
                )

        if self.max_features and len(cols3) > self.max_features:
            sel = self._top_k_with_essentials(
                scores3, cols3, self.max_features
            )
            self.keep_cols = [cols3[i] for i in sel]
        else:
            self.keep_cols = list(cols3)

        self.fitted = True

        if self.verbose:
            n0 = len(self.all_original_cols)
            n1 = len(self.keep_cols)
            logger.info(
                f"[{self.symbol}] 🔧 Sanitizer fit: {n0} → {n1} features "
                f"({100 * (1 - n1 / max(n0, 1)):.1f}% reduction)"
            )
        return self
        
    # ══════════════════════════════════════════════════════════════════
    # transform
    # ══════════════════════════════════════════════════════════════════
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        اعمال sanitization با تحمل به ترتیب ستون‌ها (ORDER-AGNOSTIC).

        ✅ v5.2 (اصلاح باگ schema-order که در لاگ دیده شد):
            نسخهٔ قبلی:
                if feature_cols_in_input != expected:
            مقایسهٔ لیستی (order-sensitive) بود. اما PHASE 2.5 در
            `_process_single_split_with_enhancement` ستون‌های frozen-fundamental
            گمشده را با مقدار 0.0 به انتهای DataFrame اضافه می‌کند (مثلاً
            `fear_greed_extreme` در split=test که به‌دلیل constant بودن در
            FundamentalFeatureBuilder افتاده بود). در نتیجه SET ستون‌ها
            برابر بود ولی ORDER متفاوت → خطای گمراه‌کنندهٔ:
                ValueError: schema mismatch: missing=[], extra=[]

            رفع: مقایسه بر اساس SET انجام می‌شود. سپس `X.loc[:, expected]`
            ستون‌ها را به ترتیب فریز شدهٔ TRAIN بازچینی می‌کند.

        قرارداد:
            - ستون‌های target در X مجاز و حذف می‌شوند.
            - ستون‌های feature باید دقیقاً SET مورد انتظار TRAIN را داشته
              باشند (نه کمتر، نه بیشتر).
            - ترتیب داخلی هرچند متفاوت باشد، به ترتیب TRAIN بازچینی می‌شود.

        Raises:
            RuntimeError: اگر fit نشده باشد یا keep_cols ناسازگار باشد.
            TypeError: اگر X DataFrame نباشد.
            ValueError: اگر duplicate ستون، SET ناسازگار، یا NaN/Inf در
                        خروجی باشد.
        """
        if not self.fitted or self.keep_cols is None:
            raise RuntimeError(
                f"[{self.symbol}] Call fit on TRAIN first"
            )
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"[{self.symbol}] transform expects DataFrame, "
                f"got {type(X).__name__}"
            )
        if X.columns.duplicated().any():
            raise ValueError(
                f"[{self.symbol}] duplicate columns not allowed"
            )

        actual = list(X.columns)
        expected = list(self.all_original_cols)

        target_cols_in_input: List[str] = [
            c for c in actual if c in PROJECT_TARGETS
        ]
        feature_cols_in_input: List[str] = [
            c for c in actual if c not in PROJECT_TARGETS
        ]

        # ── ✅ ORDER-AGNOSTIC schema validation ──
        expected_set = set(expected)
        input_set = set(feature_cols_in_input)

        if input_set != expected_set:
            missing = sorted(expected_set - input_set)
            extra = sorted(input_set - expected_set)
            raise ValueError(
                f"[{self.symbol}] schema mismatch: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        if len(feature_cols_in_input) != len(expected):
            # این حالت نباید رخ دهد چون set برابر است و طول list باید
            # برابر باشد، مگر duplicate در X (که قبلاً چک شد).
            raise ValueError(
                f"[{self.symbol}] schema length mismatch: "
                f"input={len(feature_cols_in_input)} vs "
                f"expected={len(expected)} — indicates duplicate "
                f"feature names after canonicalization."
            )

        # ── ✅ بازچینی ستون‌ها به ترتیب فریز شدهٔ TRAIN ──
        # (ضروری چون PHASE 2.5 ممکن است ستون‌های frozen گمشده را در
        #  انتهای DataFrame اضافه کرده باشد.)
        feature_block = X.loc[:, expected]

        # ── APPLY SAME DEDUP AS FIT ──
        if self.dedup_dropped_pairs:
            drop_cols = [b for _a, b, _c in self.dedup_dropped_pairs]
            drop_cols = [
                c for c in drop_cols if c in feature_block.columns
            ]
            if drop_cols:
                feature_block = feature_block.drop(columns=drop_cols)

        # ── APPLY SAME ORTHOGONALIZER AS FIT ──
        if self._orthogonalizer is not None:
            feature_block = self._orthogonalizer.transform(feature_block)

        # ── SELECT SAME keep_cols ──
        missing_keep = [
            c for c in self.keep_cols if c not in feature_block.columns
        ]
        if missing_keep:
            raise RuntimeError(
                f"[{self.symbol}] transform: keep_cols missing after "
                f"dedup/orthogonalization: {missing_keep[:10]}"
            )

        out = feature_block.loc[:, self.keep_cols].copy()

        arr = out.to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(arr).all():
            bad_cols = [
                out.columns[i]
                for i in range(arr.shape[1])
                if not np.isfinite(arr[:, i]).all()
            ]
            raise ValueError(
                f"[{self.symbol}] transform contains NaN/Inf in feature "
                f"columns: {bad_cols[:10]}"
            )

        out = out.astype(np.float32)

        if target_cols_in_input:
            target_block = X.loc[:, target_cols_in_input]
            out = pd.concat([out, target_block], axis=1)

        return out
    # ══════════════════════════════════════════════════════════════════
    # remaining helpers (بدون تغییر نسبت به V4)
    # ══════════════════════════════════════════════════════════════════
    def _variance_mask(self, X: np.ndarray) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            stds = np.nanstd(X, axis=0)
        return stds > self.var_threshold

    def _relevance_scores(
        self, X: np.ndarray, y: Any = None
    ) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            variances = np.nanvar(X, axis=0)
        variances = np.where(np.isfinite(variances), variances, 0.0)
        total = float(variances.sum()) + 1e-12
        return (variances / total).astype(np.float32)

    def _drop_high_corr_residual(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        cols: List[str],
    ) -> np.ndarray:
        n = X.shape[1]
        keep = np.ones(n, dtype=bool)
        if n < 2:
            return keep

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        protected = np.array(
            [c in self.essential_features for c in cols],
            dtype=bool,
        )

        for i in range(n):
            if not keep[i]:
                continue
            candidates = np.where(corr[i] > self.corr_threshold)[0]
            for j in candidates:
                if i == j or not keep[j]:
                    continue
                if protected[i] and protected[j]:
                    continue
                if protected[i]:
                    keep[j] = False
                elif protected[j]:
                    keep[i] = False
                    break
                elif scores[i] >= scores[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
        return keep

    def _top_k_with_essentials(
        self,
        scores: np.ndarray,
        cols: List[str],
        k: int,
    ) -> List[int]:
        essential_idx = [
            i
            for i, c in enumerate(cols)
            if c in self.essential_features
        ]
        non_essential = [
            i for i in range(len(cols)) if i not in essential_idx
        ]
        need = max(0, k - len(essential_idx))
        if need == 0 or not non_essential:
            return essential_idx[:k]
        if len(non_essential) <= need:
            return essential_idx + non_essential
        sub_scores = scores[non_essential]
        top_local = np.argpartition(sub_scores, -need)[-need:]
        return essential_idx + [non_essential[i] for i in top_local]

    def _to_float64(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[str]]:
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(
                f"[{self.symbol}] X must be non-empty DataFrame"
            )
        if df.columns.duplicated().any():
            raise ValueError(
                f"[{self.symbol}] duplicate columns not allowed"
            )
        arr = df.to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(arr).all():
            raise ValueError(
                f"[{self.symbol}] X contains NaN/Inf; "
                f"impute with TRAIN stats first"
            )
        return arr, list(df.columns)

    def fit_transform(
        self, X: pd.DataFrame, y: Any = None
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)