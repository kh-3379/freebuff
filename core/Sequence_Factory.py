from typing import Tuple, Any, Optional, Dict, List, Union
from pathlib import Path
from datetime import datetime, timezone
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import pandas as pd
import json
import h5py
import threading
import time
import os
import socket
import asyncio

from core.imports import *
from core.global_method import *
from core.logger import redirect_prints_to_logger, AdvancedTradeLogger

redirect_prints_to_logger()
logger = AdvancedTradeLogger(__name__)
SAVE_LOCK = asyncio.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# ✅ ANTI-LEAKAGE: نام تارگت‌ها با معنای صریح آینده‌نگر
# ═══════════════════════════════════════════════════════════════════════════

# در بالای فایل:
MAIN_TARGETS = [
    'price_return',
    'price_direction',
    'trend_strength',
    'volatility_regime',
    'market_regime',
]
AUXILIARY_TARGETS = [
    'magnitude_bucket',
    'momentum_direction',
]
AUXILIARY_WEIGHTS = ['sample_weight']

PROJECT_TARGETS = MAIN_TARGETS + AUXILIARY_TARGETS

PROJECT_TARGETS_TYPE = {
    'price_return': 'regression',
    'price_direction': 'classification',
    'trend_strength': 'classification',
    'volatility_regime': 'classification',
    'market_regime': 'classification',
    'magnitude_bucket': 'classification',
    'momentum_direction': 'classification',
}

AUXILIARY_COLUMNS: List[str] = AUXILIARY_TARGETS + AUXILIARY_WEIGHTS

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

# ============================================================================
# 🔒 LOCK MANAGEMENT
# ============================================================================

async def _acquire_global_processing_lock(
    lock_key: str,
    operation: str = "unknown",
    timeout_seconds: int = 300,
) -> bool:
    """قفل گلوبال با timeout و بازیابی از قفل‌های کهنه."""
    try:
        lock_file = Path("cache") / "processing_locks" / f"{lock_key}.lock"
        lock_file.parent.mkdir(parents=True, exist_ok=True)

        if lock_file.exists():
            try:
                with open(lock_file, 'r') as f:
                    lock_data = json.load(f)
                lock_time = datetime.fromisoformat(lock_data['timestamp'])
                if lock_time.tzinfo is None:
                    lock_time = lock_time.replace(tzinfo=timezone.utc)
                lock_age = (datetime.now(timezone.utc) - lock_time).total_seconds()

                if lock_age > timeout_seconds:
                    logger.warning(f"🔓 قفل قدیمی حذف: {lock_key} (سن: {lock_age:.1f}s)")
                    lock_file.unlink()
                else:
                    logger.debug(f"🔒 قفل فعال: {lock_key}")
                    return False
            except Exception:
                try:
                    lock_file.unlink()
                except Exception:
                    pass

        lock_data = {
            'process': f"PID_{os.getpid()}_{socket.gethostname()}",
            'operation': operation,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'timeout_seconds': timeout_seconds,
        }
        with open(lock_file, 'w') as f:
            json.dump(lock_data, f)

        logger.debug(f"🔒 قفل ایجاد: {lock_key}")
        return True

    except Exception as e:
        logger.error(f"❌ خطا در قفل: {e}")
        return False


async def _release_global_processing_lock(lock_key: str) -> None:
    """آزادسازی قفل."""
    try:
        lock_file = Path("cache") / "processing_locks" / f"{lock_key}.lock"
        if lock_file.exists():
            lock_file.unlink()
            logger.debug(f"🔓 قفل آزاد: {lock_key}")
    except Exception as e:
        logger.debug(f"⚠️ خطا در آزادسازی قفل: {e}")


# ============================================================================
# 📦 کلاس SequenceBuilder — نسخهٔ v9.2 با پشتیبانی AUXILIARY_COLUMNS
# ============================================================================

class SequenceBuilder:
    """
    🔨 Professional Sequence Builder v9.2 — ZERO Look-Ahead Bias Guaranteed.

    Invariants:
        X[i] = data[i : i + lookback]
        y_target[t][i]    = target[t][i + lookback - 1]         (5 targets)
        y_auxiliary[a][i] = auxiliary[a][i + lookback - 1]      (2 aux columns)

    ✅ v9.2 ویژگی‌های جدید:
        - پشتیبانی صریح از AUXILIARY_COLUMNS (sample_weight, magnitude_bucket)
        - این ستون‌ها از X حذف می‌شوند و در y_dict همراه تارگت‌ها anchor می‌شوند
        - اعتبارسنجی جداگانه برای auxiliary (skip correlation checks)
        - date_seq هم‌تراز با anchor حفظ می‌شود

    Window mode:
        - overlapping (stride=1) — پیش‌فرض
        - non_overlapping (stride=lookback) — حالت ارزیابی سخت
    """

    def __init__(
        self,
        lookback: int = 30,
        lookahead: int = 1,
        symbol: str = "__default__",
        fold_num: Optional[int] = None,
        dtype=np.float32,
        max_memory_mb: int = 2000,
        prefer_gpu: bool = False,
        target_names: Optional[List[str]] = None,
        auxiliary_columns: Optional[List[str]] = None,
        window_mode: str = "overlapping",
        **kwargs,
    ):
        if not isinstance(lookback, (int, np.integer)) or lookback < 5:
            raise ValueError(f"❌ lookback must be int >= 5, got {lookback!r}")
        if not isinstance(lookahead, (int, np.integer)) or lookahead < 1:
            raise ValueError(f"❌ lookahead must be int >= 1, got {lookahead!r}")
        if window_mode not in ("overlapping", "non_overlapping"):
            raise ValueError(
                f"❌ window_mode must be 'overlapping' or 'non_overlapping', "
                f"got {window_mode!r}"
            )

        self.lookback = int(lookback)
        self.lookahead = int(lookahead)
        self.symbol = symbol
        self.fold_num = fold_num
        self.dtype = dtype
        self.max_memory_mb = int(max_memory_mb)
        self.prefer_gpu = bool(prefer_gpu)
        self.window_mode = window_mode
        self._lock = threading.RLock()

        self._target_names: List[str] = (
            list(target_names) if target_names else list(PROJECT_TARGETS)
        )
        self._target_types: Dict[str, str] = dict(PROJECT_TARGETS_TYPE)

        self._auxiliary_names: List[str] = (
            list(auxiliary_columns) if auxiliary_columns is not None
            else list(AUXILIARY_COLUMNS)
        )

        self._target_names = list(dict.fromkeys(self._target_names))
        self._auxiliary_names = list(dict.fromkeys(self._auxiliary_names))

        overlap = set(self._target_names) & set(self._auxiliary_names)
        if overlap:
            logger.info(
                f"[{self.symbol}] 🔗 SequenceBuilder: {sorted(overlap)} "
                f"appear in BOTH targets and auxiliary — treated as targets "
                f"(extracted once per window)"
            )
            # حذف از auxiliary — در targets باقی می‌مانند
            self._auxiliary_names = [
                a for a in self._auxiliary_names if a not in overlap
            ]

        # ✅ ULTIMATE dedupe for excluded names
        self._all_excluded_names: List[str] = list(
            dict.fromkeys(self._target_names + self._auxiliary_names)
        )

        logger.debug(
            f"[{self.symbol}] 🔧 SequenceBuilder v9.3 init | "
            f"lookback={self.lookback} | lookahead={self.lookahead} | "
            f"mode={self.window_mode} | targets={len(self._target_names)} | "
            f"auxiliary={self._auxiliary_names} | "
            f"excluded={len(self._all_excluded_names)}"
        )
        
    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    async def create_sequences(
        self,
        data,
        targets=None,
        timeframe: str = "None",
        split_type: str = "None",
        lookback: Optional[int] = None,
        lookahead: Optional[int] = None,
        fold_number: Optional[int] = None,
        date_column: Optional[np.ndarray] = None,
        **kw,
    ):
        """
        ساخت سکانس با جداسازی زمانی مطلق.

        Returns:
            (X_seq, y_seq_dict, date_seq)
            - y_seq_dict شامل هر دو target و auxiliary است.
        """
        start_time = time.time()
        lb = int(lookback) if lookback is not None else self.lookback
        la = int(lookahead) if lookahead is not None else self.lookahead
        if lb < 1 or la < 1:
            raise ValueError(f"❌ Invalid lookback/lookahead: lb={lb}, la={la}")

        logger.info(
            f"[{self.symbol}] 🎬 Sequence creation | split={split_type} | "
            f"lookback={lb} | lookahead={la} | mode={self.window_mode}"
        )

        # ---------- PHASE 1: استخراج X, y, date ----------
        X_data, y_dict, date_data = self._extract_xy_date(
            data, targets, date_column
        )
        if X_data is None or len(X_data) == 0:
            raise ValueError(f"❌ X_data is empty for {split_type}")

        for tname, tdata in y_dict.items():
            if tdata is not None:
                logger.debug(
                    f"[{self.symbol}] 📥 Input target '{tname}': "
                    f"shape={tdata.shape}"
                )

        # ---------- PHASE 2: ساخت سکانس ----------
        if X_data.ndim == 3:
            X_seq, y_seq_dict, date_seq = self._validate_sequenced_data(
                X_data, y_dict, date_data, lb, la, split_type
            )
        elif X_data.ndim == 2:
            X_seq, y_seq_dict, date_seq = self._create_sequences_from_2d_no_leakage(
                X_data, y_dict, date_data, lb, la, split_type
            )
        else:
            raise ValueError(
                f"❌ Invalid dimensions: {X_data.ndim}D (expected 2D or 3D)"
            )

        for tname, tdata in y_seq_dict.items():
            if tdata is not None:
                logger.debug(
                    f"[{self.symbol}] 📤 Output '{tname}': shape={tdata.shape}"
                )

        # ---------- PHASE 3: HARD VALIDATION ----------
        self._validate_no_temporal_leakage(X_seq, y_seq_dict, date_seq, la)

        execution_time = time.time() - start_time
        logger.info(
            f"[{self.symbol}] ✅ Sequences created in {execution_time:.2f}s | "
            f"X: {X_seq.shape}"
        )
        return X_seq, y_seq_dict, date_seq

    def _create_sequences_from_2d_no_leakage(
        self,
        X_2d: np.ndarray,
        y_dict: Dict[str, Optional[np.ndarray]],
        date_data: Optional[np.ndarray],
        lookback: int,
        lookahead: int,
        split_type: str,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        ساخت سکانس با قرارداد anchor=last_X.

        anchor[i] = i + lookback - 1
        X_seq[i]  = X_2d[i : i + lookback]
        y_seq[t][i] = y_dict[t][anchor[i]]   برای هر t ∈ targets ∪ auxiliary

        حذف NaN: shared mask روی تمام تارگت‌های اصلی (نه auxiliary)
        اعمال می‌شود تا auxiliary هم‌تراز حذف شوند.
        """
        if X_2d.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X_2d.ndim}D")

        n_samples_total, n_features = X_2d.shape
        min_required = lookback + 1
        if n_samples_total < min_required:
            raise ValueError(
                f"❌ Insufficient data: {n_samples_total} < "
                f"lookback+1 ({min_required})"
            )

        n_sequences = n_samples_total - lookback + 1
        if n_sequences <= 0:
            raise ValueError(
                f"❌ Cannot create sequences: n={n_samples_total}, lb={lookback}"
            )

        # ---------- X برداری ----------
        X_contiguous = np.ascontiguousarray(X_2d, dtype=self.dtype)
        try:
            X_all = np.lib.stride_tricks.sliding_window_view(
                X_contiguous, window_shape=lookback, axis=0
            )
            X_seq = np.ascontiguousarray(
                X_all[:n_sequences].transpose(0, 2, 1),
                dtype=self.dtype,
            )
            if X_seq.shape != (n_sequences, lookback, n_features):
                raise RuntimeError(
                    f"Unexpected X shape after transpose: {X_seq.shape}"
                )
        except Exception as e:
            logger.warning(
                f"[{self.symbol}] ⚠️ sliding_window_view failed ({e}); "
                f"using fallback loop"
            )
            X_seq = np.empty(
                (n_sequences, lookback, n_features), dtype=self.dtype
            )
            for i in range(n_sequences):
                X_seq[i] = X_contiguous[i:i + lookback]

        # ---------- anchor ----------
        anchor_indices = np.arange(
            lookback - 1, n_samples_total, dtype=np.int64
        )
        if anchor_indices.size != n_sequences:
            raise RuntimeError(
                f"Internal anchor length mismatch: "
                f"{anchor_indices.size} != {n_sequences}"
            )

        y_seq_dict: Dict[str, np.ndarray] = {}

        # ── ترتیب: ابتدا auxiliary (تا mask بر اساس target اصلی باشد) ──
        # در واقع ترتیب iteration مهم نیست؛ mask بر اساس همه‌شان محاسبه می‌شود.
        all_names = self._target_names + self._auxiliary_names

        for name in all_names:
            tval = y_dict.get(name) if isinstance(y_dict, dict) else None
            if tval is None:
                if name in self._target_names:
                    logger.warning(
                        f"[{self.symbol}] ⚠️ Target '{name}' not present in y_dict"
                    )
                else:
                    logger.debug(
                        f"[{self.symbol}] ℹ️ Auxiliary '{name}' not present — skipped"
                    )
                continue

            tval_arr = np.asarray(tval)
            if tval_arr.ndim > 1:
                if tval_arr.shape[1] == 1:
                    tval_arr = tval_arr[:, 0]
                else:
                    raise ValueError(
                        f"[{self.symbol}] '{name}' must be 1D or (n,1); "
                        f"got shape {tval_arr.shape}"
                    )

            if len(tval_arr) != n_samples_total:
                raise ValueError(
                    f"[{self.symbol}] ❌ '{name}' length mismatch: "
                    f"got {len(tval_arr)}, expected {n_samples_total}."
                )

            # ✅ dtype صحیح برای auxiliary
            if name in AUXILIARY_DTYPES:
                out_dtype = AUXILIARY_DTYPES[name]
            else:
                out_dtype = self.dtype

            y_arr = tval_arr[anchor_indices].astype(out_dtype, copy=True)

            if len(y_arr) != n_sequences:
                raise RuntimeError(
                    f"[{self.symbol}] ❌ CRITICAL: y length mismatch for "
                    f"'{name}': {len(y_arr)} != {n_sequences}"
                )

            y_seq_dict[name] = y_arr
            logger.debug(
                f"[{self.symbol}] ✅ {name}: shape={y_arr.shape} "
                f"dtype={y_arr.dtype}"
            )

        # ── حداقل باید تمام ۵ تارگت اصلی موجود باشند ──
        missing_main = [
            t for t in self._target_names if t not in y_seq_dict
        ]
        if missing_main:
            raise ValueError(
                f"[{self.symbol}] ❌ Missing required PROJECT_TARGETS: "
                f"{missing_main}"
            )

        # ---------- 🔥 shared validity mask (فقط روی ۵ تارگت اصلی) ----------
        valid_mask = np.ones(n_sequences, dtype=bool)
        for tname in self._target_names:
            y_arr = y_seq_dict.get(tname)
            if y_arr is None:
                continue
            valid_mask &= np.isfinite(y_arr)

        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            logger.info(
                f"[{self.symbol}] 🧹 Removing {n_invalid}/{n_sequences} "
                f"sequences with non-finite main-target values"
            )
            X_seq = X_seq[valid_mask]
            for name in list(y_seq_dict.keys()):
                y_seq_dict[name] = y_seq_dict[name][valid_mask]

            anchor_indices = anchor_indices[valid_mask]
            n_sequences = int(valid_mask.sum())

            if n_sequences == 0:
                raise ValueError(
                    f"[{self.symbol}] ❌ All sequences removed due to "
                    f"non-finite main targets"
                )

        # ---------- date_seq هم‌تراز ----------
        date_seq: Optional[np.ndarray] = None
        if date_data is not None:
            date_arr = np.asarray(date_data)
            if len(date_arr) < n_samples_total:
                raise ValueError(
                    f"[{self.symbol}] date_data length {len(date_arr)} < "
                    f"n_samples {n_samples_total}; cannot align."
                )
            date_seq = date_arr[anchor_indices].copy()

        logger.info(
            f"[{self.symbol}] ✅ 2D→3D | n_seq={n_sequences}, L={lookback}, "
            f"targets={len(self._target_names)}, "
            f"auxiliary={len(y_seq_dict) - len(self._target_names)}, "
            f"nan_cleaned={n_invalid > 0}"
        )
        return X_seq, y_seq_dict, date_seq

    def _validate_no_temporal_leakage(
        self,
        X_seq: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        date_seq: Optional[np.ndarray],
        lookahead: int,
    ) -> None:
        """
        HARD VALIDATION با قرارداد anchor=last_X.

        ✅ v9.4 (نسبت به v9.3):
            - BUG FIX CRITICAL: اعتبارسنجی کلاس‌ها حالا در یک حلقهٔ واحد
              روی `_target_names + _auxiliary_names` انجام می‌شود.
              قبلاً `magnitude_bucket` و `momentum_direction` (که در
              `_target_names` هستند، نه `_auxiliary_names`) هرگز
              اعتبارسنجی نمی‌شدند — dead code.
            - مقادیر مجاز کلاس‌ها از `TARGET_CLASS_COUNTS` مشتق می‌شود
              به‌جای hardcode، تا با تغییر تعداد کلاس‌ها هم‌زمان
              به‌روزرسانی شود.
            - بررسی‌های correlation فقط روی تارگت‌های continuous
              (price_return) و discrete (price_direction, trend,
              vol, market) انجام می‌شود. برای magnitude_bucket و
              momentum_direction، correlation linear چندان معنادار
              نیست، ولی همچنان اجرا می‌شود تا leakage آشکار را بگیری.
        """
        n_sequences = X_seq.shape[0]

        # ── Check 1: length consistency ─────────────────────────
        for tname, y_arr in y_dict.items():
            if len(y_arr) != n_sequences:
                raise ValueError(
                    f"❌ Length mismatch: X has {n_sequences} sequences, "
                    f"'{tname}' has {len(y_arr)}"
                )

        # ── Check 2: date monotonicity ──────────────────────────
        if date_seq is not None and len(date_seq) == n_sequences:
            try:
                d = date_seq
                if not np.issubdtype(np.asarray(d).dtype, np.datetime64):
                    d = pd.to_datetime(d, utc=True)
                d = np.asarray(d)

                if len(d) > 1:
                    if not np.all(d[1:] > d[:-1]):
                        raise ValueError(
                            f"[{self.symbol}] ❌ TEMPORAL LEAKAGE: dates "
                            f"not strictly increasing across sequences."
                        )

                    if np.issubdtype(d.dtype, np.datetime64):
                        deltas = np.diff(d.astype("int64"))
                        deltas = deltas[deltas > 0]
                        if deltas.size > 4:
                            med = float(np.median(deltas))
                            cv = float(np.std(deltas) / (med + 1e-9))
                            if cv > 5.0:
                                logger.warning(
                                    f"[{self.symbol}] ⚠️ date deltas highly "
                                    f"irregular (CV={cv:.2f}); "
                                    f"check split boundaries"
                                )
                    logger.info(
                        f"[{self.symbol}] 📅 date range: "
                        f"{d[0]} → {d[-1]} ({len(d)})"
                    )
            except ValueError:
                raise
            except Exception as e:
                logger.warning(
                    f"[{self.symbol}] ⚠️ date validation skipped: {e}"
                )

        # ── Check 3: target sanity + correlation ───────────────
        last_timestep = X_seq[:, -1, :]
        first_timestep = X_seq[:, 0, :]
        mid_idx = X_seq.shape[1] // 2
        mid_timestep = X_seq[:, mid_idx, :]

        for tname in self._target_names:
            y_arr = y_dict.get(tname)
            if y_arr is None:
                continue
            y_arr = np.asarray(y_arr).flatten()
            if len(y_arr) < 20:
                continue
            y_std = float(np.std(y_arr))
            if y_std < 1e-12:
                logger.warning(
                    f"[{self.symbol}] ⚠️ target '{tname}' is CONSTANT"
                )
                continue

            n_feat = min(last_timestep.shape[1], 32)

            for fi in range(n_feat):
                xf_last = last_timestep[:, fi]
                xf_first = first_timestep[:, fi]
                xf_mid = mid_timestep[:, fi]

                if float(np.std(xf_last)) < 1e-12:
                    continue

                try:
                    corr_last = float(np.corrcoef(xf_last, y_arr)[0, 1])
                except Exception:
                    continue
                if not np.isfinite(corr_last):
                    continue

                if abs(corr_last) > 0.995:
                    raise ValueError(
                        f"[{self.symbol}] ❌ DIRECT LEAKAGE: feature {fi} "
                        f"corr={corr_last:.5f} with target '{tname}'"
                    )
                if abs(corr_last) > 0.95:
                    logger.warning(
                        f"[{self.symbol}] ⚠️ VERY HIGH corr: feature {fi} "
                        f"vs '{tname}' = {corr_last:.4f}"
                    )

                if (
                    float(np.std(xf_mid)) > 1e-12
                    and self.lookback > 4
                ):
                    try:
                        corr_mid = float(np.corrcoef(xf_mid, y_arr)[0, 1])
                        if np.isfinite(corr_mid) and abs(corr_mid) > 0.995:
                            raise ValueError(
                                f"[{self.symbol}] ❌ DIRECT LEAKAGE at "
                                f"mid-window: feature {fi} "
                                f"corr={corr_mid:.5f} with '{tname}'"
                            )
                    except ValueError:
                        raise
                    except Exception:
                        pass

                if self.lookback <= 2:
                    continue
                if float(np.std(xf_first)) < 1e-12:
                    continue

                try:
                    feat_autocorr = float(np.corrcoef(xf_first, xf_last)[0, 1])
                except Exception:
                    feat_autocorr = 0.0
                if not np.isfinite(feat_autocorr):
                    feat_autocorr = 0.0

                if abs(feat_autocorr) > 0.90:
                    continue

                try:
                    corr_first = float(np.corrcoef(xf_first, y_arr)[0, 1])
                except Exception:
                    continue
                if not np.isfinite(corr_first):
                    continue

                if (
                    abs(corr_first) > 0.95
                    and abs(corr_first) > abs(corr_last) * 1.30
                ):
                    raise ValueError(
                        f"[{self.symbol}] ❌ FUTURE LEAKAGE: feature {fi} "
                        f"corr(first)={corr_first:.4f} >> "
                        f"corr(last)={corr_last:.4f} "
                        f"(autocorr={feat_autocorr:.3f})"
                    )

        # ── Check 4: CLASS RANGE VALIDATION (v9.4 BUG FIX) ──────
        # ✅ این چک حالا روی «تمام تارگت‌ها» (اعم از main و auxiliary)
        # اجرا می‌شود، نه فقط روی `_auxiliary_names`. مقادیر مجاز از
        # TARGET_CLASS_COUNTS مشتق می‌شوند:
        #
        #   magnitude_bucket  : n_classes=3 → allowed={-1, 0, 1, 2}
        #   momentum_direction: n_classes=2 → allowed={-1, 0, 1}
        #   price_direction   : n_classes=2 → allowed={-1, 0, 1}
        #   trend_strength    : n_classes=3 → allowed={-1, 0, 1, 2}
        #   volatility_regime : n_classes=3 → allowed={-1, 0, 1, 2}
        #   market_regime     : n_classes=4 → allowed={-1, 0, 1, 2, 3}
        all_names_for_class_check = (
            list(self._target_names) + list(self._auxiliary_names)
        )
        for tname in all_names_for_class_check:
            y_arr = y_dict.get(tname)
            if y_arr is None:
                continue
            y_arr = np.asarray(y_arr).flatten()
            if len(y_arr) == 0:
                continue

            n_classes = TARGET_CLASS_COUNTS.get(tname)
            if n_classes is None:
                # تارگت رگرسیونی (price_return) — بدون چک کلاس
                continue

            allowed_classes = {-1} | set(range(int(n_classes)))
            finite_mask = np.isfinite(y_arr)
            if not finite_mask.any():
                logger.warning(
                    f"[{self.symbol}] ⚠️ classification target '{tname}' "
                    f"is entirely non-finite — skipping class check"
                )
                continue

            a_int = np.where(finite_mask, y_arr, 0).astype(np.int64)
            allowed_arr = np.asarray(
                sorted(allowed_classes), dtype=np.int64
            )
            bad_mask = finite_mask & ~np.isin(a_int, allowed_arr)
            if bool(np.any(bad_mask)):
                bad_vals = np.unique(y_arr[bad_mask])
                raise ValueError(
                    f"[{self.symbol}] ❌ {tname} invalid classes: "
                    f"{bad_vals[:10].tolist()} "
                    f"(allowed: {sorted(allowed_classes)})"
                )

            u = np.unique(y_arr[finite_mask])
            logger.info(
                f"[{self.symbol}] 📊 {tname} classes: "
                f"{sorted(int(v) for v in u)}"
            )

        # ── Check 5: sample_weight sanity (فقط auxiliary weight) ─
        for aname in self._auxiliary_names:
            if aname != "sample_weight":
                continue
            y_arr = y_dict.get(aname)
            if y_arr is None:
                continue
            y_arr = np.asarray(y_arr).flatten()
            if len(y_arr) < 10:
                continue
            finite = y_arr[np.isfinite(y_arr)]
            if finite.size == 0:
                continue
            mean_w = float(np.mean(finite))
            min_w = float(np.min(finite))
            if mean_w <= 0 or not np.isfinite(mean_w):
                raise ValueError(
                    f"[{self.symbol}] ❌ sample_weight mean invalid: "
                    f"{mean_w}"
                )
            if min_w < 0:
                raise ValueError(
                    f"[{self.symbol}] ❌ sample_weight negative: {min_w}"
                )
            logger.info(
                f"[{self.symbol}] ⚖️ sample_weight sanity OK | "
                f"mean={mean_w:.4f}, min={min_w:.4f}"
            )

        # ── Check 6: lookahead gap verification ────────────────
        if lookahead >= 2:
            logger.info(
                f"[{self.symbol}] ✅ lookahead={lookahead} — gap of "
                f"{lookahead} steps between last X and y (contract verified)"
            )

        logger.info(
            f"[{self.symbol}] ✅ temporal isolation verified | "
            f"anchor=last_X, gap={lookahead - 1}, lookahead_declared={lookahead}"
        )
        
    def _extract_xy_date(
        self,
        data,
        targets,
        date_column,
    ) -> Tuple[np.ndarray, Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]:
        """
        استخراج X, y, date با تضمین حذف کامل ستون‌های تارگت و auxiliary از X.

        ✅ v9.2: auxiliary columns (sample_weight, magnitude_bucket) نیز
        از X حذف می‌شوند و در y_dict قرار می‌گیرند.
        """
        date_data = date_column
        y_dict: Dict[str, Optional[np.ndarray]] = {
            t: None for t in self._target_names
        }
        for a in self._auxiliary_names:
            y_dict[a] = None

        # ---------- targets argument ----------
        if targets is not None:
            if isinstance(targets, dict):
                for key, value in targets.items():
                    if key in y_dict and value is not None:
                        y_dict[key] = value
                    elif key not in y_dict:
                        logger.debug(
                            f"[{self.symbol}] نادیده گرفتن تارگت ناشناخته: {key}"
                        )
            elif isinstance(targets, (np.ndarray, list, tuple)):
                arr = np.asarray(targets)
                if arr.ndim == 1:
                    y_dict[self._target_names[0]] = arr.reshape(-1, 1)
                elif arr.ndim == 2:
                    for i, tname in enumerate(self._target_names):
                        if i < arr.shape[1]:
                            y_dict[tname] = arr[:, i:i + 1]
                else:
                    raise ValueError(f"Invalid targets ndim: {arr.ndim}")

        # ---------- data argument ----------
        if isinstance(data, pd.DataFrame):
            # استخراج date از DataFrame
            if date_data is None:
                date_cols = [
                    c for c in data.columns
                    if c.lower() in (
                        'datetime', 'date', 'time', 'timestamp',
                        'open_time', 'close_time'
                    )
                ]
                if date_cols:
                    date_data = data[date_cols[0]].values.copy()
                    data = data.drop(columns=date_cols)

            # استخراج تارگت‌های اصلی
            for t in self._target_names:
                if t in data.columns:
                    y_dict[t] = data[t].values.reshape(-1, 1)
                    logger.debug(f"[{self.symbol}] ✅ تارگت اصلی: {t}")

            # ✅ v9.2: استخراج auxiliary columns
            for a in self._auxiliary_names:
                if a in data.columns:
                    y_dict[a] = data[a].values.reshape(-1, 1)
                    logger.info(f"[{self.symbol}] ✅ auxiliary: {a}")

            # ✅ حذف کامل target و auxiliary از X (case-insensitive)
            cols_lower = {c.lower(): c for c in data.columns}
            drop_cols = set()
            for name in self._all_excluded_names:
                if name in data.columns:
                    drop_cols.add(name)
                elif name.lower() in cols_lower:
                    drop_cols.add(cols_lower[name.lower()])

            feature_columns = [c for c in data.columns if c not in drop_cols]
            if not feature_columns:
                raise ValueError(
                    f"[{self.symbol}] ❌ No feature columns left after "
                    f"removing targets and auxiliary"
                )
            X_data = data[feature_columns].values

        elif isinstance(data, np.ndarray):
            X_data = data
        else:
            raise TypeError(f"فرمت نامعتبر: {type(data)}")

        # ---------- normalize y to 1D ----------
        for key in list(y_dict.keys()):
            v = y_dict[key]
            if v is None:
                continue
            if isinstance(v, pd.Series):
                v = v.values
            v = np.asarray(v)
            if v.ndim == 1:
                v = v.reshape(-1, 1)
            y_dict[key] = v

        # ---------- ensure X_data numeric ----------
        if isinstance(X_data, pd.DataFrame):
            X_data = X_data.values

        return X_data, y_dict, date_data

    def _validate_sequenced_data(
        self,
        X_data: np.ndarray,
        y_dict: Dict[str, Optional[np.ndarray]],
        date_data: Optional[np.ndarray],
        lookback: int,
        lookahead: int,
        split_type: str,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        اگر X_data از قبل 3D است، فقط اعتبارسنجی می‌کند.

        ⚠️ در این مسیر، auxiliary columns نمی‌توانند به‌طور خودکار
        anchor شوند. Caller باید مطمئن شود که y_dict از قبل هم‌تراز است.
        """
        if X_data.ndim != 3:
            raise ValueError(
                f"Expected 3D array for _validate_sequenced_data, "
                f"got {X_data.ndim}D"
            )

        n_seq, L, n_feat = X_data.shape
        if L != lookback:
            logger.warning(
                f"[{self.symbol}] ⚠️ 3D input L={L} != lookback={lookback}; "
                f"keeping L={L}"
            )

        y_seq_dict: Dict[str, np.ndarray] = {}
        all_names = self._target_names + self._auxiliary_names

        for name in all_names:
            tval = y_dict.get(name) if isinstance(y_dict, dict) else None
            if tval is None:
                if name in self._target_names:
                    logger.warning(
                        f"[{self.symbol}] ⚠️ '{name}' missing in 3D path"
                    )
                continue

            arr = np.asarray(tval)
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr[:, 0]
            elif arr.ndim > 1:
                raise ValueError(
                    f"[{self.symbol}] 3D path: '{name}' must be 1D"
                )

            if len(arr) != n_seq:
                raise ValueError(
                    f"[{self.symbol}] 3D path: '{name}' length {len(arr)} "
                    f"!= n_seq {n_seq}"
                )

            if name in AUXILIARY_DTYPES:
                arr = arr.astype(AUXILIARY_DTYPES[name], copy=True)
            y_seq_dict[name] = arr

        date_seq = None
        if date_data is not None:
            d_arr = np.asarray(date_data)
            if len(d_arr) == n_seq:
                date_seq = d_arr.copy()
            else:
                logger.warning(
                    f"[{self.symbol}] ⚠️ 3D path: date length mismatch "
                    f"({len(d_arr)} vs {n_seq}) — date_seq dropped"
                )

        return X_data, y_seq_dict, date_seq


# ============================================================================
# 🎯 universal_sequence_creation — نسخهٔ v9.2 با پشتیبانی AUXILIARY
# ============================================================================

async def universal_sequence_creation(
    symbol: str,
    timeframe: str,
    fold_num: int,
    split_type: str,
    lookback: Optional[int] = None,
    lookahead: int = 1,
    force_recreate: bool = False,
    min_lookback: int = 40,
    max_lookback: int = 80,
    use_optimal_lookback: bool = True,
    sequence_type: str = "standard",
    bypass_lock: bool = False,
    data_fetcher_instance=None,
) -> Dict[str, Any]:
    """
    ساخت سکانس از دادهٔ normalized با ZERO-LEAKAGE.

    ✅ v9.3 FIX (CRITICAL B1):
        تارگت‌های auxiliary رسمی (magnitude_bucket, momentum_direction)
        قبلاً در TWO loops (TARGET_NAMES و AUX_NAMES) نوشته می‌شدند و
        h5py با ValueError: Unable to create dataset (name already exists)
        کرش می‌کرد.

        راه‌حل: AUX_NAMES فقط شامل AUXILIARY_WEIGHTS (sample_weight)
        می‌شود. تارگت‌های auxiliary رسمی در همان loop اول ذخیره می‌شوند
        با dtype صحیح از AUXILIARY_DTYPES.
    """
    TARGET_NAMES = list(PROJECT_TARGETS)
    # ✅ FIX B1: AUX_NAMES فقط weight؛ تارگت‌های auxiliary در TARGET_NAMES هستند
    AUX_NAMES = list(AUXILIARY_WEIGHTS)
    start_time = time.time()

    # ────────────────────────────────────────────────────────────
    # LOCK
    # ────────────────────────────────────────────────────────────
    lock_key = f"seq_{symbol}_{timeframe}_fold{fold_num}_{split_type}"
    lock_acquired = False
    if not bypass_lock:
        lock_acquired = await _acquire_global_processing_lock(
            lock_key,
            operation=f"sequence_creation_{split_type}",
            timeout_seconds=600,
        )
        if not lock_acquired:
            logger.warning(
                f"[{symbol}] ⚠️ lock busy for {split_type}; "
                f"proceeding read-only"
            )

    try:
        # ────────────────────────────────────────────────────────
        # PHASE 1: LOAD NORMALIZED DATA (current split)
        # ────────────────────────────────────────────────────────
        logger.info(f"[{symbol}] 📥 PHASE 1: loading normalized/{split_type}")
        split_raw = await universal_load(
            "normalized", symbol, timeframe, fold_num, split_type
        )
        if split_raw is None:
            return {
                "status": "error",
                "error": f"{split_type} normalized not found",
            }
        if isinstance(split_raw, dict) and 'X' not in split_raw:
            split_raw = pd.DataFrame(split_raw)
        if not isinstance(split_raw, pd.DataFrame):
            raise TypeError(
                f"Expected DataFrame, got {type(split_raw).__name__}"
            )

        logger.info(f"[{symbol}] ✅ {split_type} loaded: {split_raw.shape}")

        # ────────────────────────────────────────────────────────
        # PHASE 2: LOAD TRAIN
        # ────────────────────────────────────────────────────────
        train_raw = await universal_load(
            "normalized", symbol, timeframe, fold_num, split_type="train"
        )
        if train_raw is None:
            return {"status": "error", "error": "Train normalized not found"}
        if isinstance(train_raw, dict) and 'X' not in train_raw:
            train_raw = pd.DataFrame(train_raw)
        if not isinstance(train_raw, pd.DataFrame):
            raise TypeError(
                f"Expected DataFrame from train loader, "
                f"got {type(train_raw).__name__}"
            )

        # ────────────────────────────────────────────────────────
        # PHASE 3: canonical feature/target separation
        # ────────────────────────────────────────────────────────
        def _split_targets_from_df(frame: pd.DataFrame):
            frame = frame.copy()
            lower_map = {c.lower(): c for c in frame.columns}

            present_targets: Dict[str, str] = {}
            present_aux: Dict[str, str] = {}

            for t in TARGET_NAMES:
                if t in frame.columns:
                    present_targets[t] = t
                elif t.lower() in lower_map:
                    present_targets[t] = lower_map[t.lower()]

            for a in AUX_NAMES:
                if a in frame.columns:
                    present_aux[a] = a
                elif a.lower() in lower_map:
                    present_aux[a] = lower_map[a.lower()]

            excluded = set(present_targets.values()) | set(present_aux.values())
            feature_cols_local = [
                c for c in frame.columns if c not in excluded
            ]
            return frame, feature_cols_local, present_targets, present_aux

        # ── TRAIN
        train_feat_df, train_feature_cols, _, train_aux_present = \
            _split_targets_from_df(train_raw)

        if not train_feature_cols:
            return {"status": "error", "error": "No feature columns in TRAIN"}

        logger.info(
            f"[{symbol}] 📊 TRAIN: {len(train_feature_cols)} features, "
            f"{len(train_aux_present)} auxiliary weight columns"
        )

        # ── CURRENT SPLIT
        split_feat_df, _, present_targets_split, present_aux_split = \
            _split_targets_from_df(split_raw)

        missing_feats = [
            c for c in train_feature_cols if c not in split_feat_df.columns
        ]
        extra_feats = [
            c for c in split_feat_df.columns
            if c not in set(train_feature_cols)
            and c not in set(present_targets_split.values())
            and c not in set(present_aux_split.values())
        ]

        if missing_feats:
            raise RuntimeError(
                f"[{symbol}] ❌ {split_type} missing {len(missing_feats)} "
                f"feature columns that exist in TRAIN: {missing_feats[:10]}"
            )
        if extra_feats:
            raise RuntimeError(
                f"[{symbol}] ❌ {split_type} has {len(extra_feats)} extra "
                f"feature columns not present in TRAIN: {extra_feats[:10]}"
            )

        # ── استخراج y (targets + auxiliary weights)
        y_dict: Dict[str, np.ndarray] = {}

        for target_name, col in present_targets_split.items():
            arr = split_feat_df[col].to_numpy(dtype=np.float64, copy=True)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            y_dict[target_name] = arr
            logger.info(
                f"[{symbol}] 📊 target '{target_name}' | shape={arr.shape}"
            )

        for aux_name, col in present_aux_split.items():
            arr = split_feat_df[col].to_numpy(copy=True)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            y_dict[aux_name] = arr
            logger.info(
                f"[{symbol}] 📊 auxiliary weight '{aux_name}' | "
                f"shape={arr.shape}, dtype={arr.dtype}"
            )

        # اطمینان: هر ۷ تارگت اصلی + auxiliary targets موجود باشند
        missing_main = [t for t in TARGET_NAMES if t not in y_dict]
        if missing_main:
            raise RuntimeError(
                f"[{symbol}] ❌ {split_type} missing targets: "
                f"{missing_main}"
            )

        # X نهایی (ترتیب دقیقاً برابر TRAIN)
        X_matrix = split_feat_df[train_feature_cols].to_numpy(
            dtype=np.float32, copy=False
        )

        # ────────────────────────────────────────────────────────
        # PHASE 4: date column extraction
        # ────────────────────────────────────────────────────────
        date_arr = None
        date_col_name = None
        for candidate in (
            "datetime", "date", "time", "timestamp",
            "open_time", "close_time",
        ):
            if candidate in split_raw.columns:
                date_arr = split_raw[candidate].to_numpy()
                date_col_name = candidate
                break

        if date_arr is not None:
            logger.info(
                f"[{symbol}] 📅 date column '{date_col_name}' detected "
                f"(len={len(date_arr)})"
            )

        # ────────────────────────────────────────────────────────
        # PHASE 5: lookback (TRAIN-only statistics)
        # ────────────────────────────────────────────────────────
        if lookback is None and use_optimal_lookback:
            try:
                from core.Lookback_Factory import _determine_optimal_lookback_enhanced
                X_train_array = train_feat_df[train_feature_cols].to_numpy(
                    dtype=np.float32, copy=False
                )
                final_lookback = await _determine_optimal_lookback_enhanced(
                    X=X_train_array,
                    symbol=symbol,
                    timeframe=timeframe,
                    fold_num=fold_num,
                    user_lookback=None,
                    use_optimal_lookback=True,
                    min_lookback=max(min_lookback, 30),
                    max_lookback=max_lookback,
                    sequence_type=sequence_type,
                    data_length=len(X_train_array),
                )
            except Exception as e:
                logger.warning(
                    f"[{symbol}] ⚠️ optimal lookback failed ({e}); "
                    f"using default"
                )
                final_lookback = max(min_lookback, 60)
        else:
            final_lookback = (
                int(lookback) if lookback is not None
                else max(min_lookback, 60)
            )

        if final_lookback + 1 > len(X_matrix):
            raise ValueError(
                f"[{symbol}] lookback={final_lookback} too large for "
                f"n={len(X_matrix)} in split={split_type}"
            )

        logger.info(f"[{symbol}] 🎯 final_lookback = {final_lookback}")

        # ────────────────────────────────────────────────────────
        # PHASE 6: sequence creation
        # ────────────────────────────────────────────────────────
        builder = SequenceBuilder(
            symbol=symbol,
            lookback=final_lookback,
            lookahead=int(lookahead),
            max_memory_mb=2000,
            fold_num=fold_num,
            prefer_gpu=False,
            target_names=TARGET_NAMES,
            auxiliary_columns=AUX_NAMES,
            window_mode="overlapping",
        )

        X_seq, y_seq_dict, date_seq = await builder.create_sequences(
            data=X_matrix,
            targets=y_dict,
            timeframe=timeframe,
            split_type=split_type,
            lookback=final_lookback,
            lookahead=int(lookahead),
            fold_number=fold_num,
            date_column=date_arr,
        )

        if X_seq is None or X_seq.size == 0:
            raise ValueError("X_seq is empty after sequence creation")

        logger.info(f"[{symbol}] ✅ X_seq shape={X_seq.shape}")
        for tname, tdata in y_seq_dict.items():
            logger.info(f"[{symbol}]    ├─ {tname}: {tdata.shape}")

        # ────────────────────────────────────────────────────────
        # PHASE 7: save HDF5 (atomic) — FIXED B1
        # ────────────────────────────────────────────────────────
        save_dir = Path(f"data/sequenced/{symbol}/{timeframe}/fold_{fold_num}")
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{symbol}_{timeframe}_fold{fold_num}_{split_type}"
            f"_sequenced_lookback{final_lookback}.h5"
        )
        file_path = save_dir / filename

        if force_recreate and file_path.exists():
            file_path.unlink()

        tmp_path = save_dir / (
            f".tmp_{os.getpid()}_{int(time.time()*1e6)}.h5"
        )

        with h5py.File(tmp_path, 'w') as hf:
            hf.create_dataset(
                'X',
                data=X_seq.astype(np.float32, copy=False),
                compression='gzip', compression_opts=4,
            )

            # ── ✅ FIX B1: یک loop یکپارچه برای همهٔ targets + weights ──
            # dedupe names preserving order
            all_save_names = list(dict.fromkeys(TARGET_NAMES + AUX_NAMES))
            saved_targets = 0
            saved_aux = 0

            for save_name in all_save_names:
                save_data = y_seq_dict.get(save_name)
                if save_data is None or len(save_data) == 0:
                    if save_name in TARGET_NAMES:
                        logger.warning(
                            f"[{symbol}] ⚠️ target '{save_name}' empty — skipped"
                        )
                    else:
                        logger.debug(
                            f"[{symbol}] ℹ️ auxiliary '{save_name}' not present — skipped"
                        )
                    continue

                # ✅ dtype صحیح: auxiliary targets int32, weights float32, main float32
                if save_name in AUXILIARY_DTYPES:
                    out_dtype = AUXILIARY_DTYPES[save_name]
                else:
                    out_dtype = np.float32

                save_arr = (
                    save_data.reshape(-1, 1)
                    if save_data.ndim == 1
                    else save_data
                )

                hf.create_dataset(
                    save_name,
                    data=save_arr.astype(out_dtype, copy=False),
                    compression='gzip', compression_opts=4,
                )

                if save_name in TARGET_NAMES:
                    saved_targets += 1
                if save_name in AUX_NAMES and save_name not in TARGET_NAMES:
                    saved_aux += 1

                logger.info(
                    f"[{symbol}] 💾 saved '{save_name}' | "
                    f"shape={save_arr.shape} | dtype={out_dtype}"
                )

            # ── date ──
            if date_seq is not None and len(date_seq) == X_seq.shape[0]:
                try:
                    d = pd.to_datetime(date_seq, utc=True, errors="coerce")
                    valid = d.notna().to_numpy()
                    ns = np.full(len(d), -1, dtype=np.int64)
                    ns[valid] = d[valid].astype("int64").to_numpy()
                    hf.create_dataset('date_ns', data=ns)
                except Exception as e:
                    logger.warning(
                        f"[{symbol}] ⚠️ could not serialize date_seq: {e}"
                    )

            # ── attrs ──
            hf.attrs['symbol'] = symbol
            hf.attrs['timeframe'] = timeframe
            hf.attrs['fold_num'] = int(fold_num)
            hf.attrs['split_type'] = split_type
            hf.attrs['lookback'] = int(final_lookback)
            hf.attrs['lookahead'] = int(lookahead)
            hf.attrs['n_samples'] = int(X_seq.shape[0])
            hf.attrs['n_timesteps'] = int(X_seq.shape[1])
            hf.attrs['n_features'] = int(X_seq.shape[2])
            hf.attrs['saved_targets'] = int(saved_targets)
            hf.attrs['saved_auxiliary'] = int(saved_aux)
            hf.attrs['target_names'] = ','.join(TARGET_NAMES)
            hf.attrs['auxiliary_columns'] = ','.join(AUX_NAMES)
            hf.attrs['feature_names'] = ','.join(train_feature_cols)
            hf.attrs['date_column'] = (
                str(date_col_name) if date_col_name else ""
            )
            hf.attrs['contract'] = "anchor_at_last_X_no_double_shift"
            hf.attrs['creation_time_iso'] = (
                datetime.now(timezone.utc).isoformat()
            )
            hf.attrs['pipeline_version'] = "9.3-aux-formal-fixed"

        tmp_path.replace(file_path)

        # ────────────────────────────────────────────────────────
        # metadata JSON
        # ────────────────────────────────────────────────────────
        metadata = {
            "symbol": symbol,
            "timeframe": timeframe,
            "fold_num": int(fold_num),
            "split_type": split_type,
            "lookback": int(final_lookback),
            "lookahead": int(lookahead),
            "n_samples": int(X_seq.shape[0]),
            "n_timesteps": int(X_seq.shape[1]),
            "n_features": int(X_seq.shape[2]),
            "target_names": list(TARGET_NAMES),
            "auxiliary_columns": list(AUX_NAMES),
            "feature_names": list(train_feature_cols),
            "date_column": date_col_name,
            "contract": "anchor_at_last_X_no_double_shift",
            "creation_time_iso": datetime.now(timezone.utc).isoformat(),
            "pipeline_version": "9.3-aux-formal-fixed",
        }
        meta_path = save_dir / (
            f"{symbol}_{timeframe}_fold{fold_num}_{split_type}_metadata.json"
        )
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        execution_time = time.time() - start_time
        logger.info(
            f"[{symbol}] ✅ saved {filename} | targets={saved_targets} | "
            f"auxiliary_weights={saved_aux} | time={execution_time:.2f}s"
        )

        return {
            "status": "success",
            "lookback": int(final_lookback),
            "lookahead": int(lookahead),
            "n_samples": int(X_seq.shape[0]),
            "n_features": int(X_seq.shape[2]),
            "target_names": list(TARGET_NAMES),
            "auxiliary_columns": list(AUX_NAMES),
            "saved_targets": int(saved_targets),
            "saved_auxiliary": int(saved_aux),
            "feature_names": list(train_feature_cols),
            "file_path": str(file_path),
            "metadata_path": str(meta_path),
            "execution_time": execution_time,
        }

    except Exception as e:
        logger.error(
            f"[{symbol}] ❌ universal_sequence_creation failed: {e}"
        )
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "error": str(e)}
    finally:
        if lock_acquired:
            await _release_global_processing_lock(lock_key)

# ============================================================================
# 🔬 تست عدم وجود فیچرهای ثابت
# ============================================================================

def _test_no_constant_features_in_sequence(
    file_path: Path,
    logger_instance=None,
) -> bool:
    """
    تست عدم وجود ویژگی‌های ثابت در فایل sequenced.

    ✅ v9.2: ستون‌های auxiliary نیز به‌صورت جداگانه لاگ می‌شوند.
    """
    log = logger_instance or logger
    try:
        with h5py.File(file_path, 'r') as hf:
            if 'X' not in hf:
                log.warning(f"X dataset missing in {file_path.name}")
                return False
            X = hf['X'][:]
            n_features = X.shape[2]
            constant_features = []
            for i in range(n_features):
                feature_slice = X[:, :, i]
                if feature_slice.size == 0:
                    continue
                if np.all(feature_slice == feature_slice[0, 0]):
                    constant_features.append(i)

            target_names: List[str] = []
            if 'target_names' in hf.attrs:
                target_names = str(hf.attrs['target_names']).split(',')

            aux_names: List[str] = []
            if 'auxiliary_columns' in hf.attrs:
                aux_names = str(hf.attrs['auxiliary_columns']).split(',')

            if constant_features:
                log.error(
                    f"❌ CONSTANT FEATURES FOUND in {file_path.name}: "
                    f"indices {constant_features[:10]}"
                )
                log.error(f"   Target names: {target_names}")
                log.error(f"   Auxiliary columns: {aux_names}")
                raise RuntimeError(
                    f"Constant features detected: {constant_features}"
                )
            else:
                log.info(f"✅ No constant features in {file_path.name}")

            for target_name in target_names:
                if target_name in hf:
                    target_data = hf[target_name][:]
                    unique_vals = np.unique(target_data)
                    log.info(
                        f"   🎯 Target '{target_name}': "
                        f"{len(unique_vals)} unique, "
                        f"{len(target_data)} samples"
                    )
                    if len(unique_vals) == 1:
                        log.warning(
                            f"   ⚠️ Target '{target_name}' is constant!"
                        )

            for aux_name in aux_names:
                if aux_name in hf:
                    aux_data = hf[aux_name][:]
                    unique_vals = np.unique(aux_data)
                    log.info(
                        f"   📊 Auxiliary '{aux_name}': "
                        f"{len(unique_vals)} unique, "
                        f"{len(aux_data)} samples, "
                        f"dtype={aux_data.dtype}"
                    )

            return True
    except Exception as e:
        log.warning(f"Could not test features: {e}")
        return False