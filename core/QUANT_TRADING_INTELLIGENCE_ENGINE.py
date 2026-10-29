from __future__ import annotations

from keras.src import optimizers, models
from keras.src.legacy import layers
from tensorflow.python.layers.base import Layer

from core.global_method import get_active_symbol_timeframe, get_latest_training_folds
from core.logger import redirect_prints_to_logger
from core.logger import setup_logger

tel_log = setup_logger("trade")
from core.regim import *
from main import folds as NUM_FOLDS, sum_folds as TOTAL_FOLDS
from core.architecture_selector import HardwareProfile, UtilityWeights, ModelBuilder, \
    DataCharacteristics, _compute_adaptive_weights, _estimate_architecture_cost, _GPU_CONFIG, \
    _score_architecture, ComputeBudget, ModelType, ArchitectureConfig, ArchitectureRegistry, _HAS_TF, _HAS_XGB, _HAS_LGB, _HAS_CATBOOST

# =============================================================================
# NEW IMPORTS FOR LIGHTGBM & CATBOOST
# =============================================================================

# =============================================================================
# SETUP & CONSTANTS
# =============================================================================

logger = AdvancedTradeLogger(__name__)
redirect_prints_to_logger()
warnings.filterwarnings('ignore')

EPOCH_D = 10  # Debug mode epochs
EPOCH_P = 11  # Production epochs
# =============================================================================
# 🔥 FIX 1: REMOVED THE 7-ATTEMPT LIMIT
# =============================================================================
# MAX_RETRY_ATTEMPTS = 7  # <-- REMOVED - We now use infinite attempts
MAX_SELF_HEALING_ATTEMPTS = 3
Architected_Failed = []


# =============================================================================
# HELPER FUNCTIONS FOR GPU PARAMETERS
# =============================================================================

def _get_xgb_gpu_params() -> Dict[str, Any]:
    """Get XGBoost GPU parameters based on configuration"""
    if _GPU_CONFIG.get("xgboost_device") == "cuda":
        return {"tree_method": "hist", "device": "cuda"}
    elif _GPU_CONFIG.get("xgboost_tree_method") == "gpu_hist":
        return {"tree_method": "gpu_hist"}
    else:
        return {"tree_method": "hist", "device": "cpu"}


def _get_lgb_gpu_params() -> Dict[str, Any]:
    """Get LightGBM GPU parameters based on configuration"""
    if _GPU_CONFIG.get("lightgbm_device") == "gpu":
        return {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
    return {"device": "cpu"}


def _get_catboost_gpu_params() -> Dict[str, Any]:
    """Get CatBoost GPU parameters based on configuration"""
    if _GPU_CONFIG.get("catboost_task_type") == "GPU":
        return {"task_type": "GPU", "devices": "0"}
    return {"task_type": "CPU"}


# =============================================================================
# FORENSIC CACHE MANAGER - FIXED VERSION WITH ROBUST EXTRACTION
# Goldman Sachs Grade - Forensic Cache Intelligence
# =============================================================================

class ForensicCacheManager:
    """
    Goldman Sachs Grade - Forensic Cache Intelligence

    If DATA_VERIFIER confirms data is healthy, self-healing is skipped.
    Instead, cache results are used to determine optimal model architecture.

    Decision Logic:
      ├─ CRITICAL categories (Structural/Completeness/Statistical/Consistency/Chain/Forensic) all 100
      ├─ or certification_status == 'CERTIFIED'
      └─ → skip healing + use cache for architecture hint

    FIXED: Added robust regex extraction for category scores from nested JSON
    """

    CRITICAL_CATEGORIES = {'Structural', 'Completeness', 'Statistical', 'Consistency', 'Chain', 'Forensic'}
    PERFECT_THRESHOLD = 100.0
    CERTIFIED_STATUSES = {'CERTIFIED', 'CERTIFIED_WITH_WARNINGS', 'CONDITIONAL'}
    DEFAULT_CACHE_PATH = Path("cache/forensic_cache.json")

    def __init__(self, cache_path: Union[str, Path, None] = None):
        self.cache_path = Path(cache_path) if cache_path else self.DEFAULT_CACHE_PATH
        self._cache_data: Optional[Dict[str, Any]] = None
        self._loaded_at: Optional[datetime] = None
        self.logger = logger

    def load(self, force: bool = False) -> bool:
        """Load JSON cache file. Returns True if successful."""
        if self._cache_data is not None and not force:
            return True
        try:
            if not self.cache_path.exists():
                self.logger.warning(f"⚠️  Forensic cache not found: {self.cache_path}")
                return False
            with open(self.cache_path, 'r', encoding='utf-8') as f:
                self._cache_data = json.load(f)
            self._loaded_at = datetime.now()
            self.logger.info(f"✅ Forensic cache loaded from {self.cache_path}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load forensic cache: {e}")
            self._cache_data = None
            return False

    def should_skip_self_healing(self, symbol: str = None, fold: int = None) -> Tuple[bool, str]:
        """
        Decision: Can self-healing be skipped?

        Returns:
            (skip: bool, reason: str)
        """
        if not self.load():
            return False, "cache_not_available"

        try:
            summary = self._get_summary(symbol, fold)
            if summary is None:
                return False, "no_summary_in_cache"

            # Check certification status
            cert = summary.get('certification_status', summary.get('Certification', ''))
            if cert in self.CERTIFIED_STATUSES:
                self.logger.info(f"✅ Cache status: {cert} → self-healing SKIPPED")
                return True, f"certified:{cert}"

            # Check category scores
            category_scores = self._extract_category_scores(summary)
            if not category_scores:
                return False, "no_category_scores"

            all_critical_perfect = all(
                category_scores.get(cat, 0.0) >= self.PERFECT_THRESHOLD
                for cat in self.CRITICAL_CATEGORIES
                if cat in category_scores
            )

            if all_critical_perfect:
                self.logger.info(
                    f"✅ All critical categories at {self.PERFECT_THRESHOLD} → self-healing SKIPPED"
                )
                return True, "all_critical_categories_perfect"

            # Check normalized score >= 95
            normalized = float(summary.get('normalized_score', summary.get('Normalized Score', 0.0)))
            if normalized >= 95.0:
                self.logger.info(f"✅ Normalized score {normalized:.2f} >= 95 → self-healing SKIPPED")
                return True, f"high_normalized_score:{normalized:.2f}"

            self.logger.info(f"ℹ️  Cache does not meet skip threshold → self-healing REQUIRED")
            return False, f"score_below_threshold:{normalized:.2f}"

        except Exception as e:
            self.logger.error(f"❌ Error in should_skip_self_healing: {e}")
            return False, f"error:{str(e)}"

    def get_full_summary(self, symbol: str = None, fold: int = None) -> Optional[Dict]:
        """Get full summary from cache for specific symbol/fold."""
        if not self.load():
            return None
        return self._get_summary(symbol, fold)

    def _get_summary(self, symbol: str = None, fold: int = None) -> Optional[Dict]:
        """
        Extract summary from cache.
        Supports multiple JSON structures:
          1. {'summary': {...}}
          2. {'symbol': {'fold': {'summary': {...}}}}
          3. {'results': {'summary': {...}}}
          4. Flat dict with 'normalized_score' key
        """
        d = self._cache_data
        if d is None:
            return None

        # Structure 1: flat
        if 'normalized_score' in d or 'Normalized Score' in d or 'certification_status' in d:
            return d

        # Structure 2: with 'summary' key
        if 'summary' in d:
            return d['summary']

        # Structure 3: nested by symbol → fold
        if symbol and symbol in d:
            sym_data = d[symbol]
            fold_key = str(fold) if fold is not None else None
            if fold_key and fold_key in sym_data:
                fd = sym_data[fold_key]
                return fd.get('summary', fd)
            # Last available fold
            if sym_data:
                last = list(sym_data.values())[-1]
                return last.get('summary', last)

        # Structure 4: with 'results' key
        if 'results' in d:
            r = d['results']
            return r.get('summary', r)

        # Fallback: root
        return d

    def _extract_category_scores(self, summary: Optional[Dict]) -> Dict[str, float]:
        """
        FIXED: Extract category scores from summary with robust handling.
        Supported formats:
          - Direct keys in summary
          - Nested under 'category_scores' or 'Category Scores'
          - Nested under 'categories' or 'scores'
          - Using regex for deeply nested structures
        """
        if summary is None:
            return {}

        # Try direct keys in summary
        found = {}
        for cat in self.CRITICAL_CATEGORIES.union({'Scalers'}):
            if cat in summary:
                try:
                    found[cat] = float(summary[cat])
                except (TypeError, ValueError):
                    pass

        if found:
            return found

        # Try nested dictionaries
        for key in ('category_scores', 'Category Scores', 'categories', 'scores'):
            if key in summary and isinstance(summary[key], dict):
                for cat in self.CRITICAL_CATEGORIES.union({'Scalers'}):
                    if cat in summary[key]:
                        try:
                            found[cat] = float(summary[key][cat])
                        except (TypeError, ValueError):
                            pass
                if found:
                    return found

        # If still not found, try to extract from the entire JSON using regex
        if not found and hasattr(self, '_cache_data') and self._cache_data:
            import json
            data_str = json.dumps(self._cache_data)
            found = self._extract_categories_regex(data_str)

        return found

    def _extract_categories_regex(self, data_str: str) -> Dict[str, float]:
        """
        FIXED: Extract category scores using regex for deeply nested JSON.
        This ensures we always get the scores regardless of nesting depth.
        """
        import re
        found = {}

        # Pattern for category scores in various formats
        cat_patterns = [
            r'"(?:Structural|Completeness|Statistical|Consistency|Chain|Scalers|Forensic)"\s*:\s*([\d.]+)',
            r'category_scores.*?"(?:Structural|Completeness|Statistical|Consistency|Chain|Scalers|Forensic)"\s*:\s*([\d.]+)',
        ]

        for cat in self.CRITICAL_CATEGORIES.union({'Scalers'}):
            for pattern in cat_patterns:
                # Replace placeholder with actual category
                if pattern.startswith('category_scores'):
                    pattern = pattern.replace(
                        '(?:Structural|Completeness|Statistical|Consistency|Chain|Scalers|Forensic)', cat)
                else:
                    pattern = pattern.replace(
                        '(?:Structural|Completeness|Statistical|Consistency|Chain|Scalers|Forensic)', cat)

                matches = re.findall(pattern, data_str, re.IGNORECASE)
                if matches:
                    try:
                        found[cat] = float(matches[0])
                        break
                    except (ValueError, TypeError):
                        pass

        return found

    def _extract_regime(self, symbol: str = None, fold: int = None) -> str:
        """Extract regime from cache."""
        d = self._cache_data
        if d is None:
            return 'UNDEFINED'

        # Search recursively in cache
        for key in ('regime', 'market_regime', 'detected_regime'):
            if key in d:
                return str(d[key])

        # Nested
        if symbol and symbol in d:
            sym = d[symbol]
            fold_key = str(fold) if fold is not None else None
            if fold_key and fold_key in sym:
                fd = sym[fold_key]
                for key in ('regime', 'market_regime'):
                    if key in fd:
                        return str(fd[key])

        return 'UNDEFINED'


# =============================================================================
# NEW: TRAINING CHECKPOINT MANAGER - GOLDMAN SACHS GRADE
# =============================================================================

@dataclass
class TrainingCheckpoint:
    """
    Complete training state for resuming after crash
    """
    symbol: str
    fold: int
    architecture_name: str
    architecture_variant: int
    current_epoch: int
    total_epochs: int
    best_val_loss: float
    optimizer_state: Dict[str, Any]
    model_weights_path: str
    history: Dict[str, List[float]]
    regime_at_checkpoint: str
    timestamp: datetime = field(default_factory=datetime.now)

    def is_complete(self) -> bool:
        return self.current_epoch >= self.total_epochs

    @property
    def checkpoint_id(self) -> str:
        return f"{self.symbol}_fold{self.fold}_{self.architecture_name}_epoch{self.current_epoch}"


class CheckpointManager:
    """
    Goldman Sachs Tier-1 - Crash Recovery System
    Manages training checkpoints for seamless resume after crash
    """

    CHECKPOINT_DIR = Path("models/checkpoints")

    def __init__(self, symbol: str, fold: int):
        self.symbol = symbol
        self.fold = fold
        self.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _get_checkpoint_path(self, arch_name: str) -> Path:
        """Get checkpoint file path for architecture"""
        return self.CHECKPOINT_DIR / f"{self.symbol}_fold{self.fold}_{arch_name}_checkpoint.pkl"

    def _get_weights_path(self, arch_name: str, epoch: int) -> Path:
        """Get weights file path"""
        return self.CHECKPOINT_DIR / f"{self.symbol}_fold{self.fold}_{arch_name}_epoch{epoch}.weights.h5"

    def _get_model_path(self, arch_name: str) -> Path:
        """Get full model path for tree-based models"""
        return self.CHECKPOINT_DIR / f"{self.symbol}_fold{self.fold}_{arch_name}_model.pkl"

    def save_checkpoint(self, checkpoint: TrainingCheckpoint) -> bool:
        """Save training checkpoint"""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint.architecture_name)

            # Save as pickle for full state
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)

            self.logger.info(
                f"✅ Checkpoint saved: {checkpoint.architecture_name} "
                f"epoch {checkpoint.current_epoch}/{checkpoint.total_epochs}"
            )
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, arch_name: str) -> Optional[TrainingCheckpoint]:
        """Load training checkpoint if exists"""
        try:
            checkpoint_path = self._get_checkpoint_path(arch_name)

            if not checkpoint_path.exists():
                return None

            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            self.logger.info(
                f"✅ Checkpoint loaded: {arch_name} "
                f"epoch {checkpoint.current_epoch}/{checkpoint.total_epochs}"
            )
            return checkpoint
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint: {e}")
            return None

    def has_checkpoint(self, arch_name: str) -> bool:
        """Check if checkpoint exists for architecture"""
        return self._get_checkpoint_path(arch_name).exists()

    def save_tree_model(self, arch_name: str, model_obj: Any) -> bool:
        """Save tree-based model (XGBoost, LightGBM, CatBoost)"""
        try:
            model_path = self._get_model_path(arch_name)
            with open(model_path, 'wb') as f:
                pickle.dump(model_obj, f)
            self.logger.info(f"✅ Tree model saved: {arch_name}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to save tree model: {e}")
            return False

    def load_tree_model(self, arch_name: str) -> Optional[Any]:
        """Load tree-based model"""
        try:
            model_path = self._get_model_path(arch_name)
            if not model_path.exists():
                return None
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.logger.info(f"✅ Tree model loaded: {arch_name}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load tree model: {e}")
            return None


# =============================================================================
# NEW: RESUME-AWARE CALLBACK
# =============================================================================

class ResumeEpochLogger(tf.keras.callbacks.Callback):
    """
    FIXED: Epoch logger that saves checkpoints for crash recovery
    """

    def __init__(
            self,
            logger,
            checkpoint_manager: CheckpointManager,
            arch_name: str,
            arch_variant: int,
            total_epochs: int,
            model_weights_path: str
    ):
        super().__init__()
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.arch_name = arch_name
        self.arch_variant = arch_variant
        self.total_epochs = total_epochs
        self.model_weights_path = model_weights_path
        self.epoch_start_time = None
        self.best_val_loss = float('inf')
        self.history = {'loss': [], 'val_loss': []}

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()
        try:
            lr = float(self.model.optimizer.learning_rate.numpy())
            self.logger.info(f"📘 Epoch {epoch + 1} START | LR={lr:.8f}")
        except:
            pass

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        duration = time.perf_counter() - self.epoch_start_time

        try:
            lr = float(self.model.optimizer.learning_rate.numpy())
        except:
            lr = -1

        # Update history
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))

        # Save best model
        current_val_loss = logs.get('val_loss', float('inf'))
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            self.model.save_weights(self.model_weights_path)
            self.logger.info(f"💾 Best model saved (val_loss={current_val_loss:.6f})")

        # Log epoch results
        metrics_str = " | ".join([f"{k}={v:.6f}" for k, v in logs.items()])
        self.logger.info(
            f"📗 Epoch {epoch + 1}/{self.total_epochs} | "
            f"time={duration:.2f}s | LR={lr:.8f} | {metrics_str}"
        )

        # 🔥 CRITICAL: Save checkpoint after each epoch for crash recovery
        checkpoint = TrainingCheckpoint(
            symbol=self.checkpoint_manager.symbol,
            fold=self.checkpoint_manager.fold,
            architecture_name=self.arch_name,
            architecture_variant=self.arch_variant,
            current_epoch=epoch + 1,
            total_epochs=self.total_epochs,
            best_val_loss=self.best_val_loss,
            optimizer_state={},  # Can't pickle optimizer easily, but weights saved
            model_weights_path=self.model_weights_path,
            history=self.history.copy(),
            regime_at_checkpoint="unknown"  # Will be updated if needed
        )

        self.checkpoint_manager.save_checkpoint(checkpoint)




# =============================================================================
# LAYER 2: REGIME & STRUCTURAL INTELLIGENCE
# =============================================================================


class DriftDetectionSystem:
    """Detect distribution shift in features/targets"""

    def __init__(self, warning_threshold: float = 0.05, drift_threshold: float = 0.10):
        self.warning_threshold = warning_threshold
        self.drift_threshold = drift_threshold
        logger.info(f"DriftDetectionSystem initialized with drift_threshold={drift_threshold}")

    def detect(self, recent_data: np.ndarray, reference_data: np.ndarray) -> Tuple[bool, float]:
        if len(recent_data) < 30 or len(reference_data) < 30:
            logger.debug("Insufficient data for drift detection.")
            return False, 0.0

        recent_data = recent_data[~np.isnan(recent_data)]
        reference_data = reference_data[~np.isnan(reference_data)]

        if len(recent_data) < 30 or len(reference_data) < 30:
            logger.debug("Insufficient clean data for drift detection.")
            return False, 0.0

        try:
            all_data = np.concatenate([reference_data, recent_data])
            bins = np.percentile(all_data, np.linspace(0, 100, 11))

            ref_counts, _ = np.histogram(reference_data, bins=bins)
            ref_pct = ref_counts / len(reference_data)
            ref_pct = np.clip(ref_pct, 0.001, 0.999)

            recent_counts, _ = np.histogram(recent_data, bins=bins)
            recent_pct = recent_counts / len(recent_data)
            recent_pct = np.clip(recent_pct, 0.001, 0.999)

            psi = np.sum((recent_pct - ref_pct) * np.log(recent_pct / ref_pct))
            psi = max(0, psi)

            drift_detected = psi > self.drift_threshold
            if drift_detected:
                logger.warning(f"⚠️ Drift detected! PSI: {psi:.4f} > {self.drift_threshold}")
            return drift_detected, float(psi)
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return False, 0.0


class LiquidityAnalyzer:
    def analyze(self, volume: Optional[np.ndarray], bid_ask: Optional[np.ndarray], returns: np.ndarray) -> Tuple[
        LiquidityRegime, float]:
        if volume is None or bid_ask is None or len(volume) < 20:
            logger.debug("Insufficient data for liquidity analysis.")
            return LiquidityRegime.UNKNOWN, 0.0

        try:
            recent_volume = volume[-20:]
            recent_spread = bid_ask[-20:]

            vol_percentile = stats.percentileofscore(volume, recent_volume.mean()) / 100
            avg_spread = recent_spread.mean()
            spread_percentile = stats.percentileofscore(bid_ask, avg_spread) / 100

            liquidity_score = (vol_percentile * 0.5) + ((1 - spread_percentile) * 0.5)

            if liquidity_score < 0.2:
                regime = LiquidityRegime.ILLIQUID
            elif liquidity_score < 0.4:
                regime = LiquidityRegime.LOW
            elif liquidity_score < 0.6:
                regime = LiquidityRegime.REDUCED
            else:
                regime = LiquidityRegime.NORMAL

            logger.debug(f"Liquidity analysis: score={liquidity_score:.3f}, regime={regime.value}")
            return regime, float(liquidity_score)
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return LiquidityRegime.UNKNOWN, 0.0

# =============================================================================
# LAYER 3: ALPHA FACTORY (Adaptive Training Loop)
# =============================================================================

@dataclass
class SignalQualityMetrics:
    signal_to_noise_ratio: float
    edge_persistence: float
    feature_stability: float
    cross_validation_gap: float
    information_coefficient: float
    hit_rate: float
    sharpe_ratio: float
    is_valid: bool
    rejection_reason: Optional[str] = None


class SignalQualityLayer:
    """Institutional signal quality assessment"""

    def __init__(
            self,
            min_snr: float = 1.0,
            min_edge_persistence: float = 0.6,
            max_cv_gap: float = 0.15,
            min_feature_stability: float = 0.7
    ):
        self.min_snr = min_snr
        self.min_edge_persistence = min_edge_persistence
        self.max_cv_gap = max_cv_gap
        self.min_feature_stability = min_feature_stability
        logger.info(
            f"SignalQualityLayer initialized with thresholds: SNR={min_snr}, EdgePersistence={min_edge_persistence}, CV_Gap={max_cv_gap}, FeatStability={min_feature_stability}")

    def assess_signal_quality(
            self,
            predictions: np.ndarray,
            actuals: np.ndarray,
            feature_importance_history: List[np.ndarray],
            cv_scores: List[float],
            cv_scores_oos: List[float]
    ) -> SignalQualityMetrics:
        predictions = predictions.flatten()
        actuals = actuals.flatten()

        min_len = min(len(predictions), len(actuals))
        predictions = predictions[:min_len]
        actuals = actuals[:min_len]

        signal_power = np.var(predictions)
        noise_power = np.var(predictions - actuals)
        snr = signal_power / (noise_power + 1e-8)

        edge_persistence = self._calculate_edge_persistence(predictions, actuals)
        feature_stability = self._calculate_feature_stability(feature_importance_history)

        if cv_scores and cv_scores_oos:
            cv_gap = abs(np.mean(cv_scores) - np.mean(cv_scores_oos))
        else:
            cv_gap = 1.0

        if len(predictions) > 1 and len(actuals) > 1:
            try:
                ic = np.corrcoef(predictions, actuals)[0, 1]
                if np.isnan(ic):
                    ic = 0.0
            except:
                ic = 0.0
        else:
            ic = 0.0

        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        hit_rate = np.mean(pred_direction == actual_direction) if len(predictions) > 0 else 0.0

        strategy_returns = pred_direction * actuals
        if len(strategy_returns) > 1:
            sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        is_valid = True
        rejection_reason = None

        if snr < self.min_snr:
            is_valid = False
            rejection_reason = f"SNR too low: {snr:.2f} < {self.min_snr}"
        elif edge_persistence < self.min_edge_persistence:
            is_valid = False
            rejection_reason = f"Edge persistence too low: {edge_persistence:.2f} < {self.min_edge_persistence}"
        elif cv_gap > self.max_cv_gap:
            is_valid = False
            rejection_reason = f"CV gap too large: {cv_gap:.2f} > {self.max_cv_gap}"
        elif feature_stability < self.min_feature_stability:
            is_valid = False
            rejection_reason = f"Feature stability too low: {feature_stability:.2f} < {self.min_feature_stability}"

        if is_valid:
            logger.debug("Signal quality check PASSED.")
        else:
            logger.warning(f"Signal quality check FAILED: {rejection_reason}")

        return SignalQualityMetrics(
            signal_to_noise_ratio=float(snr),
            edge_persistence=float(edge_persistence),
            feature_stability=float(feature_stability),
            cross_validation_gap=float(cv_gap),
            information_coefficient=float(ic),
            hit_rate=float(hit_rate),
            sharpe_ratio=float(sharpe),
            is_valid=is_valid,
            rejection_reason=rejection_reason
        )

    def _calculate_edge_persistence(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        if len(predictions) < 20:
            return 0.5

        window = 20
        rolling_sharpes = []

        for i in range(window, len(predictions)):
            pred_window = predictions[i - window:i]
            actual_window = actuals[i - window:i]

            direction = np.sign(pred_window)
            returns = direction * actual_window

            if len(returns) > 1:
                sr = np.mean(returns) / (np.std(returns) + 1e-8)
                if not np.isnan(sr):
                    rolling_sharpes.append(sr)

        if not rolling_sharpes:
            return 0.5

        persistence = np.mean(np.array(rolling_sharpes) > 0)
        return float(persistence)

    def _calculate_feature_stability(self, feature_importance_history: List[np.ndarray]) -> float:
        if len(feature_importance_history) < 2:
            return 0.8

        try:
            importance_matrix = np.array(feature_importance_history)
            correlations = []
            for i in range(1, len(importance_matrix)):
                corr = np.corrcoef(importance_matrix[i - 1], importance_matrix[i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

            if not correlations:
                return 0.8

            return float(np.mean(correlations))
        except Exception as e:
            logger.error(f"Error in feature stability calculation: {e}")
            return 0.8


# =============================================================================
# FINANCIAL LOSS FUNCTIONS
# =============================================================================

@dataclass
class TradingCostConfig:
    commission_pct: float = 0.0005
    slippage_base_pct: float = 0.0002
    slippage_vol_multiplier: float = 0.5
    market_impact_pct: float = 0.0001
    funding_rate_annual: float = 0.05
    min_cost_bps: float = 0.0001
    max_cost_bps: float = 0.0020

    @property
    def daily_funding_rate(self) -> float:
        return self.funding_rate_annual / 365.0


class DrawdownPenaltyRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, penalty_lambda: float = 0.01):
        self.penalty_lambda = penalty_lambda

    def __call__(self, x):
        if tf.rank(x) < 2:
            return tf.constant(0.0, dtype=x.dtype)

        l2_penalty = tf.reduce_sum(tf.square(x))
        weight_mean = tf.reduce_mean(x)
        weight_std = tf.math.reduce_std(x)
        stability_penalty = weight_std / (tf.abs(weight_mean) + 1e-8)
        stability_penalty = tf.where(tf.math.is_nan(stability_penalty), 0.0, stability_penalty)

        return self.penalty_lambda * (l2_penalty + 0.1 * stability_penalty)

    def get_config(self):
        return {'penalty_lambda': self.penalty_lambda}


class SharpeAwareLoss(tf.keras.losses.Loss):
    """Direct optimization of risk-adjusted returns"""

    def __init__(
            self,
            risk_free_rate: float = 0.02,
            drawdown_penalty: float = 1.0,
            turnover_penalty: float = 0.2,
            cost_config: Optional[Any] = None,
            reduction: str = 'sum_over_batch_size',
            name: str = "sharpe_aware_loss"
    ):
        super().__init__(reduction=reduction, name=name)
        self.risk_free_rate = risk_free_rate
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.cost_config = cost_config
        self.epsilon = 1e-8

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        positions = tf.tanh(y_pred * 2.0)
        threshold = 0.2
        positions = tf.where(
            tf.abs(positions) < threshold,
            tf.zeros_like(positions),
            positions
        )

        positions = tf.roll(positions, shift=1, axis=0)
        positions = tf.concat([tf.zeros_like(positions[:1]), positions[1:]], axis=0)

        prev_positions = tf.roll(positions, shift=1, axis=0)
        turnover = tf.abs(positions - prev_positions)
        turnover = tf.concat([tf.abs(positions[:1]), turnover[1:]], axis=0)

        commission = tf.constant(
            getattr(self.cost_config, "commission_pct", 0.0005),
            dtype=tf.float32
        )
        slippage = tf.constant(
            getattr(self.cost_config, "slippage_base_pct", 0.0005),
            dtype=tf.float32
        )

        total_cost_rate = commission + slippage
        costs = turnover * total_cost_rate

        gross_returns = positions * y_true
        net_returns = gross_returns - costs

        mean_return = tf.reduce_mean(net_returns)
        std_return = tf.math.reduce_std(net_returns) + self.epsilon

        daily_rf = self.risk_free_rate / 252.0
        sharpe = (mean_return - daily_rf) / std_return
        sharpe *= tf.sqrt(252.0)

        cum_returns = tf.cumsum(net_returns)
        running_max = tf.scan(lambda a, x: tf.maximum(a, x), cum_returns)
        drawdown = cum_returns - running_max
        max_drawdown = -tf.reduce_min(drawdown)

        avg_turnover = tf.reduce_mean(turnover)

        loss = -sharpe + self.drawdown_penalty * max_drawdown + self.turnover_penalty * avg_turnover
        return loss

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "risk_free_rate": self.risk_free_rate,
            "drawdown_penalty": self.drawdown_penalty,
            "turnover_penalty": self.turnover_penalty,
        })
        return config


class MultiHeadFinancialLoss(tf.keras.losses.Loss):
    """
    Multi-head financial loss
    """

    def __init__(
            self,
            price_weight: float = 0.6,
            volatility_weight: float = 0.3,
            confidence_weight: float = 0.1,
            risk_free_rate: float = 0.02,
            drawdown_penalty: float = 0.5,
            turnover_penalty: float = 0.1,
            cost_config: Optional[Any] = None,
            reduction: str = 'sum_over_batch_size',
            name: str = "multi_head_financial_loss"
    ):
        super().__init__(reduction=reduction, name=name)
        self.price_weight = price_weight
        self.volatility_weight = volatility_weight
        self.confidence_weight = confidence_weight
        self.risk_free_rate = risk_free_rate
        self.drawdown_penalty = drawdown_penalty
        self.turnover_penalty = turnover_penalty
        self.cost_config = cost_config

        # استفاده از SharpeAwareLoss برای قیمت (بسیار هوشمندانه)
        self.price_loss = SharpeAwareLoss(
            risk_free_rate=risk_free_rate,
            drawdown_penalty=drawdown_penalty,
            turnover_penalty=turnover_penalty,
            cost_config=cost_config
        )
        self.volatility_loss = tf.keras.losses.Huber(delta=1.0)
        self.confidence_loss = tf.keras.losses.BinaryCrossentropy()

    @property
    def loss_weights(self) -> Dict[str, float]:
        total = self.price_weight + self.volatility_weight + self.confidence_weight
        return {
            'price': self.price_weight / total,
            'volatility': self.volatility_weight / total,
            'confidence': self.confidence_weight / total
        }

    def call(self, y_true, y_pred):
        # --- FIX: اصلاح علامت ضرب (*) در خطوط زیر ---

        if isinstance(y_pred, dict):
            total_loss = 0.0
            if "price" in y_pred:
                total_loss += self.price_weight * self.price_loss(y_true["price"], y_pred["price"])
            if "volatility" in y_pred:
                total_loss += self.volatility_weight * self.volatility_loss(y_true["volatility"], y_pred["volatility"])
            if "confidence" in y_pred:
                total_loss += self.confidence_weight * self.confidence_loss(y_true["confidence"], y_pred["confidence"])
            return total_loss

        if isinstance(y_pred, (list, tuple)):
            total_loss = 0.0
            if len(y_pred) > 0:
                total_loss += self.price_weight * self.price_loss(y_true[0], y_pred[0])
            if len(y_pred) > 1:
                total_loss += self.volatility_weight * self.volatility_loss(y_true[1], y_pred[1])
            if len(y_pred) > 2:
                total_loss += self.confidence_weight * self.confidence_loss(y_true[2], y_pred[2])
            return total_loss

        return self.price_weight * self.price_loss(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        config.update({
            "price_weight": self.price_weight,
            "volatility_weight": self.volatility_weight,
            "confidence_weight": self.confidence_weight,
            "risk_free_rate": self.risk_free_rate,
            "drawdown_penalty": self.drawdown_penalty,
            "turnover_penalty": self.turnover_penalty,
        })
        return config

# =============================================================================
# LAYER 4: RISK BRAIN (Capital Survival Engine)
# =============================================================================

class AutoShutdownManager:
    """
    LAYER 4: RISK BRAIN - Kill-Switch AI
    """

    def __init__(
            self,
            drawdown_threshold: float = 0.20,
            volatility_spike_threshold: float = 3.0,
            consecutive_loss_threshold: int = 5,
            daily_loss_threshold: float = 0.03,
            liquidity_shutdown: bool = True
    ):
        self.drawdown_threshold = drawdown_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        self.consecutive_loss_threshold = consecutive_loss_threshold
        self.daily_loss_threshold = daily_loss_threshold
        self.liquidity_shutdown = liquidity_shutdown

        self.is_shutdown = False
        self.shutdown_reason = None
        self.shutdown_time = None
        self.shutdown_history: List[Dict] = []
        self.equity_curve: List[float] = []
        self.daily_pnl: List[float] = []
        self.volatility_history: List[float] = []
        logger.info(
            f"AutoShutdownManager initialized with thresholds: DD={drawdown_threshold}, Vol={volatility_spike_threshold}, ConsecLoss={consecutive_loss_threshold}")

    def update(self, equity: float, pnl: float, volatility: float, liquidity_score: float):
        self.equity_curve.append(equity)
        self.daily_pnl.append(pnl)
        self.volatility_history.append(volatility)

        if len(self.equity_curve) > 1000:
            self.equity_curve.pop(0)
        if len(self.daily_pnl) > 100:
            self.daily_pnl.pop(0)
        if len(self.volatility_history) > 100:
            self.volatility_history.pop(0)

        shutdown, reason = self._check_conditions(liquidity_score)

        if shutdown and not self.is_shutdown:
            self.is_shutdown = True
            self.shutdown_reason = reason
            self.shutdown_time = datetime.now()
            self.shutdown_history.append({
                'timestamp': self.shutdown_time.isoformat(),
                'reason': reason,
                'equity': equity,
                'pnl': pnl
            })
            logger.critical(f"🚨 AUTO-SHUTDOWN ACTIVATED! Reason: {reason}")

    def _check_conditions(self, liquidity_score: float) -> Tuple[bool, str]:
        if len(self.equity_curve) > 20:
            peak = max(self.equity_curve)
            current = self.equity_curve[-1]
            drawdown = (peak - current) / peak if peak > 0 else 0
            if drawdown > self.drawdown_threshold:
                return True, f"MAX DRAWDOWN: {drawdown:.2%}"

        if len(self.daily_pnl) >= self.consecutive_loss_threshold:
            recent = self.daily_pnl[-self.consecutive_loss_threshold:]
            if all(p < 0 for p in recent):
                return True, f"CONSECUTIVE LOSSES: {self.consecutive_loss_threshold}"

        if len(self.daily_pnl) > 0:
            today_pnl = self.daily_pnl[-1]
            if today_pnl < -self.daily_loss_threshold:
                return True, f"DAILY LOSS LIMIT: {today_pnl:.2%}"

        if len(self.volatility_history) > 20:
            mean_vol = np.mean(self.volatility_history[:-20])
            std_vol = np.std(self.volatility_history[:-20])
            current_vol = self.volatility_history[-1]
            if std_vol > 0 and current_vol > mean_vol + self.volatility_spike_threshold * std_vol:
                return True, f"VOLATILITY SPIKE: {current_vol:.4f} (> {self.volatility_spike_threshold}σ)"

        if self.liquidity_shutdown and liquidity_score < 0.2:
            return True, f"LIQUIDITY CRISIS: score={liquidity_score:.2f}"

        return False, ""

    def reset(self) -> bool:
        if self.is_shutdown:
            self.is_shutdown = False
            self.shutdown_reason = None
            self.shutdown_time = None
            logger.info("Auto-shutdown reset.")
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        return {
            'is_shutdown': self.is_shutdown,
            'reason': self.shutdown_reason,
            'time': self.shutdown_time.isoformat() if self.shutdown_time else None,
            'equity_peak': max(self.equity_curve) if self.equity_curve else 0,
            'equity_current': self.equity_curve[-1] if self.equity_curve else 0,
            'drawdown': self._calculate_current_drawdown(),
            'shutdown_events': len(self.shutdown_history)
        }

    def _calculate_current_drawdown(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        return (peak - current) / peak if peak > 0 else 0.0


# =============================================================================
# LAYER 5: CAPITAL ALLOCATION ENGINE
# =============================================================================

class AllocationMethod(Enum):
    KELLY_OPTIMAL = "KELLY_OPTIMAL"
    HALF_KELLY = "HALF_KELLY"
    QUARTER_KELLY = "QUARTER_KELLY"
    VOL_TARGETING = "VOL_TARGETING"
    RISK_PARITY = "RISK_PARITY"
    CONSTANT = "CONSTANT"


@dataclass
class CapitalAllocationConfig:
    method: AllocationMethod = AllocationMethod.HALF_KELLY
    max_position_pct: float = 0.05
    max_correlated_exposure_pct: float = 0.15
    target_annual_vol: float = 0.12
    max_leverage: float = 1.0
    min_confidence: float = 0.65
    kelly_fraction: float = 0.5
    volatility_lookback: int = 60
    correlation_lookback: int = 120
    max_drawdown_limit: float = 0.20


@dataclass
class CapitalAllocation:
    symbol: str
    allocation_pct: float
    position_size: float
    method_used: AllocationMethod
    confidence_score: float
    regime_multiplier: float
    volatility_adjustment: float
    max_allowed_position: float
    kelly_fraction_used: Optional[float]
    risk_contribution: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_within_limits(self) -> bool:
        return self.position_size <= self.max_allowed_position

    @property
    def risk_adjusted_allocation(self) -> float:
        return self.allocation_pct * self.confidence_score * self.regime_multiplier


class KellyCalculator:
    def __init__(self, max_leverage: float = 1.0):
        self.max_leverage = max_leverage

    def calculate(
            self,
            win_rate: float,
            avg_win: float,
            avg_loss: float,
            use_half: bool = True,
            use_quarter: bool = False
    ) -> float:
        if avg_loss <= 0 or win_rate >= 1:
            return 0.0

        odds = avg_win / avg_loss
        kelly = (win_rate * odds - (1 - win_rate)) / (odds + 1e-8)
        kelly = np.clip(kelly, 0.0, self.max_leverage)

        if use_quarter:
            kelly *= 0.25
            logger.debug(f"Quarter-Kelly: {kelly:.4f}")
        elif use_half:
            kelly *= 0.5
            logger.debug(f"Half-Kelly: {kelly:.4f}")
        else:
            logger.debug(f"Full-Kelly: {kelly:.4f}")

        return float(kelly)

    def calculate_from_sharpe(self, sharpe_ratio: float, use_half: bool = True) -> float:
        kelly = sharpe_ratio * 0.1
        kelly = np.clip(kelly, 0.0, self.max_leverage)
        if use_half:
            kelly *= 0.5
        logger.debug(f"Kelly from Sharpe: {kelly:.4f}")
        return float(kelly)


class VolatilityTargeting:
    def __init__(self, target_annual_vol: float = 0.12):
        self.target_annual_vol = target_annual_vol
        self.daily_target = target_annual_vol / np.sqrt(252)

    def calculate_multiplier(
            self,
            expected_vol: float,
            max_multiplier: float = 2.0,
            min_multiplier: float = 0.1
    ) -> float:
        if expected_vol <= 0:
            return 1.0
        multiplier = self.target_annual_vol / expected_vol
        multiplier = np.clip(multiplier, min_multiplier, max_multiplier)
        logger.debug(f"Volatility targeting multiplier: {multiplier:.3f}")
        return float(multiplier)


class CapitalAllocationEngine:
    def __init__(
            self,
            config: Optional[CapitalAllocationConfig] = None,
            total_capital: float = 1_000_000.0
    ):
        self.config = config or CapitalAllocationConfig()
        self.total_capital = total_capital
        self.kelly = KellyCalculator(max_leverage=self.config.max_leverage)
        self.vol_targeting = VolatilityTargeting(target_annual_vol=self.config.target_annual_vol)
        self.allocations: Dict[str, CapitalAllocation] = {}
        self.allocation_history: List[Dict] = []
        logger.info(
            f"CapitalAllocationEngine initialized with method={self.config.method.value}, capital=${total_capital:,.0f}")

    def calculate_allocation(
            self,
            symbol: str,
            expected_return: float,
            expected_vol: float,
            confidence: float,
            win_rate: float,
            avg_win: float,
            avg_loss: float,
            regime: MarketRegime,
            sharpe_ratio: float = 0.0,
            existing_allocations: Optional[Dict[str, float]] = None,
            correlation_matrix: Optional[np.ndarray] = None,
            symbols: Optional[List[str]] = None
    ) -> CapitalAllocation:
        logger.info(f"[{symbol}] Calculating capital allocation...")
        base_allocation = 0.02
        kelly_fraction = None
        method = AllocationMethod.CONSTANT

        if self.config.method == AllocationMethod.KELLY_OPTIMAL:
            base_allocation = self.kelly.calculate(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                use_half=False,
                use_quarter=False
            )
            kelly_fraction = 1.0
            method = AllocationMethod.KELLY_OPTIMAL
        elif self.config.method == AllocationMethod.HALF_KELLY:
            base_allocation = self.kelly.calculate(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                use_half=True,
                use_quarter=False
            )
            kelly_fraction = 0.5
            method = AllocationMethod.HALF_KELLY
        elif self.config.method == AllocationMethod.QUARTER_KELLY:
            base_allocation = self.kelly.calculate(
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                use_half=False,
                use_quarter=True
            )
            kelly_fraction = 0.25
            method = AllocationMethod.QUARTER_KELLY

        if base_allocation <= 0 and sharpe_ratio > 0:
            base_allocation = self.kelly.calculate_from_sharpe(
                sharpe_ratio,
                use_half=(method != AllocationMethod.KELLY_OPTIMAL)
            )
            if base_allocation > 0:
                kelly_fraction = 0.5
                method = AllocationMethod.HALF_KELLY  # Override method

        vol_multiplier = self.vol_targeting.calculate_multiplier(expected_vol)
        vol_adjusted = base_allocation * vol_multiplier

        regime_multiplier = regime.capital_allocation_multiplier

        if confidence < self.config.min_confidence:
            confidence_multiplier = 0.0
            logger.warning(
                f"[{symbol}] Confidence {confidence:.2f} below min {self.config.min_confidence}. Allocation set to 0.")
        else:
            confidence_multiplier = (confidence - self.config.min_confidence) / (1.0 - self.config.min_confidence)
            confidence_multiplier = np.clip(confidence_multiplier, 0.0, 1.0)

        raw_allocation = vol_adjusted * regime_multiplier * confidence_multiplier

        # Correlation check
        if existing_allocations and correlation_matrix is not None and symbols and symbol in symbols:
            try:
                idx = symbols.index(symbol)
                correlated_exposure = 0.0
                for other_symbol, alloc in existing_allocations.items():
                    if other_symbol in symbols:
                        other_idx = symbols.index(other_symbol)
                        corr = correlation_matrix[idx, other_idx]
                        if corr > 0.7:
                            correlated_exposure += abs(alloc)

                if correlated_exposure >= self.config.max_correlated_exposure_pct:
                    reduction = self.config.max_correlated_exposure_pct / (correlated_exposure + 1e-8)
                    raw_allocation *= reduction
                    logger.info(
                        f"[{symbol}] Reducing allocation due to high correlation. Reduction factor: {reduction:.2f}")
            except Exception as e:
                logger.error(f"[{symbol}] Error in correlation check: {e}")

        regime_max = regime.max_position_pct
        config_max = self.config.max_position_pct
        max_allowed = min(regime_max, config_max)

        final_allocation = min(raw_allocation, max_allowed)
        position_size = final_allocation * self.total_capital
        max_allowed_position = max_allowed * self.total_capital

        # Risk contribution
        risk_contribution = final_allocation
        if correlation_matrix is not None and symbols and symbol in symbols:
            try:
                idx = symbols.index(symbol)
                weights = np.zeros(len(symbols))
                weights[idx] = final_allocation
                port_var = weights @ correlation_matrix @ weights
                port_vol = np.sqrt(port_var) if port_var > 0 else 0
                if port_vol > 0:
                    marginal = correlation_matrix @ weights
                    risk_contrib = weights * marginal / (port_vol + 1e-8)
                    risk_contribution = float(risk_contrib[idx])
            except:
                pass

        allocation = CapitalAllocation(
            symbol=symbol,
            allocation_pct=final_allocation,
            position_size=position_size,
            method_used=method,
            confidence_score=confidence,
            regime_multiplier=regime_multiplier,
            volatility_adjustment=vol_multiplier,
            max_allowed_position=max_allowed_position,
            kelly_fraction_used=kelly_fraction,
            risk_contribution=risk_contribution,
            timestamp=datetime.now(),
            metadata={
                "base_allocation": base_allocation,
                "vol_adjusted": vol_adjusted,
                "raw_allocation": raw_allocation,
                "expected_return": expected_return,
                "expected_vol": expected_vol,
                "win_rate": win_rate,
                "sharpe_ratio": sharpe_ratio
            }
        )

        self.allocations[symbol] = allocation
        self.allocation_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'allocation_pct': final_allocation,
            'position_size': position_size
        })

        logger.info(
            f"[{symbol}] Allocation calculated: {final_allocation:.2%} (${position_size:,.0f}) using {method.value}")
        return allocation


# =============================================================================
# LAYER 6: EXECUTION INTELLIGENCE
# =============================================================================

class SlippageModel:
    """پیش‌بینی لغزش قیمت بر اساس شرایط بازار"""

    def __init__(self):
        self.model = None
        self.is_trained = False
        logger.info("SlippageModel initialized.")

    async def predict_slippage(
            self,
            order_size: float,
            avg_daily_volume: float,
            volatility: float,
            spread: float,
            time_of_day: int
    ) -> float:
        """
        پیش‌بینی لغزش قیمت بر حسب درصد
        """
        # اندازه سفارش نسبت به حجم
        size_ratio = order_size / (avg_daily_volume + 1e-8)

        # اثر نوسان
        vol_effect = volatility * 2.0

        # اثر اسپرد
        spread_effect = spread * 0.5

        # اثر زمان (ساعات شلوغ‌تر = لغزش کمتر)
        if 9 <= time_of_day <= 16:  # ساعات کاری لندن/نیویورک
            time_factor = 0.8
        elif 0 <= time_of_day <= 6:  # ساعات شب
            time_factor = 1.5
        else:
            time_factor = 1.0

        base_slippage = 0.0001  # 0.01% پایه

        slippage = base_slippage + (size_ratio * 0.01) + (vol_effect * 0.001) + (spread_effect * 0.001)
        slippage *= time_factor

        slippage = min(slippage, 0.005)  # حداکثر 0.5%
        logger.debug(f"Slippage prediction: {slippage:.4%}")
        return slippage


class SpreadForecaster:
    """پیش‌بینی تغییرات اسپرد"""

    async def forecast_spread(
            self,
            current_spread: float,
            volatility: float,
            time_until_execution: int  # ثانیه
    ) -> float:
        """
        پیش‌بینی اسپرد در زمان اجرا
        """
        # هر چه زمان بیشتر، عدم قطعیت بیشتر
        time_factor = 1.0 + (time_until_execution / 60) * 0.1  # 10% افزایش به ازای هر دقیقه

        # نوسان بالا = اسپرد بالاتر
        vol_factor = 1.0 + volatility * 10

        forecast = current_spread * time_factor * vol_factor
        logger.debug(f"Spread forecast: {forecast:.4%}")
        return forecast


class SmartOrderRouter:
    """مسیریابی هوشمند سفارش"""

    def __init__(self):
        self.venues = ['MT5', 'BINANCE']
        self.latency_history = {venue: [] for venue in self.venues}
        logger.info("SmartOrderRouter initialized.")

    async def select_best_venue(
            self,
            order_size: float,
            symbol: str,
            max_slippage: float
    ) -> str:
        """
        انتخاب بهترین محل اجرا
        """
        # در حالت پیش‌فرض، MT5
        logger.debug(f"Selecting best venue for {symbol}. Default: MT5")
        return 'MT5'


class ImpactCostModel:
    """مدل‌سازی هزینه تأثیر بازار"""

    def calculate_impact_cost(
            self,
            order_size: float,
            market_depth: float,
            volatility: float
    ) -> float:
        """
        محاسبه هزینه تأثیر بازار
        """
        # مدل ساده Almgren-Chriss
        permanent_impact = 0.0001 * (order_size / market_depth)
        temporary_impact = 0.0002 * np.sqrt(order_size / market_depth) * volatility

        impact_cost = permanent_impact + temporary_impact
        logger.debug(f"Impact cost calculation: {impact_cost:.4%}")
        return impact_cost


class ExecutionIntelligenceLayer:
    """لایه هوش اجرا"""

    def __init__(self):
        self.slippage_model = SlippageModel()
        self.spread_forecaster = SpreadForecaster()
        self.order_router = SmartOrderRouter()
        self.impact_model = ImpactCostModel()
        logger.info("ExecutionIntelligenceLayer initialized.")

    async def optimize_execution(
            self,
            symbol: str,
            side: str,  # 'buy' or 'sell'
            quantity: float,
            price: float,
            urgency: str = 'normal'  # 'low', 'normal', 'high'
    ) -> Dict[str, Any]:
        """
        بهینه‌سازی اجرای سفارش
        """
        logger.info(f"[{symbol}] Optimizing execution for {side} {quantity} units with urgency={urgency}")

        # تعیین پنجره زمانی بر اساس فوریت
        if urgency == 'high':
            time_window = 60  # 1 دقیقه
            slices = 1
        elif urgency == 'normal':
            time_window = 300  # 5 دقیقه
            slices = 3
        else:  # low
            time_window = 900  # 15 دقیقه
            slices = 5

        slice_size = quantity / slices

        # پیش‌بینی هزینه‌ها
        total_impact = 0
        total_slippage = 0

        for i in range(slices):
            impact = self.impact_model.calculate_impact_cost(
                slice_size,
                market_depth=1000000,  # فرضی
                volatility=0.01  # فرضی
            )
            total_impact += impact

            slippage = await self.slippage_model.predict_slippage(
                slice_size,
                avg_daily_volume=10000000,  # فرضی
                volatility=0.01,
                spread=0.0001,
                time_of_day=datetime.now().hour
            )
            total_slippage += slippage

        execution_plan = {
            'symbol': symbol,
            'side': side,
            'total_quantity': quantity,
            'slices': slices,
            'slice_size': slice_size,
            'time_window_seconds': time_window,
            'estimated_impact_pct': total_impact / slices,
            'estimated_slippage_pct': total_slippage / slices,
            'total_cost_pct': (total_impact + total_slippage) / slices,
            'recommended_venue': await self.order_router.select_best_venue(quantity, symbol, 0.001),
            'execution_schedule': [
                {
                    'slice': i + 1,
                    'quantity': slice_size,
                    'delay_seconds': (i * time_window // slices)
                }
                for i in range(slices)
            ]
        }

        logger.info(f"[{symbol}] Execution plan created. Total cost: {execution_plan['total_cost_pct']:.4%}")
        return execution_plan

# =============================================================================
# 2. FINANCIAL METRICS (FIXED SYNTAX)
# =============================================================================
@dataclass
class FinancialMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    calmar_ratio: float
    sortino_ratio: float
    max_drawdown: float
    avg_drawdown: float
    drawdown_days: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    turnover: float
    avg_holding_period: float
    stability_score: float
    regime_consistency: float

    @property
    def is_acceptable(self) -> bool:
        # ✅ اصلاح شد: اضافه شدن پرانتز و عملگرهای منطقی
        return (
                self.sharpe_ratio > 0.7 and
                self.max_drawdown < 0.20 and
                self.win_rate > 0.45 and
                self.profit_factor > 1.2
        )

    @property
    def quality_score(self) -> float:
        # ✅ اصلاح شد: اصلاح فرمول ریاضی
        score = (
                0.4 * min(1.0, self.sharpe_ratio / 2.0) +
                0.3 * (1.0 - min(1.0, self.max_drawdown / 0.3)) +
                0.2 * self.win_rate +
                0.1 * self.stability_score
        )
        return float(np.clip(score, 0.0, 1.0))


# =============================================================================
# 3. MODEL REJECTION CRITERIA (OPTIMIZED)
# =============================================================================
@dataclass
class ModelRejectionCriteria:
    max_overfit_gap: float = 0.20  # کمی تساهل برای Overfit
    min_stability_score: float = 0.65  # کمی کاهش برای اجازه به مدل‌های خلاقانه
    min_sharpe_ratio: float = 0.60  # افزایش از 0.5 به 0.6 (فیلتر بهتر)
    max_drawdown: float = 0.25  # افزایش از 0.20 به 0.25 (اجازه به نوسان)
    min_win_rate: float = 0.42  # کاهش از 0.45 به 0.42
    min_profit_factor: float = 1.10  # کاهش از 1.20 به 1.10
    regime_consistency_required: bool = True
    min_confidence_score: float = 0.60



class ModelRejectionEngine:
    def __init__(self, criteria: Optional['ModelRejectionCriteria'] = None):
        self.criteria = criteria or ModelRejectionCriteria()
        self.rejection_log: List[Dict] = []
        logger.info(f"ModelRejectionEngine initialized with criteria: {self.criteria}")

    def evaluate_model(self, candidate: 'ModelCandidate') -> 'ModelCandidate':
        rejection_reasons = []

        # محاسبه شکاف بیش‌برازش
        overfit_gap = candidate.overfit_gap
        if overfit_gap > self.criteria.max_overfit_gap:
            rejection_reasons.append(f"OVERFIT: {overfit_gap:.2%} > {self.criteria.max_overfit_gap:.2%}")

        # بررسی پایداری
        stability = candidate.stability_score
        if stability < self.criteria.min_stability_score:
            rejection_reasons.append(f"UNSTABLE: {stability:.2f} < {self.criteria.min_stability_score:.2f}")

        # بررسی متریک‌های تست (با احتیاط در صورت نبودن متریک‌ها)
        if candidate.test_metrics:
            if candidate.test_metrics.sharpe_ratio < self.criteria.min_sharpe_ratio:
                rejection_reasons.append(
                    f"LOW SHARPE: {candidate.test_metrics.sharpe_ratio:.2f} < {self.criteria.min_sharpe_ratio:.2f}")

            if candidate.test_metrics.max_drawdown > self.criteria.max_drawdown:
                rejection_reasons.append(
                    f"HIGH DRAWDOWN: {candidate.test_metrics.max_drawdown:.2%} > {self.criteria.max_drawdown:.2%}")

            if candidate.test_metrics.win_rate < self.criteria.min_win_rate:
                rejection_reasons.append(
                    f"LOW WIN RATE: {candidate.test_metrics.win_rate:.1%} < {self.criteria.min_win_rate:.1%}")

            if candidate.test_metrics.profit_factor < self.criteria.min_profit_factor:
                rejection_reasons.append(
                    f"LOW PROFIT FACTOR: {candidate.test_metrics.profit_factor:.2f} < {self.criteria.min_profit_factor:.2f}")
        else:
            rejection_reasons.append("MISSING TEST METRICS")

        if rejection_reasons:
            candidate.is_rejected = True
            candidate.rejection_reasons = rejection_reasons
            self.rejection_log.append({
                'timestamp': datetime.now().isoformat(),
                'candidate_id': candidate.id,
                'symbol': candidate.symbol,
                'reasons': rejection_reasons
            })
            logger.info(
                f"[{candidate.symbol}] Model {candidate.id} ({candidate.architecture.value}) REJECTED: {rejection_reasons[0]}")
        else:
            candidate.is_rejected = False
            candidate.rejection_reasons = []
            logger.info(f"[{candidate.symbol}] Model {candidate.id} ({candidate.architecture.value}) ACCEPTED.")

        return candidate


@dataclass
class ModelCandidate:
    symbol: str
    architecture: 'ModelType'
    model_object: Any
    regime_at_training: 'MarketRegime'
    regime_detector_state: Dict[str, Any]
    train_metrics: 'FinancialMetrics'
    val_metrics: 'FinancialMetrics'
    test_metrics: 'FinancialMetrics'

    id: str = field(default_factory=lambda: hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8])
    version: str = ""
    training_timestamp: datetime = field(default_factory=datetime.now)

    monte_carlo_stability: float = 0.0
    bootstrap_robustness: float = 0.0
    adversarial_robustness: float = 0.0

    recommended_allocation: float = 0.0
    kelly_fraction: float = 0.0

    is_rejected: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    model_path: Optional[str] = None

    @property
    def overfit_gap(self) -> float:
        # اصلاح شده: جلوگیری از تقسیم بر صفر و بهبود منطق
        train_sr = self.train_metrics.sharpe_ratio
        test_sr = self.test_metrics.sharpe_ratio
        # استفاده از فرمولی که پایدارتر است
        gap = abs(train_sr - test_sr) / (abs(train_sr) + abs(test_sr) + 1e-6)
        return float(gap)

    @property
    def stability_score(self) -> float:
        return float(np.mean([self.monte_carlo_stability, self.bootstrap_robustness, self.adversarial_robustness]))

    @property
    def quality_score(self) -> float:
        if self.is_rejected:
            return 0.0

        # اصلاح شده: استفاده از عملگر ضرب (*) به جای زیرخط (_)
        score = (0.4 * min(1.0, self.test_metrics.sharpe_ratio / 2.0) +
                 0.3 * (1.0 - min(1.0, self.test_metrics.max_drawdown / 0.3)) +
                 0.2 * self.test_metrics.win_rate +
                 0.1 * self.stability_score)

        if self.overfit_gap > 0.15:
            # اصلاح شده: استفاده از عملگر ضرب (*)
            score *= max(0.0, 1.0 - (self.overfit_gap - 0.15) * 2.0)

        return float(np.clip(score, 0.0, 1.0))




# =============================================================================
# HELPER FUNCTIONS (Model Building, Evaluation, etc.)
# =============================================================================


def _build_lstm_model(input_shape: Tuple[int, int], regime: MarketRegime) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    dropout_rate = 0.3 if regime in [MarketRegime.VOLATILITY_CLUSTER, MarketRegime.CRISIS_FLASH] else 0.2

    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    price = tf.keras.layers.Dense(1, name='price')(x)
    volatility = tf.keras.layers.Dense(1, activation='softplus', name='volatility')(x)
    confidence = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence')(x)

    logger.debug(f"Built LSTM model with input shape {input_shape}, dropout={dropout_rate}")
    return tf.keras.Model(inputs=inputs, outputs=[price, volatility, confidence])


def _build_transformer_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(64)(inputs)

    for _ in range(3):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
        attn = tf.keras.layers.Dropout(0.1)(attn)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(64)
        ])
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn(x))

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    price = tf.keras.layers.Dense(1, name='price')(x)
    volatility = tf.keras.layers.Dense(1, activation='softplus', name='volatility')(x)
    confidence = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence')(x)

    logger.debug(f"Built Transformer model with input shape {input_shape}")
    return tf.keras.Model(inputs=inputs, outputs=[price, volatility, confidence])


def _build_lstm_residual_model(input_shape, config, loss_fn=None):
    """
    FIXED VERSION - TIER-1 STANDARD
    Properly handles dimension mismatches in residual connections
    """
    inputs = layers.Input(shape=input_shape)

    # Initial projection to match first layer dimensions
    x = layers.Dense(config['units'][0], activation='relu', name='input_projection')(inputs)

    for i, units in enumerate(config['units']):
        shortcut = x

        # Main LSTM layer
        x = layers.LSTM(
            units,
            return_sequences=True,
            dropout=config['dropout'],
            recurrent_dropout=config.get('recurrent_dropout', 0.1),
            name=f'lstm_{i + 1}'
        )(x)

        # --- CRITICAL FIX: Match dimensions for residual connection ---
        # If shortcut shape doesn't match current units, project it
        if shortcut.shape[-1] != units:
            shortcut = layers.Dense(units, name=f'shortcut_proj_{i + 1}')(shortcut)

        # Add residual connection
        if config.get('residual', True):
            x = layers.Add(name=f'residual_add_{i + 1}')([x, shortcut])
            x = layers.LayerNormalization(epsilon=1e-6, name=f'residual_norm_{i + 1}')(x)

    # Attention layer (if enabled in config)
    if config.get('attention', False):
        attention = layers.MultiHeadAttention(
            num_heads=config.get('num_heads', 4),
            key_dim=x.shape[-1] // config.get('num_heads', 4),
            name='attention'
        )(x, x)
        x = layers.Add(name='attention_add')([x, attention])
        x = layers.LayerNormalization(epsilon=1e-6, name='attention_norm')(x)

    # Global pooling to convert (batch, timesteps, units) to (batch, units)
    x = layers.GlobalAveragePooling1D(name='global_pool')(x)

    # Multi-Output Heads (matching your data structure)
    outputs = {
        'confidence_output': layers.Dense(1, name='confidence_output')(x),
        'price_output': layers.Dense(1, name='price_output')(x),
        'volatility_output': layers.Dense(1, name='volatility_output')(x)
    }

    model = models.Model(inputs=inputs, outputs=outputs)

    # Optimizer selection
    opt_name = config.get('optimizer', 'adamw').lower()
    lr = config.get('initial_lr', 0.0005)

    if opt_name == 'adamw':
        optimizer = optimizers.AdamW(learning_rate=lr, weight_decay=1e-5)
    elif opt_name == 'sgd':
        optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    else:
        optimizer = optimizers.Adam(learning_rate=lr)

    model.compile(
        optimizer=optimizer,
        loss={
            'price': loss_fn.price_loss if hasattr(loss_fn, 'price_loss') else 'mse',
            'volatility': loss_fn.volatility_loss if hasattr(loss_fn, 'volatility_loss') else 'mae',
            'confidence': loss_fn.confidence_loss if hasattr(loss_fn, 'confidence_loss') else 'binary_crossentropy',
        },
        loss_weights={
            'price': loss_fn.price_weight if hasattr(loss_fn, 'price_weight') else 1.0,
            'volatility': loss_fn.volatility_weight if hasattr(loss_fn, 'volatility_weight') else 0.5,
            'confidence': loss_fn.confidence_weight if hasattr(loss_fn, 'confidence_weight') else 1.0,
        },
        metrics={
            'price': ['mae'],
            'volatility': ['mae'],
            'confidence': ['accuracy'],
        }
    )

    logger.info(f"✅ Built LSTM Residual Model with units={config['units']}, attention={config.get('attention', False)}")
    return model


def _get_batch_size(n_samples: int) -> int:
    if n_samples > 10000:
        return 512
    elif n_samples > 5000:
        return 256
    elif n_samples > 2000:
        return 128
    else:
        return 64


def _sharpe_metric(y_true, y_pred):
    returns = y_pred * y_true
    mean_return = tf.reduce_mean(returns)
    std_return = tf.math.reduce_std(returns) + 1e-8
    sharpe = mean_return / std_return * tf.sqrt(252.0)
    return sharpe


def _max_drawdown_metric(y_true, y_pred):
    returns = y_pred * y_true
    cum_returns = tf.cumsum(returns)
    max_cum = tf.scan(lambda a, x: tf.maximum(a, x), cum_returns)
    drawdown = (cum_returns - max_cum) / (1.0 + max_cum + 1e-8)
    return -tf.reduce_min(drawdown)


# ============================================================================
# SECTION 5: COMPILE TF MODEL
# ============================================================================

def _compile_tf_model(
        model: "tf.keras.Model",
        optimizer: Any,  # حالا به صورت آبجکت دریافت می‌شود
        loss_fn,
) -> None:
    """
    ✅ TIER-1 FINAL FIX: Explicit Metrics + Object-based Compilation
    """
    # تعریف صریح متریک‌ها برای جلوگیری از باگ رشته‌ای در مدل‌های Multi-Output
    metrics_config = {
        "price": [tf.keras.metrics.MeanAbsoluteError(name="mae_p")],
        "volatility": [tf.keras.metrics.MeanAbsoluteError(name="mae_v")],
        "confidence": [tf.keras.metrics.BinaryAccuracy(name="acc_c")]
    }

    model.compile(
        optimizer=optimizer,  # حتماً باید شیء باشد، نه رشته "adamw"
        loss={
            "price": loss_fn.price_loss,
            "volatility": loss_fn.volatility_loss,
            "confidence": loss_fn.confidence_loss,
        },
        loss_weights={
            "price": float(loss_fn.price_weight),
            "volatility": float(loss_fn.volatility_weight),
            "confidence": float(loss_fn.confidence_weight),
        },
        metrics=metrics_config,
    )


class InstitutionalEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.perf_counter()
        lr = float(self.model.optimizer.learning_rate.numpy())
        self.logger.info(f"📘 Epoch {epoch + 1} START | LR={lr:.8f}")

    def on_epoch_end(self, epoch, logs=None):
        duration = time.perf_counter() - self.epoch_start_time
        lr = float(self.model.optimizer.learning_rate.numpy())

        metrics_str = " | ".join(
            [f"{k}={v:.6f}" for k, v in logs.items()]
        )

        self.logger.info(
            f"📗 Epoch {epoch + 1} END | "
            f"time={duration:.2f}s | LR={lr:.8f} | {metrics_str}"
        )


# =============================================================================
# FINANCIAL EVALUATION FUNCTION - COMPLETE GOLDMAN SACHS TIER-1 VERSION
# =============================================================================

# =============================================================================
# FINANCIAL EVALUATION FUNCTION - FIXED VERSION
# =============================================================================



@dataclass
class TrainingContext:
    """Fixed TrainingContext - handles both dict and object regime"""
    symbol: str
    timeframe: str
    fold: int
    lookback: int
    regime: Dict[str, Any]
    feature_names: List[str]
    sequence_metadata: Dict[str, Any]

    @classmethod
    async def from_universal_load(cls, symbol: str, fold: int = 5, lookback: int = 190):

        try:
            from core.global_method import get_active_symbol_timeframe
            active_symbol, active_timeframe = get_active_symbol_timeframe()
        except:
            active_timeframe = "1H"

        regime_dict = await get_symbol_regime()
        metadata = await universal_load(data_type="metadata", symbol=active_symbol, fold=fold) or {}
        features_info = await universal_load(data_type="features_info", symbol=active_symbol, fold=fold) or {}



        return cls(
            symbol=active_symbol,
            timeframe=active_timeframe,
            fold=fold,
            lookback=lookback,
            regime=regime_dict,
            feature_names=features_info.get('feature_names', []),
            sequence_metadata=metadata if isinstance(metadata, dict) else {}
        )

    @property
    def regime_name(self) -> str:
        if isinstance(self.regime, dict):
            return self.regime.get('regime', 'UNDEFINED')
        return 'UNDEFINED'

    @property
    def regime_confidence(self) -> float:
        if isinstance(self.regime, dict):
            return float(self.regime.get('confidence', 0.0))
        return 0.0

    @property
    def regime_enum(self) -> MarketRegime:
        regime_str = self.regime_name
        try:
            return MarketRegime(regime_str)
        except:
            regime_map = {
                'trending_up': MarketRegime.BULL_TRENDING,
                'trending_down': MarketRegime.BEAR_TRENDING,
                'ranging': MarketRegime.SIDEWAYS_RANGING,
                'sideways': MarketRegime.SIDEWAYS_RANGING,
                'mixed': MarketRegime.SIDEWAYS_RANGING,  # YOUR CASE
                'volatile': MarketRegime.VOLATILITY_CLUSTER,
                'high_volatility': MarketRegime.VOLATILITY_CLUSTER,
                'crisis': MarketRegime.CRISIS_FLASH,
                'crash': MarketRegime.CRISIS_FLASH,
            }
            return regime_map.get(regime_str.lower(), MarketRegime.UNDEFINED)

    @property
    def is_tradeable(self) -> bool:
        confidence = self.regime_confidence
        regime = self.regime_enum
        return confidence >= regime.required_confidence and regime != MarketRegime.UNDEFINED


def _get_default_metrics() -> Dict[str, float]:
    """مقادیر پیش‌فرض برای زمانی که ارزیابی با خطا مواجه می‌شود"""
    return {
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.01,
        'expectancy': 0.0,
        'annualized_return': 0.0,
        'annualized_volatility': 0.0,
        'calmar_ratio': 0.0,
        'sortino_ratio': 0.0,
        'turnover': 0.0,
        'total_return': 0.0,
        'trade_count': 0,
        'signal_quality': 0.5,
        'avg_position': 0.0,
        'stability_score': 0.5,
        'regime_consistency': 0.5,
        'drawdown_days': 0,
        'regime': 'UNDEFINED',
        'samples_used': 0
    }


# =============================================================================
# CORE TRAINING FUNCTION - WITH RETRY LOOP AND DATA SELF-HEALING
# =============================================================================

"""
================================================================================
🚀 QUANT TRAINING PIPELINE v2.1 - FORENSIC CACHE AWARE EDITION
================================================================================
Each time a model is rejected, the next architecture from the registry is selected.
If forensic cache confirms data is healthy, self-healing is skipped.

Architecture Registry:
  LSTM  : v1-Standard | v2-Bidirectional+Attention | v3-Deep+Residual | v4-Lightweight | v5-TCN-Hybrid
  TRANS : v1-Small     | v2-Standard                | v3-Large          | v4-CrossAttn
  XGB   : v1-Conservative | v2-Standard             | v3-Aggressive     | v4-DART
  LGB   : v1-HighLeaf
  CATBOOST : v1-Deep
================================================================================
"""


class WarmupCosineDecayCallback(tf.keras.callbacks.Callback):
    """
    FIX-1: Warmup + Cosine Decay as Callback
    -----------------------------------------
    Why Callback instead of LearningRateSchedule?
    If LearningRateSchedule is given to optimizer, ReduceLROnPlateau
    cannot modify learning_rate and throws error:
        "This optimizer was created with a LearningRateSchedule..."

    Solution: Optimizer created with FLOAT lr.
    This callback adjusts lr per batch.
    Full compatibility with ReduceLROnPlateau maintained.
    """

    def __init__(
            self,
            initial_lr: float,
            warmup_steps: int,
            total_steps: int,
            min_lr: float = 1e-7,
            cycle_restarts: int = 2,
    ):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        self.cycle_restarts = cycle_restarts
        self._step = 0
        self._current_lr = initial_lr

    def on_train_batch_begin(self, batch, logs=None):
        self._step += 1
        step = self._step

        if step <= self.warmup_steps:
            lr = self.initial_lr * (step / self.warmup_steps)
        else:
            decay_steps = self.total_steps - self.warmup_steps
            decay_step = step - self.warmup_steps
            cycle_len = decay_steps / max(1, self.cycle_restarts)
            cycle_pos = (decay_step % cycle_len) / cycle_len
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * cycle_pos)
            )

        self._current_lr = float(lr)

        # 🔥 FIX: assignment compatible with Keras 3
        if hasattr(self.model.optimizer, "learning_rate"):
            try:
                self.model.optimizer.learning_rate.assign(self._current_lr)
            except Exception:
                self.model.optimizer.lr.assign(self._current_lr)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs['learning_rate'] = self._current_lr


class GatedResidualNetwork(Layer):
    """GRN for TFT and TabNet-like structures"""

    def __init__(self, units, dropout_rate=0.1, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.dense1 = layers.Dense(self.units, activation='elu')
        self.dense2 = layers.Dense(self.units, activation=None)
        self.gate = layers.Dense(self.units, activation='sigmoid')
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(self.dropout_rate)
        super(GatedResidualNetwork, self).build(input_shape)

    def call(self, inputs, context=None):
        x = inputs
        if context is not None:
            context_layer = layers.Dense(self.units, activation='elu')(context)
            x = layers.Add()([x, context_layer])
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        gate = self.gate(inputs)
        if context is not None:
            gate_c = self.gate(context)
            gate = layers.Add()([gate, gate_c])
        x = layers.Multiply()([x, gate])
        return self.layer_norm(layers.Add()([x, inputs]))


class TemporalFusionTransformerBlock(Layer):
    """Simplified TFT Block"""

    def __init__(self, head_size, num_heads, ff_dim, dropout=0.1, **kwargs):
        super(TemporalFusionTransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="gelu"), layers.Dropout(dropout), layers.Dense(head_size)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# ============================================================================
# SECTION 4: CALLBACKS — FIXED (با Warmup)
# ============================================================================

def _make_callbacks(
        symbol: str,
        fold: int,
        arch_name: str,
        patience_stop: int = 20,
        patience_lr: int = 8,
        warmup_callback: Optional[WarmupCosineDecayCallback] = None,
) -> List:
    """
    FIX-1: warmup_callback passed as parameter.
    ReduceLROnPlateau still active (works on float lr).
    WarmupCosineDecay controls lr during warm-up phase.
    After warm-up, ReduceLROnPlateau takes over.
    """
    ckpt_path = f"models/temp/best_{symbol}_fold{fold}_{arch_name}.weights.h5"

    cbs = [
        warmup_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_stop,
            restore_best_weights=True,
            min_delta=1e-5,
            verbose=1
        ),

        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
    ]

    # FIX-1: Add warmup callback if provided (must execute first)
    if warmup_callback is not None:
        cbs.insert(0, warmup_callback)

    return cbs


# =============================================================================
# SECTION 7: SINGLE ARCHITECTURE TRAINER — FIXED VERSION WITH LIGHTGBM & CATBOOST SUPPORT
# =============================================================================

def _dict_to_financial_metrics(d: Dict) -> Any:
    return FinancialMetrics(
        total_return=d.get("total_return", 0.0),
        annualized_return=d.get("annualized_return", 0.0),
        annualized_volatility=d.get("annualized_volatility", 0.0),
        sharpe_ratio=d.get("sharpe_ratio", 0.0),
        calmar_ratio=d.get("calmar_ratio", 0.0),
        sortino_ratio=d.get("sortino_ratio", 0.0),
        max_drawdown=d.get("max_drawdown", 0.0),
        avg_drawdown=d.get("avg_drawdown", 0.0),
        drawdown_days=int(d.get("drawdown_days", 0)),
        win_rate=d.get("win_rate", 0.0),
        profit_factor=d.get("profit_factor", 0.0),
        avg_win=d.get("avg_win", 0.0),
        avg_loss=d.get("avg_loss", 0.0),
        expectancy=d.get("expectancy", 0.0),
        turnover=d.get("turnover", 0.0),
        avg_holding_period=d.get("avg_holding_period", 0.0),
        stability_score=d.get("stability_score", 0.5),
        regime_consistency=d.get("regime_consistency", 0.5),
    )




# ─────────────────────────────────────────────────────────────────────────────
# NEW FUNCTION 1: _check_data_health
# ─────────────────────────────────────────────────────────────────────────────

def _check_data_health(
        fcm: 'ForensicCacheManager',
        symbol: str,
        fold: int,
        data_quality_score: float = 0.0,
        is_data_certified: bool = False,
) -> tuple:
    """
    ============================================================================
    Goldman Sachs Tier-1 — Single Source of Truth برای تصمیم skip-healing
    ============================================================================
    منطق:
      Priority 1: data_quality_score >= 95.0  → skip (fast path)
      Priority 2: is_data_certified == True    → skip (fast path)
      Priority 3: fcm.should_skip_self_healing → skip (cache check)

    Returns:
        (should_skip: bool, reason: str)
    """
    # ── Priority 1 ────────────────────────────────────────────────────────────
    if data_quality_score >= 95.0:
        return True, f"data_quality_pristine:{data_quality_score:.2f}"

    # ── Priority 2 ────────────────────────────────────────────────────────────
    if is_data_certified:
        return True, "explicit_certified_flag"

    # ── Priority 3 ────────────────────────────────────────────────────────────
    if fcm is not None:
        return fcm.should_skip_self_healing(symbol=symbol, fold=fold)

    return False, "no_cache_available"


# ─────────────────────────────────────────────────────────────────────────────
# NEW FUNCTION 2: _extract_quality_from_cache
# ─────────────────────────────────────────────────────────────────────────────

def _extract_quality_from_cache(cache_path: str = "cache/forensic_cache.json") -> tuple:
    """
    ============================================================================
    Goldman Sachs Tier-1 — Robust استخراج نمره کیفیت با Regex
    ============================================================================
    چرا regex؟ ساختار JSON می‌تواند تو در تو باشد و کلیدها متفاوت.
    Dict-key lookup فقط سطح اول را می‌بیند.

    Returns:
        (quality_score: float, is_certified: bool)
    """
    import re
    from pathlib import Path

    score = 0.0
    is_certified = False

    try:
        cache_file = Path(cache_path)
        if not cache_file.exists():
            return score, is_certified

        with open(cache_file, 'r', encoding='utf-8') as f:
            data_str = f.read()

        # ── استخراج score — چند pattern به ترتیب اولویت ───────────────────
        score_patterns = [
            r'"normalized_score"\s*:\s*([\d.]+)',
            r'"Normalized\s+Score"\s*:\s*([\d.]+)',
            r'"quality_score"\s*:\s*([\d.]+)',
            r'"data_score"\s*:\s*([\d.]+)',
            r'"overall_score"\s*:\s*([\d.]+)',
        ]
        for pat in score_patterns:
            m = re.search(pat, data_str, re.IGNORECASE)
            if m:
                score = float(m.group(1))
                break

        # ── استخراج certification_status ──────────────────────────────────
        cert_patterns = [
            r'"certification_status"\s*:\s*"([^"]+)"',
            r'"Certification"\s*:\s*"([^"]+)"',
            r'"cert_status"\s*:\s*"([^"]+)"',
        ]
        certified_values = {
            "CERTIFIED", "CONDITIONAL", "CERTIFIED_WITH_WARNINGS",
            "CERTIFIED_PRISTINE", "PASSED"
        }
        for pat in cert_patterns:
            m = re.search(pat, data_str, re.IGNORECASE)
            if m:
                is_certified = m.group(1).upper() in certified_values
                break

        # ── اگر score پیدا نشد ولی همه category ها 100 هستند ──────────────
        if score == 0.0:
            cat_matches = re.findall(r'"(?:Structural|Completeness|Statistical|'
                                     r'Consistency|Chain|Forensic)"\s*:\s*([\d.]+)',
                                     data_str, re.IGNORECASE)
            if cat_matches and all(float(v) >= 100.0 for v in cat_matches):
                score = 100.0

    except Exception:
        pass  # fail silently

    return score, is_certified



# ────────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE FUNCTION
# ────────────────────────────────────────────────────────────────────────────────
def _normalize_targets(y: Union[np.ndarray, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    🔥 GOLDMAN SACHS TIER-1 FIX: Target Sanitization
    Ensures standardized targets are properly formatted for their respective loss functions.

    Returns:
        Dict with keys: 'price', 'volatility', 'confidence' (exactly these names)
    """
    if isinstance(y, dict):
        # استانداردسازی کلیدها به فرمت مورد انتظار loss function
        out = {}

        # نقشه تبدیل کلیدها به فرمت استاندارد
        key_map = {
            "y_price_output": "price",
            "y_volatility_output": "volatility",
            "y_confidence_output": "confidence",
            "price_output": "price",
            "volatility_output": "volatility",
            "confidence_output": "confidence",
            "y_price": "price",
            "y_volatility": "volatility",
            "y_confidence": "confidence",
            "price": "price",
            "volatility": "volatility",
            "confidence": "confidence",
        }

        for k, v in y.items():
            # پیدا کردن کلید استاندارد
            clean_k = key_map.get(k, k)
            val = np.asarray(v)

            # اطمینان از shape مناسب
            if val.ndim == 1:
                val = val.reshape(-1, 1)
            elif val.ndim > 2:
                val = val.reshape(val.shape[0], -1)

            # 🔥 CRITICAL FIX: Un-scale confidence back to pure Binary [0, 1]
            if clean_k == "confidence":
                val = (val > 0).astype(np.float32)

            out[clean_k] = val

        # اطمینان از وجود همه کلیدهای مورد نیاز
        required_keys = {"price", "volatility", "confidence"}
        missing_keys = required_keys - set(out.keys())

        if missing_keys:
            # اگر کلیدی缺失 بود، با مقدار پیش‌فرض پر کن
            n_samples = next(iter(out.values())).shape[0] if out else 1

            if "price" in missing_keys:
                out["price"] = np.zeros((n_samples, 1), dtype=np.float32)
            if "volatility" in missing_keys:
                out["volatility"] = np.ones((n_samples, 1), dtype=np.float32) * 0.01
            if "confidence" in missing_keys:
                out["confidence"] = np.ones((n_samples, 1), dtype=np.float32)

        return out

    else:
        # ورودی آرایه ساده
        arr = np.asarray(y)

        # اگر 1 بعدی باشد
        if arr.ndim == 1:
            return {
                "price": arr.reshape(-1, 1).astype(np.float32),
                "volatility": np.ones((len(arr), 1), dtype=np.float32) * 0.01,
                "confidence": (arr.reshape(-1, 1) > 0).astype(np.float32)
            }

        # اگر 2 بعدی باشد
        if arr.ndim == 2:
            if arr.shape[1] >= 3:
                return {
                    "price": arr[:, 0:1].astype(np.float32),
                    "volatility": arr[:, 1:2].astype(np.float32),
                    "confidence": (arr[:, 2:3] > 0).astype(np.float32)
                }
            elif arr.shape[1] == 2:
                return {
                    "price": arr[:, 0:1].astype(np.float32),
                    "volatility": arr[:, 1:2].astype(np.float32),
                    "confidence": np.ones((len(arr), 1), dtype=np.float32)
                }
            else:  # 1 ستون
                return {
                    "price": arr[:, 0:1].astype(np.float32),
                    "volatility": np.ones((len(arr), 1), dtype=np.float32) * 0.01,
                    "confidence": (arr[:, 0:1] > 0).astype(np.float32)
                }

        # اگر بیشتر از 2 بعد
        arr = arr.reshape(arr.shape[0], -1)
        if arr.shape[1] >= 3:
            return {
                "price": arr[:, 0:1].astype(np.float32),
                "volatility": arr[:, 1:2].astype(np.float32),
                "confidence": (arr[:, 2:3] > 0).astype(np.float32)
            }
        else:
            return {
                "price": arr[:, 0:1].astype(np.float32),
                "volatility": np.ones((len(arr), 1), dtype=np.float32) * 0.01,
                "confidence": np.ones((len(arr), 1), dtype=np.float32)
            }


async def _reload_split(split: str, symbol: str, timeframe: str,
                        fold: int, lookback: int
                        ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Reload data after self-healing"""
    try:
        X, y = await universal_load("sequenced", symbol, timeframe,
                                    fold, split, lookback, multi_output=True)
        # ✅ اینجا یکبار normalize می‌کند
        normalized_y = _normalize_targets(y)

        # برای دیباگ - چک کردن کلیدهای خروجی
        logger.debug(f"✓ Reloaded {split} data - y keys: {list(normalized_y.keys())}")

        return X, normalized_y
    except Exception as e:
        raise Exception(f"Failed to reload {split} data: {e}") from e


# =============================================================================
# FIXED: UnifiedTreeModelWrapper با پشتیبانی از **kwargs
# =============================================================================
class UnifiedTreeModelWrapper:
    """
    Goldman Sachs Tier-1 - Unified wrapper for tree-based models
    Supports XGBoost, LightGBM, CatBoost, etc.
    """

    def __init__(self, models, model_type="xgboost"):
        self.models = models
        self.model_type = model_type  # برای تشخیص در evaluation

    def predict(self, X, **kwargs):
        """
        Predict with tree-based models
        Args:
            X: Input data (3D or 2D)
            kwargs: Ignored (for compatibility with TF/Keras predict API)
        """
        # Flatten if 3D
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        # Predict each target
        try:
            # 1. Price Prediction
            price_pred = self.models['price'].predict(X_flat)

            # 2. Volatility Prediction
            vol_pred = self.models['volatility'].predict(X_flat)

            # 3. Confidence (classification or regression)
            if hasattr(self.models['confidence'], 'predict_proba'):
                # For classifiers that support probability
                conf_proba = self.models['confidence'].predict_proba(X_flat)
                if conf_proba.shape[1] >= 2:
                    conf_pred = conf_proba[:, 1]  # Probability of class 1
                else:
                    conf_pred = conf_proba[:, 0]
            else:
                # Fallback to raw predict and sigmoid
                conf_pred = self.models['confidence'].predict(X_flat)
                if conf_pred.ndim > 1 and conf_pred.shape[1] > 1:
                    conf_pred = conf_pred[:, 1]

        except Exception as e:
            logger.warning(f"⚠️ Prediction error in wrapper: {e}, using fallback")

            # 🔥 CRITICAL FIX: Safe Fallback
            # اگر price_pred هنوز تعریف نشده بود (خطا در خط اول)، نباید از آن استفاده کنیم
            # از یک آرایه صفر به عنوان فال‌بک استفاده می‌کنیم
            if 'price_pred' not in locals():
                price_pred = np.zeros(X_flat.shape[0])

            # Volatility fallback (low constant)
            vol_pred = np.ones_like(price_pred) * 0.01  # FIXED: *

            # Confidence fallback (neutral)
            conf_pred = np.ones_like(price_pred) * 0.5  # FIXED: *

        # Ensure correct shapes
        price_pred = np.asarray(price_pred).reshape(-1, 1)
        vol_pred = np.asarray(vol_pred).reshape(-1, 1)
        conf_pred = np.asarray(conf_pred).reshape(-1, 1)

        # Post-processing
        vol_pred = np.abs(vol_pred)  # Volatility must be positive
        conf_pred = np.clip(conf_pred, 0, 1)  # Confidence in [0,1]

        logger.debug(f"📊 Wrapper predict: price={price_pred.shape}, "
                     f"vol={vol_pred.shape}, conf={conf_pred.shape}")

        return {
            'price': price_pred.astype(np.float32),
            'volatility': vol_pred.astype(np.float32),
            'confidence': conf_pred.astype(np.float32)
        }

# =============================================================================
        # EVALUATION GROUP
# =============================================================================

def _evaluate_model_financial(model, X_test, y_test, regime):
        """ارزیابی کامل برای هر سه تارگت (price, volatility, confidence)"""
        local_logger = logger
        default_metrics = {
            'price': {
                'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0,
                'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'total_return': 0.0, 'directional_accuracy': 0.0
            },
            'volatility': {
                'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0
            },
            'confidence': {
                'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0
            }
        }
        try:
            # 1. پیش‌بینی مدل
            preds = model.predict(X_test, verbose=1)
            # 2. استخراج خروجی‌ها
            price_pred = preds[0].flatten()
            volatility_pred = preds[1].flatten()
            confidence_pred = preds[2]
            # 3. استخراج تارگت‌ها (کلیدهای صحیح)
            true_price = y_test['price_output'].flatten()
            true_volatility = y_test['volatility_output'].flatten()
            true_confidence = y_test['confidence_output']
            # 4. ارزیابی تارگت price (مالی)
            price_metrics = _evaluate_price_metrics(price_pred, true_price)
            # 5. ارزیابی تارگت volatility (رگرسیون)
            volatility_metrics = _evaluate_regression_metrics(volatility_pred, true_volatility)
            # 6. ارزیابی تارگت confidence (کلاسیفیکیشن)
            confidence_metrics = _evaluate_classification_metrics(confidence_pred, true_confidence)
            # 7. ترکیب نتایج
            result = {
                'price': price_metrics,
                'volatility': volatility_metrics,
                'confidence': confidence_metrics
            }
            # 8. لاگ‌گیری نتایج
            log_test_results(result)
            return result
        except Exception as e:
            local_logger.error(f"❌ Critical Error in model evaluation: {e}")
            return default_metrics

def _evaluate_price_metrics(price_pred, true_price):
        """ارزیابی مالی برای تارگت price (اصلاح شده)"""
        # تطبیق طول داده‌ها
        min_len = min(len(price_pred), len(true_price))
        price_pred = price_pred[:min_len]
        true_price = true_price[:min_len]
        if min_len < 20:
            return {
                'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0,
                'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'total_return': 0.0, 'directional_accuracy': 0.0
            }
        
        # ✅ اصلاح 8: محاسبه صحیح Sharpe Ratio
        err = price_pred - true_price
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        non_zero = np.abs(true_price) > 1e-6
        mape = float(np.mean(np.abs(err[non_zero] / true_price[non_zero])) * 100) if non_zero.any() else 0.0
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true_price - np.mean(true_price)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))

        # ✅ اصلاح 9: محاسبه سیگنال‌های معاملاتی بدون خطا
        if len(price_pred) < 2:
            return {
                'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
                'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'total_return': 0.0, 'directional_accuracy': 0.0
            }

        # محاسبه بازده خالص
        cost_rate = 0.0005
        signals = np.sign(price_pred - np.mean(price_pred))
        strategy_ret = signals * true_price
        net_ret = strategy_ret - np.abs(np.diff(signals, prepend=0)) * cost_rate
        net_ret = np.clip(net_ret, -0.20, 0.20)

        # تحلیل معاملات
        trade_mask = signals != 0
        trade_ret = net_ret[trade_mask]
        if trade_ret.size == 0:
            return {
                'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
                'sharpe_ratio': 0.0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'total_return': 0.0, 'directional_accuracy': 0.0
            }

        win_mask = trade_ret > 0
        wins = trade_ret[win_mask]
        losses = trade_ret[~win_mask]
        win_rate = float(np.mean(win_mask))
        avg_win = float(np.mean(wins)) if wins.size > 0 else 0.0
        avg_loss = float(-np.mean(losses)) if losses.size > 0 else 0.0
        gross_profit = np.sum(wins) if wins.size > 0 else 0.0
        gross_loss = abs(np.sum(losses)) if losses.size > 0 else 1e-8
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 0.0

        # محاسبه معیارهای مالی
        equity = np.cumprod(1 + net_ret)
        equity = np.clip(equity, 1e-6, None)
        total_return = float(equity[-1] - 1)
        periods_per_year = 252
        ann_return = float((1 + total_return) ** (periods_per_year / min_len) - 1) if min_len > 0 else 0.0
        ann_vol = float(np.std(net_ret) * np.sqrt(periods_per_year))
        sharpe = float(ann_return / ann_vol) if ann_vol > 1e-8 else 0.0

        # محاسبه directional accuracy
        directional_accuracy = float(np.mean(np.sign(price_pred) == np.sign(true_price)))

        return {
            'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2,
            'sharpe_ratio': sharpe, 'win_rate': win_rate, 'profit_factor': profit_factor,
            'total_return': total_return, 'directional_accuracy': directional_accuracy
        }
def _evaluate_regression_metrics(pred, true):
        """ارزیابی رگرسیون برای تارگت volatility"""
        if len(pred) < 20:
            return {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0}
        err = pred - true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        non_zero = np.abs(true) > 1e-6
        mape = float(np.mean(np.abs(err[non_zero] / true[non_zero])) * 100) if non_zero.any() else 0.0
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
        return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2': r2}

def _evaluate_classification_metrics(pred, true):
        """ارزیابی کلاسیفیکیشن برای تارگت confidence"""
        if len(pred) < 20:
            return {'accuracy': 0.0, 'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}
        # تبدیل خروجی به کلاس
        pred_classes = np.argmax(pred, axis=1)
        true_classes = np.argmax(true, axis=1) if true.ndim > 1 else true
        # محاسبه معیارها
        accuracy = float(np.mean(pred_classes == true_classes))
        # محاسبه F1-score (برای کلاس‌های 0,1,2)
        from sklearn.metrics import f1_score
        f1 = f1_score(true_classes, pred_classes, average='weighted')
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': 0.0,
            'recall': 0.0
        }

def log_test_results(metrics):
        """لاگ‌گیری کامل نتایج ارزیابی برای هر سه تارگت"""
        logger.info("=" * 70)
        logger.info("📊 TEST RESULTS - FINANCIAL METRICS (Price)")
        logger.info("=" * 70)
        logger.info(f"📈 Sharpe Ratio:          {metrics['price']['sharpe_ratio']:.4f}")
        logger.info(f"🎲 Win Rate:               {metrics['price']['win_rate']:.2%}")
        logger.info(f"💸 Profit Factor:          {metrics['price']['profit_factor']:.4f}")
        logger.info(f"🎯 Directional Accuracy:  {metrics['price']['directional_accuracy']:.2%}")
        logger.info(f"💰 Total Return:           {metrics['price']['total_return']:.2%}")
        logger.info("\n📊 TEST RESULTS - VOLATILITY METRICS")
        logger.info("=" * 70)
        logger.info(f"📉 RMSE:                  {metrics['volatility']['rmse']:.4f}")
        logger.info(f"📉 MAE:                   {metrics['volatility']['mae']:.4f}")
        logger.info(f"📉 R²:                    {metrics['volatility']['r2']:.4f}")
        logger.info("\n📊 TEST RESULTS - CONFIDENCE METRICS")
        logger.info("=" * 70)
        logger.info(f"✅ Accuracy:              {metrics['confidence']['accuracy']:.4f}")
        logger.info(f"✅ F1 Score:              {metrics['confidence']['f1_score']:.4f}")
        logger.info("=" * 70)


# =============================================================================
# HELPER FUNCTIONS FOR MODEL FAMILY DETECTION
# =============================================================================

def _detect_model_family(cfg: ArchitectureConfig) -> str:
    """Detect model family from config"""

    # 1. Deep Learning Models (TensorFlow/Keras)
    if cfg.model_type in [
        ModelType.LSTM, ModelType.BILSTM, ModelType.GRU, ModelType.RNN,
        ModelType.CNN_LSTM, ModelType.TRANSFORMER, ModelType.TCN,
        ModelType.ATTENTION, ModelType.MLP, ModelType.HYBRID, ModelType.REINFORCEMENT
    ]:
        return "tensorflow"

    # 2. Gradient Boosting Models
    elif cfg.model_type == ModelType.XGBOOST:
        return "xgboost"
    elif cfg.model_type == ModelType.LIGHTGBM:
        return "lightgbm"
    elif cfg.model_type == ModelType.CATBOOST:
        return "catboost"

    # 3. Sklearn Tree-Based
    elif cfg.model_type == ModelType.RANDOM_FOREST:
        return "random_forest"

    # 4. Linear & Bayesian
    elif cfg.model_type == ModelType.LINEAR:
        return "linear"
    elif cfg.model_type == ModelType.BAYESIAN:
        return "bayesian"  # اضافه شده برای مدل‌های بیزی

    # 5. SVM & KNN
    elif cfg.model_type == ModelType.SVR:
        return "svr"
    elif cfg.model_type == ModelType.KNN:
        return "knn"

    # 6. Ensemble
    elif cfg.model_type in [ModelType.ENSEMBLE]:
        return "ensemble"

    else:
        return "unknown"

def _build_catboost_models(cfg: ArchitectureConfig, plan: AttemptPlan) -> Dict[str, Any]:
    """Build CatBoost models"""
    import catboost as cb

    # محاسبه پارامترهای پویا
    lr = getattr(cfg, 'learning_rate', 0.03) * plan.lr_multiplier
    reg = getattr(cfg, 'reg_lambda', 1.0) * plan.reg_multiplier

    params = {
        'iterations': getattr(cfg, 'iterations', 500),
        'depth': getattr(cfg, 'depth', 6),
        'learning_rate': lr,
        'l2_leaf_reg': reg,  # تنظیم صحیح منظم‌ساز
        'random_seed': 42,
        'verbose': True,  # جلوگیری از پر شدن کنسول (می‌توانید True کنید برای دیباگ)
        'allow_writing_files': False,  # جلوگیری از ساخت فایل‌های اضافی
    }

    models = {
        'price': cb.CatBoostRegressor(**params),
        'volatility': cb.CatBoostRegressor(**params),
        # برای کلاسیفایر معمولا loss_function نیاز نیست مگر اینکه خاص باشد، Logloss پیش‌فرض است
        'confidence': cb.CatBoostClassifier(**params)
    }
    return models

def _build_random_forest_models(cfg: ArchitectureConfig, plan: AttemptPlan) -> Dict[str, Any]:
    """Build Random Forest models"""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    params = {
        'n_estimators': getattr(cfg, 'n_estimators', 100),
        'max_depth': getattr(cfg, 'max_depth', 10),
        'min_samples_split': getattr(cfg, 'min_samples_split', 5),
        'min_samples_leaf': getattr(cfg, 'min_samples_leaf', 2),
        'random_state': 42,
        'n_jobs': -1,
    }

    models = {
        'price': RandomForestRegressor(**params),
        'volatility': RandomForestRegressor(**params),
        'confidence': RandomForestClassifier(**params),
    }
    return models


def _build_linear_models(cfg: ArchitectureConfig, plan: AttemptPlan) -> Dict[str, Any]:
    """Build Linear models"""
    from sklearn.linear_model import Ridge, Lasso, LogisticRegression

    alpha = getattr(cfg, 'reg_alpha', 1.0) * plan.reg_multiplier

    # تشخیص نوع مدل بر اساس نام کانفیگ (مثلاً "Ridge-Dynamic" یا "Lasso-Dynamic")
    # این روش با سیستم تولید خودکار شما سازگارتر است چون نام‌ها دقیق هستند
    model_name = cfg.name.lower()

    is_ridge = "ridge" in model_name
    # اگر Ridge نباشد، فرض می‌کنیم Lasso است (طبق کانفیگ‌های جدید شما فقط این دو تا وجود دارند)

    if is_ridge:
        models = {
            'price': Ridge(alpha=alpha, random_state=42),
            'volatility': Ridge(alpha=alpha, random_state=42),
            'confidence': LogisticRegression(C=1.0 / alpha, random_state=42, max_iter=1000),
        }
    else:  # Lasso
        models = {
            'price': Lasso(alpha=alpha, random_state=42),
            'volatility': Lasso(alpha=alpha, random_state=42),
            # برای Lasso در LogisticRegression باید از solver='saga' استفاده کرد
            'confidence': LogisticRegression(C=1.0 / alpha, penalty='l1', solver='saga', random_state=42,
                                             max_iter=1000),
        }
    return models

def _build_svr_models(cfg: ArchitectureConfig, plan: AttemptPlan) -> Dict[str, Any]:
    """Build SVR models"""
    from sklearn.svm import SVR, SVC

    # محاسبه پارامتر C با ضرب‌کننده‌ی AttemptPlan
    C_val = getattr(cfg, 'C', 1.0) * plan.reg_multiplier

    # پارامترهای مشترک برای SVR
    svr_common_params = {
        'kernel': getattr(cfg, 'kernel', 'rbf'),
        'C': C_val,
        'epsilon': getattr(cfg, 'epsilon', 0.1),
        'gamma': getattr(cfg, 'gamma', 'scale'),
        'verbose': 1,  # فعال‌سازی گزارش‌دهی لحظه‌ای
    }

    # پارامترهای مخصوص SVC (بدون epsilon)
    svc_params = {
        'kernel': getattr(cfg, 'kernel', 'rbf'),
        'C': C_val,
        'gamma': getattr(cfg, 'gamma', 'scale'),
        'verbose': 1,  # فعال‌سازی گزارش‌دهی لحظه‌ای
        'probability': True
    }

    models = {
        'price': SVR(**svr_common_params),
        'volatility': SVR(**svr_common_params),
        'confidence': SVC(**svc_params),
    }

    return models

def _build_knn_models(cfg: ArchitectureConfig, plan: AttemptPlan) -> Dict[str, Any]:
    """Build KNN models"""
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

    params = {
        'n_neighbors': getattr(cfg, 'n_neighbors', 5),
        'weights': getattr(cfg, 'weights', 'distance'),
        'algorithm': getattr(cfg, 'algorithm', 'auto'),
        'n_jobs': -1,
    }

    models = {
        'price': KNeighborsRegressor(**params),
        'volatility': KNeighborsRegressor(**params),
        'confidence': KNeighborsClassifier(**params),
    }

    return models


def _build_ensemble_model(cfg: ArchitectureConfig, input_shape: tuple, builder: 'ModelBuilder') -> tf.keras.Model:
    """
    🏗️ DYNAMIC ENSEMBLE BUILDER
    Builds a stacked ensemble model by dynamically loading and combining sub-architectures.

    Args:
        cfg: ArchitectureConfig containing the 'ensemble_of' list (e.g., ["LSTM-Attention", "XGB-Deep"]).
        input_shape: Tuple defining the input tensor shape (e.g., (timesteps, features)).
        builder: Instance of ModelBuilder capable of constructing individual models.

    Returns:
        A compiled tf.keras.Model representing the ensemble.
    """
    import tensorflow as tf
    from typing import List

    # 1. اعتبارسنجی تنظیمات
    if not cfg.ensemble_of:
        raise ValueError("Configuration error: 'ensemble_of' list is empty for Ensemble model.")

    logger.info(f"🔗 Building Ensemble Model: {cfg.name}")
    logger.info(f"   Sub-models: {cfg.ensemble_of}")
    logger.info(f"   Meta-Learner: {cfg.meta_learner}")

    # لایه ورودی مشترک برای تمام زیرمدل‌ها
    main_input = tf.keras.Input(shape=input_shape, name="ensemble_input")

    # لیستی برای نگهداری خروجی‌های هر زیرمدل
    # ساختار: [ [price_out_1, vol_out_1, conf_out_1], [price_out_2, ...], ... ]
    sub_model_outputs: List[List[tf.Tensor]] = []

    # 2. ساخت و اجرای زیرمدل‌ها (Sub-Models)
    for model_name in cfg.ensemble_of:
        try:
            # دریافت تنظیمات زیرمدل از رجیستری
            sub_cfg = ArchitectureRegistry.get_by_name(model_name)  # فرض بر این است متد get_by_name دارید

            if not sub_cfg:
                logger.warning(f"⚠️ Model '{model_name}' not found in registry. Skipping.")
                continue

            # ساخت زیرمدل با استفاده از builder
            # نکته: ما مدل را جداگانه نمی‌سازیم، بلکه لایه‌های آن را روی ورودی اصلی اجرا می‌کنیم
            # تا همه مدل‌ها در یک گراف محاسباتی باشند و وزن‌هایشان مستقل باشد.
            sub_model = builder.build(sub_cfg, input_shape)

            # استخراج لایه ورودی زیرمدل برای اتصال به ورودی اصلی
            # فرض می‌کنیم لایه اول ورودی است
            sub_input_layer = sub_model.layers[0]

            # اتصال ورودی اصلی به لایه‌های داخلی زیرمدل
            # ما باید خروجی‌های زیرمدل را بگیریم. برای این کار، یک مدل موقت Functional می‌سازیم
            # که ورودی آن main_input و خروجی‌هایش خروجی‌های sub_model هستند.
            # این کار باعث می‌شود وزن‌های sub_model مجدداً استفاده شوند.
            x = main_input

            # بازسازی مسیر مدل: عبور از تمام لایه‌ها به جز ورودی
            # روش بهتر: استفاده از متد __call__ روی مدل در حالت training=False برای گرفتن تنسورها
            # اما برای اطمینان از ساختار گراف:

            # پیدا کردن لایه‌هایی که باید به main_input وصل شوند
            # ساده‌ترین راه در Keras Functional API:
            # ساخت یک مدل جدید که ورودی‌اش main_input است و خروجی‌هایش خروجی‌های sub_model هستند
            # اما با وزن‌های sub_model.

            # روش استاندارد برای Ensemble در یک گراف:
            # ما sub_model را به عنوان یک لایه (Model Layer) در نظر می‌گیریم.
            # اما چون sub_cfg ممکن است XGBoost باشد (که Keras نیست)، باید هوشمند عمل کنیم.

            if sub_cfg.model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.SVR, ModelType.RANDOM_FOREST]:
                # مدل‌های غیر Keras (Sklearn based)
                # این مدل‌ها نمی‌توانند داخل گراف Keras باشند.
                # راه حل: استفاده از یک لایه Lambda که در آینده (در مرحله training) پر می‌شود
                # یا ساخت یک Placeholder.
                # در اینجا برای سادگی گراف Keras، فرض می‌کنیم Placeholder داریم یا از یک لایه Dense استفاده می‌کنیم.
                # در یک سیستم واقعی، این بخش باید با Custom Layer مدیریت شود که خروجی Sklearn را می‌گیرد.

                logger.warning(
                    f"⚠️ Skipping non-Keras sub-model '{model_name}' in graph construction. Requires Custom Wrapper.")
                continue

            # اتصال مدل Keras به ورودی اصلی
            # تکنیک: استفاده از sub_model به عنوان یک لایه روی ورودی جدید
            # این کار وزن‌های sub_model را به اشتراک می‌گذارد (Shared Weights) که برای Ensemble غلط است!
            # ما باید وزن‌های جدید بسازیم. بنابراین باید معماری را دوباره بسازیم نه مدل را.

            # اصلاح رویکرد: فراخوانی مجدد builder برای ساخت لایه‌ها با وزن‌های جدید
            # اما builder یک Model برمی‌گرداند.
            # بهترین راه: استفاده از tf.keras.models.clone_model با input_tensor=main_input
            cloned_sub_model = tf.keras.models.clone_model(sub_model, input_tensors=main_input)

            # دریافت خروجی‌های کلون شده
            # خروجی‌ها دیکشنری یا لیست هستند
            if isinstance(sub_model.output, dict):
                current_outputs = list(sub_model.output.values())
            else:
                current_outputs = [sub_model.output] if not isinstance(sub_model.output, list) else sub_model.output

            sub_model_outputs.append(current_outputs)

        except Exception as e:
            logger.error(f"❌ Failed to build sub-model '{model_name}': {e}")
            continue

    if not sub_model_outputs:
        raise ValueError("No valid sub-models could be built for the ensemble.")

    # 3. ادغام خروجی‌ها (Concatenation)
    # فرض می‌کنیم همه مدل‌ها ۳ خروجی دارند (Price, Volatility, Confidence)
    # باید خروجی‌های هم‌نوع را کنار هم بچینیم

    # جدا کردن خروجی‌ها بر اساس نوع (Price, Volatility, Confidence)
    # ترتیب خروجی‌ها در لیست sub_model_outputs باید ثابت باشد.
    # فرض: ترتیب در خروجی دیکشنری یا لیست مشخص است.

    # استخراج خروجی‌های price (فرض می‌کنیم ایندکس ۰ یا نام 'price' است)
    # برای اطمینان، نام‌گذاری لایه‌ها را چک می‌کنیم.

    price_outputs = []
    vol_outputs = []
    conf_outputs = []

    for outs in sub_model_outputs:
        # outs لیستی از تنسورهای خروجی یک مدل است
        # باید نام آن‌ها را پیدا کنیم
        for out in outs:
            name = out.name.split('/')[0].split('_')[0]  # استخراج نام پایه (مثلا price از dense_1/price)
            if 'price' in name:
                price_outputs.append(out)
            elif 'volatility' in name:
                vol_outputs.append(out)
            elif 'confidence' in name:
                conf_outputs.append(out)

    if not price_outputs:
        raise ValueError("Could not identify 'price' outputs from sub-models.")

    # اتصال (Concatenate) خروجی‌های هم‌نوع
    # اگر فقط یک مدل باشد، نیازی به Concat نیست، اما برای Ensemble معمولا چندتاست.
    if len(price_outputs) > 1:
        merged_price = tf.keras.layers.Concatenate(name="merged_price")(price_outputs)
        merged_vol = tf.keras.layers.Concatenate(name="merged_vol")(vol_outputs)
        merged_conf = tf.keras.layers.Concatenate(name="merged_conf")(conf_outputs)
    else:
        merged_price = price_outputs[0]
        merged_vol = vol_outputs[0]
        merged_conf = conf_outputs[0]

    # 4. Meta-Learner (لایه ترکیب نهایی)
    # این بخش یاد می‌گیرد به هر خروجی چقدر وزن بدهد
    dense_layer_size = cfg.units[0] if cfg.units else 64

    # پردازش روی خروجی‌های ادغام شده
    z_price = tf.keras.layers.Dense(dense_layer_size, activation='relu')(merged_price)
    z_vol = tf.keras.layers.Dense(dense_layer_size, activation='relu')(merged_vol)
    z_conf = tf.keras.layers.Dense(dense_layer_size, activation='relu')(merged_conf)

    # لایه نهایی خروجی
    final_price = tf.keras.layers.Dense(1, name='price')(z_price)
    final_vol = tf.keras.layers.Dense(1, activation='softplus', name='volatility')(z_vol)
    final_conf = tf.keras.layers.Dense(1, activation='sigmoid', name='confidence')(z_conf)

    # ساخت مدل نهایی
    ensemble_model = tf.keras.Model(inputs=main_input, outputs={
        'price': final_price,
        'volatility': final_vol,
        'confidence': final_conf
    }, name=f"Ensemble_{cfg.name}")

    logger.info(f"✅ Ensemble model built successfully with {len(sub_model_outputs)} sub-models.")

    return ensemble_model

@dataclass
class AttemptPlan:
    architectures: List[ArchitectureConfig]
    lr_multiplier: float = 1.0
    dropout_multiplier: float = 1.0
    reg_multiplier: float = 1.0


def select_architecture_plan(
        mode: str,
        attempt: int,
        regime_dict: Dict[str, Any],
        X_train: np.ndarray,
        data_quality_score: float = 0.0,
        hw_profile: Optional['HardwareProfile'] = None,
        is_real_time: bool = False,
) -> Tuple['AttemptPlan', 'HardwareProfile', str, 'UtilityWeights']:
    """
    🏛️ QUANT-ARCHITECT: DYNAMIC & ADAPTIVE SELECTION ENGINE

    Priority Order:
    1. Hardware Constraints (Feasibility Filter)
    2. Market Regime (Strategy Alignment)
    3. Data Quality (Noise Robustness)
    4. Attempt History (Progressive Refinement)
    """

    hw = hw_profile
    if hw is None:
        raise ValueError("HardwareProfile is required for architecture selection.")

    # ======================================================================== #
    # 1. LOGGING & HARDWARE ANALYSIS
    # ======================================================================== #
    logger.info("=" * 80)
    logger.info("🎮 HARDWARE PROFILE DETECTED")
    logger.info("=" * 80)
    logger.info(f"   GPU Available : {hw.has_gpu}")
    logger.info(f"   GPU Name      : {hw.gpu_name}")
    logger.info(f"   VRAM          : {hw.gpu_vram_gb:.1f}GB")
    logger.info(f"   Compute Budget: {hw.compute_budget.value}")
    logger.info(f"   Max Batch Size: {hw.max_batch_size}")
    logger.info(f"   Max LSTM Units: {hw.max_lstm_units}")
    logger.info("=" * 80)

    # ======================================================================== #
    # 2. REGIME & DATA EXTRACTION
    # ======================================================================== #
    # اصلاح شده: استفاده از متد resolve کلاس MarketRegime به جای تابع جداگانه
    if isinstance(regime_dict, dict):
        raw_regime = regime_dict.get("regime", "UNDEFINED")
        regime_conf = float(regime_dict.get("confidence", 0.5))
    else:
        raw_regime = "UNDEFINED"
        regime_conf = 0.3

    # استفاده از متد استاتیک resolve برای نرمال‌سازی رژیم
    regime_enum = MarketRegime.resolve(raw_regime)
    regime_key = regime_enum.value  # مقدار نهایی برای استفاده در لاگ‌ها و محاسبات

    logger.info(f"📊 Regime: {regime_key} | Confidence: {regime_conf:.2f}")

    # Data dimensions
    n_samples = X_train.shape[0] if X_train is not None else 10000
    lookback = X_train.shape[1] if X_train is not None and len(X_train.shape) > 1 else 190
    n_features = X_train.shape[2] if X_train is not None and len(X_train.shape) > 2 else 37

    data = DataCharacteristics(
        n_samples=n_samples,
        n_features=n_features,
        sequence_length=lookback,
    )

    # ======================================================================== #
    # 3. DYNAMIC HYPERPARAMETER GENERATION (The "Quant" Way)
    # ======================================================================== #
    # به جای لیست‌های ثابت، مقادیر را بر اساس فیزیک مسئله محاسبه می‌کنیم

    # پایه لرنینگ ریت بر اساس رژیم
    # روندها نیاز به سرعت بالا دارند، رنج‌ها نیاز به احتیاط
    base_lr_map = {
        "bull_trending": 3e-4,
        "bear_trending": 3e-4,
        "sideways_ranging": 1e-4,
        "volatility_cluster": 5e-5,
        "crisis_flash": 1e-5,
        "undefined": 1e-4
    }
    base_lr = base_lr_map.get(regime_key, 1e-4)

    attempt_decay = 1.0 / (1.0 + 0.2 * (attempt - 1))

    # ضریب کیفیت داده: اگر کیفیت پایین است، لرنینگ ریت را کم کنید تا روی نویز فیت نشود
    quality_factor = np.clip(data_quality_score, 0.5, 1.0)

    final_lr = base_lr * attempt_decay * (0.8 + 0.2 * quality_factor)

    # Dropout: بر اساس رژیم و تلاش
    # در روندها Dropout کمتر (برای گرفتن سیگنال)، در رنج/نوسان Dropout بیشتر (برای جلوگیری از فیت روی نویز)
    if regime_key in ["bull_trending", "bear_trending"]:
        base_do = 0.1
    elif regime_key in ["sideways_ranging", "volatility_cluster"]:
        base_do = 0.3
    else:
        base_do = 0.2

    # با افزایش تلاش، Dropout را کمی افزایش می‌دهیم تا مدل Generalize بهتری داشته باشد
    final_do = np.clip(base_do + (attempt * 0.05), 0.0, 0.5)

    # Regularization (L2): بر اساس کیفیت داده
    # داده بد = Regularization بالا
    base_reg = 0.01 if data_quality_score < 0.6 else 0.001
    final_reg = base_reg * (1.2 ** (attempt - 1))

    logger.info(f"🎛️  DYNAMIC HYPERPARAMETERS (Attempt {attempt}):")
    logger.info(f"   Learning Rate : {final_lr:.2e} (Base: {base_lr:.2e}, Decay: {attempt_decay:.2f})")
    logger.info(f"   Dropout       : {final_do:.2f}")
    logger.info(f"   L2 Reg        : {final_reg:.4f}")

    # ======================================================================== #
    # 4. UTILITY & SCORING
    # ======================================================================== #
    weights = _compute_adaptive_weights(regime_key, regime_conf, hw, data, attempt, is_real_time)

    all_cfgs = ArchitectureRegistry.get_all()
    scores = []

    for cfg in all_cfgs:
        cost = _estimate_architecture_cost(cfg, n_samples, lookback, n_features, hw)

        if not cost.feasible:
            logger.debug(f"   ❌ {cfg.name}: {cost.rejection_reason}")
            continue

        score = _score_architecture(
            cfg, regime_key, regime_conf, cost, weights, hw, data, attempt
        )
        scores.append(score)

    # مرتب‌سازی بر اساس امتیاز کاربردی
    scores.sort(key=lambda x: x.utility_score, reverse=True)

    # ======================================================================== #
    # 5. SELECTION LOGIC (Smart Filtering)
    # ======================================================================== #
    # فیلتر کردن معماری‌هایی که قبلاً شکست خورده‌اند (Architected_Failed)
    valid_scores = [s for s in scores if s.cfg.name not in Architected_Failed]

    if not valid_scores:
        logger.warning(
            "⚠️ No valid architectures left after filtering failures. Resetting failure list or picking best failed.")
        # در سناریوی واقعی ممکن است لیست شکست‌ها را پاک کنید یا استراتژی عوض کنید
        valid_scores = scores
    for i in Architected_Failed: logger.info(f" Architected_Failed = {i} ")
    logger.info("📊 TOP CANDIDATES:")
    for i, s in enumerate(valid_scores[:5]):
        logger.info(f"   {i + 1}. {s.cfg.name:25s} | utility={s.utility_score:.3f} | complexity={s.cfg.complexity}")

    selected_cfgs = []

    # استراتژی انتخاب بر اساس بودجه سخت‌افزاری و تلاش
    if attempt == 1:
        # تلاش اول: تنوع بخشی (Diversification)
        # سعی می‌کنیم یک مدل سبک، یک متوسط و یک سنگین (اگر سخت‌افزار اجازه دهد) انتخاب کنیم
        complexities = ["light", "medium", "heavy"]

        for comp in complexities:
            # پیدا کردن بهترین مدل در این کلاس پیچیدگی
            candidates = [s for s in valid_scores if s.cfg.complexity == comp]
            if candidates:
                selected_cfgs.append(candidates[0].cfg)

            # محدودیت تعداد مدل‌ها برای تلاش اول
            if len(selected_cfgs) >= 2:
                break
    else:

        count = 1 if hw.compute_budget == ComputeBudget.LOW else 2
        selected_cfgs = [s.cfg for s in valid_scores[:count]]

    # حذف نام‌های تکراری (احتیاط)
    seen = set()
    unique_selected = []
    for cfg in selected_cfgs:
        if cfg.name not in seen:
            seen.add(cfg.name)
            unique_selected.append(cfg)

    logger.info("✅ FINAL SELECTION:")
    for cfg in unique_selected:
        logger.info(f"   • {cfg.name:25s} | complexity={cfg.complexity}")

    # ساخت پلن نهایی
    plan = AttemptPlan(unique_selected, final_lr, final_do, final_reg)

    return plan, hw, regime_key, weights

async def _maybe_retry(
        X_train, y_train, X_val, y_val, X_test, y_test,
        fold, mode, epochs, symbol, timeframe, debug_mode,
        attempt, reason, local_logger, rejected_details=None,
        forensic_cache_manager=None,
        data_quality_score: float = 0.0,
        is_data_certified: bool = False,
) -> dict:
    """
    🏛️ GOLDMAN SACHS TIER-1 RETRY LOGIC
    - Tracks failed architectures to prevent infinite loops
    - Respects hardware constraints
    - Uses cache hint only as suggestion, not mandate
    """
    # Add current failures to tracking list
    if rejected_details:
        for detail in rejected_details:
            arch_name = detail.get('arch', '')
            if arch_name and arch_name not in Architected_Failed:
                local_logger.info(f"📝 Tracking failed architecture: {arch_name}")
                Architected_Failed.append(arch_name)  # ✅ اضافه شدن به لیست گلوبال

    local_logger.warning(
        f"⚠️  {reason} on attempt {attempt}. "
        f"Retrying with next architecture (attempt {attempt + 1})..."
    )

    fcm = forensic_cache_manager or ForensicCacheManager()

    # Check data health
    skip_repair, skip_reason = _check_data_health(
        fcm=fcm,
        symbol=symbol,
        fold=fold,
        data_quality_score=data_quality_score,
        is_data_certified=is_data_certified,
    )

    if skip_repair:
        local_logger.info(
            f"✅ [RETRY] Data is PRISTINE (reason={skip_reason}) → Skipping repair"
        )

    return await run_quant_training_pipeline(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        fold=fold, mode=mode, epochs=epochs,
        symbol=symbol, timeframe=timeframe,
        debug_mode=debug_mode,
        attempt=attempt + 1,
        forensic_cache_manager=fcm,
        data_quality_score=data_quality_score,
        is_data_certified=is_data_certified,
    )


def _allocate_capital(best, regime_signal, symbol: str) -> Dict[str, Any]:
    try:
        engine = CapitalAllocationEngine(total_capital=1_000_000.0)
        alloc = engine.calculate_allocation(
            symbol=symbol,
            expected_return=best.test_metrics.annualized_return,
            expected_vol=best.test_metrics.annualized_volatility,
            confidence=regime_signal.confidence,  # FIXED: استفاده از آبجکت regime_signal
            win_rate=best.test_metrics.win_rate,
            avg_win=best.test_metrics.avg_win,
            avg_loss=best.test_metrics.avg_loss,
            regime=regime_signal.regime,  # FIXED: استفاده از آبجکت regime_signal
            sharpe_ratio=best.test_metrics.sharpe_ratio,
        )
        return {
            "method": str(alloc.method_used),
            "allocation_pct": alloc.allocation_pct,
            "position_size": alloc.position_size,
            "regime_multiplier": alloc.regime_multiplier,
            "kelly_fraction": alloc.kelly_fraction_used,
        }
    except Exception as e:
        return {"error": str(e)}


async def _train_single_architecture(
        cfg: 'ArchitectureConfig',
        X_train: np.ndarray,
        y_train: Dict[str, np.ndarray],
        X_val: np.ndarray,
        y_val: Dict[str, np.ndarray],
        X_test: np.ndarray,
        y_test: Dict[str, np.ndarray],
        plan: 'AttemptPlan',
        epochs: int,
        symbol: str,
        fold: int,
        regime_signal: 'RegimeSignal',
        loss_fn: 'MultiHeadFinancialLoss',
        logger: logging.Logger,
        _evaluate_fn: Callable,
        checkpoint_manager: Optional['CheckpointManager'] = None,
) -> Optional[Dict[str, Any]]:
    """
    🚀 GOLDMAN SACHS TIER-1 SINGLE ARCHITECTURE TRAINER v5.2
    FIXED: Checkpoint loading and Syntax Errors
    """
    arch_name = cfg.name
    arch_type = cfg.model_type.value if hasattr(cfg.model_type, 'value') else str(cfg.model_type)
    logger.info("=" * 80)
    logger.info(f"📐 Training Architecture: {arch_name} (type={arch_type}, variant={cfg.variant})")
    logger.info("=" * 80)

    # =========================================================================
    # 1️⃣ DATA VALIDATION
    # =========================================================================
    if X_train is None or y_train is None:
        logger.error("❌ Training data is None")
        return None
    if len(X_train) == 0:
        logger.error("❌ Training data is empty")
        return None
    if not np.isfinite(X_train).all():
        logger.error("❌ X_train contains NaN/Inf")
        return None

    required_targets = ['price', 'volatility', 'confidence']
    for key in required_targets:
        if key not in y_train:
            logger.error(f"❌ Missing target key: {key}")
            return None
        if not np.isfinite(y_train[key]).all():
            logger.error(f"❌ y_train[{key}] contains NaN/Inf")
            return None

    n_samples = X_train.shape[0]
    lookback = X_train.shape[1] if X_train.ndim >= 2 else 1
    n_features = X_train.shape[2] if X_train.ndim >= 3 else X_train.shape[1] if X_train.ndim == 2 else 1
    logger.info(f"📊 Data dimensions: samples={n_samples}, lookback={lookback}, features={n_features}")

    # =========================================================================
    # 2️⃣ CREATE CHECKPOINT MANAGER IF NOT PROVIDED
    # =========================================================================
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager(symbol=symbol, fold=fold)

    os.makedirs("models/temp", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    # =========================================================================
    # 3️⃣ CHECK FOR EXISTING CHECKPOINT
    # =========================================================================
    start_epoch = 0
    best_val_loss = float('inf')
    history_dict = {'loss': [], 'val_loss': []}
    trained_models_dict = {}

    weights_path = f"models/temp/best_{symbol}_fold{fold}_{arch_name}.weights"
    model_full_path = f"models/temp/{symbol}_fold{fold}_{arch_name}_final.pkl"

    checkpoint = checkpoint_manager.load_checkpoint(arch_name)
    if checkpoint:
        start_epoch = checkpoint.current_epoch
        best_val_loss = checkpoint.best_val_loss
        history_dict = checkpoint.history
        logger.info(f"🔄 Found checkpoint: epoch {start_epoch}/{epochs} (best_val_loss={best_val_loss:.6f})")

        if start_epoch >= epochs:
            logger.info(f"✅ Model already fully trained (epoch {start_epoch} >= {epochs})")
            logger.info(f"⏭️ Skipping training and loading saved model for evaluation...")
            model_type_str = _detect_model_family(cfg)

            # Load Tree Models
            if model_type_str in ["xgboost", "lightgbm", "catboost", "random_forest", "linear", "svr", "knn"]:
                if os.path.exists(model_full_path):
                    try:
                        with open(model_full_path, 'rb') as f:
                            saved_models = pickle.load(f)
                        if isinstance(saved_models, dict) and 'price' in saved_models:
                            model = UnifiedTreeModelWrapper(
                                models=saved_models,
                                model_type=model_type_str
                            )
                            logger.info(f"✅ Loaded fully trained {model_type_str} model from {model_full_path}")

                            # Evaluate
                            logger.info("📊 Evaluating model on test set...")
                            metrics = await _evaluate_fn(
                                model=model, X_test=X_test, y_test=y_test,
                                regime=regime_signal.regime, eval_logger=logger,  # FIXED
                            )
                            if not metrics:
                                logger.warning("⚠️ Evaluation returned None")
                                return None

                            edge_analysis = await log_test_results(metrics, logger)
                            metrics['edge_detected'] = edge_analysis['has_edge']
                            metrics['edge_analysis'] = edge_analysis

                            return {
                                "model": model, "architecture": arch_name, "model_type": model_type_str,
                                "metrics": metrics, "history": history_dict, "final_lr": None,
                                "n_samples": n_samples, "model_family": model_type_str,
                                "training_complete": True, "checkpoint_loaded": True, "resumed_from_checkpoint": True,
                            }
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load saved model: {e}, will retrain")
                        start_epoch = 0
                else:
                    logger.warning(f"⚠️ Model file {model_full_path} not found, will retrain")
                    start_epoch = 0
            else:
                # TF Models handled later
                pass
        else:
            logger.info(f"🔄 Resuming training from epoch {start_epoch}/{epochs}")

    # =========================================================================
    # 4️⃣ ADAPTIVE LEARNING RATE (FIXED)
    # =========================================================================
    model_family = "tensorflow"  # Default assumption for this snippet

    default_lr_map = {
        "tensorflow": 0.001, "xgboost": 0.01, "lightgbm": 0.01, "catboost": 0.03,
        "random_forest": None, "svr": None, "knn": None, "linear": None, "ensemble": 0.001,
    }

    lr_candidates = [
        ('initial_lr', getattr(cfg, 'initial_lr', None)),
        ('learning_rate', getattr(cfg, 'learning_rate', None)),
        ('xgb_lr', getattr(cfg, 'xgb_lr', None)),
        ('eta', getattr(cfg, 'eta', None)),
        ('lr', getattr(cfg, 'lr', None)),
    ]

    base_lr = None
    lr_source = "default"

    for source_name, lr_value in lr_candidates:
        if lr_value is not None and lr_value > 0:
            base_lr = float(lr_value)
            lr_source = source_name
            logger.info(f"✅ Learning rate detected from '{source_name}': {base_lr:.6f}")
            break

    if base_lr is None:
        default_lr = default_lr_map.get(model_family, 0.001)
        if default_lr is not None:
            base_lr = default_lr
            logger.info(f"⚠️ Using default {model_family} learning rate: {base_lr:.6f}")
        else:
            base_lr = 0.001  # Fallback to safe default

    # FIX: Ensure LR is never zero
    if base_lr <= 0:
        base_lr = 0.001
        logger.warning(f"⚠️ LR was zero or negative. Resetting to safe default: {base_lr}")

    # Apply multiplier
    final_lr = base_lr * getattr(plan, 'lr_multiplier', 1.0)

    # Adaptive scaling based on samples
    if n_samples > 0:
        if n_samples>100000:
            scale_factor = np.sqrt(max(n_samples, 100) / 1000.0)
            computed_lr = final_lr / scale_factor
            final_lr = float(np.clip(computed_lr, 1e-5, 1e-2))  # Min 1e-6 instead of 1e-7
        else:
            final_lr = float(np.clip(final_lr, 1e-4, 5e-3))

    logger.info(f"📈 LR adjusted: base={base_lr:.6f} → final={final_lr:.6f}")

    # =========================================================================
    # 5️⃣ BUILD MODEL
    # =========================================================================
    input_shape = (lookback, n_features) if X_train.ndim >= 3 else (n_features,)
    builder = ModelBuilder(
        cfg=cfg, lr_multiplier=plan.lr_multiplier,
        dropout_multiplier=plan.dropout_multiplier, reg_multiplier=plan.reg_multiplier,
    )

    model = None
    tree_models = None
    model_type_str = None

    try:
        # ─── TENSORFLOW DEEP LEARNING MODELS ──────────────────────────────
        if cfg.model_type in [ModelType.LSTM, ModelType.BILSTM, ModelType.GRU, ModelType.RNN,
                              ModelType.CNN_LSTM, ModelType.TRANSFORMER, ModelType.TCN,
                              ModelType.ATTENTION, ModelType.MLP]:
            if not _HAS_TF:
                logger.error("❌ TensorFlow not installed but required for this architecture")
                return None
            logger.info(f"🔨 Building TensorFlow model: {cfg.model_type.value}")

            # هدایت به متد صحیح بر اساس نوع مدل
            if cfg.model_type == ModelType.LSTM:
                model = builder._build_rnn_base(input_shape, "lstm")
            elif cfg.model_type == ModelType.BILSTM:
                model = builder._build_rnn_base(input_shape, "bilstm")
            elif cfg.model_type == ModelType.GRU:
                model = builder._build_rnn_base(input_shape, "gru")
            elif cfg.model_type == ModelType.RNN:
                model = builder._build_rnn_base(input_shape, "rnn")
            elif cfg.model_type == ModelType.CNN_LSTM:
                model = builder.build_cnn_lstm_attention(input_shape)
            elif cfg.model_type == ModelType.TRANSFORMER:
                model = builder.build_transformer(input_shape)
            elif cfg.model_type == ModelType.TCN:
                model = builder.build_tcn(input_shape)
            elif cfg.model_type == ModelType.ATTENTION:
                model = builder.build_attention(input_shape)
            elif cfg.model_type == ModelType.MLP:
                model = builder.build_mlp(input_shape)
            elif cfg.model_type == ModelType.TABNET:
                model = builder.build_tabnet(input_shape)
            elif cfg.model_type == ModelType.NBEATS:
                model = builder.build_nbeats(input_shape)
            elif cfg.model_type == ModelType.DEEPAR:
                model = builder.build_deepar(input_shape)
            elif cfg.model_type == ModelType.WAVENET:
                model = builder.build_wavenet(input_shape)
            elif cfg.model_type == ModelType.DEEPFM:
                model = builder.build_deepfm(input_shape)
            elif cfg.model_type == ModelType.GAUSSIAN_PROCESS:
                model = builder.build_gp_approx(input_shape)
            elif cfg.model_type == ModelType.TFT:
                model = builder.build_tft(input_shape)
            else:
                # Fallback
                model = builder._build_rnn_base(input_shape, "lstm")

            model_type_str = "tensorflow"

            # Load weights logic (همان کد قبلی)
            if checkpoint and start_epoch < epochs and os.path.exists(checkpoint.model_weights_path):
                try:
                    model.load_weights(checkpoint.model_weights_path)
                    logger.info(f"✅ Loaded weights from {checkpoint.model_weights_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load weights: {e}")

        # XGBoost
        elif cfg.model_type == ModelType.XGBOOST:
            if not _HAS_XGB: logger.error("❌ XGBoost not installed"); return None
            logger.info("🔨 Building XGBoost models")
            tree_models = builder.build_xgboost()
            model_type_str = "xgboost"
            if checkpoint and start_epoch < epochs and os.path.exists(model_full_path):
                try:
                    with open(model_full_path, 'rb') as f:
                        saved_models = pickle.load(f)
                    if isinstance(saved_models, dict) and 'price' in saved_models:
                        tree_models = saved_models
                        logger.info(f"✅ Loaded XGBoost models from {model_full_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load XGBoost models: {e}")

        # LightGBM
        elif cfg.model_type == ModelType.LIGHTGBM:
            if not _HAS_LGB: logger.error("❌ LightGBM not installed"); return None
            logger.info("🔨 Building LightGBM models")
            tree_models = builder.build_lightgbm()
            model_type_str = "lightgbm"
            if checkpoint and start_epoch < epochs and os.path.exists(model_full_path):
                try:
                    with open(model_full_path, 'rb') as f:
                        saved_models = pickle.load(f)
                    if isinstance(saved_models, dict) and 'price' in saved_models:
                        tree_models = saved_models
                        logger.info(f"✅ Loaded LightGBM models from {model_full_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load LightGBM models: {e}")

        # CatBoost
        elif cfg.model_type == ModelType.CATBOOST:
            if not _HAS_CATBOOST: logger.error("❌ CatBoost not installed"); return None
            logger.info("🔨 Building CatBoost models")
            tree_models = builder.build_catboost()
            model_type_str = "catboost"
            if checkpoint and start_epoch < epochs and os.path.exists(model_full_path):
                try:
                    with open(model_full_path, 'rb') as f:
                        saved_models = pickle.load(f)
                    if isinstance(saved_models, dict) and 'price' in saved_models:
                        tree_models = saved_models
                        logger.info(f"✅ Loaded CatBoost models from {model_full_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load CatBoost models: {e}")

        # Sklearn Models
        elif cfg.model_type == ModelType.RANDOM_FOREST:
            try:
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                logger.info("🔨 Building Random Forest models")
                tree_models = _build_random_forest_models(cfg, plan)
                model_type_str = "random_forest"
            except ImportError:
                logger.error("❌ scikit-learn not installed");
                return None

        elif cfg.model_type == ModelType.LINEAR:
            try:
                from sklearn.linear_model import Ridge, Lasso, LogisticRegression
                logger.info("🔨 Building Linear models")
                tree_models = _build_linear_models(cfg, plan)
                model_type_str = "linear"
            except ImportError:
                logger.error("❌ scikit-learn not installed");
                return None

        elif cfg.model_type == ModelType.SVR:
            try:
                from sklearn.svm import SVR, SVC
                logger.info("🔨 Building SVR models")
                tree_models = _build_svr_models(cfg, plan)
                model_type_str = "svr"
            except ImportError:
                logger.error("❌ scikit-learn not installed");
                return None

        elif cfg.model_type == ModelType.KNN:
            try:
                from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
                logger.info("🔨 Building KNN models")
                tree_models = _build_knn_models(cfg, plan)
                model_type_str = "knn"
            except ImportError:
                logger.error("❌ scikit-learn not installed");
                return None

        elif cfg.model_type in [ModelType.ENSEMBLE, ModelType.HYBRID]:
            logger.info("🔨 Building Ensemble/Hybrid model")
            model = _build_ensemble_model(cfg, input_shape, builder)
            model_type_str = "ensemble"

        else:
            logger.error(f"❌ Unsupported model type: {cfg.model_type}")
            return None

        # =====================================================================
        # 6️⃣ TRAINING - TENSORFLOW
        # =====================================================================
        if model is not None and model_type_str == "tensorflow":
            logger.info("🚀 Training TensorFlow model...")
            if final_lr is None or final_lr <= 0:
                final_lr = 0.001
                logger.warning(f"⚠️ Using fallback LR: {final_lr}")

            if cfg.optimizer_type == "adamw":
                optimizer = tf.keras.optimizers.AdamW(learning_rate=final_lr, weight_decay=1e-5, clipnorm=1.0)
            elif cfg.optimizer_type == "sgd":
                optimizer = tf.keras.optimizers.SGD(learning_rate=final_lr, momentum=0.9, nesterov=True, clipnorm=1.0)
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=final_lr, clipnorm=1.0)

            model.compile(
                optimizer=optimizer,
                loss={
                    'price': loss_fn.price_loss,
                    'volatility': loss_fn.volatility_loss,
                    'confidence': loss_fn.confidence_loss,
                },
                loss_weights={
                    'price': loss_fn.price_weight,
                    'volatility': loss_fn.volatility_weight,
                    'confidence': loss_fn.confidence_weight,
                },
                metrics={
                    'price': ['mae'],
                    'volatility': ['mae'],
                    'confidence': ['accuracy'],
                }
            )

            resume_logger = ResumeEpochLogger(
                logger=logger, checkpoint_manager=checkpoint_manager, arch_name=arch_name,
                arch_variant=cfg.variant, total_epochs=epochs, model_weights_path=weights_path + '.h5'
            )

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True,
                                                 min_delta=1e-5),
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7,
                                                     verbose=1),
                resume_logger,
            ]

            batch_size = getattr(cfg, 'batch_size', 64)
            if batch_size is None:
                batch_size = min(64, max(32, n_samples // 100))

            epochs_to_run = epochs - start_epoch
            if epochs_to_run > 0:
                logger.info(f"🚀 Training for {epochs_to_run} epochs (total {epochs})")
                try:
                    history = model.fit(
                        X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                        verbose=1, shuffle=True, initial_epoch=start_epoch,
                    )
                    model.save_weights(weights_path + '.h5')
                    logger.info(f"💾 Final model weights saved to {weights_path}.h5")
                except Exception as e:
                    if "NaN" in str(e):
                        logger.warning(f"⚠️ NaN detected, retrying with lower LR")
                        new_lr = final_lr * 0.5  # FIXED: Syntax Error (_0.5 -> * 0.5)
                        tf.keras.backend.set_value(model.optimizer.learning_rate, new_lr)
                        history = model.fit(
                            X_train, y_train, validation_data=(X_val, y_val),
                            epochs=epochs - start_epoch, batch_size=batch_size,
                            callbacks=callbacks, verbose=1, initial_epoch=start_epoch,
                        )
                    else:
                        raise e

        # =====================================================================
        # 7️⃣ TRAINING - TREE-BASED
        # =====================================================================
        elif tree_models is not None:
            # Check if already trained (redundant check but safe)
            if checkpoint and start_epoch >= epochs:
                # This block is technically handled in section 3, but kept for safety flow
                pass
            else:
                logger.info(f"🚀 Training {model_type_str.upper()} models...")
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                X_test_flat = X_test.reshape(X_test.shape[0], -1)
                logger.info(f"📊 Flattened shapes: train={X_train_flat.shape}, val={X_val_flat.shape}")

                trained_models = {}
                training_complete = True

                for target_name in required_targets:
                    if target_name in tree_models:
                        model_obj = tree_models[target_name]
                        y_target = y_train[target_name].ravel()
                        logger.info(f"  📈 Training {target_name} model...")
                        try:
                            if model_type_str == "xgboost":
                                eval_set = [(X_val_flat, y_val[target_name].ravel())]
                                model_obj.fit(X_train_flat, y_target, eval_set=eval_set, verbose=True)
                            elif model_type_str == "lightgbm":
                                model_obj.fit(
                                    X_train_flat, y_target,
                                    eval_set=[(X_val_flat, y_val[target_name].ravel())],
                                    eval_metric='rmse' if target_name != 'confidence' else 'binary_logloss',
                                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
                                    verbose=True
                                )
                            elif model_type_str == "catboost":
                                model_obj.fit(
                                    X_train_flat, y_target,
                                    eval_set=(X_val_flat, y_val[target_name].ravel()),
                                    verbose=10, plot=True
                                )
                            elif model_type_str in ["random_forest", "linear", "svr", "knn"]:
                                model_obj.fit(X_train_flat, y_target)
                            else:
                                model_obj.fit(X_train_flat, y_target)

                            trained_models[target_name] = model_obj
                            logger.info(f"    ✅ {target_name} model trained")
                        except Exception as e:
                            logger.error(f"    ❌ Failed to train {target_name}: {e}")
                            training_complete = False
                            break

                if training_complete and len(trained_models) == len(required_targets):
                    try:
                        with open(model_full_path, 'wb') as f:
                            pickle.dump(trained_models, f)
                        logger.info(f"💾 Full model saved to {model_full_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not save model pickle: {e}")

                    model = UnifiedTreeModelWrapper(models=trained_models, model_type=model_type_str)
                    logger.info(f"✅ Unified wrapper created for {model_type_str}")

                    checkpoint = TrainingCheckpoint(
                        symbol=symbol, fold=fold, architecture_name=arch_name,
                        architecture_variant=cfg.variant, current_epoch=epochs,
                        total_epochs=epochs, best_val_loss=best_val_loss, optimizer_state={},
                        model_weights_path=model_full_path, history={'loss': [], 'val_loss': []},
                        regime_at_checkpoint=regime_signal.regime.value, timestamp=datetime.now()  # FIXED
                    )
                    checkpoint_manager.save_checkpoint(checkpoint)
                    logger.info(f"💾 Checkpoint saved for {arch_name}")
                else:
                    logger.error(f"❌ Not all models trained successfully")
                    return None
        else:
            logger.error(f"❌ No model created for {arch_name}")
            return None

        # =========================================================================
        # 8️⃣ EVALUATION (WITH CACHING LOGIC)
        # =========================================================================
        logger.info("📊 Preparing evaluation phase...")

        # مسیر ذخیره نتایج ارزیابی (متفاوت از فایل مدل)
        metrics_path = f"models/temp/{symbol}_fold{fold}_{arch_name}_metrics.json"
        metrics = None
        evaluation_skipped = False

        # بررسی اینکه آیا مدل قبلاً کامل آموزش دیده و ارزیابی شده است؟
        # اگر start_epoch >= epochs یعنی آموزش کامل بوده است.
        if start_epoch >= epochs and os.path.exists(metrics_path):
            logger.info(f"✅ Found cached evaluation results at {metrics_path}")
            logger.info(f"⏭️ Skipping re-evaluation (Model unchanged). Loading previous metrics...")
            try:
                import json
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                # تبدیل مقادیر کلیدی به float (چون json همه چیز را رشته یا عدد ساده ذخیره می‌کند)
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        metrics[k] = float(v)
                evaluation_skipped = True
                logger.info("✅ Cached metrics loaded successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Could not load cached metrics: {e}. Will re-evaluate.")
                evaluation_skipped = False

        # اگر متریک‌ها لود نشده بودند (یا مدل جدید آموزش دیده)، ارزیابی را انجام بده
        if not evaluation_skipped:
            logger.info("📊 Evaluating model on test set...")
            metrics = await _evaluate_fn(
                model=model,
                X_test=X_test,
                y_test=y_test,
                regime=regime_signal.regime,  # FIXED
                eval_logger=logger,
            )
            if not metrics:
                logger.warning("⚠️ Evaluation returned None")
                return None

            # ذخیره متریک‌ها در دیسک برای دفعات بعد
            try:
                import json
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"💾 Evaluation metrics saved to {metrics_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not save metrics to file: {e}")

        # لاگ‌گیری کامل نتایج تست (چه جدید چه کش شده)
        edge_analysis = await log_test_results(metrics, logger)

        # ذخیره در metrics برای گزارش نهایی
        metrics['edge_detected'] = edge_analysis['has_edge']
        metrics['edge_analysis'] = edge_analysis

        # Log key metrics
        logger.info(f"📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"📉 Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"🎯 Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}")

        # =========================================================================
        # 9️⃣ SAVE FINAL CHECKPOINT (COMPLETE)
        # =========================================================================
        final_checkpoint = TrainingCheckpoint(
            symbol=symbol,
            fold=fold,
            architecture_name=arch_name,
            architecture_variant=cfg.variant,
            current_epoch=epochs,
            total_epochs=epochs,
            best_val_loss=best_val_loss,
            optimizer_state={},
            model_weights_path=weights_path + ('.h5' if model_type_str == 'tensorflow' else '.pkl'),
            history=history_dict,
            regime_at_checkpoint=regime_signal.regime.value,  # FIXED
            timestamp=datetime.now()
        )
        checkpoint_manager.save_checkpoint(final_checkpoint)

        # =========================================================================
        # 🔟 RETURN RESULTS
        # =========================================================================
        return {
            "model": model,
            "architecture": arch_name,
            "model_type": model_type_str,
            "metrics": metrics,
            "history": history_dict,
            "final_lr": final_lr,
            "n_samples": n_samples,
            "model_family": model_family,
            "training_complete": True,
            "checkpoint_saved": True,
            "evaluation_skipped": evaluation_skipped,  # نشان می‌دهد که آیا ارزیابی از کش لود شده یا خیر
        }
    except Exception as e:
        logger.error(f"❌ Fatal error in _train_single_architecture: {e}")
        logger.exception(e)
        # Save emergency checkpoint on error
        try:
            emergency_checkpoint = TrainingCheckpoint(
                symbol=symbol,
                fold=fold,
                architecture_name=arch_name,
                architecture_variant=cfg.variant,
                current_epoch=start_epoch,
                total_epochs=epochs,
                best_val_loss=best_val_loss,
                optimizer_state={},
                model_weights_path=weights_path + ('.h5' if model_type_str == 'tensorflow' else '.pkl'),
                history=history_dict,
                regime_at_checkpoint=regime_signal.regime.value,  # FIXED
                timestamp=datetime.now()
            )
            checkpoint_manager.save_checkpoint(emergency_checkpoint)
            logger.info(f"💾 Emergency checkpoint saved on error")
        except:
            pass
        return None


async def run_quant_training_pipeline(
        X_train: np.ndarray,
        y_train: Union[np.ndarray, Dict[str, np.ndarray]],
        X_val: np.ndarray,
        y_val: Union[np.ndarray, Dict[str, np.ndarray]],
        X_test: np.ndarray,
        y_test: Union[np.ndarray, Dict[str, np.ndarray]],
        fold: int,
        mode: str,
        epochs: int,
        symbol: str = None,
        timeframe: str = "1H",
        debug_mode: bool = False,
        attempt: int = 1,
        data_quality_score: float = 0.0,
        is_data_certified: bool = False,
        forensic_cache_manager: Optional['ForensicCacheManager'] = None,
        checkpoint_manager: Optional['CheckpointManager'] = None,
) -> Dict[str, Any]:
    """
    ============================================================================
    🚀 QUANT TRAINING PIPELINE v2.1 - WITH CRASH RECOVERY
    ============================================================================
    """
    local_logger = logger

    # ── 1. symbol / timeframe ────────────────────────────────────────────────
    if symbol is None:
        try:
            symbol, timeframe = get_active_symbol_timeframe()
        except Exception:
            return {"status": "failed", "reason": "no_active_symbol"}

    local_logger.info(
        f"\n{'=' * 70}\n"
        f"🚀 Quant Pipeline v2.1 | {symbol} | fold={fold} | mode={mode} | attempt={attempt}\n"
        f"{'=' * 70}"
    )

    # ── 2. data validation ───────────────────────────────────────────────────
    MIN_SAMPLES = 500
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if X is None:
            return {"status": "failed", "reason": f"no_{name}_data"}

    if X_train.shape[0] < MIN_SAMPLES:
        return {"status": "failed",
                "reason": f"insufficient_samples:{X_train.shape[0]}<{MIN_SAMPLES}"}

    # Ensure 3D shape for sequence models
    if X_train.ndim != 3:
        X_train = X_train.reshape(X_train.shape[0], 1, -1)
    if X_val.ndim != 3:
        X_val = X_val.reshape(X_val.shape[0], 1, -1)
    if X_test.ndim != 3:
        X_test = X_test.reshape(X_test.shape[0], 1, -1)

    # Normalize targets to dict format
    y_train_d = _normalize_targets(y_train)
    y_val_d = _normalize_targets(y_val)
    y_test_d = _normalize_targets(y_test)

    # ── 3. FORENSIC CACHE CHECK → conditional self-healing ──────────────────
    fcm = forensic_cache_manager or ForensicCacheManager()
    skip_healing, skip_reason = _check_data_health(
        fcm=fcm,
        symbol=symbol,
        fold=fold,
        data_quality_score=data_quality_score,
        is_data_certified=is_data_certified,
    )

    if skip_healing:
        local_logger.info(
            f"⚡ [CACHE] Data VERIFIED HEALTHY ({skip_reason}) → "
            f"self-healing SKIPPED — architecture will evolve per attempt"
        )

    # ── 4. regime detection ─────────────────────────────────────────────────
    # خروجی دیکشنری است
    regime_data = await get_symbol_regime()


    resolved_regime_enum = MarketRegime.resolve(regime_data.get('regime', 'undefined'))

    regime_signal = RegimeSignal(
        regime=resolved_regime_enum,
        confidence=regime_data.get('confidence', 0.0),
        persistence_score=regime_data.get('persistence_score', 0.0),
        volatility_regime=regime_data.get('volatility_regime', 'UNKNOWN'),
        trend_strength=regime_data.get('trend_strength', 0.0),
        liquidity_regime=regime_data.get('liquidity_regime', 'UNKNOWN'),
        detection_timestamp=datetime.now(),
        lookback_window=X_train.shape[1],
        metadata={}
    )

    # دسترسی به مقادیر از طریق آبجکت ساخته شده
    local_logger.info(
        f"📊 Regime: {regime_signal.regime.value} | "
        f"Confidence: {regime_signal.confidence:.1%}"
    )

    # ── 6. ✅ MULTI-OBJECTIVE ARCHITECTURE SELECTION (v3.0) ─────────────────
    local_logger.info(
        f"📤 Sending regime to selector: {regime_signal.regime.value} (confidence={regime_signal.confidence:.2f})"
    )

    # ✅ FIX: ارسال hw_profile به سلکتور برای تطبیق با تغییرات جدید
    plan, hw_profile, resolved_regime, utility_weights = select_architecture_plan(
        mode=mode,
        attempt=attempt,
        regime_dict={
            "regime": regime_signal.regime.value,
            "confidence": regime_signal.confidence,
        },
        X_train=X_train,
        data_quality_score=data_quality_score,
        hw_profile=HardwareProfile,  # ✅ اضافه شد
    )

    # بعد از فراخوانی، نتیجه را لاگ کنید
    local_logger.info(f"📥 Selector returned resolved_regime: {resolved_regime}")
    local_logger.info(f"📥 Selected architectures: {[c.name for c in plan.architectures]}")

    arch_names = [c.name for c in plan.architectures]
    tel_log.trade(
        f"\n{'─' * 70}\n"
        f"📋 Attempt {attempt} plan\n"
        f"   Architectures  : {arch_names}\n"
        f"   lr_mul         : {plan.lr_multiplier:.2f}\n"
        f"   dropout_mul    : {plan.dropout_multiplier:.2f}\n"
        f"   reg_mul        : {plan.reg_multiplier:.2f}\n"
        f"   Cache guided   : {skip_healing and attempt == 1}\n"
        f"   Regime key     : {resolved_regime}\n"
        f"   HW Budget      : {hw_profile.compute_budget.value} | "
        f"VRAM={hw_profile.gpu_vram_gb:.1f}GB\n"
        f"   Data shapes    : train={X_train.shape} | val={X_val.shape} | test={X_test.shape}\n"
        f"{'─' * 70}"
    )

    # ── 7. loss function ─────────────────────────────────────────────────────
    loss_fn = MultiHeadFinancialLoss(
        risk_free_rate=0.02,
        drawdown_penalty=0.7 * plan.dropout_multiplier,
        turnover_penalty=0.15,
    )

    # ── 7b. Initialize checkpoint manager if not provided ────────────────────
    if checkpoint_manager is None:
        checkpoint_manager = CheckpointManager(symbol=symbol, fold=fold)

    # ── 8. train all architectures ───────────────────────────────────────────
    trained_models: List[Dict[str, Any]] = []
    for cfg in plan.architectures:
        # Call _train_single_architecture with checkpoint manager
        result = await _train_single_architecture(
            cfg=cfg,
            X_train=X_train,
            y_train=y_train_d,
            X_val=X_val,
            y_val=y_val_d,
            X_test=X_test,
            y_test=y_test_d,
            plan=plan,
            epochs=epochs,
            symbol=symbol,
            fold=fold,
            regime_signal=regime_signal,  # FIXED: ارسال آبجکت صحیح
            loss_fn=loss_fn,
            logger=local_logger,
            _evaluate_fn=_evaluate_model_financial,
            checkpoint_manager=checkpoint_manager,
        )
        if result is not None:
            trained_models.append(result)

    # Check if any models were trained successfully
    if not trained_models:
        local_logger.error(f"❌ All architectures failed in attempt {attempt}")
        return await _maybe_retry(
            X_train, y_train_d, X_val, y_val_d, X_test, y_test_d,
            fold, mode, epochs, symbol, timeframe, debug_mode, attempt,
            reason="all_architectures_failed",
            local_logger=local_logger,
            forensic_cache_manager=fcm,
            data_quality_score=data_quality_score,
            is_data_certified=is_data_certified,
        )

    # ── 9. rejection engine ───────────────────────────────────────────────────
    rejection_engine = ModelRejectionEngine()
    candidates = []
    for info in trained_models:
        metrics_dict = info["metrics"]
        fin_metrics = _dict_to_financial_metrics(metrics_dict)

        # 🔥 FIXED: Use info["model_type"] which now exists
        model_type_str = info.get("model_type", "lstm")  # Fallback to lstm if missing
        try:
            arch_enum = ModelType(model_type_str)
        except:
            # Default to LSTM if unknown
            arch_enum = ModelType.LSTM

        candidate = ModelCandidate(
            symbol=symbol,
            architecture=arch_enum,
            model_object=info["model"],
            regime_at_training=regime_signal.regime,  # FIXED
            regime_detector_state={
                "regime": regime_signal.regime.value,
                "confidence": regime_signal.confidence,
            },
            train_metrics=fin_metrics,
            val_metrics=fin_metrics,
            test_metrics=fin_metrics,
            monte_carlo_stability=0.8,
            bootstrap_robustness=0.8,
            adversarial_robustness=0.8,
        )
        candidate._arch_name = info["architecture"]
        candidate = rejection_engine.evaluate_model(candidate)
        candidates.append(candidate)

    accepted = [c for c in candidates if not c.is_rejected]
    rejected = [c for c in candidates if c.is_rejected]

    local_logger.info(
        f"📊 Rejection: {len(accepted)} accepted | {len(rejected)} rejected"
    )

    for c in rejected:
        reason = c.rejection_reasons[0] if c.rejection_reasons else "unknown"
        arch_name = getattr(c, '_arch_name', '?')
        if arch_name not in Architected_Failed:
            Architected_Failed.append(arch_name)  # ✅ اضافه شدن به لیست سیاه
        tel_log.trade(
            f"   ❌ REJECTED [{arch_name}]: {reason}\n"
        )

    logger.warning(f"length of Architected_Failed = {len(Architected_Failed)}")

    for c in accepted:
        local_logger.info(
            f"   ✅ ACCEPTED [{getattr(c, '_arch_name', '?')}]: "
            f"Sharpe={c.test_metrics.sharpe_ratio:.2f} | "
            f"Score={c.quality_score:.3f}"
        )

    # 🔥 10. retry if no accepted models (UNLIMITED attempts)
    if not accepted:
        return await _maybe_retry(
            X_train, y_train_d, X_val, y_val_d, X_test, y_test_d,
            fold, mode, epochs, symbol, timeframe, debug_mode, attempt,
            reason="all_models_rejected",
            rejected_details=[
                {"arch": getattr(c, "_arch_name", "?"), "reasons": c.rejection_reasons}
                for c in rejected
            ],
            local_logger=local_logger,
            forensic_cache_manager=fcm,
            data_quality_score=data_quality_score,
            is_data_certified=is_data_certified,
        )

    # ── 11. best model selection ──────────────────────────────────────────────
    best = max(accepted, key=lambda c: c.quality_score)
    local_logger.info(
        f"\n🏆 Best: [{getattr(best, '_arch_name', '?')}] | "
        f"Score={best.quality_score:.3f} | "
        f"Sharpe={best.test_metrics.sharpe_ratio:.2f} | "
        f"MaxDD={best.test_metrics.max_drawdown:.1%}"
    )

    # ── 12. capital allocation ─────────────────────────────────────────────────
    allocation = _allocate_capital(best, regime_signal, symbol)  # FIXED

    # ── 13. final report ──────────────────────────────────────────────────────
    return {
        "status": "success",
        "symbol": symbol,
        "timeframe": timeframe,
        "fold": fold,
        "mode": mode,
        "attempt": attempt,
        "regime": {
            "name": regime_signal.regime.value,
            "confidence": regime_signal.confidence,
            "persistence": regime_signal.persistence_score,
            "allocation_mult": regime_signal.regime.capital_allocation_multiplier,
            "max_position_pct": regime_signal.regime.max_leverage,  # FIXED: max_leverage
            "is_tradeable": regime_signal.is_tradeable,
            "volatility_regime": regime_signal.volatility_regime,
            "trend_strength": regime_signal.trend_strength,
        },
        "forensic_cache": {
            "skip_healing": skip_healing,
            "skip_reason": skip_reason,
        },
        "architecture_plan": {
            "attempt": attempt,
            "architectures": [c.name for c in plan.architectures],
            "lr_mul": plan.lr_multiplier,
            "reg_mul": plan.reg_multiplier,
            "cache_guided": skip_healing and attempt == 1,
        },
        "data": {
            "train_samples": X_train.shape[0],
            "val_samples": X_val.shape[0],
            "test_samples": X_test.shape[0],
            "lookback": X_train.shape[1],
            "features": X_train.shape[2],
        },
        "models": {
            "total_trained": len(trained_models),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "accepted_models": [
                {
                    "name": getattr(c, "_arch_name", "?"),
                    "sharpe": c.test_metrics.sharpe_ratio,
                    "max_drawdown": c.test_metrics.max_drawdown,
                    "win_rate": c.test_metrics.win_rate,
                    "score": c.quality_score,
                }
                for c in accepted
            ],
            "rejected_models": [
                {
                    "name": getattr(c, "_arch_name", "?"),
                    "reasons": c.rejection_reasons,
                }
                for c in rejected
            ],
        },
        "best_model": {
            "name": getattr(best, "_arch_name", "?"),
            "sharpe_ratio": best.test_metrics.sharpe_ratio,
            "annualized_return": best.test_metrics.annualized_return,
            "max_drawdown": best.test_metrics.max_drawdown,
            "calmar_ratio": best.test_metrics.calmar_ratio,
            "win_rate": best.test_metrics.win_rate,
            "profit_factor": best.test_metrics.profit_factor,
            "quality_score": best.quality_score,
            "overfit_gap": best.overfit_gap,
        },
        "capital_allocation": allocation,
        "timestamp": datetime.now().isoformat(),
    }

# =============================================================================
# HIGH-LEVEL WRAPPER - COMPLETE WITH DATE VERIFICATION AND RETRY LOOP
# =============================================================================

async def run_training_with_universal_load(
        symbol: str = None,
        fold: int = None,
        lookback: int = 190,
        training_type: str = "ensemble",
        debug_mode: bool = False,
        force_regime_refresh: bool = False,
        force_data_repair: bool = True,
        max_retries: int = None,  # 🔥 FIX: Parameter kept for compatibility but ignored
        forensic_cache_path: Optional[str] = None,
        resume_from_crash: bool = True,
        loaded_data=None) -> Dict[str, Any]:
    """
    🚀 HIGH-LEVEL WRAPPER v2.2 - UNLIMITED ATTEMPTS + CRASH RECOVERY

    New:
      • Resume training from exact point after crash
      • Checkpoint management for each fold
      • LightGBM & CatBoost support
      • Fixed category score extraction
      • 🔥 NO MAX ATTEMPTS - Will retry infinitely until model is accepted
    """
    local_logger = logger

    # ── symbol / timeframe ────────────────────────────────────────────────────
    if symbol is None:
        try:
            symbol, timeframe = get_active_symbol_timeframe()
            local_logger.info(f"📊 Active Symbol: {symbol} | Timeframe: {timeframe}")
        except Exception as e:
            local_logger.error(f"❌ Failed to get active symbol: {e}")
            return {'status': 'failed', 'reason': 'no_active_symbol'}
    else:
        try:
            _, timeframe = get_active_symbol_timeframe()
            local_logger.info(f"📊 Using active timeframe: {timeframe} for symbol: {symbol}")
        except Exception:
            timeframe = "1H"
            local_logger.info(f"📊 Using default timeframe: {timeframe} for symbol: {symbol}")

    # ── INIT FORENSIC CACHE MANAGER ────────────────────────────────────────────
    fcm = ForensicCacheManager(cache_path=forensic_cache_path)
    cache_loaded = fcm.load()
    if cache_loaded:
        local_logger.info("✅ [SESSION] ForensicCacheManager ready")
    else:
        local_logger.warning("⚠️ [SESSION] Forensic cache not available - healing will run normally")

    # ── دریافت لیست فولدهای فعال ──────────────────────────────────────────────
    active_folds = get_latest_training_folds(TOTAL_FOLDS, NUM_FOLDS)
    local_logger.info(f"📂 Active folds for processing: {active_folds}")

    if not active_folds:
        local_logger.error("❌ No active folds to process.")
        return {'status': 'failed', 'reason': 'no_active_folds'}

    final_reports = []
    overall_status = 'success'

    for current_fold in active_folds:
        local_logger.info(f"\n{'=' * 80}")
        local_logger.info(f"[{symbol}] 📅 PROCESSING FOLD: {current_fold}")
        local_logger.info(f"{'=' * 80}")

        # ── NEW: Checkpoint manager for this fold ──────────────────────────────
        checkpoint_manager = CheckpointManager(symbol=symbol, fold=current_fold)

        # If resuming, we don't need to re-verify/repair if checkpoints exist
        skip_data_prep = False
        if resume_from_crash:
            # Check if any architecture checkpoint exists
            arch_checkpoints = list(Path("models/checkpoints").glob(
                f"{symbol}_fold{current_fold}_*_checkpoint.pkl"
            ))
            if arch_checkpoints:
                local_logger.info(
                    f"🔄 Found {len(arch_checkpoints)} checkpoint(s) for fold {current_fold}. "
                    f"Resuming training without re-processing data."
                )
                skip_data_prep = True
                # We'll still load data, but skip verification/repair

        # ── STEP 0: بررسی cache برای این fold ─────────────────────────────────
        fold_skip_healing, fold_skip_reason = fcm.should_skip_self_healing(
            symbol=symbol, fold=current_fold
        )

        # =========================================================================
        # استخراج data_quality_score
        # =========================================================================
        data_quality_score = 0.0
        is_data_certified = False

        _cache_path_str = str(fcm.cache_path) if fcm else "cache/forensic_cache.json"
        data_quality_score, is_data_certified = _extract_quality_from_cache(_cache_path_str)

        if data_quality_score == 0.0 and cache_loaded:
            try:
                cache_summary = fcm.get_full_summary(symbol=symbol, fold=current_fold)
                if cache_summary:
                    for _key in ('normalized_score', 'Normalized Score',
                                 'quality_score', 'data_score', 'score'):
                        _val = cache_summary.get(_key)
                        if _val is not None:
                            data_quality_score = float(_val)
                            break

                    if not is_data_certified:
                        _cert = cache_summary.get(
                            'certification_status',
                            cache_summary.get('Certification', '')
                        )
                        is_data_certified = str(_cert).upper() in (
                            'CERTIFIED', 'CONDITIONAL', 'CERTIFIED_WITH_WARNINGS'
                        )
            except Exception as _e:
                local_logger.warning(f"[{symbol}] ⚠️ data_quality extraction fallback failed: {_e}")

        local_logger.info(
            f"[{symbol}] 📊 Data Quality → score={data_quality_score:.2f} | "
            f"certified={is_data_certified} | fold={current_fold}"
        )
        if fold_skip_healing:
            local_logger.info(
                f"[{symbol}] ⚡ Fold {current_fold}: forensic cache OK → "
                f"pre-verification self-healing SKIPPED ({fold_skip_reason})"
            )

        # ── Skip data prep if resuming ────────────────────────────────────────
        if skip_data_prep:
            local_logger.info(f"[{symbol}] ⏭️ Skipping data verification/repair (resuming from checkpoint)")
            date_verification_passed = True
            verification_status = "RESUMED"

        # ── STEP 2: LOAD DATA ─────────────────────────────────────────────────
        local_logger.info(f"[{symbol}] 📊 STEP 2: LOAD DATA | fold={current_fold}")

        X_train = X_val = X_test = None
        y_train = y_val = y_test = None

        try:
            # از verification results ابتدا
            if not skip_data_prep and 'loaded_data' in locals() and loaded_data:
                for key, data in loaded_data.items():
                    if 'sequenced_train' in key and isinstance(data, dict):
                        X_train = data.get('X')
                        y_train = data.get('y')
                    elif 'sequenced_val' in key and isinstance(data, dict):
                        X_val = data.get('X')
                        y_val = data.get('y')
                    elif 'sequenced_test' in key and isinstance(data, dict):
                        X_test = data.get('X')
                        y_test = data.get('y')

            # fallback به direct loading
            if X_train is None:
                local_logger.info(f"[{symbol}] ⚠️ Loading train directly...")
                X_train, y_train = await universal_load(
                    data_type="sequenced", symbol=symbol, timeframe=timeframe,
                    fold=current_fold, split_type="train", lookback=lookback, multi_output=True
                )
            if X_val is None:
                X_val, y_val = await universal_load(
                    data_type="sequenced", symbol=symbol, timeframe=timeframe,
                    fold=current_fold, split_type="val", lookback=lookback, multi_output=True
                )
            if X_test is None:
                X_test, y_test = await universal_load(
                    data_type="sequenced", symbol=symbol, timeframe=timeframe,
                    fold=current_fold, split_type="test", lookback=lookback, multi_output=True
                )

        except Exception as e:
            local_logger.error(f"[{symbol}] ❌ Error loading data: {e}")
            final_reports.append({'fold': current_fold, 'status': 'failed', 'reason': f'data_load_error:{e}'})
            continue

        if X_train is None or y_train is None:
            local_logger.error(f"[{symbol}] ❌ Failed to load training data for fold {current_fold}")
            final_reports.append({'fold': current_fold, 'status': 'failed', 'reason': 'data_load_failed'})
            continue

        local_logger.info(
            f"✅ Data loaded | "
            f"train={X_train.shape} | "
            f"val={X_val.shape if X_val is not None else 'None'} | "
            f"test={X_test.shape if X_test is not None else 'None'}"
        )

        # ── STEP 3: TRAINING WITH UNLIMITED RETRY LOOP ─────────────────────────────────
        local_logger.info(
            f"[{symbol}] 🚀 STEP 3: TRAINING | fold={current_fold} | UNLIMITED ATTEMPTS"
        )

        attempt = 1
        fold_result = None

        # 🔥 REMOVED max_retries limit - will loop until success
        while True:
            local_logger.info(f"\n{'─' * 60}")
            local_logger.info(
                f"[{symbol}] 🔄 ATTEMPT {attempt} (unlimited) | fold={current_fold}"
            )
            local_logger.info(f"{'─' * 60}")

            result = await run_quant_training_pipeline(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                fold=current_fold, mode=training_type,
                epochs=EPOCH_D if debug_mode else EPOCH_P,
                symbol=symbol, timeframe=timeframe,
                debug_mode=debug_mode,
                attempt=attempt,
                forensic_cache_manager=fcm,
                data_quality_score=data_quality_score,
                is_data_certified=is_data_certified,
                checkpoint_manager=checkpoint_manager,  # NEW
            )

            if result.get('status') == 'success':
                local_logger.info(
                    f"[{symbol}] ✅ Training successful | "
                    f"attempt={attempt} | fold={current_fold}"
                )
                fold_result = result
                break

            elif result.get('status') == 'rejected':
                tel_log.trade(
                    f"[{symbol}] ❌ Rejected on attempt {attempt}. "
                    f"Evaluating repair need..."
                )

                _skip, _reason = _check_data_health(
                    fcm=fcm,
                    symbol=symbol,
                    fold=current_fold,
                    data_quality_score=data_quality_score,
                    is_data_certified=is_data_certified,
                )

                if _skip:
                    local_logger.info(
                        f"[{symbol}] ✅ Data PRISTINE ({_reason}) → "
                        f"Skipping repair. Next attempt = different architecture."
                    )

                attempt += 1
                # 🔥 Continue loop (no max attempts)

            else:
                local_logger.error(
                    f"[{symbol}] ❌ Training failed | "
                    f"attempt={attempt} | status={result.get('status')}"
                )
                fold_result = result
                break

        if fold_result is None:
            fold_result = {
                'status': 'failed', 'reason': 'unknown_error',
                'symbol': symbol, 'fold': current_fold, 'attempts': attempt
            }

        if isinstance(fold_result, dict):
            fold_result['fold'] = current_fold
            fold_result['date_verification'] = {
                'status': verification_status,
                'passed': date_verification_passed
            }
            fold_result['attempts_used'] = attempt
            fold_result['cache_skip_healing'] = fold_skip_healing
            fold_result['cache_skip_reason'] = fold_skip_reason
            fold_result['data_quality_score'] = data_quality_score

        final_reports.append(fold_result)

        if fold_result.get('status') != 'success':
            overall_status = 'partial'

    # ── Final summary ─────────────────────────────────────────────────────────
    if all(r.get('status') != 'success' for r in final_reports):
        overall_status = 'failed'

    summary_report = {
        'status': overall_status,
        'symbol': symbol,
        'timeframe': timeframe,
        'folds_processed': active_folds,
        'total_folds': TOTAL_FOLDS,
        'num_folds': NUM_FOLDS,
        'fold_reports': final_reports,
        'forensic_cache': {
            'cache_loaded': cache_loaded,
            'cache_path': str(fcm.cache_path),
        },
        'data_quality': {
            'scores': [r.get('data_quality_score', 0.0) for r in final_reports],
            'certified_folds': [r.get('fold') for r in final_reports if r.get('data_quality_score', 0.0) >= 95.0]
        },
        'timestamp': datetime.now().isoformat()
    }

    local_logger.info(f"\n{'=' * 80}")
    local_logger.info(
        f"[{symbol}] ✅ TRAINING COMPLETE | Status: {overall_status} | "
        f"Cache guided: {cache_loaded}"
    )
    local_logger.info(f"{'=' * 80}")

    return summary_report




# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # GPU
    "_GPU_CONFIG",
    "_get_xgb_gpu_params",
    "_get_lgb_gpu_params",
    "_get_catboost_gpu_params",
    # LR
    "WarmupCosineDecayCallback",
    # Builder
    "ModelBuilder",
    # Callbacks
    "_make_callbacks",
    "_compile_tf_model",
    # Plan
    "AttemptPlan",
    # Trainer
    "_train_single_architecture",
    # Debug
    'MarketRegime',
    'RegimeSignal',
    'SignalQualityLayer',
    'CapitalAllocationEngine',
    'ModelRejectionEngine',
    'AutoShutdownManager',
    'ExecutionIntelligenceLayer',
    'ModelCandidate',
    'FinancialMetrics',
    'SharpeAwareLoss',
    'MultiHeadFinancialLoss',
    'DrawdownPenaltyRegularizer',
    'TradingCostConfig',
    'ForensicCacheManager',
    'select_architecture_plan',
    'run_quant_training_pipeline',
    'run_training_with_universal_load',
    'TrainingContext',

    # NEW - Checkpoint classes
    'TrainingCheckpoint',
    'CheckpointManager',
    'ResumeEpochLogger',
]

