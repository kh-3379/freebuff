# architecture_selector.py - ENHANCED VERSION

from __future__ import annotations
from core.imports import *
from core.logger import *
from core.regim import *
from core.regim import _get_default_robustness

logger = AdvancedTradeLogger(__name__)
try:
    import lightgbm as lgb

    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import catboost as cb

    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    _HAS_TF = True
except ImportError:
    _HAS_TF = False

# Lazy XGB import
try:
    import xgboost as xgb

    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

class ComputeBudget(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    INSTITUTIONAL = "INSTITUTIONAL"


@dataclass
class HardwareProfile:
    """
    کلاس پروفایل سخت‌افزاری با قابلیت اطمینان بالا.
    این کلاس به عنوان منبع اصلی حقیقت (Single Source of Truth) عمل می‌کند.
    """
    cpu_cores: int = 4
    ram_gb: float = 8.0
    has_gpu: bool = False
    gpu_vram_gb: float = 0.0
    gpu_compute_cap: float = 0.0
    gpu_name: str = "CPU"
    compute_budget: ComputeBudget = ComputeBudget.LOW
    training_time_constraint_hours: float = 24.0
    inference_latency_constraint_ms: float = 100.0

    # محدودیت‌های مشتق شده (Dynamic Caps)
    max_batch_size: int = 64
    max_lstm_units: int = 128
    max_cnn_filters: int = 64
    max_transformer_dmodel: int = 64
    training_time_multiplier: float = 8.0

    def __post_init__(self):
        # اعتبارسنجی سخت‌گیرانه پس از مقداردهی اولیه
        if not self.has_gpu or self.gpu_vram_gb < 0.5:
            self.has_gpu = False
            self.gpu_vram_gb = 0.0
            self.gpu_name = "CPU"
            self.compute_budget = ComputeBudget.LOW

    @property
    def is_low_end(self) -> bool:
        return self.compute_budget == ComputeBudget.LOW or not self.has_gpu

    @property
    def can_run_transformers(self) -> bool:
        return self.compute_budget in [ComputeBudget.HIGH, ComputeBudget.INSTITUTIONAL]


def _detect_hardware_profile() -> HardwareProfile:
    """
    🔥 GOLDMAN SACHS TIER-1 HARDWARE DETECTION (UPDATED)
    تشخیص سخت‌افزار با اولویت‌بندی دقیق برای جلوگیری از ناهماهنگی.
    اولویت: nvidia-smi > PyTorch > TensorFlow
    """
    logger.info("🔍 شروع عملیات تشخیص سخت‌افزار سیستم کنونی...")
    profile = HardwareProfile()

    # 1. CPU Detection
    try:
        profile.cpu_cores = os.cpu_count() or 4
    except Exception:
        profile.cpu_cores = 4

    # 2. RAM Detection
    try:
        import psutil
        profile.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        profile.ram_gb = 8.0

    # 3. GPU Detection (Robust Multi-Method)
    vram_detected = 0.0
    gpu_name = "CPU"
    has_gpu = False

    # Method 1: nvidia-smi (بسیار دقیق برای VRAM و نام کارت)
    # این روش بهترین روش برای جلوگیری از ناهماهنگی است چون مستقل از فریمورک پایتون کار می‌کند.
    try:
        cmd = ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                gpu_name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                vram_detected = vram_mb / 1024.0
                has_gpu = True
                logger.info(f"✅ GPU شناسایی شد (nvidia-smi): {gpu_name} با {vram_detected:.1f}GB VRAM")
    except FileNotFoundError:
        logger.debug("nvidia-smi یافت نشد.")
    except Exception as e:
        logger.debug(f"nvidia-smi خطا داد: {e}")

    # Method 2: PyTorch (پشتیبان)
    if not has_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                has_gpu = True
                gpu_name = torch.cuda.get_device_name(0)
                vram_detected = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                logger.info(f"✅ GPU شناسایی شد (PyTorch): {gpu_name} با {vram_detected:.1f}GB VRAM")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"PyTorch GPU detection failed: {e}")

    # Method 3: TensorFlow (پشتیبان ثانویه)
    if not has_gpu:
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                has_gpu = True
                try:
                    details = tf.config.experimental.get_device_details(gpus[0])
                    gpu_name = details.get('device_name', 'Unknown GPU')
                except:
                    gpu_name = "Unknown TensorFlow GPU"

                # تخمین VRAM اگر nvidia-smi و PyTorch نبودند
                if vram_detected == 0:
                    if '3090' in gpu_name or '4090' in gpu_name:
                        vram_detected = 24.0
                    elif '3080' in gpu_name:
                        vram_detected = 10.0
                    else:
                        vram_detected = 8.0
                logger.info(f"✅ GPU شناسایی شد (TensorFlow): {gpu_name}")
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"TensorFlow GPU detection failed: {e}")

    # تنظیم پروفایل نهایی
    profile.has_gpu = has_gpu
    profile.gpu_name = gpu_name
    profile.gpu_vram_gb = vram_detected

    # 4. Budget Classification Logic (هوشمند)
    if not profile.has_gpu or profile.gpu_vram_gb < 4.0:
        profile.compute_budget = ComputeBudget.LOW
        profile.max_batch_size = 32
        profile.max_lstm_units = 64
        profile.max_cnn_filters = 32
        profile.max_transformer_dmodel = 32
        profile.training_time_multiplier = 8.0
    elif profile.gpu_vram_gb <= 8.0:
        profile.compute_budget = ComputeBudget.MEDIUM
        profile.max_batch_size = 128
        profile.max_lstm_units = 256
        profile.max_cnn_filters = 128
        profile.max_transformer_dmodel = 128
        profile.training_time_multiplier = 2.0
    elif profile.gpu_vram_gb <= 20.0:
        profile.compute_budget = ComputeBudget.HIGH
        profile.max_batch_size = 256
        profile.max_lstm_units = 512
        profile.max_cnn_filters = 256
        profile.max_transformer_dmodel = 256
        profile.training_time_multiplier = 1.0
    else:
        profile.compute_budget = ComputeBudget.INSTITUTIONAL
        profile.max_batch_size = 512
        profile.max_lstm_units = 1024
        profile.max_cnn_filters = 512
        profile.max_transformer_dmodel = 512
        profile.training_time_multiplier = 0.5

    logger.info(
        f"🏛️ پروفایل نهایی سخت‌افزار: {profile.gpu_name} | VRAM: {profile.gpu_vram_gb:.1f}GB | Budget: {profile.compute_budget.value}")
    return profile


def configure_gpu_robust(profile: HardwareProfile) -> Dict[str, Any]:
    """
    🔥 Goldman Sachs Tier-1 GPU Configuration (UPDATED)

    این تابع دیگر خودش سخت‌افزار را اسکن نمی‌کند.
    بلکه بر اساس پروفایل دریافتی (Single Source of Truth) عمل می‌کند.
    این کار تضمین می‌کند که تمامی بخش‌های سیستم هماهنگ عمل کنند.
    """
    report = {
        "tensorflow": "not_configured",
        "xgboost": "not_configured",
        "lightgbm": "not_configured",
        "catboost": "not_configured",
        "gpu_available": profile.has_gpu,
        "gpu_name": profile.gpu_name,
        "vram_gb": profile.gpu_vram_gb,
        "compute_budget": profile.compute_budget.value
    }

    if not profile.has_gpu:
        logger.info("💻 GPU در پروفایل تشخیص داده نشد. سیستم در حالت CPU اجرا می‌شود.")
        return report

    # ===== TensorFlow GPU Configuration =====
    try:
        import tensorflow as tf
        # تنظیم رشد حافظه برای جلوگیری از اشغال کل VRAM توسط TensorFlow
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"✅ TensorFlow Memory Growth برای {len(gpus)} GPU فعال شد.")
            except RuntimeError as e:
                logger.warning(f"⚠️ Memory Growth قبلاً تنظیم شده: {e}")

            # تنظیم دقت بر اساس بودجه سخت‌افزاری
            if profile.compute_budget == ComputeBudget.INSTITUTIONAL:
                try:
                    policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    tf.keras.mixed_precision.set_global_policy(policy)
                    report["tensorflow"] = f"GPU enabled: Mixed Precision (FP16)"
                    logger.info("🚀 Mixed Precision (FP16) فعال شد.")
                except Exception as e:
                    tf.keras.mixed_precision.set_global_policy('float32')
                    report["tensorflow"] = f"GPU enabled: FP32 (Mixed precision failed)"
            else:
                tf.keras.mixed_precision.set_global_policy('float32')
                report["tensorflow"] = f"GPU enabled: FP32 Stable"
        else:
            report["tensorflow"] = "CPU fallback (TF list_physical_devices empty)"
            logger.warning("⚠️ پروفایل GPU دارد اما TensorFlow خالی است!")
    except ImportError:
        report["tensorflow"] = "TensorFlow not installed"
    except Exception as e:
        report["tensorflow"] = f"config_error: {e}"
        logger.error(f"❌ خطای پیکربندی TensorFlow: {e}")

    # ===== CatBoost GPU Verification =====
    try:
        import catboost as cb
        # تست کوچک برای اطمینان از درایورهای GPU
        X_test = np.random.randn(100, 5).astype(np.float32)
        y_test = np.random.randn(100).astype(np.float32)
        try:
            params = {
                'iterations': 2,
                'task_type': 'GPU',
                'devices': '0',
                'verbose': False
            }
            model = cb.CatBoostRegressor(**params)
            model.fit(X_test, y_test, verbose=False)
            report["catboost"] = "GPU verified"
            logger.info("✅ CatBoost GPU تایید شد.")
        except Exception as e:
            report["catboost"] = f"GPU failed, CPU fallback: {str(e)[:50]}"
            logger.warning(f"⚠️ CatBoost GPU ناموفق، استفاده از CPU: {e}")
    except ImportError:
        report["catboost"] = "CatBoost not installed"
    except Exception as e:
        report["catboost"] = f"error: {e}"

    # ===== XGBoost & LightGBM Placeholders =====
    # در اینجا می‌توانید منطق XGBoost و LightGBM را مشابه CatBoost اضافه کنید
    # فعلاً بر اساس پروفایل پر می‌شوند
    if profile.has_gpu:
        report["xgboost"] = "GPU available (check config)"
        report["lightgbm"] = "GPU available (check config)"

    # Final Summary
    logger.info("=" * 60)
    logger.info("🎮 خلاصه پیکربندی GPU")
    logger.info("=" * 60)
    logger.info(f"Hardware Profile: {profile.gpu_name}")
    logger.info(f"VRAM: {profile.gpu_vram_gb:.1f} GB")
    logger.info(f"Budget: {profile.compute_budget.value}")
    logger.info(f"TensorFlow: {report['tensorflow']}")
    logger.info(f"CatBoost: {report['catboost']}")
    logger.info("=" * 60)

    return report


# =============================================================================
# 🔥 EXECUTION BLOCK
# =============================================================================

# 1. ابتدا پروفایل سخت‌افزار را دقیق تشخیص می‌دهیم
HWPROFILE: HardwareProfile = _detect_hardware_profile()

# 2. سپس تنظیمات GPU را بر اساس همان پروفایل اعمال می‌کنیم
_GPU_CONFIG = configure_gpu_robust(HWPROFILE)




class ModelType(Enum):
    LSTM = "lstm"
    GRU = "gru"
    BILSTM = "bilstm"
    RNN = "rnn"
    TRANSFORMER = "transformer"
    ATTENTION = "attention"
    CNN_LSTM = "cnn_lstm"
    TCN = "tcn"
    TABNET = "tabnet"
    NBEATS = "nbeats"
    DEEPAR = "deepar"
    WAVENET = "wavenet"
    TFT = "tft"
    DEEPFM = "deepfm"
    GAUSSIAN_PROCESS = "gaussian_process"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    LINEAR = "linear"
    SVR = "svr"
    RANDOM_FOREST = "random_forest"
    KNN = "knn"
    MLP = "mlp"
    BAYESIAN = "bayesian"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"
    HYBRID = "hybrid"


class ModelBuilder:
    """
    جامع‌ترین سازنده مدل برای بازارهای مالی.
    پشتیبانی کامل از: LSTM, GRU, BiLSTM, CNN-LSTM, TCN, Transformer, TFT,
    TabNet, N-BEATS, DeepAR, WaveNet, DeepFM, GP, XGB, LGBM, CatBoost, MLP, Attention
    """

    def __init__(self, cfg: 'ArchitectureConfig', lr_multiplier: float = 1.0, dropout_multiplier: float = 1.0,
                 reg_multiplier: float = 1.0):
        self.cfg = cfg
        self.lr_mul = lr_multiplier
        self.do_mul = min(dropout_multiplier, 2.0)
        self.reg_mul = min(reg_multiplier, 1.8)

    def _get_reg(self):
        return regularizers.l2(getattr(self.cfg, 'l2_reg', 1e-4) * self.reg_mul)

    def _get_do(self):
        return min(getattr(self.cfg, 'dropout', 0.2) * self.do_mul, 0.5)

    def _build_output_heads(self, inputs, features, do: float, reg):
        # اصلاح سینتکس ضرب: do _ 0.5 تبدیل شد به do * 0.5
        x = layers.GaussianNoise(0.01, name="structural_noise")(features)
        x = layers.Dense(64, activation="swish", kernel_regularizer=reg, name="shared_dense")(x)
        x = layers.BatchNormalization(name="shared_bn")(x)
        x = layers.Dropout(do * 0.5, name="shared_drop")(x)  # FIX: Syntax Error

        # FIX: اضافه کردن activation='linear' برای خروجی قیمت
        # این کار باعث می‌شود مدل بتواند گرادیان‌ها را به درستی منتقل کند
        price_out = layers.Dense(1, activation='linear', name="price",
                                 kernel_initializer='he_normal', kernel_regularizer=reg)(x)

        vol_out = layers.Dense(1, activation="softplus", name="volatility",
                               kernel_initializer='he_normal', kernel_regularizer=reg)(x)

        conf_out = layers.Dense(1, activation="sigmoid", name="confidence",
                                kernel_initializer='glorot_uniform', kernel_regularizer=reg)(x)

        return models.Model(inputs=inputs, outputs={"price": price_out, "volatility": vol_out, "confidence": conf_out})

    # --- Helpers ---
    def _apply_attention_block(self, x, do, reg):
        num_heads = getattr(self.cfg, 'num_heads', 4)
        input_shape = tf.keras.backend.int_shape(x)
        feature_dim = input_shape[-1]
        if feature_dim is None:
            feature_dim = 64
            logging.warning(f"⚠️ Attention block detected None feature dimension, defaulting to {feature_dim}")
        key_dim = max(8, feature_dim // num_heads)
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=do)(x, x)
        x = layers.Add()([x, attn_out])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x

    # --- 1. RNN Family ---
    def _build_rnn_base(self, input_shape: Tuple[int, int], rnn_layer_type: str) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = inputs
        rnn_units = cfg.units if hasattr(cfg, 'units') and cfg.units else [128]
        n_layers = len(rnn_units) if isinstance(rnn_units, list) else 1
        for i in range(n_layers):
            units = rnn_units[i] if isinstance(rnn_units, list) else rnn_units
            return_seq = (i < n_layers - 1) or cfg.use_attention
            if rnn_layer_type == "lstm":
                x = layers.LSTM(units, return_sequences=return_seq, dropout=do, kernel_regularizer=reg)(x)
            elif rnn_layer_type == "gru":
                x = layers.GRU(units, return_sequences=return_seq, dropout=do, kernel_regularizer=reg)(x)
            elif rnn_layer_type == "bilstm":
                x = layers.Bidirectional(
                    layers.LSTM(units, return_sequences=return_seq, dropout=do, kernel_regularizer=reg))(x)
            else:
                x = layers.SimpleRNN(units, return_sequences=return_seq, dropout=do, kernel_regularizer=reg)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
        if cfg.use_attention and len(x.shape) == 3:
            x = self._apply_attention_block(x, do, reg)
        if len(x.shape) == 3:
            x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 2. CNN-LSTM ---
    def build_cnn_lstm_attention(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.Conv1D(64, 1, padding="causal", activation="relu", kernel_regularizer=reg)(inputs)
        for f in (cfg.cnn_filters if cfg.cnn_filters else [64, 128]):
            x_res = x
            x = layers.Conv1D(f, 3, padding="causal", activation="gelu", kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(do)(x)
            if x_res.shape[-1] != f:
                x_res = layers.Conv1D(f, 1, padding="causal")(x_res)
            x = layers.Add()([x, x_res])
        x = layers.LSTM(128, return_sequences=cfg.use_attention, dropout=do, kernel_regularizer=reg)(x)
        if cfg.use_attention:
            x = self._apply_attention_block(x, do, reg)
        if len(x.shape) == 3:
            x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 3. Transformer & TFT ---
    def build_transformer(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.Dense(cfg.d_model, kernel_regularizer=reg)(inputs)
        seq_len = input_shape[0]
        pos_emb = layers.Embedding(seq_len, cfg.d_model)(tf.range(start=0, limit=seq_len, delta=1))
        x = x + pos_emb
        # اصلاح تایپو: _in به _
        for _ in range(cfg.num_layers):
            attn_output = layers.MultiHeadAttention(num_heads=cfg.num_heads, key_dim=cfg.d_model // cfg.num_heads)(x, x)
            x = layers.Add()([x, attn_output])
            x = layers.LayerNormalization()(x)
            ffn = layers.Dense(cfg.ff_dim, activation="relu")(x)
            ffn = layers.Dense(cfg.d_model)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    def build_tft(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.Dense(cfg.d_model, activation='elu')(inputs)
        for _ in range(cfg.num_layers):
            x = layers.Dense(cfg.d_model, activation='elu')(x)
            attn_out = layers.MultiHeadAttention(cfg.num_heads, cfg.d_model // cfg.num_heads)(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 4. TCN ---
    def build_tcn(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = inputs
        num_filters = cfg.cnn_filters if cfg.cnn_filters else [64, 128]
        for i, f in enumerate(num_filters):
            res = layers.Conv1D(f, 1, padding='causal', kernel_regularizer=reg)(x) if x.shape[-1] != f else x
            for dilation in [1, 2, 4, 8]:
                x = layers.Conv1D(f, 3, padding='causal', dilation_rate=dilation, kernel_regularizer=reg)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.SpatialDropout1D(do)(x)
            x = layers.Add()([x, res])
        x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 5. N-BEATS ---
    def build_nbeats(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.Reshape((input_shape[0], input_shape[1]))(inputs)
        n_stacks = 3 if cfg.complexity == "heavy" else 2
        for _ in range(n_stacks):
            for _ in range(3):
                h = layers.Dense(512, activation='relu')(x)
                h = layers.Dropout(do)(h)
                theta_b = layers.Dense(input_shape[0])(h)
                theta_f = layers.Dense(1)(h)
                backcast = layers.Dense(input_shape[0])(theta_b)
                x = layers.Subtract()([x, backcast])
        x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 6. TabNet ---
    def build_tabnet(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.Reshape((-1,))(inputs)
        x = layers.Dense(128, activation='relu')(x)
        n_steps = 5 if cfg.complexity == "heavy" else 3
        for _ in range(n_steps):
            x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(do)(x)
        decision = layers.Dense(64, activation='sigmoid', name='decision_mask')(x)
        x = layers.Multiply()([x, decision])
        return self._build_output_heads(inputs, x, do, reg)

    # --- 7. DeepAR ---
    def build_deepar(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.LSTM(128, return_sequences=True, dropout=do, kernel_regularizer=reg)(inputs)
        x = layers.LSTM(64, return_sequences=False, dropout=do, kernel_regularizer=reg)(x)
        mu = layers.Dense(1, name="price_mu")(x)
        sigma = layers.Dense(1, activation='softplus', name="price_sigma")(x)
        conf = layers.Dense(1, activation='sigmoid', name="confidence")(x)
        return models.Model(inputs=inputs, outputs={"price": mu, "volatility": sigma, "confidence": conf})

    # --- 8. WaveNet ---
    def build_wavenet(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = inputs
        num_blocks = 2 if cfg.complexity == "light" else 3
        filters = 32 if cfg.complexity == "light" else 64
        for b in range(num_blocks):
            for l in range(6):
                dilation_rate = 2 ** l
                res = x
                x = layers.Conv1D(filters, 2, padding='causal', dilation_rate=dilation_rate, activation='tanh')(x)
                x = layers.Conv1D(filters, 1, padding='causal', activation='sigmoid')(x)
                x = layers.Multiply()([x, res])
                x = layers.Conv1D(filters, 1, padding='causal')(x)
                x = layers.Add()([x, res])
        x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 9. DeepFM ---
    def build_deepfm(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.Flatten()(inputs)
        fm_first_order = layers.Dense(1, use_bias=False)(x)
        x_deep = layers.Dense(128, activation='relu')(x)
        x_deep = layers.BatchNormalization()(x_deep)
        x_deep = layers.Dropout(do)(x_deep)
        x_deep = layers.Dense(64, activation='relu')(x_deep)
        combined = layers.Concatenate()([fm_first_order, x_deep])
        return self._build_output_heads(inputs, combined, do, reg)

    # --- 10. Gaussian Process (Approx) ---
    def build_gp_approx(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")
        x = layers.Flatten()(inputs)
        x = layers.Dense(100, activation='relu')(x)
        x = layers.Dropout(do)(x)
        mean = layers.Dense(1, name="price")(x)
        variance = layers.Dense(1, activation='softplus', name="volatility")(x)
        conf = layers.Dense(1, activation='sigmoid', name="confidence")(x)
        return models.Model(inputs=inputs, outputs={"price": mean, "volatility": variance, "confidence": conf})

    # --- 11. MLP ---
    def build_mlp(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        if len(input_shape) == 3:
            inputs = layers.Input(shape=input_shape, name="sequence_input")
            x = layers.Flatten()(inputs)
        else:
            inputs = layers.Input(shape=input_shape, name="flat_input")
            x = inputs
        units_list = getattr(cfg, 'units', [256, 128, 64])
        if not isinstance(units_list, list):
            units_list = [units_list]
        for i, units in enumerate(units_list):
            x = layers.Dense(units, activation='swish', kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(do)(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 12. Attention (Pure Attention Model) ---
    def build_attention(self, input_shape: Tuple[int, int]) -> "tf.keras.Model":
        """
        مدل مبتنی بر Attention خالص برای سری‌های زمانی.
        این متد برای رفع مشکل کرش در ModelType.ATTENTION اضافه شد.
        """
        cfg = self.cfg
        do = self._get_do()
        reg = self._get_reg()
        inputs = layers.Input(shape=input_shape, name="sequence_input")

        # پروجکشن اولیه برای تطبیق ابعاد
        x = layers.Dense(cfg.d_model, kernel_regularizer=reg)(inputs)

        # بلوک‌های توجه چند سره
        for _ in range(getattr(cfg, 'num_layers', 2)):
            attn_out = layers.MultiHeadAttention(
                num_heads=cfg.num_heads,
                key_dim=cfg.d_model // cfg.num_heads,
                dropout=do
            )(x, x)
            x = layers.Add()([x, attn_out])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

            # Feed Forward Network
            ffn = layers.Dense(cfg.ff_dim, activation="relu")(x)
            ffn = layers.Dropout(do)(ffn)
            ffn = layers.Dense(cfg.d_model)(ffn)
            x = layers.Add()([x, ffn])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

        x = layers.GlobalAveragePooling1D()(x)
        return self._build_output_heads(inputs, x, do, reg)

    # --- 13. Tree Models ---
    def build_xgboost(self) -> Dict[str, Any]:
        cfg = self.cfg
        params = {
            'n_estimators': getattr(cfg, 'n_estimators', 500),
            'max_depth': getattr(cfg, 'max_depth', 6),
            'learning_rate': getattr(cfg, 'learning_rate', 0.05) * self.lr_mul,
            'subsample': getattr(cfg, 'subsample', 0.8),
            'colsample_bytree': getattr(cfg, 'colsample', 0.8),
            'reg_alpha': getattr(cfg, 'reg_alpha', 0.1) * self.reg_mul,
            'reg_lambda': getattr(cfg, 'reg_lambda', 1.0) * self.reg_mul,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        return {
            "price": xgb.XGBRegressor(**params),
            "volatility": xgb.XGBRegressor(**params),
            "confidence": xgb.XGBClassifier(**{**params, 'objective': 'binary:logistic'})
        }

    def build_lightgbm(self) -> Dict[str, Any]:
        cfg = self.cfg
        params = {
            'n_estimators': getattr(cfg, 'n_estimators', 500),
            'num_leaves': getattr(cfg, 'num_leaves', 31),
            'learning_rate': getattr(cfg, 'learning_rate', 0.05) * self.lr_mul,
            'feature_fraction': getattr(cfg, 'feature_fraction', 0.8),
            'bagging_fraction': getattr(cfg, 'bagging_fraction', 0.8),
            'bagging_freq': 5,
            'reg_alpha': getattr(cfg, 'reg_alpha', 0.1) * self.reg_mul,
            'reg_lambda': getattr(cfg, 'reg_lambda', 1.0) * self.reg_mul,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        return {
            "price": lgb.LGBMRegressor(**params),
            "volatility": lgb.LGBMRegressor(**params),
            "confidence": lgb.LGBMClassifier(**{**params, 'objective': 'binary'})
        }

    def build_catboost(self) -> Dict[str, Any]:
        cfg = self.cfg
        params = {
            'iterations': getattr(cfg, 'iterations', 500),
            'depth': getattr(cfg, 'depth', 6),
            'learning_rate': getattr(cfg, 'learning_rate', 0.05) * self.lr_mul,
            'l2_leaf_reg': getattr(cfg, 'reg_lambda', 3.0) * self.reg_mul,
            'random_seed': 42,
            'verbose': False
        }
        return {
            "price": cb.CatBoostRegressor(**params),
            "volatility": cb.CatBoostRegressor(**params),
            "confidence": cb.CatBoostClassifier(**{**params, 'loss_function': 'Logloss'})
        }

    def make_optimizer(self, steps_per_epoch: int = 100):
        cfg = self.cfg
        lr = float(getattr(cfg, 'initial_lr', 0.001) * self.lr_mul)
        if cfg.optimizer_type == "adamw":
            opt = optimizers.AdamW(learning_rate=lr, weight_decay=1e-4)
        elif cfg.optimizer_type == "nadam":
            opt = optimizers.Nadam(learning_rate=lr)
        else:
            opt = optimizers.Adam(learning_rate=lr)
        return opt

@dataclass
class ArchitectureConfig:
    """🏛️ GOLDMAN SACHS TIER-1 ARCHITECTURE CONFIG"""
    name: str
    model_type: ModelType
    variant: int
    regime_robustness: Dict[str, float]

    # Deep Learning Shared
    use_cosine_lr: bool = True
    price_weight: float = 0.6
    volatility_weight: float = 0.3
    confidence_weight: float = 0.1
    ensemble_of: List[str] = field(default_factory=list)
    suitable_for: List[str] = field(default_factory=list)
    description: str = ""
    complexity: str = "medium"

    # RNN / LSTM / GRU
    units: Optional[List[int]] = None
    use_bidirectional: bool = False
    use_attention: bool = False
    use_residual: bool = False
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    dropout: float = 0.2
    recurrent_dropout: float = 0.1
    optimizer_type: str = "adamw"
    initial_lr: Optional[float] = None

    # Regularization
    l2_reg: float = 1e-4
    weight_decay: float = 1e-5
    clipnorm: float = 1.0

    # CNN Specific
    cnn_filters: Optional[List[int]] = None
    cnn_kernel_sizes: Optional[List[int]] = None
    use_separable_conv: bool = False

    # Transformer / Attention Specific
    d_model: Optional[int] = None
    num_heads: Optional[int] = None
    ff_dim: Optional[int] = None
    num_layers: Optional[int] = None

    # Tree Models (XGB/LGBM/CAT)
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    tree_method: str = "hist"
    device: str = "cpu"
    dart: bool = False
    use_dart: bool = False
    xgb_lr: Optional[float] = None
    subsample: float = 0.8
    colsample: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    min_child_weight: int = 1

    # LightGBM Specific
    num_leaves: Optional[int] = None
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_data_in_leaf: int = 20

    # CatBoost Specific
    depth: Optional[int] = None
    iterations: Optional[int] = None

    # SVR / KNN / RF
    min_samples_split: Optional[int] = None
    min_samples_leaf: Optional[int] = None
    n_neighbors: Optional[int] = None
    weights: Optional[str] = None
    algorithm: Optional[str] = None
    kernel: Optional[str] = None
    C: Optional[float] = None
    epsilon: Optional[float] = None
    gamma: Optional[str] = None

    # Hybrid / Ensemble
    meta_learner: Optional[str] = None
    blending_weight: float = 0.5

    # Performance Metrics
    base_predictive_power: float = 0.7
    generalization_stability: float = 0.7
    flops_per_sample: float = 100.0
    memory_per_sample_mb: float = 5.0
    training_time_ratio: float = 1.0
    inference_latency_ms: float = 2.0
    hyperparameter_sensitivity: float = 0.5
    min_samples_required: int = 500

    def __post_init__(self):
        if not self.description:
            self.description = f"{self.name} - {self.complexity} complexity"
        if not self.regime_robustness:
            self.regime_robustness = {k: 0.7 for k in
                                      ["trending", "mean_reverting", "high_volatility", "low_volatility",
                                       "structural_break", "regime_transition", "low_liquidity", "event_driven"]}
        if not self.ensemble_of:
            self.ensemble_of = []
        if not self.suitable_for:
            self.suitable_for = ["all"]
        if self.dart and not self.use_dart: self.use_dart = self.dart
        if self.use_dart and not self.dart: self.dart = self.use_dart


class ArchitectureRegistry:
    registry: Dict[Tuple[ModelType, int], ArchitectureConfig] = {}
    name_index: Dict[str, ArchitectureConfig] = {}

    @classmethod
    def register(cls, config: ArchitectureConfig):
        cls.registry[(config.model_type, config.variant)] = config
        cls.name_index[config.name] = config

    @classmethod
    def get(cls, model_type: ModelType, variant: int) -> Optional[ArchitectureConfig]:
        return cls.registry.get((model_type, variant))

    @classmethod
    def get_by_name(cls, name: str) -> Optional[ArchitectureConfig]:
        return cls.name_index.get(name)

    @classmethod
    def get_all(cls) -> List[ArchitectureConfig]:
        return list(cls.registry.values())



def register_dynamic_architectures():
    logger.info("🚀 Initiating Dynamic Architecture Generation based on Hardware Profile...")

    default_regime_signal = RegimeSignal(
        regime=MarketRegime.SIDEWAYS_RANGING,
        confidence=0.8,
        persistence_score=0.7,
        volatility_regime="NORMAL",
        trend_strength=0.5,
        liquidity_regime="HIGH",
        detection_timestamp=datetime.now(),
        lookback_window=50
    )
    profile: HardwareProfile = HWPROFILE
    budget = profile.compute_budget

    # تنظیم سقف‌ها بر اساس بودجه سخت‌افزاری
    if budget == ComputeBudget.LOW:
        max_layers, base_units, max_estimators, complexity_filter = 2, 32, 100, ["light"]
    elif budget == ComputeBudget.MEDIUM:
        max_layers, base_units, max_estimators, complexity_filter = 3, 64, 300, ["light", "medium"]
    elif budget == ComputeBudget.HIGH:
        max_layers, base_units, max_estimators, complexity_filter = 4, 128, 800, ["medium", "heavy"]
    else:  # INSTITUTIONAL
        max_layers, base_units, max_estimators, complexity_filter = 6, 256, 2000, ["heavy"]

    variant_counter = 1

    # 1. RNN Family (LSTM, GRU, BILSTM, RNN)
    rnn_types = [
        (ModelType.LSTM, "LSTM"),
        (ModelType.BILSTM, "BiLSTM"),
        (ModelType.GRU, "GRU"),
        (ModelType.RNN, "RNN")
    ]

    layer_configs = [
        ([base_units], "Light"),
        ([base_units * 2, base_units], "Medium"),
        ([base_units * 4, base_units * 2, base_units], "Heavy")
    ]

    mechanisms = [
        (False, False, "Standard"),
        (True, False, "Attention"),
        (False, True, "Residual"),
        (True, True, "Full")
    ]

    for m_type, m_str in rnn_types:
        for units, comp in layer_configs:
            if comp.lower() not in complexity_filter and comp.lower() != "medium":
                continue

            for use_attn, use_res, suffix in mechanisms:
                if budget == ComputeBudget.LOW and suffix in ["Residual", "Full"]:
                    continue

                # استفاده صحیح از _get_default_robustness با ارسال RegimeSignal
                robustness = _get_default_robustness(default_regime_signal, trending=0.8 if use_attn else 0.7)

                ArchitectureRegistry.register(ArchitectureConfig(
                    name=f"{m_str}-{comp}-{suffix}",
                    model_type=m_type,
                    variant=variant_counter,
                    units=units,
                    use_attention=use_attn,
                    use_residual=use_res,
                    batch_size=min(profile.max_batch_size, 64 if comp == "Light" else 32),
                    learning_rate=0.001 / (1 if comp == "Light" else 2),
                    complexity=comp.lower(),
                    regime_robustness=robustness
                ))
                variant_counter += 1

    # 2. Transformer & Attention
    if profile.can_run_transformers:
        d_models = [base_units, base_units * 2, base_units * 4]
        for d_model in d_models:
            heads = 4 if d_model < 100 else 8
            layers = 2 if d_model == base_units else 4

            robustness = _get_default_robustness(default_regime_signal, trending=0.85)

            ArchitectureRegistry.register(ArchitectureConfig(
                name=f"Transformer-{d_model}d",
                model_type=ModelType.TRANSFORMER,
                variant=variant_counter,
                d_model=d_model,
                num_heads=heads,
                num_layers=layers,
                ff_dim=d_model * 4,
                batch_size=profile.max_batch_size // 2,
                learning_rate=0.0001,
                complexity="heavy",
                regime_robustness=robustness
            ))
            variant_counter += 1
    else:
        # Fallback to Attention Lite
        robustness = _get_default_robustness(default_regime_signal)
        ArchitectureRegistry.register(ArchitectureConfig(
            name="Attention-Lite",
            model_type=ModelType.ATTENTION,
            variant=variant_counter,
            units=[base_units],
            use_attention=True,
            batch_size=profile.max_batch_size,
            complexity="light",
            regime_robustness=robustness
        ))
        variant_counter += 1

    # 3. CNN-LSTM & TCN
    for filters, comp in [([16, 32], "Lite"), ([32, 64, 128], "Standard"), ([64, 128, 256], "Deep")]:
        if budget == ComputeBudget.LOW and comp == "Deep":
            continue

        robustness = _get_default_robustness(default_regime_signal)

        ArchitectureRegistry.register(ArchitectureConfig(
            name=f"CNN-LSTM-{comp}",
            model_type=ModelType.CNN_LSTM,
            variant=variant_counter,
            units=[base_units * 2, base_units],
            cnn_filters=filters,
            cnn_kernel_sizes=[3, 3, 3],
            use_attention=(comp == "Deep"),
            batch_size=profile.max_batch_size,
            complexity=comp.lower(),
            regime_robustness=robustness
        ))
        variant_counter += 1

    robustness = _get_default_robustness(default_regime_signal)
    ArchitectureRegistry.register(ArchitectureConfig(
        name="TCN-Dynamic",
        model_type=ModelType.TCN,
        variant=variant_counter,
        units=[base_units * 2, base_units * 2],
        use_residual=True,
        batch_size=profile.max_batch_size,
        complexity="medium",
        regime_robustness=robustness
    ))
    variant_counter += 1

    # 4. SOTA Models (TabNet, N-BEATS, DeepAR, WaveNet, TFT, DeepFM, GP)
    sota_models = [
        (ModelType.TABNET, "TabNet", "heavy"),
        (ModelType.NBEATS, "N-BEATS", "heavy"),
        (ModelType.DEEPAR, "DeepAR", "medium"),
        (ModelType.WAVENET, "WaveNet", "medium"),
        (ModelType.TFT, "TFT", "heavy"),
        (ModelType.DEEPFM, "DeepFM", "light"),
        (ModelType.GAUSSIAN_PROCESS, "GaussianProcess", "light")
    ]

    for m_type, name, comp_req in sota_models:
        if budget == ComputeBudget.LOW and comp_req == "heavy":
            robustness = _get_default_robustness(default_regime_signal)
            ArchitectureRegistry.register(ArchitectureConfig(
                name=f"{name}-Lite",
                model_type=m_type,
                variant=variant_counter,
                units=[32],
                batch_size=32,
                complexity="light",
                description=f"{name} optimized for Low-End Hardware",
                regime_robustness=robustness
            ))
        else:
            robustness = _get_default_robustness(default_regime_signal)
            ArchitectureRegistry.register(ArchitectureConfig(
                name=name,
                model_type=m_type,
                variant=variant_counter,
                units=[base_units * 2],
                batch_size=profile.max_batch_size,
                complexity=comp_req,
                regime_robustness=robustness
            ))
        variant_counter += 1

    # 5. Tree Models (XGB, LGBM, CatBoost)
    for m_type, prefix in [(ModelType.XGBOOST, "XGB"), (ModelType.LIGHTGBM, "LGBM"), (ModelType.CATBOOST, "CatBoost")]:
        for n_est, depth, comp in [(100, 4, "Fast"), (300, 6, "Balanced"), (max_estimators, 8, "Deep")]:
            device = "cuda" if (profile.has_gpu and budget != ComputeBudget.LOW) else "cpu"
            robustness = _get_default_robustness(default_regime_signal)

            ArchitectureRegistry.register(ArchitectureConfig(
                name=f"{prefix}-{comp}",
                model_type=m_type,
                variant=variant_counter,
                n_estimators=n_est,
                max_depth=depth,
                device=device,
                learning_rate=0.1 if comp == "Fast" else 0.01,
                complexity=comp.lower(),
                regime_robustness=robustness
            ))
            variant_counter += 1

    # 6. Classical & Statistical Models (Linear, SVR, RF, KNN, Bayesian)
    classical_models = [
        (ModelType.LINEAR, "Linear", "light"),
        (ModelType.SVR, "SVR", "light"),
        (ModelType.RANDOM_FOREST, "RandomForest", "medium"),
        (ModelType.KNN, "KNN", "light"),
        (ModelType.BAYESIAN, "Bayesian", "light")
    ]

    for m_type, name, comp in classical_models:
        robustness = _get_default_robustness(default_regime_signal)
        ArchitectureRegistry.register(ArchitectureConfig(
            name=name,
            model_type=m_type,
            variant=variant_counter,
            complexity=comp,
            regime_robustness=robustness
        ))
        variant_counter += 1

    # 7. MLP
    robustness = _get_default_robustness(default_regime_signal)
    ArchitectureRegistry.register(
        ArchitectureConfig(
            name="MLP-Dynamic",
            model_type=ModelType.MLP,
            variant=variant_counter,
            units=[128, 64],
            complexity="medium",
            regime_robustness=robustness
        )
    )
    variant_counter += 1

    # 8. Advanced Strategies (Reinforcement, Ensemble, Hybrid)
    # این بخش برای تکمیل بودن تمام آیتم‌های ModelType اضافه شد
    advanced_models = [
        (ModelType.REINFORCEMENT, "RL-Agent", "heavy"),
        (ModelType.ENSEMBLE, "Super-Ensemble", "heavy"),
        (ModelType.HYBRID, "Hybrid-Alpha", "heavy")
    ]

    for m_type, name, comp in advanced_models:
        # فقط در بودجه‌های بالا اجازه ساخت می‌دهیم
        if budget in [ComputeBudget.LOW, ComputeBudget.MEDIUM]:
            continue

        robustness = _get_default_robustness(default_regime_signal)
        ArchitectureRegistry.register(ArchitectureConfig(
            name=name,
            model_type=m_type,
            variant=variant_counter,
            complexity=comp,
            description="Advanced strategy requiring high compute",
            regime_robustness=robustness
        ))
        variant_counter += 1

    logger.info(f"✅ Dynamic Architecture Generation Complete. Total Registered: {len(ArchitectureRegistry.get_all())}")


# اجرای اولیه
register_dynamic_architectures()
# =============================================================================
# III. DATA CHARACTERISTICS ANALYZER
# =============================================================================

@dataclass
class DataCharacteristics:
    """Comprehensive data characteristics for architecture selection"""
    n_samples: int
    n_features: int
    is_time_series: bool = True
    sequence_length: Optional[int] = None
    signal_to_noise_ratio: float = 1.0
    long_range_dependency: bool = False
    is_stationary: bool = True
    feature_sparsity: float = 0.0
    autocorrelation_lag1: float = 0.0
    hurst_exponent: float = 0.5
    nonlinearity_score: float = 0.5
    target_type: str = "regression"

    @property
    def sample_size_category(self) -> str:
        if self.n_samples < 1000:
            return "tiny"
        elif self.n_samples < 5000:
            return "small"
        elif self.n_samples < 20000:
            return "medium"
        elif self.n_samples < 100000:
            return "large"
        else:
            return "huge"

    @property
    def feature_dimensionality(self) -> str:
        ratio = self.n_samples / max(self.n_features, 1)
        if ratio < 10:
            return "high_dimensional"
        elif ratio < 50:
            return "medium_dimensional"
        else:
            return "low_dimensional"

    @property
    def is_noisy(self) -> bool:
        return self.signal_to_noise_ratio < 0.5

    @classmethod
    def from_arrays(cls, X_train: np.ndarray, y_train: np.ndarray) -> 'DataCharacteristics':
        """Extract characteristics from actual data"""
        n_samples = X_train.shape[0]
        n_features = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
        seq_len = X_train.shape[1] if X_train.ndim == 3 else 1

        # Approximate SNR from returns
        if y_train.ndim > 1:
            y_flat = y_train[:, 0] if y_train.shape[1] > 0 else y_train.flatten()
        else:
            y_flat = y_train.flatten()

        signal = np.mean(np.abs(y_flat))
        noise = np.std(y_flat)
        snr = signal / (noise + 1e-8)

        return cls(
            n_samples=n_samples,
            n_features=n_features,
            sequence_length=seq_len,
            signal_to_noise_ratio=snr,
            long_range_dependency=seq_len > 100,
        )




# =============================================================================
# V. WEIGHT ADAPTATION ENGINE
# =============================================================================

@dataclass
class UtilityWeights:
    """Dynamic weights for multi-objective utility function"""
    predictive_power: float = 0.25
    generalization_stability: float = 0.20
    regime_robustness: float = 0.15
    overfitting_risk: float = 0.10
    compute_cost: float = 0.10
    latency_cost: float = 0.08
    memory_pressure: float = 0.07
    instability_penalty: float = 0.05

    def normalize(self) -> 'UtilityWeights':
        total = sum([
            self.predictive_power, self.generalization_stability,
            self.regime_robustness, self.overfitting_risk,
            self.compute_cost, self.latency_cost,
            self.memory_pressure, self.instability_penalty
        ])
        if total > 0:
            self.predictive_power /= total
            self.generalization_stability /= total
            self.regime_robustness /= total
            self.overfitting_risk /= total
            self.compute_cost /= total
            self.latency_cost /= total
            self.memory_pressure /= total
            self.instability_penalty /= total
        return self


def _compute_adaptive_weights(
    regime_key: str,
    regime_confidence: float,
    hw: HardwareProfile,
    data: DataCharacteristics,
    attempt: int,
    is_real_time: bool = False
) -> UtilityWeights:
    """
    Compute adaptive weights based on context
    """
    w = UtilityWeights()

    # ===== REGIME-BASED ADAPTATION =====
    if regime_key in ["BULL_TRENDING", "BEAR_TRENDING", "trending"]:
        w.regime_robustness += 0.15
        w.predictive_power += 0.10

    elif regime_key in ["SIDEWAYS_RANGING", "mean_reverting"]:
        w.generalization_stability += 0.15
        w.overfitting_risk += 0.10

    elif regime_key in ["VOLATILITY_CLUSTER", "high_volatility"]:
        w.overfitting_risk += 0.15
        w.instability_penalty += 0.10
        w.generalization_stability += 0.10

    elif regime_key in ["CRISIS_FLASH", "structural_break"]:
        w.instability_penalty += 0.20
        w.compute_cost += 0.10
        w.overfitting_risk += 0.15

    # ===== HARDWARE-BASED ADAPTATION =====
    if hw.compute_budget == ComputeBudget.LOW:
        w.compute_cost += 0.20
        w.memory_pressure += 0.15
        w.latency_cost += 0.10
    elif hw.compute_budget == ComputeBudget.MEDIUM:
        w.compute_cost += 0.10
        w.memory_pressure += 0.05
    elif hw.compute_budget == ComputeBudget.HIGH:
        w.compute_cost -= 0.05
        w.memory_pressure -= 0.05
    elif hw.compute_budget == ComputeBudget.INSTITUTIONAL:
        w.compute_cost -= 0.10
        w.memory_pressure -= 0.10

    # ===== DATA-BASED ADAPTATION =====
    if data.sample_size_category == "tiny":
        w.overfitting_risk += 0.20
        w.compute_cost += 0.10
    elif data.sample_size_category == "small":
        w.overfitting_risk += 0.10
    elif data.sample_size_category == "huge":
        w.predictive_power += 0.10

    if data.is_noisy:
        w.overfitting_risk += 0.15
        w.generalization_stability += 0.10

    if data.feature_dimensionality == "high_dimensional":
        w.overfitting_risk += 0.15

    if data.long_range_dependency:
        w.regime_robustness += 0.10

    # ===== ATTEMPT-BASED ADAPTATION =====
    if attempt > 1:
        penalty = min((attempt - 1) * 0.05, 0.20)
        w.overfitting_risk = min(0.6, w.overfitting_risk + penalty)

    # ===== DEPLOYMENT ADAPTATION =====
    if is_real_time:
        w.latency_cost += 0.20
        w.compute_cost += 0.10

    # Ensure non-negative and normalize
    for attr in ['predictive_power', 'generalization_stability', 'regime_robustness',
                 'overfitting_risk', 'compute_cost', 'latency_cost',
                 'memory_pressure', 'instability_penalty']:
        setattr(w, attr, max(0.0, getattr(w, attr)))

    return w.normalize()




# =============================================================================
# VII. COST ESTIMATION ENGINE
# =============================================================================

@dataclass
class ArchitectureCostEstimate:
    """Detailed cost estimates for an architecture"""
    flops_score: float = 0.5
    memory_score: float = 0.5
    latency_score: float = 0.5
    training_time_score: float = 0.5
    exceeds_vram: bool = False
    exceeds_ram: bool = False
    cpu_incompatible: bool = False
    gpu_required: bool = False
    feasible: bool = True
    rejection_reason: str = ""


def _estimate_architecture_cost(
    cfg: ArchitectureConfig,
    n_samples: int,
    lookback: int,
    n_features: int,
    hw: HardwareProfile,
) -> ArchitectureCostEstimate:
    """
    Estimate resource consumption with hardware constraints
    """
    est = ArchitectureCostEstimate()

    # GPU requirement check
    if cfg.device == "cuda" and not hw.has_gpu:
        est.cpu_incompatible = True
        est.gpu_required = True
        est.feasible = False
        est.rejection_reason = "GPU_REQUIRED_BUT_UNAVAILABLE"
        return est

    # Memory estimation for DL models
    if cfg.model_type in [ModelType.LSTM, ModelType.BILSTM, ModelType.GRU,
                          ModelType.CNN_LSTM, ModelType.TRANSFORMER, ModelType.TCN]:
        # Parameter count estimation
        param_count = 0
        if cfg.units:
            for i, units in enumerate(cfg.units):
                input_dim = n_features if i == 0 else cfg.units[i-1]
                if cfg.model_type == ModelType.LSTM:
                    param_count += 4 * units * (input_dim + units + 1)
                elif cfg.model_type == ModelType.GRU:
                    param_count += 3 * units * (input_dim + units + 1)
                elif cfg.model_type == ModelType.BILSTM:
                    param_count += 8 * units * (input_dim + units + 1)
                else:
                    param_count += units * input_dim

        # Memory = params + activations
        param_bytes = param_count * 4 * 3  # FP32 + grad + momentum
        max_units = max(cfg.units) if cfg.units else 64
        batch = min(cfg.batch_size or 32, hw.max_batch_size)
        activation_bytes = batch * lookback * max_units * 4 * 2

        total_gb = (param_bytes + activation_bytes) / (1024**3)

        # Check VRAM
        if hw.has_gpu:
            est.exceeds_vram = total_gb > hw.gpu_vram_gb * 0.8
            if est.exceeds_vram:
                est.feasible = False
                est.rejection_reason = f"VRAM_EXCEEDED: needs {total_gb:.1f}GB > {hw.gpu_vram_gb:.1f}GB"

        # Check RAM
        est.exceeds_ram = total_gb > hw.ram_gb * 0.5
        if est.exceeds_ram and not hw.has_gpu:
            est.feasible = False
            est.rejection_reason = f"RAM_EXCEEDED: needs {total_gb:.1f}GB > {hw.ram_gb:.1f}GB"

        # Score calculations
        est.memory_score = min(1.0, total_gb / max(1.0, hw.gpu_vram_gb if hw.has_gpu else hw.ram_gb))

        # Training time
        steps = n_samples / (cfg.batch_size or 32)
        est.training_time_score = min(1.0, (param_count / 1e6) * steps / 1000)
        est.flops_score = min(1.0, param_count / 1e6)

    else:  # Tree-based models
        n_est = cfg.n_estimators or 100
        depth = cfg.max_depth or 6
        est.flops_score = min(0.9, n_est / 1000)
        est.memory_score = min(0.8, (n_est * depth) / 5000)
        est.training_time_score = min(0.9, (n_est * depth) / 5000)
        est.latency_score = min(0.8, n_est / 1000)

    # Sample size check
    if n_samples < cfg.min_samples_required:
        est.feasible = False
        est.rejection_reason = f"INSUFFICIENT_SAMPLES: need {cfg.min_samples_required}, have {n_samples}"

    return est


# =============================================================================
# VIII. ARCHITECTURE SCORING ENGINE
# =============================================================================

@dataclass
class ArchitectureScore:
    """Complete score for an architecture candidate"""
    cfg: ArchitectureConfig
    utility_score: float
    predictive_score: float
    stability_score: float
    regime_score: float
    overfit_penalty: float
    compute_penalty: float
    latency_penalty: float
    memory_penalty: float
    instability_penalty: float
    cost_estimate: ArchitectureCostEstimate
    is_pareto_optimal: bool = False
    rejection_reason: Optional[str] = None


def _score_architecture(
    cfg: ArchitectureConfig,
    regime_key: str,
    regime_confidence: float,
    cost_est: ArchitectureCostEstimate,
    weights: UtilityWeights,
    hw: HardwareProfile,
    data: DataCharacteristics,
    attempt: int,
) -> ArchitectureScore:
    """
    Compute multi-objective utility score
    """
    # Hard feasibility check
    if not cost_est.feasible:
        return ArchitectureScore(
            cfg=cfg,
            utility_score=0.0,
            predictive_score=0.0,
            stability_score=0.0,
            regime_score=0.0,
            overfit_penalty=1.0,
            compute_penalty=1.0,
            latency_penalty=1.0,
            memory_penalty=1.0,
            instability_penalty=1.0,
            cost_estimate=cost_est,
            rejection_reason=cost_est.rejection_reason
        )

    # 1. Predictive power score
    predictive_score = cfg.base_predictive_power
    if data.sample_size_category == "tiny":
        predictive_score *= 0.8
    elif data.sample_size_category == "huge":
        predictive_score *= 1.1
    predictive_score = min(1.0, predictive_score)

    # 2. Generalization stability
    stability_score = cfg.generalization_stability
    stability_score *= (1.0 - cfg.hyperparameter_sensitivity * 0.3)
    if data.feature_dimensionality == "high_dimensional":
        if cfg.model_type in [ModelType.XGBOOST, ModelType.LIGHTGBM, ModelType.RANDOM_FOREST]:
            stability_score *= 1.1
        else:
            stability_score *= 0.9
    stability_score = min(1.0, max(0.0, stability_score))

    # 3. Regime robustness
    regime_lower = regime_key.lower().replace("_", "")
    regime_score = cfg.regime_robustness.get(regime_lower, 0.5)

    # 4. Overfitting penalty
    overfit_penalty = 1.0 - cfg.generalization_stability
    if data.sample_size_category in ["tiny", "small"]:
        overfit_penalty *= 1.5
    if data.is_noisy:
        overfit_penalty *= 1.3
    overfit_penalty = min(1.0, overfit_penalty)

    # 5. Compute penalty
    compute_penalty = cost_est.training_time_score
    if hw.compute_budget == ComputeBudget.LOW:
        compute_penalty *= 1.5
    compute_penalty = min(1.0, compute_penalty)

    # 6. Latency penalty
    latency_penalty = cfg.inference_latency_ms / hw.inference_latency_constraint_ms
    latency_penalty = min(1.0, latency_penalty)

    # 7. Memory penalty
    memory_penalty = cost_est.memory_score

    # 8. Instability penalty
    instability_penalty = 1.0 - cfg.generalization_stability
    if regime_key in ["high_volatility", "structural_break", "regime_transition"]:
        instability_penalty *= 1.5
    instability_penalty = min(1.0, instability_penalty)

    # Final utility
    utility_score = (
        weights.predictive_power * predictive_score +
        weights.generalization_stability * stability_score +
        weights.regime_robustness * regime_score -
        weights.overfitting_risk * overfit_penalty -
        weights.compute_cost * compute_penalty -
        weights.latency_cost * latency_penalty -
        weights.memory_pressure * memory_penalty -
        weights.instability_penalty * instability_penalty
    )
    utility_score = max(0.0, utility_score)

    return ArchitectureScore(
        cfg=cfg,
        utility_score=utility_score,
        predictive_score=predictive_score,
        stability_score=stability_score,
        regime_score=regime_score,
        overfit_penalty=overfit_penalty,
        compute_penalty=compute_penalty,
        latency_penalty=latency_penalty,
        memory_penalty=memory_penalty,
        instability_penalty=instability_penalty,
        cost_estimate=cost_est,
    )

# =============================================================================
# XII. EXPORTS
# =============================================================================

__all__ = [
    "ComputeBudget",
    "HardwareProfile",
    "ModelType",
    "ArchitectureConfig",
    "ArchitectureRegistry",
    "UtilityWeights",
    "DataCharacteristics",
]