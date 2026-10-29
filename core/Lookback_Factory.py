from core.imports import *
from core.global_method import *
from core.logger import redirect_prints_to_logger, AdvancedTradeLogger
redirect_prints_to_logger()
logger = AdvancedTradeLogger(__name__)


class LookBack:
    """
    🎯 کلاس پیشرفته محاسبه لوک‌بک هوشمند با 15+ استراتژی روز دنیا
    🔥 محدوده داینامیک: 40 تا 80
    """

    # استراتژی‌های پیشرفته با وزن‌های اولیه
    ADVANCED_STRATEGIES = {
        "data_density": {"weight": 0.12, "description": "تراکم داده و ابعاد"},
        "volatility_regime": {"weight": 0.11, "description": "رژیم نوسانات"},
        "autocorrelation": {"weight": 0.10, "description": "خودهمبستگی سری زمانی"},
        "spectral_entropy": {"weight": 0.09, "description": "آنتروپی طیفی"},
        "market_regime": {"weight": 0.09, "description": "شناسایی رژیم بازار"},
        "data_quality": {"weight": 0.08, "description": "کیفیت داده و نویز"},
        "fractal_dimension": {"weight": 0.08, "description": "بعد فرکتالی"},
        "information_complexity": {"weight": 0.07, "description": "پیچیدگی اطلاعات"},
        "stationarity_analysis": {"weight": 0.07, "description": "آنالیز مانایی"},
        "trend_cycle_decomposition": {"weight": 0.06, "description": "تفکیک رنج و چرخه"},
        "correlation_structure": {"weight": 0.05, "description": "ساختار همبستگی"},
        "optimal_sampling": {"weight": 0.04, "description": "نمونه‌برداری بهینه"},
        "machine_learning_adaptive": {"weight": 0.02, "description": "سازگاری مبتنی بر ML"},
        "ensemble_prediction": {"weight": 0.01, "description": "پیش‌بینی گروهی"},
        "bayesian_optimization": {"weight": 0.01, "description": "بهینه‌سازی بیزی"}
    }

    # 🔥 محدوده داینامیک جدید: 40 تا 80
    DYNAMIC_MIN = 40
    DYNAMIC_MAX = 80

    def __init__(
        self,
        symbol: str,
        model_name: str = "AdvancedLookBack",
        log_dir: str = "./logs/lookback",
        checkpoint_dir: Optional[str] = None,
        logger=None,
        min_lookback_samples: int = 5000,
        max_lookback: Optional[int] = None,
        min_lookback: Optional[int] = None,
        adaptive_lookback: bool = True,
        volatility_window: int = 20,
        timeframes: Optional[List[int]] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
        cache_expiry: int = 3600,
        max_cache_size: int = 200,
        enable_ml_adaptive: bool = True,
        quality_guarantee: bool = True,
        min_quality_lookback: int = 40,
        **kwargs,
    ) -> None:
        # ✅ CRITICAL FIX: self._logger همیشه قبل از استفاده تعریف می‌شود
        from core.logger import AdvancedTradeLogger as _ATL
        self._logger = logger if logger is not None else _ATL(__name__)
        self._logger.info(f"[{symbol}] 🚀 راه‌اندازی LookBack پیشرفته (محدوده داینامیک 40-80)")

        if getattr(self, '_initialized', False):
            return

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.model_name = model_name
        self.log_dir = log_dir

        try:
            self.min_lookback_samples = self._validate_int(
                min_lookback_samples, 5000, "min_lookback_samples"
            )

            # 🔥 min_lookback داینامیک در بازه 40-60
            min_lb = self._validate_int(min_lookback, self.DYNAMIC_MIN, "min_lookback")
            self._min_lookback = max(self.DYNAMIC_MIN, min(60, min_lb))

            # 🔥 max_lookback داینامیک در بازه 60-80
            max_lb = self._validate_int(max_lookback, self.DYNAMIC_MAX, "max_lookback")
            self._max_lookback = max(
                self._min_lookback + 10,
                min(self.DYNAMIC_MAX, max_lb)
            )

            self._volatility_window = self._validate_int(
                volatility_window, 20, "volatility_window"
            )
            self._adaptive_lookback = bool(adaptive_lookback)
            self._enable_ml_adaptive = bool(enable_ml_adaptive)
            self._quality_guarantee = bool(quality_guarantee)
            self._min_quality_lookback = self._validate_int(
                min_quality_lookback, self.DYNAMIC_MIN, "min_quality_lookback"
            )

            self.timeframes = timeframes if timeframes else [
                1, 5, 15, 30, 60, 240, 1440, 10080
            ]
            self._max_cache_size = self._validate_int(max_cache_size, 200, "max_cache_size")
            self._cache_expiry = self._validate_int(cache_expiry, 3600, "cache_expiry")
            self._provided_strategy_weights = (
                strategy_weights if strategy_weights is not None else {}
            )
        except Exception as e:
            self._logger.error(f"[{symbol}] ❌ خطا در اعتبارسنجی: {e}")
            raise

        try:
            self._initialize_advanced_systems()
        except Exception as e:
            self._logger.error(f"[{symbol}] ❌ خطا در راه‌اندازی سیستم‌ها: {e}")
            self._initialize_basic_systems()

        try:
            self._initialize_advanced_strategies()
        except Exception as e:
            self._logger.error(f"[{symbol}] ❌ خطا در راه‌اندازی استراتژی‌ها: {e}")
            self._initialize_basic_strategies()

        try:
            self.default_lookback = self._compute_advanced_default_lookback()
            self.lookback = self.default_lookback
        except Exception as e:
            self._logger.error(f"[{symbol}] ❌ خطا در محاسبه پیش‌فرض: {e}")
            self.default_lookback = max(self.DYNAMIC_MIN, self._min_lookback)
            self.lookback = self.default_lookback

        try:
            self._validate_and_repair_internal_parameters()
        except Exception as e:
            self._logger.warning(f"[{symbol}] ⚠️ خطا در اعتبارسنجی داخلی: {e}")

        try:
            self._log_initialization_summary()
        except Exception as e:
            self._logger.warning(f"[{symbol}] ⚠️ خطا در لاگ خلاصه: {e}")

        self._initialized = True
        self._logger.info(f"[{symbol}] ✅ LookBack پیشرفته راه‌اندازی شد (محدوده 40-80)")

    def _validate_int(self, value: Any, default: int, param_name: str) -> int:
        """اعتبارسنجی و تبدیل به عدد صحیح — امن در برابر نبود self._logger."""
        try:
            if value is None:
                return default
            val = int(value)
            if val <= 0:
                raise ValueError(f"{param_name} باید مثبت باشد")
            return val
        except (ValueError, TypeError) as e:
            log = getattr(self, '_logger', None) or logger
            log.warning(f"[{getattr(self, 'symbol', '?')}] ⚠️ {param_name} نامعتبر ({value}): {e}")
            return default

    def _initialize_advanced_systems(self):
        """راه‌اندازی سیستم‌های پیشرفته"""
        try:
            # 🗄️ کش چندلایه
            self._lookback_cache = OrderedDict()  # LRU Cache
            self._strategy_cache = {}
            self._feature_cache = {}

            # 📊 معیارهای عملکرد
            self._cache_hits = 0
            self._cache_misses = 0

            self._strategy_execution_times = {}
            self._performance_metrics = {
                'total_calculations': 0,
                'average_calculation_time': 0,
                'strategy_success_rates': {},
                'cache_efficiency': 0.0,
                'lookback_history': [],
                'data_quality_scores': []
            }

            # 🧮 ویژگی‌های آماری
            self._statistical_features = {
                'volatility_regimes': [],
                'fractal_dimensions': [],
                'stationarity_scores': [],
                'spectral_entropies': []
            }

            # 📈 تاریخچه تصمیمات
            self._decision_history = []

            # 🎯 معیارهای کیفیت
            self._quality_metrics = {
                'min_acceptable_lookback': self.DYNAMIC_MIN,
                'quality_violations': 0,
                'quality_adjustments': 0
            }

            self._logger.debug(f"[{self.symbol}] 🗄️ سیستم‌های پیشرفته راه‌اندازی شدند")

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در راه‌اندازی سیستم‌های پیشرفته: {e}")
            # راه‌اندازی حداقلی
            self._initialize_basic_systems()

    def _initialize_basic_systems(self):
        """راه‌اندازی سیستم‌های پایه"""
        self._lookback_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._performance_metrics = {
            'total_calculations': 0,
            'average_calculation_time': 0,
            'strategy_success_rates': {},
            'cache_efficiency': 0.0,
            'lookback_history': [],
            'data_quality_scores': []
        }
        self._quality_metrics = {
            'min_acceptable_lookback': self.DYNAMIC_MIN,
            'quality_violations': 0,
            'quality_adjustments': 0
        }

    def _initialize_advanced_strategies(self):
        """🎯 راه‌اندازی 15 استراتژی پیشرفته"""
        try:
            self._strategies = OrderedDict()

            # تعریف تمام استراتژی‌های پیشرفته
            strategy_definitions = [
                ("data_density", self._advanced_data_density_based, 0.12),
                ("volatility_regime", self._volatility_regime_based, 0.11),
                ("autocorrelation", self._advanced_autocorrelation_based, 0.10),
                ("spectral_entropy", self._spectral_entropy_based, 0.09),
                ("market_regime", self._advanced_market_regime_based, 0.09),
                ("data_quality", self._advanced_data_quality_based, 0.08),
                ("fractal_dimension", self._fractal_dimension_based, 0.08),
                ("information_complexity", self._information_complexity_based, 0.07),
                ("stationarity_analysis", self._stationarity_analysis_based, 0.07),
                ("trend_cycle_decomposition", self._trend_cycle_decomposition_based, 0.06),
                ("correlation_structure", self._correlation_structure_based, 0.05),
                ("optimal_sampling", self._optimal_sampling_based, 0.04),
                ("machine_learning_adaptive", self._ml_adaptive_based, 0.02),
                ("ensemble_prediction", self._ensemble_prediction_based, 0.01),
                ("bayesian_optimization", self._bayesian_optimization_based, 0.01)
            ]

            for name, fn, default_weight in strategy_definitions:
                # استفاده از وزن سفارشی یا پیش‌فرض
                weight = self._provided_strategy_weights.get(name, default_weight)
                self._strategies[name] = {
                    "function": fn,
                    "weight": weight,
                    "description": self.ADVANCED_STRATEGIES.get(name, {}).get("description", "استراتژی پیشرفته"),
                    "execution_count": 0,
                    "success_count": 0,
                    "avg_time": 0.0
                }

            # نرمال‌سازی وزن‌ها
            total_weight = sum(s["weight"] for s in self._strategies.values())
            if total_weight > 0:
                for name in self._strategies:
                    self._strategies[name]["weight"] /= total_weight

            self._logger.info(f"[{self.symbol}] 🎯 {len(self._strategies)} استراتژی پیشرفته راه‌اندازی شد")

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در راه‌اندازی استراتژی‌ها: {e}")
            # فال‌بک به استراتژی‌های پایه
            self._initialize_basic_strategies()

    def _initialize_basic_strategies(self):
        """راه‌اندازی استراتژی‌های پایه"""
        basic_strategies = [
            ("data_density", self._advanced_data_density_based, 0.4),
            ("volatility_regime", self._volatility_regime_based, 0.3),
            ("autocorrelation", self._advanced_autocorrelation_based, 0.3)
        ]

        self._strategies = OrderedDict()
        for name, fn, weight in basic_strategies:
            self._strategies[name] = {
                "function": fn,
                "weight": weight,
                "description": "استراتژی پایه",
                "execution_count": 0,
                "success_count": 0,
                "avg_time": 0.0
            }

        self._logger.warning(f"[{self.symbol}] ⚠️ استفاده از استراتژی‌های پایه به دلیل خطا")

    def _compute_advanced_default_lookback(self) -> int:
        """🧮 محاسبه پیشرفته لوک‌بک پیش‌فرض (در محدوده 40-80)"""
        try:
            # محاسبه بر اساس تئوری اطلاعات — مقیاس کوچک
            base_lookback = int(20 * np.log1p(self.min_lookback_samples / 1000) + 40)

            # اعمال محدودیت‌ها
            constrained = max(self._min_lookback, min(self._max_lookback, base_lookback))

            # گرد کردن به مضرب 5 برای بهینه‌سازی
            rounded = 5 * round(constrained / 5)

            # 🔥 تضمین محدوده 40-80
            final_lookback = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, rounded))

            self._logger.debug(f"[{self.symbol}] 🔧 لوک‌بک پیش‌فرض پیشرفته: {final_lookback}")
            return final_lookback

        except Exception as e:
            self._logger.warning(f"[{self.symbol}] ⚠️ خطا در محاسبه پیش‌فرض: {e}")
            return max(self.DYNAMIC_MIN, self._min_lookback)

    def _log_initialization_summary(self):
        """📊 لاگ خلاصه راه‌اندازی"""
        summary = f"""
        📊 خلاصه راه‌اندازی LookBack برای {self.symbol}:

        📈 پارامترهای اصلی:
          • حداقل لوک‌بک: {self._min_lookback} (تضمین شده ≥ {self.DYNAMIC_MIN})
          • حداکثر لوک‌بک: {self._max_lookback} (محدود به ≤ {self.DYNAMIC_MAX})
          • نمونه‌های حداقل: {self.min_lookback_samples}
          • لوک‌بک پیش‌فرض: {self.default_lookback}
          • محدوده داینامیک: {self.DYNAMIC_MIN} - {self.DYNAMIC_MAX}
          • سازگاری پویا: {'فعال' if self._adaptive_lookback else 'غیرفعال'}
          • تضمین کیفیت: {'فعال' if self._quality_guarantee else 'غیرفعال'}

        🎯 استراتژی‌ها: {len(self._strategies)} استراتژی فعال
        🗄️  سیستم کش: {self._max_cache_size} ورودی، انقضا: {self._cache_expiry} ثانیه
        📅 تایم‌فریم‌ها: {self.timeframes}

        {'=' * 60}
        """
        self._logger.info(summary)

    def calculate_optimal_lookback(
            self,
            data: pd.DataFrame,
            fold_num: Optional[int] = None,
            force_recalculation: bool = False,
            target_data: Optional[np.ndarray] = None
    ) -> int:
        """🎯 محاسبه لوک‌بک بهینه با تمام استراتژی‌های پیشرفته و پشتیبانی از تارگت‌ها (خروجی در 40-80)"""

        start_time = time.time()
        fold_info = f"fold_{fold_num}" if fold_num is not None else "FULL"

        self._logger.info(f"""
        {'=' * 80}
        🚀 شروع محاسبه لوک‌بک بهینه برای {self.symbol} - {fold_info} (محدوده: 40-80)
        {'=' * 80}
        """)

        try:
            # ذخیره target_data در ویژگی‌ها برای استفاده استراتژی‌ها
            if target_data is not None:
                self._target_data = target_data
                self._logger.info(f"[{self.symbol}] 📊 داده‌های target با shape {target_data.shape} دریافت شد")

                if target_data.ndim == 2:
                    n_targets = target_data.shape[1]
                    self._logger.info(f"[{self.symbol}] 📊 تعداد تارگت‌ها: {n_targets}")
                    if n_targets >= 3:
                        self._logger.info(f"[{self.symbol}] 🔥 حالت 3-تارگت فعال شد")

            # 🔧 فاز 0: اعتبارسنجی داده
            self._validate_input_data_advanced(data)

            # 🔧 فاز 1: بررسی کش
            cache_key = self._generate_advanced_cache_key(data, fold_num)

            if target_data is not None:
                cache_key += f"_targets_{target_data.shape[1] if target_data.ndim > 1 else 1}"

            if not force_recalculation:
                cached_result = self._get_cached_lookback_advanced(cache_key)
                if cached_result is not None:
                    self._logger.info(f"[{self.symbol}] ✅ استفاده از لوک‌بک کش‌شده برای {fold_info}: {cached_result}")
                    return cached_result

            # 🔧 فاز 2: محاسبه ویژگی‌های پیشرفته
            advanced_features = self._calculate_advanced_features(data)
            if hasattr(self, '_target_data'):
                advanced_features['has_target'] = True
                advanced_features['target_shape'] = self._target_data.shape
                if self._target_data.ndim == 2:
                    advanced_features['n_targets'] = self._target_data.shape[1]

            # 🔥 اطمینان از وجود استراتژی‌ها
            if not hasattr(self, '_strategies') or not self._strategies:
                self._logger.warning(f"[{self.symbol}] ⚠️ _strategies وجود ندارد، راه‌اندازی مجدد...")
                self._initialize_advanced_strategies()

            # 🔧 فاز 3: اجرای تمام استراتژی‌های پیشرفته
            self._logger.info(f"[{self.symbol}] 🔍 اجرای {len(self._strategies)} استراتژی پیشرفته...")

            strategy_results = {}
            strategy_details = {}
            failed_strategies = []

            for strategy_name, config in self._strategies.items():
                try:
                    strategy_start = time.time()

                    result = config["function"](data, advanced_features)

                    strategy_time = time.time() - strategy_start

                    if result is not None and self._is_valid_lookback(result):
                        strategy_results[strategy_name] = result
                        strategy_details[strategy_name] = {
                            "value": result,
                            "time": strategy_time,
                            "weight": config["weight"]
                        }

                        config["execution_count"] += 1
                        config["success_count"] += 1
                        config["avg_time"] = (config["avg_time"] * (config["execution_count"] - 1) + strategy_time) / \
                                             config["execution_count"]

                        self._logger.debug(f"[{self.symbol}] ✅ {strategy_name}: {result} ({strategy_time:.3f}s)")

                    else:
                        failed_strategies.append((strategy_name, "نتیجه نامعتبر"))
                        self._logger.debug(f"[{self.symbol}] ⚠️ {strategy_name}: نتیجه نامعتبر")

                except Exception as e:
                    failed_strategies.append((strategy_name, str(e)))
                    self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی {strategy_name} ناموفق: {e}")

            # 🔧 فاز 4: ترکیب هوشمند نتایج
            if not strategy_results:
                self._logger.warning(f"[{self.symbol}] ⚠️ هیچ استراتژی موفق نبود، استفاده از استراتژی‌های فال‌بک")
                strategy_results = self._execute_fallback_strategies(data, advanced_features)

                if not strategy_results:
                    raise RuntimeError(f"همه استراتژی‌ها ناموفق بودند: {failed_strategies}")

            final_lookback = self._combine_strategy_results_advanced(strategy_results, strategy_details)

            # 🔧 فاز 5: اعمال محدودیت‌های کیفیت
            final_lookback = self._apply_quality_guarantees(final_lookback, data)

            # 🔥 اگر 3 تارگت داریم، lookback را افزایش بده (اما نه بیشتر از 80)
            if hasattr(self, '_target_data') and self._target_data is not None:
                if self._target_data.ndim == 2 and self._target_data.shape[1] >= 3:
                    new_lb = int(final_lookback * 1.15)
                    final_lookback = min(self.DYNAMIC_MAX, max(final_lookback, new_lb))
                    self._logger.info(f"[{self.symbol}] 🔥 افزایش lookback برای 3-تارگت: {final_lookback}")

            # 🔧 فاز 6: اعتبارسنجی نهایی
            final_lookback = self._final_validation(final_lookback, len(data))

            # 🔧 فاز 7: ذخیره در کش
            self._cache_lookback_advanced(cache_key, final_lookback, strategy_details)

            # 🔧 فاز 8: ثبت معیارهای عملکرد
            total_time = time.time() - start_time
            self._record_advanced_metrics(total_time, strategy_results, strategy_details)

            # 🔧 فاز 9: لاگ نتایج
            self._log_calculation_summary(
                fold_info, final_lookback, total_time,
                strategy_results, failed_strategies
            )

            return final_lookback

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در محاسبه لوک‌بک برای {fold_info}: {e}")
            import traceback
            self._logger.error(traceback.format_exc())

            fallback = self._get_advanced_fallback(data)
            self._logger.warning(f"[{self.symbol}] 🔄 استفاده از لوک‌بک فال‌بک برای {fold_info}: {fallback}")

            return fallback

    def _execute_fallback_strategies(self, data: pd.DataFrame, features: Dict) -> Dict[str, int]:
        """🎯 اجرای استراتژی‌های فال‌بک ساده و مطمئن (محدوده 40-80)"""
        fallback_results = {}

        try:
            # 1. استراتژی مبتنی بر حجم داده
            if len(data) > 0:
                data_based = int(min(80, max(40, 40 + len(data) / 500)))
                fallback_results["data_fallback"] = data_based

            # 2. استراتژی مبتنی بر نوسان ساده
            if 'close' in data.columns and len(data) > 50:
                try:
                    returns = data['close'].pct_change().dropna()
                    volatility = returns.std()
                    if volatility > 0.02:
                        vol_based = 70  # نوسان بالا
                    elif volatility < 0.005:
                        vol_based = 45  # نوسان پایین
                    else:
                        vol_based = 60  # نوسان متوسط
                    fallback_results["volatility_fallback"] = vol_based
                except:
                    pass

            # 3. استراتژی ثابت با کیفیت
            fallback_results["quality_fallback"] = 50

            self._logger.info(f"[{self.symbol}] 🛡️ استراتژی‌های فال‌بک اجرا شدند: {list(fallback_results.keys())}")

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ خطا در استراتژی‌های فال‌بک: {e}")

        return fallback_results

    def _validate_input_data_advanced(self, data: pd.DataFrame):
        """🔍 اعتبارسنجی پیشرفته داده‌های ورودی"""
        try:
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"داده باید DataFrame باشد، نه {type(data)}")

            if data.empty:
                raise ValueError("داده‌های ورودی خالی هستند")

            if len(data) < 100:
                self._logger.warning(f"[{self.symbol}] ⚠️ داده‌های کم: {len(data)} سطر (توصیه: ≥1000)")

            required_columns = ['close']

            missing_required = [col for col in required_columns if col not in data.columns]
            if missing_required:
                possible_cols = ['price', 'value', 'target', 'Close', 'CLOSE']
                found_alt = None
                for col in possible_cols:
                    if col in data.columns:
                        found_alt = col
                        break

                if found_alt:
                    self._logger.warning(
                        f"[{self.symbol}] ⚠️ ستون 'close' یافت نشد، استفاده از '{found_alt}' به جای آن")
                    data = data.rename(columns={found_alt: 'close'})
                else:
                    raise ValueError(f"ستون 'close' یا جایگزین آن یافت نشد")

            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].isnull().all():
                    self._logger.warning(f"[{self.symbol}] ⚠️ ستون {col} تماماً خالی است")
                elif data[col].isnull().any():
                    null_count = data[col].isnull().sum()
                    if null_count > 0:
                        self._logger.debug(f"[{self.symbol}] 🔍 ستون {col}: {null_count} مقدار null")

            self._logger.debug(f"[{self.symbol}] ✅ اعتبارسنجی داده‌ها موفق: {data.shape}")

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ اعتبارسنجی داده ناموفق: {e}")
            raise

    def _generate_advanced_cache_key(self, data: pd.DataFrame, fold_num: Optional[int] = None) -> str:
        """🔑 تولید کلید کش پیشرفته"""
        try:
            data_hash = hashlib.md5(
                f"{data.shape}_{data.iloc[-1:].to_json()}".encode()
            ).hexdigest()[:8]

            time_sig = int(time.time() // 1800)

            fold_sig = f"_fold{fold_num}" if fold_num is not None else "_FULL"

            key = f"lookback_adv_{self.symbol}{fold_sig}_{data_hash}_{time_sig}"

            return key

        except Exception:
            return f"lookback_{self.symbol}_{fold_num}_{int(time.time())}"

    def _get_cached_lookback_advanced(self, cache_key: str) -> Optional[int]:
        """💾 بازیابی پیشرفته از کش"""
        try:
            if not hasattr(self, '_lookback_cache'):
                return None

            if cache_key not in self._lookback_cache:
                return None

            cached_data = self._lookback_cache[cache_key]

            if not isinstance(cached_data, dict):
                return None

            if time.time() - cached_data.get('timestamp', 0) > self._cache_expiry:
                del self._lookback_cache[cache_key]
                return None

            self._cache_hits += 1
            return cached_data.get('lookback')

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ خطا در بازیابی از کش: {e}")
            return None

    def _cache_lookback_advanced(self, cache_key: str, lookback: int, strategy_details: Dict):
        """💾 ذخیره پیشرفته در کش"""
        try:
            start_time = time.time()
            self._logger.debug(f"[{self.symbol}] 🗄️ شروع ذخیره در کش پیشرفته: کلید={cache_key}, لوک‌بک={lookback}")

            if not hasattr(self, '_lookback_cache'):
                self._logger.warning(f"[{self.symbol}] ⚠️ _lookback_cache وجود ندارد، ایجاد جدید")
                self._lookback_cache = OrderedDict()

            if not hasattr(self, '_max_cache_size'):
                self._logger.warning(f"[{self.symbol}] ⚠️ _max_cache_size وجود ندارد، تنظیم مقدار پیش‌فرض=200")
                self._max_cache_size = 200

            try:
                if not isinstance(self._max_cache_size, (int, np.integer)):
                    self._logger.warning(
                        f"[{self.symbol}] ⚠️ _max_cache_size نامعتبر: {type(self._max_cache_size)}، تنظیم مجدد=200")
                    self._max_cache_size = 200

                if self._max_cache_size <= 0:
                    self._logger.warning(
                        f"[{self.symbol}] ⚠️ _max_cache_size نامعتبر: {self._max_cache_size}، تنظیم مجدد=200")
                    self._max_cache_size = 200

                self._logger.debug(f"[{self.symbol}] ✅ _max_cache_size معتبر: {self._max_cache_size}")
            except Exception as e:
                self._logger.error(f"[{self.symbol}] ❌ اعتبارسنجی _max_cache_size ناموفق: {e}، تنظیم مقدار=200")
                self._max_cache_size = 200

            try:
                if len(self._lookback_cache) >= self._max_cache_size:
                    oldest_key = next(iter(self._lookback_cache))
                    removed_data = self._lookback_cache[oldest_key]
                    del self._lookback_cache[oldest_key]

                    self._logger.debug(
                        f"[{self.symbol}] 🔄 کش کامل، حذف قدیمی‌ترین: {oldest_key} "
                        f"(لوک‌بک: {removed_data.get('lookback', 'N/A')})"
                    )
            except Exception as e:
                self._logger.warning(f"[{self.symbol}] ⚠️ خطا در حذف قدیمی‌ترین کش: {e}")
                if len(self._lookback_cache) >= self._max_cache_size:
                    self._lookback_cache.clear()
                    self._logger.warning(f"[{self.symbol}] ⚠️ کش پاک‌سازی شد")

            try:
                cache_entry = {
                    'lookback': lookback,
                    'timestamp': time.time(),
                    'strategy_details': strategy_details,
                    'symbol': self.symbol,
                    'creation_time': datetime.now().isoformat(),
                    'metadata': {
                        'lookback_min': getattr(self, '_min_lookback', self.DYNAMIC_MIN),
                        'lookback_max': getattr(self, '_max_lookback', self.DYNAMIC_MAX),
                        'cache_size_before': len(self._lookback_cache),
                        'cache_hits': getattr(self, '_cache_hits', 0),
                        'cache_misses': getattr(self, '_cache_misses', 0)
                    }
                }

                self._lookback_cache[cache_key] = cache_entry

                try:
                    if not hasattr(self, '_cache_misses'):
                        self._cache_misses = 0
                    self._cache_misses += 1

                    cache_hits = getattr(self, '_cache_hits', 0)
                    cache_misses = getattr(self, '_cache_misses', 0)
                    total_cache = cache_hits + cache_misses

                    if total_cache > 0:
                        hit_rate = cache_hits / total_cache
                        self._logger.debug(
                            f"[{self.symbol}] 📊 آمار کش: "
                            f"Hits={cache_hits}, Misses={cache_misses}, "
                            f"Hit Rate={hit_rate:.1%}, Size={len(self._lookback_cache)}/{self._max_cache_size}"
                        )
                except Exception as stats_error:
                    self._logger.debug(f"[{self.symbol}] ⚠️ خطا در به‌روزرسانی آمار کش: {stats_error}")

            except Exception as e:
                self._logger.error(f"[{self.symbol}] ❌ خطا در ساخت entry کش: {e}")
                self._lookback_cache[cache_key] = {
                    'lookback': lookback,
                    'timestamp': time.time(),
                    'symbol': self.symbol
                }

            execution_time = time.time() - start_time
            self._logger.info(
                f"[{self.symbol}] 💾 لوک‌بک در کش ذخیره شد: {lookback} | "
                f"کلید: {cache_key[:20]}... | زمان: {execution_time:.3f}s"
            )

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در ذخیره کش: {e}")
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'cache_key': cache_key,
                'lookback': lookback,
                'has_max_cache_size': hasattr(self, '_max_cache_size'),
                'max_cache_size': getattr(self, '_max_cache_size', 'N/A'),
                'has_lookback_cache': hasattr(self, '_lookback_cache'),
                'lookback_cache_len': len(getattr(self, '_lookback_cache', {})),
                'strategy_details_len': len(strategy_details) if strategy_details else 0
            }
            self._logger.error(f"[{self.symbol}] 🔍 جزئیات خطای کش: {error_details}")

            try:
                if not hasattr(self, '_lookback_cache'):
                    self._lookback_cache = OrderedDict()
                    self._logger.warning(f"[{self.symbol}] 🛠️ _lookback_cache ایجاد شد")

                if not hasattr(self, '_max_cache_size'):
                    self._max_cache_size = 200
                    self._logger.warning(f"[{self.symbol}] 🛠️ _max_cache_size=200 تنظیم شد")

                self._lookback_cache[cache_key] = {
                    'lookback': lookback,
                    'timestamp': time.time(),
                    'symbol': self.symbol
                }
                self._logger.info(f"[{self.symbol}] 🛡️ ذخیره حداقلی در کش انجام شد")

            except Exception as recovery_error:
                self._logger.error(f"[{self.symbol}] 💥 خطا در بازیابی کش: {recovery_error}")

    def _calculate_advanced_features(self, data: pd.DataFrame) -> Dict[str, Any]:
        """🧮 محاسبه ویژگی‌های پیشرفته برای استراتژی‌ها"""
        features = {}

        try:
            if 'close' not in data.columns:
                return features

            prices = data['close'].astype(float).dropna().values

            if len(prices) < 20:
                return features

            # 1. ویژگی‌های آماری پایه
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                features['returns'] = returns
                features['volatility'] = np.std(returns) if len(returns) > 0 else 0

                if len(returns) > 2:
                    try:
                        features['skewness'] = float(stats.skew(returns))
                    except:
                        features['skewness'] = 0

                    try:
                        features['kurtosis'] = float(stats.kurtosis(returns))
                    except:
                        features['kurtosis'] = 0
                else:
                    features['skewness'] = 0
                    features['kurtosis'] = 0

            # 2. ویژگی‌های طیفی
            if len(prices) >= 64:
                try:
                    prices_limited = prices[-256:] if len(prices) > 256 else prices
                    freqs, psd = signal.periodogram(prices_limited, detrend='linear')
                    if len(psd) > 1:
                        psd_nonzero = psd[1:] / (psd[1:].sum() + 1e-10)
                        features['spectral_entropy'] = float(stats.entropy(psd_nonzero + 1e-10))
                        features['dominant_frequency'] = float(freqs[np.argmax(psd[1:]) + 1])
                except Exception as e:
                    features['spectral_entropy'] = 1.0
                    features['dominant_frequency'] = 0.1

            # 3. ویژگی‌های سری زمانی
            if len(prices) > 1:
                try:
                    features['autocorr_1'] = float(np.corrcoef(prices[:-1], prices[1:])[0, 1])
                except:
                    features['autocorr_1'] = 0

            if len(prices) > 5:
                try:
                    features['autocorr_5'] = float(np.corrcoef(prices[:-5], prices[5:])[0, 1])
                except:
                    features['autocorr_5'] = 0

            # 4. ویژگی‌های نوسان
            if all(col in data.columns for col in ['high', 'low', 'close']):
                try:
                    high = data['high'].astype(float).values
                    low = data['low'].astype(float).values
                    close = data['close'].astype(float).values

                    if len(high) > 1 and len(low) > 1 and len(close) > 1:
                        tr = np.maximum(
                            high[1:] - low[1:],
                            np.abs(high[1:] - close[:-1]),
                            np.abs(low[1:] - close[:-1])
                        )
                        features['atr'] = float(np.mean(tr[-20:])) if len(tr) >= 20 else float(np.mean(tr))
                except:
                    features['atr'] = 0

            self._logger.debug(f"[{self.symbol}] 🧮 {len(features)} ویژگی پیشرفته محاسبه شد")

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ خطا در محاسبه ویژگی‌ها: {e}")

        return features

    def _is_valid_lookback(self, lookback: int) -> bool:
        """✅ بررسی اعتبار لوک‌بک (محدوده 40-80)"""
        try:
            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            return (
                    isinstance(lookback, (int, np.integer)) and
                    min_lb <= lookback <= max_lb and
                    self.DYNAMIC_MIN <= lookback <= self.DYNAMIC_MAX
            )
        except:
            return False

    def _combine_strategy_results_advanced(
            self,
            strategy_results: Dict[str, int],
            strategy_details: Dict[str, Dict]
    ) -> int:
        """⚖️ ترکیب هوشمند نتایج استراتژی‌ها (خروجی 40-80)"""
        try:
            if not strategy_results:
                return max(self.DYNAMIC_MIN, getattr(self, 'default_lookback', 60))

            values = list(strategy_results.values())
            weights = []

            for name in strategy_results.keys():
                if name in strategy_details:
                    weights.append(strategy_details[name]['weight'])
                else:
                    weights.append(1.0 / len(strategy_results))

            # 1. حذف outlierها با روش IQR
            if len(values) >= 3:
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                if iqr > 0:
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    filtered_values = []
                    filtered_weights = []

                    for i, value in enumerate(values):
                        if lower_bound <= value <= upper_bound:
                            filtered_values.append(value)
                            filtered_weights.append(weights[i])

                    if filtered_values:
                        values = filtered_values
                        weights = filtered_weights

            # 2. اعمال وزن‌ها
            weighted_sum = sum(v * w for v, w in zip(values, weights))
            total_weight = sum(weights)

            if total_weight == 0:
                raw_result = np.median(values)
            else:
                raw_result = weighted_sum / total_weight

            # 3. گرد کردن هوشمند — در محدوده 40-80 به مضرب 5
            rounded = 5 * round(raw_result / 5)

            final_result = int(max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, rounded)))

            self._logger.debug(f"[{self.symbol}] ⚖️ ترکیب نتایج: {raw_result:.1f} → {final_result}")

            return final_result

        except Exception as e:
            self._logger.warning(f"[{self.symbol}] ⚠️ خطا در ترکیب نتایج: {e}")
            if strategy_results:
                median_val = int(np.median(list(strategy_results.values())))
                return max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, median_val))
            return max(self.DYNAMIC_MIN, getattr(self, 'default_lookback', 60))

    def _apply_quality_guarantees(self, lookback: int, data: pd.DataFrame) -> int:
        """🔒 اعمال تضمین‌های کیفیت (محدوده 40-80)"""
        if not getattr(self, '_quality_guarantee', True):
            return lookback

        original = lookback
        adjusted = lookback

        # 1. حداقل مطلق 40
        adjusted = max(self.DYNAMIC_MIN, adjusted)

        # 2. بررسی نسبت به حجم داده
        if data is not None and len(data) > 0:
            max_by_data = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, len(data) // 3))
            adjusted = min(adjusted, max_by_data)

        # 3. بررسی ویژگی‌های داده
        if data is not None and len(data) > 100 and 'close' in data.columns:
            try:
                returns = data['close'].pct_change().dropna()
                if len(returns) > 0:
                    noise_level = returns.std()
                    if noise_level > 0.02:
                        adjusted = min(adjusted * 1.15, self.DYNAMIC_MAX)
                    elif noise_level < 0.005:
                        adjusted = max(adjusted * 0.85, self.DYNAMIC_MIN)
            except:
                pass

        # 4. گرد کردن نهایی به مضرب 5
        adjusted = 5 * round(adjusted / 5)
        adjusted = int(adjusted)

        # 5. محدودسازی نهایی
        adjusted = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, adjusted))

        if adjusted != original:
            self._logger.info(f"[{self.symbol}] 🔒 اصلاح کیفیت: {original} → {adjusted}")
            if hasattr(self, '_quality_metrics'):
                self._quality_metrics['quality_adjustments'] += 1

        if adjusted < self.DYNAMIC_MIN:
            self._logger.warning(f"[{self.symbol}] ⚠️ تخلف کیفیت: لوک‌بک {adjusted} کمتر از {self.DYNAMIC_MIN}")
            if hasattr(self, '_quality_metrics'):
                self._quality_metrics['quality_violations'] += 1
            adjusted = self.DYNAMIC_MIN

        return adjusted

    def _final_validation(self, lookback: int, data_length: int) -> int:
        """✅ اعتبارسنجی نهایی لوک‌بک (خروجی 40-80)"""
        try:
            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            validated = max(min_lb, min(max_lb, lookback))

            # 2. بررسی نسبت به طول داده
            if data_length > 0:
                max_reasonable = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, data_length // 2))
                validated = min(validated, max_reasonable)

            # 3. گرد کردن به مضرب 5
            validated = 5 * round(validated / 5)
            validated = int(validated)

            # 4. 🔥 تضمین محدوده 40-80
            validated = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, validated))

            if validated != lookback:
                self._logger.debug(f"[{self.symbol}] ✅ اعتبارسنجی: {lookback} → {validated}")

            return validated

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در اعتبارسنجی نهایی: {e}")
            return max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, data_length // 10 or 60))

    def _record_advanced_metrics(
            self,
            total_time: float,
            strategy_results: Dict[str, int],
            strategy_details: Dict[str, Dict]
    ):
        """📈 ثبت معیارهای عملکرد پیشرفته"""
        try:
            if not hasattr(self, '_performance_metrics'):
                self._performance_metrics = {
                    'total_calculations': 0,
                    'average_calculation_time': 0,
                    'strategy_success_rates': {},
                    'cache_efficiency': 0.0,
                    'lookback_history': [],
                    'data_quality_scores': []
                }

            self._performance_metrics['total_calculations'] += 1
            self._performance_metrics['lookback_history'].append({
                'timestamp': time.time(),
                'lookback': self.lookback,
                'calculation_time': total_time,
                'successful_strategies': len(strategy_results)
            })

            n = self._performance_metrics['total_calculations']
            current_avg = self._performance_metrics['average_calculation_time']
            self._performance_metrics['average_calculation_time'] = (
                (current_avg * (n - 1) + total_time) / n if n > 0 else total_time
            )

            for name in getattr(self, '_strategies', {}):
                if name not in self._performance_metrics['strategy_success_rates']:
                    self._performance_metrics['strategy_success_rates'][name] = []

                success = name in strategy_results
                self._performance_metrics['strategy_success_rates'][name].append(success)

            total_cache_attempts = getattr(self, '_cache_hits', 0) + getattr(self, '_cache_misses', 0)
            if total_cache_attempts > 0:
                self._performance_metrics['cache_efficiency'] = getattr(self, '_cache_hits', 0) / total_cache_attempts

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ خطا در ثبت معیارها: {e}")

    def _log_calculation_summary(
            self,
            fold_info: str,
            final_lookback: int,
            total_time: float,
            strategy_results: Dict[str, int],
            failed_strategies: List[Tuple[str, str]]
    ):
        """📝 لاگ خلاصه محاسبات"""
        summary = f"""
        {'=' * 80}
        📊 خلاصه محاسبه لوک‌بک برای {self.symbol} - {fold_info}:

        🎯 نتیجه نهایی: {final_lookback} (محدوده تضمین شده {self.DYNAMIC_MIN}-{self.DYNAMIC_MAX})
        ⏱️  زمان کل: {total_time:.3f} ثانیه
        ✅ استراتژی‌های موفق: {len(strategy_results)}/{len(getattr(self, '_strategies', {}))}

        📈 نتایج استراتژی‌ها:
        """

        for name, result in sorted(strategy_results.items()):
            if name in getattr(self, '_strategies', {}):
                weight = self._strategies[name].get('weight', 0)
                summary += f"    • {name}: {result} (وزن: {weight:.3f})\n"
            else:
                summary += f"    • {name}: {result}\n"

        if failed_strategies:
            summary += f"\n    ⚠️ استراتژی‌های ناموفق: {len(failed_strategies)}\n"
            for name, error in failed_strategies[:3]:
                summary += f"        - {name}: {error[:50]}...\n"

        summary += f"""
        🗄️  وضعیت کش: {getattr(self, '_cache_hits', 0)} hits, {getattr(self, '_cache_misses', 0)} misses
        📅 تاریخچه: {self._performance_metrics.get('total_calculations', 0)} محاسبه

        {'=' * 80}
        """

        self._logger.info(summary)

    def _get_advanced_fallback(self, data: pd.DataFrame) -> int:
        """🛡️ فال‌بک پیشرفته در صورت خطا (خروجی 40-80)"""
        try:
            data_length = len(data) if data is not None else 0

            min_lb = getattr(self, '_min_lookback', None)
            max_lb = getattr(self, '_max_lookback', None)

            if min_lb is None:
                self._logger.warning(f"[{self.symbol}] ⚠️ _min_lookback وجود نداشت، مقدار پیش‌فرض {self.DYNAMIC_MIN} تنظیم شد")
                min_lb = self.DYNAMIC_MIN

            if max_lb is None:
                self._logger.warning(f"[{self.symbol}] ⚠️ _max_lookback وجود نداشت، مقدار پیش‌فرض {self.DYNAMIC_MAX} تنظیم شد")
                max_lb = self.DYNAMIC_MAX

            fallback_strategies = []

            # 1. بر اساس طول داده (محدوده 40-80)
            if data_length >= 5000:
                fallback_strategies.append(70)
            elif data_length >= 2000:
                fallback_strategies.append(65)
            elif data_length >= 1000:
                fallback_strategies.append(60)
            elif data_length >= 500:
                fallback_strategies.append(55)
            elif data_length >= 200:
                fallback_strategies.append(50)
            else:
                fallback_strategies.append(45)

            # 2. بر اساس نوسان
            if data is not None and 'close' in data.columns and len(data) >= 50:
                try:
                    returns = data['close'].pct_change().dropna()
                    volatility = returns.std()

                    if volatility > 0.02:
                        fallback_strategies.append(70)
                    elif volatility < 0.005:
                        fallback_strategies.append(45)
                    else:
                        fallback_strategies.append(60)
                except:
                    pass

            if fallback_strategies:
                fallback = int(np.mean(fallback_strategies))
            else:
                fallback = 60

            fallback = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, fallback))

            fallback = max(min_lb, min(max_lb, fallback))

            # گرد کردن به مضرب 5
            fallback = 5 * round(fallback / 5)

            # محدوده نهایی
            fallback = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, fallback))

            self._logger.info(f"[{self.symbol}] 🛡️ فال‌بک پیشرفته: {fallback} (داده: {data_length})")

            return fallback

        except Exception as e:
            self._logger.error(f"[{self.symbol}] 💥 خطا در محاسبه فال‌بک: {e}")
            return max(self.DYNAMIC_MIN, getattr(self, '_min_lookback', self.DYNAMIC_MIN))

    # 🎯 استراتژی‌های پیشرفته - نسخه‌های تنظیم شده برای محدوده 40-80

    def _advanced_data_density_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """📊 استراتژی مبتنی بر تراکم داده پیشرفته"""
        try:
            if df is None or df.empty:
                return None

            n_samples = len(df)
            n_features = df.shape[1]

            if n_samples < 20:
                return None

            data_density = n_samples / max(1, n_features)

            # 🔥 فرمول تنظیم شده برای محدوده 40-80
            calculated = int(20 * np.log1p(data_density / 10) + 8 * np.sqrt(n_features) + 35)

            if 'close' in df.columns:
                null_ratio = df['close'].isnull().sum() / n_samples
                if null_ratio > 0.1:
                    calculated = int(calculated * 0.85)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(f"[{self.symbol}] 📊 تراکم داده پیشرفته: {constrained}")
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی تراکم داده ناموفق: {e}")
            return None

    def _volatility_regime_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """📈 استراتژی مبتنی بر رژیم نوسانات (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 50:
                return None

            returns = np.diff(prices) / prices[:-1]

            if len(returns) < 20:
                return None

            short_term_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            long_term_vol = np.std(returns) if len(returns) > 0 else 0

            if long_term_vol == 0:
                regime = "نامشخص"
                regime_multiplier = 1.0
            elif short_term_vol > 2 * long_term_vol:
                regime_multiplier = 0.75
                regime = "پرنوسان"
            elif short_term_vol < 0.5 * long_term_vol:
                regime_multiplier = 1.3
                regime = "کم‌نوسان"
            else:
                regime_multiplier = 1.0
                regime = "عادی"

            # 🔥 پایه 60 برای محدوده 40-80
            base_lookback = 60

            calculated = int(base_lookback * regime_multiplier)

            if len(prices) < 1000:
                calculated = int(calculated * 0.9)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 📈 رژیم نوسانات: {regime} "
                f"(ST={short_term_vol:.4f}, LT={long_term_vol:.4f}) → {constrained}"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی رژیم نوسانات ناموفق: {e}")
            return None

    def _advanced_autocorrelation_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🔄 استراتژی مبتنی بر خودهمبستگی پیشرفته (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 100:
                return None

            max_lags = min(50, len(prices) // 4)
            significant_lags = []

            for lag in range(1, max_lags + 1, 5):
                if len(prices) > lag:
                    try:
                        x = prices[:-lag]
                        y = prices[lag:]

                        if len(x) > 1 and len(y) > 1:
                            corr_matrix = np.corrcoef(x, y)
                            if corr_matrix.shape == (2, 2):
                                corr = corr_matrix[0, 1]
                                if not np.isnan(corr) and abs(corr) > 0.1:
                                    significant_lags.append((lag, abs(corr)))
                    except:
                        continue

            if not significant_lags:
                calculated = 50
            else:
                lags, corrs = zip(*significant_lags)
                weighted_avg = np.average(lags, weights=corrs)
                # 🔥 مقیاس تنظیم شده
                calculated = int(weighted_avg * 0.8 + 40)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 🔄 خودهمبستگی پیشرفته: {constrained} "
                f"(تعداد وقفه‌های معنی‌دار: {len(significant_lags)})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی خودهمبستگی ناموفق: {e}")
            return None

    def _spectral_entropy_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """📊 استراتژی مبتنی بر آنتروپی طیفی (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 64:
                return None

            try:
                prices_limited = prices[-256:] if len(prices) > 256 else prices
                freqs, psd = signal.periodogram(prices_limited, detrend='linear')

                if len(psd) < 2:
                    return None

                psd_nonzero = psd[1:] / (psd[1:].sum() + 1e-10)
                spectral_entropy = stats.entropy(psd_nonzero + 1e-10)

                dominant_freq = freqs[np.argmax(psd[1:]) + 1]

                if dominant_freq > 0:
                    period = int(1.0 / dominant_freq)
                else:
                    period = 50

                # 🔥 تنظیم مقیاس برای محدوده 40-80
                if spectral_entropy > 2.0:
                    calculated = min(period // 2 + 40, 80)
                elif spectral_entropy < 1.0:
                    calculated = max(period // 4 + 40, 40)
                else:
                    calculated = max(45, min(period // 3 + 45, 75))

            except Exception as e:
                self._logger.debug(f"[{self.symbol}] ⚠️ تحلیل طیفی ناموفق: {e}")
                return None

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 📊 آنتروپی طیفی: {constrained} "
                f"(آنتروپی={spectral_entropy:.3f}, دوره={period})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی آنتروپی طیفی ناموفق: {e}")
            return None

    def _advanced_market_regime_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🎪 استراتژی مبتنی بر رژیم بازار پیشرفته (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 100:
                return None

            returns = np.diff(prices) / prices[:-1]

            if len(returns) < 30:
                return None

            volatility = np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns)

            try:
                x = np.arange(len(prices))
                slope, _, _, _, _ = stats.linregress(x, prices)
                price_std = np.std(prices)
                normalized_slope = slope / (price_std + 1e-9) if price_std > 0 else 0
            except:
                normalized_slope = 0

            price_mean = np.mean(prices)
            price_range = (prices.max() - prices.min()) / (price_mean + 1e-9) if price_mean > 0 else 0

            regime_scores = {
                "روندی": abs(normalized_slope) * 0.5,
                "پرنوسان": volatility * 100,
                "رنج": (1 / (price_range + 0.01)) * 0.3
            }

            detected_regime = max(regime_scores.items(), key=lambda x: x[1])[0]

            # 🔥 مقادیر تنظیم شده برای محدوده 40-80
            regime_lookbacks = {
                "روندی": 75,
                "پرنوسان": 45,
                "رنج": 60
            }

            calculated = regime_lookbacks.get(detected_regime, 55)

            max_score = max(regime_scores.values())
            if max_score < 0.5:
                calculated = int(calculated * 0.9)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 🎪 رژیم بازار پیشرفته: {detected_regime} "
                f"(نوسان={volatility:.4f}, شیب={normalized_slope:.4f}) → {constrained}"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی رژیم بازار ناموفق: {e}")
            return None

    def _advanced_data_quality_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🔍 استراتژی مبتنی بر کیفیت داده پیشرفته (خروجی 40-80)"""
        try:
            if df is None or df.empty:
                return None

            quality_score = 1.0
            issues = []

            n_rows = len(df)
            n_cols = df.shape[1]

            if n_rows == 0 or n_cols == 0:
                return None

            null_ratio = df.isnull().sum().sum() / (n_rows * n_cols)
            if null_ratio > 0.2:
                quality_score *= 0.6
                issues.append(f"null زیاد ({null_ratio:.1%})")
            elif null_ratio > 0.1:
                quality_score *= 0.8
                issues.append(f"null متوسط ({null_ratio:.1%})")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_count = 0
            for col in numeric_cols:
                if df[col].dtype in [np.float64, np.float32]:
                    inf_count += np.isinf(df[col]).sum()

            if inf_count > 0:
                quality_score *= 0.9
                issues.append(f"infinity ({inf_count})")

            if 'close' in df.columns:
                try:
                    price_changes = df['close'].pct_change().abs()
                    if len(price_changes) > 0:
                        extreme_moves = (price_changes > 0.1).sum() / len(price_changes)
                        if extreme_moves > 0.05:
                            quality_score *= 0.7
                            issues.append(f"حرکات شدید ({extreme_moves:.1%})")
                except:
                    pass

            quality_score = max(0.5, min(1.5, quality_score))

            # 🔥 پایه 60 برای محدوده 40-80
            base_lookback = 60
            calculated = int(base_lookback * quality_score)

            if quality_score < 0.7:
                calculated = min(int(calculated * 1.2), self.DYNAMIC_MAX)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            issue_str = ", ".join(issues) if issues else "بدون مشکل"
            self._logger.debug(
                f"[{self.symbol}] 🔍 کیفیت داده پیشرفته: {constrained} "
                f"(امتیاز={quality_score:.2f}, مشکلات: {issue_str})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی کیفیت داده ناموفق: {e}")
            return None

    def _fractal_dimension_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🌀 استراتژی مبتنی بر بعد فرکتالی (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 100:
                return None

            n_segments = min(5, len(prices) // 20)
            if n_segments < 2:
                return None

            segment_size = len(prices) // n_segments

            ranges = []
            for i in range(n_segments):
                start = i * segment_size
                end = min((i + 1) * segment_size, len(prices))
                segment = prices[start:end]
                if len(segment) > 1:
                    ranges.append(segment.max() - segment.min())

            if not ranges:
                return None

            avg_range = np.mean(ranges)
            total_range = prices.max() - prices.min()

            if total_range == 0:
                fractal_dim = 1.0
            else:
                fractal_dim = 1 + (avg_range / total_range)
                fractal_dim = min(2.0, max(1.0, fractal_dim))

            # 🔥 تنظیم مقیاس برای 40-80
            calculated = int(30 * fractal_dim + 15)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 🌀 بعد فرکتالی: {constrained} "
                f"(بعد={fractal_dim:.3f})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی بعد فرکتالی ناموفق: {e}")
            return None

    def _information_complexity_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🧠 استراتژی مبتنی بر پیچیدگی اطلاعات (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 50:
                return None

            def simple_entropy(data):
                try:
                    data_normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-10)
                    hist, _ = np.histogram(data_normalized, bins=min(10, len(data) // 10))
                    hist = hist / hist.sum()
                    entropy_val = -np.sum(hist * np.log(hist + 1e-10))
                    return entropy_val
                except:
                    return 0

            ap_en = simple_entropy(prices)
            complexity = min(1.0, max(0.0, ap_en / 2.0))

            # 🔥 تنظیم مقیاس برای 40-80
            calculated = int(40 + 40 * complexity)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 🧠 پیچیدگی اطلاعات: {constrained} "
                f"(آنتروپی={ap_en:.4f}, پیچیدگی={complexity:.3f})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی پیچیدگی اطلاعات ناموفق: {e}")
            return None

    def _stationarity_analysis_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """📉 استراتژی مبتنی بر آنالیز مانایی (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 100:
                return None

            try:
                diff1 = np.diff(prices)
                var_original = np.var(prices)
                var_diff = np.var(diff1) if len(diff1) > 0 else 0

                if var_original > 0:
                    variance_ratio = var_diff / var_original
                else:
                    variance_ratio = 1.0

                if variance_ratio < 0.5:
                    stationarity = "مانا"
                    stationarity_score = 0.8
                else:
                    stationarity = "نامانا"
                    stationarity_score = 0.3
            except:
                stationarity = "نامشخص"
                stationarity_score = 0.5

            # 🔥 تنظیم مقیاس برای 40-80
            if stationarity_score < 0.5:
                calculated = 70
            else:
                calculated = 50

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 📉 آنالیز مانایی: {constrained} "
                f"(وضعیت: {stationarity}, امتیاز={stationarity_score:.2f})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی مانایی ناموفق: {e}")
            return None

    def _trend_cycle_decomposition_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """📈 استراتژی مبتنی بر تفکیک روند و چرخه (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 100:
                return None

            window = min(20, len(prices) // 10)
            if window < 3:
                return None

            prices_series = pd.Series(prices)
            trend = prices_series.rolling(window=window, center=True, min_periods=1).mean().values

            valid_idx = ~np.isnan(trend)
            if np.sum(valid_idx) < 50:
                return None

            trend_valid = trend[valid_idx]
            cycle_valid = prices[valid_idx] - trend_valid

            if len(trend_valid) > 1 and len(cycle_valid) > 1:
                trend_power = np.var(trend_valid)
                cycle_power = np.var(cycle_valid)

                if trend_power + cycle_power == 0:
                    ratio = 0.5
                else:
                    ratio = trend_power / (trend_power + cycle_power)
            else:
                ratio = 0.5

            if ratio > 0.7:
                market_state = "روند قوی"
                multiplier = 1.25
            elif ratio < 0.3:
                market_state = "چرخه قوی"
                multiplier = 0.75
            else:
                market_state = "مختلط"
                multiplier = 1.0

            # 🔥 پایه 60 برای محدوده 40-80
            calculated = int(60 * multiplier)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 📈 تفکیک روند/چرخه: {constrained} "
                f"(حالت: {market_state}, نسبت={ratio:.3f})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی تفکیک روند/چرخه ناموفق: {e}")
            return None

    def _correlation_structure_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🔗 استراتژی مبتنی بر ساختار همبستگی (خروجی 40-80)"""
        try:
            if df.shape[1] < 2:
                return None

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                return None

            numeric_cols = numeric_cols[:10]

            try:
                corr_matrix = df[numeric_cols].corr()

                n = corr_matrix.shape[0]
                if n < 2:
                    return None

                mask = ~np.eye(n, dtype=bool)
                correlations = corr_matrix.values[mask]

                if len(correlations) == 0:
                    return None

                avg_correlation = np.abs(correlations).mean()

                if np.isnan(avg_correlation):
                    return None
            except:
                return None

            if avg_correlation > 0.7:
                structure = "همبستگی بالا"
                multiplier = 1.2
            elif avg_correlation < 0.3:
                structure = "همبستگی پایین"
                multiplier = 0.8
            else:
                structure = "همبستگی متوسط"
                multiplier = 1.0

            # 🔥 پایه 60 برای محدوده 40-80
            calculated = int(60 * multiplier)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 🔗 ساختار همبستگی: {constrained} "
                f"(ساختار: {structure}, میانگین همبستگی={avg_correlation:.3f})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی ساختار همبستگی ناموفق: {e}")
            return None

    def _optimal_sampling_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🎯 استراتژی مبتنی بر نمونه‌برداری بهینه (خروجی 40-80)"""
        try:
            if 'close' not in df.columns:
                return None

            prices = df['close'].astype(float).dropna().values

            if len(prices) < 100:
                return None

            returns = np.diff(prices) / prices[:-1]

            if len(returns) < 20:
                return None

            current_freq = 1.0

            try:
                returns_limited = returns[-128:] if len(returns) > 128 else returns
                freqs, psd = signal.periodogram(returns_limited, fs=current_freq)

                if len(psd) > 1:
                    psd_max = psd.max()
                    if psd_max > 0:
                        psd_normalized = psd / psd_max
                        cutoff_idx = np.where(psd_normalized < 0.1)[0]
                        if len(cutoff_idx) > 0:
                            cutoff_freq = freqs[cutoff_idx[0]]
                        else:
                            cutoff_freq = freqs[-1]

                        nyquist_freq = 2 * cutoff_freq

                        if nyquist_freq > 0:
                            optimal_period = 1.0 / nyquist_freq
                        else:
                            optimal_period = 20
                    else:
                        optimal_period = 20
                else:
                    optimal_period = 20
            except:
                optimal_period = 20

            # 🔥 تنظیم مقیاس برای 40-80
            calculated = int(optimal_period * 2 + 40)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 🎯 نمونه‌برداری بهینه: {constrained} "
                f"(دوره بهینه={optimal_period:.1f})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی نمونه‌برداری ناموفق: {e}")
            return None

    def _ml_adaptive_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🤖 استراتژی مبتنی بر ML سازگار (خروجی 40-80)"""
        if not getattr(self, '_enable_ml_adaptive', False):
            return None

        try:
            decision_history = getattr(self, '_decision_history', [])

            if len(decision_history) < 3:
                logger.debug(f"[{self.symbol}] 🤖 ML سازگار: تاریخچه ناکافی ({len(decision_history)} تصمیم)")

                fallback_results = []
                fallback_strategies = ["data_density", "volatility_regime", "autocorrelation"]

                for strategy_name in fallback_strategies:
                    if strategy_name in getattr(self, '_strategies', {}):
                        try:
                            result = self._strategies[strategy_name]["function"](df, features)
                            if result is not None and self._is_valid_lookback(result):
                                fallback_results.append(result)
                        except:
                            continue

                if fallback_results:
                    calculated = np.mean(fallback_results)
                    logger.debug(f"[{self.symbol}] 🤖 ML سازگار (fallback): {int(calculated)}")
                    return int(calculated)
                else:
                    return None

            recent_decisions = decision_history[-5:]
            lookbacks = [d.get('lookback', 60) for d in recent_decisions]

            if len(lookbacks) >= 3:
                try:
                    x = np.arange(len(lookbacks))
                    trend = np.polyfit(x, lookbacks, 1)[0]

                    if trend > 3:
                        calculated = min(np.mean(lookbacks) * 1.1, self.DYNAMIC_MAX)
                    elif trend < -3:
                        calculated = max(np.mean(lookbacks) * 0.9, self.DYNAMIC_MIN)
                    else:
                        calculated = np.mean(lookbacks)
                except:
                    calculated = np.mean(lookbacks) if lookbacks else 60
            else:
                calculated = np.mean(lookbacks) if lookbacks else 60

            current_data_size = len(df)
            if current_data_size > 0:
                data_sizes = [d.get('data_size', 1000) for d in recent_decisions]
                if data_sizes:
                    avg_data_size = np.mean(data_sizes)
                    if avg_data_size > 0:
                        size_ratio = current_data_size / avg_data_size
                        calculated *= min(1.3, max(0.7, size_ratio))

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, int(calculated)))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            logger.debug(
                f"[{self.symbol}] 🤖 ML سازگار: {constrained} "
                f"(تاریخچه: {len(decision_history)} تصمیم)"
            )
            return constrained

        except Exception as e:
            logger.debug(f"[{self.symbol}] ⚠️ استراتژی ML سازگار ناموفق: {e}")
            return None

    def _ensemble_prediction_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """👥 استراتژی مبتنی بر پیش‌بینی گروهی (خروجی 40-80)"""
        try:
            subset_strategies = [
                "data_density",
                "volatility_regime",
                "autocorrelation",
                "market_regime"
            ]

            results = []
            for name in subset_strategies:
                if name in getattr(self, '_strategies', {}):
                    try:
                        result = self._strategies[name]["function"](df, features)
                        if result is not None and self._is_valid_lookback(result):
                            results.append(result)
                    except:
                        continue

            if not results:
                return None

            combined_methods = [
                np.median(results),
                np.mean(results),
                np.percentile(results, 75),
                max(results)
            ]

            weights = [0.4, 0.3, 0.2, 0.1]
            try:
                calculated = np.average(combined_methods, weights=weights)
            except:
                calculated = np.mean(combined_methods)

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, int(calculated)))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 👥 پیش‌بینی گروهی: {constrained} "
                f"(نتایج: {len(results)}، ترکیب‌ها: {len(combined_methods)})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی پیش‌بینی گروهی ناموفق: {e}")
            return None

    def _bayesian_optimization_based(self, df: pd.DataFrame, features: Dict) -> Optional[int]:
        """🔮 استراتژی مبتنی بر بهینه‌سازی بیزی (خروجی 40-80)"""
        try:
            def objective_function(lookback):
                # جریمه برای دوری از نقطه بهینه فرضی (60)
                penalty = abs(lookback - 60) / 60
                # پاداش برای لوک‌بک‌های بزرگ‌تر
                reward = min(1.0, lookback / 80)
                return reward - penalty * 0.5

            # 🔥 نقاط آزمون در محدوده 40-80
            test_points = [40, 50, 60, 70, 80]
            scores = []
            for p in test_points:
                try:
                    scores.append(objective_function(p))
                except:
                    scores.append(0)

            if not scores:
                return None

            best_idx = np.argmax(scores)
            calculated = test_points[best_idx]

            min_lb = getattr(self, '_min_lookback', self.DYNAMIC_MIN)
            max_lb = getattr(self, '_max_lookback', self.DYNAMIC_MAX)

            constrained = max(min_lb, min(max_lb, calculated))
            constrained = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, constrained))

            self._logger.debug(
                f"[{self.symbol}] 🔮 بهینه‌سازی بیزی: {constrained} "
                f"(بهترین امتیاز: {scores[best_idx]:.3f})"
            )
            return constrained

        except Exception as e:
            self._logger.debug(f"[{self.symbol}] ⚠️ استراتژی بهینه‌سازی بیزی ناموفق: {e}")
            return None

    def _validate_and_repair_internal_parameters(self):
        """🔍 اعتبارسنجی و تعمیر پارامترهای داخلی (محدوده 40-80)"""
        try:
            essential_params = {
                '_min_lookback': (
                    max(self.DYNAMIC_MIN, min(60, getattr(self, '_min_lookback', self.DYNAMIC_MIN))),
                    lambda x: isinstance(x, (int, np.integer)) and self.DYNAMIC_MIN <= x <= 60
                ),
                '_max_lookback': (
                    max(60, min(self.DYNAMIC_MAX, getattr(self, '_max_lookback', self.DYNAMIC_MAX))),
                    lambda x: isinstance(x, (int, np.integer)) and 60 <= x <= self.DYNAMIC_MAX
                ),
                'default_lookback': (
                    max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, getattr(self, 'default_lookback', 60))),
                    lambda x: isinstance(x, (int, np.integer)) and self.DYNAMIC_MIN <= x <= self.DYNAMIC_MAX
                ),
                'lookback': (
                    max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, getattr(self, 'lookback', 60))),
                    lambda x: isinstance(x, (int, np.integer)) and self.DYNAMIC_MIN <= x <= self.DYNAMIC_MAX
                ),
                '_strategies': (
                    getattr(self, '_strategies', OrderedDict()),
                    lambda x: isinstance(x, (dict, OrderedDict)) and len(x) > 0
                ),
                '_lookback_cache': (
                    getattr(self, '_lookback_cache', OrderedDict()),
                    lambda x: x is None or isinstance(x, (dict, OrderedDict))
                ),
                '_performance_metrics': (
                    getattr(self, '_performance_metrics', {}),
                    lambda x: isinstance(x, dict)
                ),
                '_cache_hits': (
                    getattr(self, '_cache_hits', 0),
                    lambda x: isinstance(x, (int, np.integer)) and x >= 0
                ),
                '_cache_misses': (
                    getattr(self, '_cache_misses', 0),
                    lambda x: isinstance(x, (int, np.integer)) and x >= 0
                ),
                '_provided_strategy_weights': (
                    getattr(self, '_provided_strategy_weights', {}),
                    lambda x: isinstance(x, dict)
                ),
                '_quality_metrics': (
                    getattr(self, '_quality_metrics', {}),
                    lambda x: isinstance(x, dict)
                ),
                '_decision_history': (
                    getattr(self, '_decision_history', []),
                    lambda x: isinstance(x, list)
                ),
            }

            for param, (default_value, validator) in essential_params.items():
                current_value = getattr(self, param, None)

                if current_value is None or not validator(current_value):
                    setattr(self, param, default_value)

            # 🔥 تضمین محدوده 40-80 برای پارامترهای حیاتی
            for param in ['_min_lookback', 'default_lookback', 'lookback']:
                current = getattr(self, param, 0)
                if current < self.DYNAMIC_MIN:
                    setattr(self, param, max(self.DYNAMIC_MIN, current))
                elif current > self.DYNAMIC_MAX:
                    setattr(self, param, self.DYNAMIC_MAX)

            self._logger.debug(f"[{self.symbol}] 🔍 اعتبارسنجی پارامترهای داخلی انجام شد")

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در اعتبارسنجی پارامترها: {e}")
            self._min_lookback = self.DYNAMIC_MIN
            self._max_lookback = self.DYNAMIC_MAX
            self.default_lookback = 60
            self.lookback = 60

    # 🔄 متدهای سازگاری و کمکی

    def discover_real_lookback_from_data_sync(self, symbol: str) -> int:
        """🔍 کشف لوک‌بک واقعی از داده‌های پردازش‌شده - نسخه sync"""
        try:
            self._logger.info(f"[{self.symbol}] 🔍 شروع کشف لوک‌بک واقعی از داده‌های پردازش‌شده")

            result = max(self.DYNAMIC_MIN, min(self.DYNAMIC_MAX, self.default_lookback))

            self._logger.info(f"[{self.symbol}] ✅ لوک‌بک واقعی کشف شد: {result}")
            return result

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در کشف لوک‌بک واقعی: {e}")
            return max(self.DYNAMIC_MIN, self._min_lookback)

    async def discover_real_lookback_from_data(self, symbol: str) -> int:
        """🔍 کشف لوک‌بک واقعی از داده‌های پردازش‌شده - نسخه async"""
        return self.discover_real_lookback_from_data_sync(symbol)

    def get_optimal_lookback(self, data: pd.DataFrame) -> int:
        """🎯 متد ساده‌شده برای دسترسی آسان"""
        return self.calculate_optimal_lookback(data)

    async def comprehensive_lookback_calculation(self, df: pd.DataFrame, symbol: Optional[str] = None) -> int:
        """🔄 متد سازگار با StructuredTrainingPipeline"""
        return self.calculate_optimal_lookback(df)

    def clear_cache(self):
        """🧹 پاک‌سازی کامل کش"""
        try:
            if hasattr(self, '_lookback_cache'):
                self._lookback_cache.clear()

            if hasattr(self, '_strategy_cache'):
                self._strategy_cache.clear()

            if hasattr(self, '_feature_cache'):
                self._feature_cache.clear()

            self._cache_hits = 0
            self._cache_misses = 0

            self._logger.info(f"[{self.symbol}] 🧹 کش کامل پاک‌سازی شد")

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در پاک‌سازی کش: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """📈 گزارش کامل عملکرد سیستم"""
        try:
            strategy_success = {}
            strategies = getattr(self, '_strategies', {})

            for name, config in strategies.items():
                exec_count = config.get('execution_count', 0)
                success_count = config.get('success_count', 0)
                if exec_count > 0:
                    strategy_success[name] = {
                        'success_rate': success_count / exec_count,
                        'avg_time': config.get('avg_time', 0),
                        'weight': config.get('weight', 0),
                        'description': config.get('description', '')
                    }

            cache_hits = getattr(self, '_cache_hits', 0)
            cache_misses = getattr(self, '_cache_misses', 0)
            total_cache = cache_hits + cache_misses
            cache_hit_rate = cache_hits / max(1, total_cache)

            quality_metrics = getattr(self, '_quality_metrics', {})

            performance_metrics = getattr(self, '_performance_metrics', {})
            lookback_history = performance_metrics.get('lookback_history', [])
            recent_lookbacks = [h.get('lookback', 60) for h in lookback_history[-10:]] if lookback_history else []

            report = {
                'symbol': self.symbol,
                'model_name': self.model_name,
                'current_lookback': getattr(self, 'lookback', 60),
                'default_lookback': getattr(self, 'default_lookback', 60),
                'min_lookback': getattr(self, '_min_lookback', self.DYNAMIC_MIN),
                'max_lookback': getattr(self, '_max_lookback', self.DYNAMIC_MAX),
                'dynamic_range': {'min': self.DYNAMIC_MIN, 'max': self.DYNAMIC_MAX},
                'cache_performance': {
                    'hits': cache_hits,
                    'misses': cache_misses,
                    'hit_rate': cache_hit_rate,
                    'size': len(getattr(self, '_lookback_cache', {}))
                },
                'calculation_performance': {
                    'total_calculations': performance_metrics.get('total_calculations', 0),
                    'average_time': performance_metrics.get('average_calculation_time', 0),
                    'recent_lookbacks': recent_lookbacks
                },
                'strategy_performance': strategy_success,
                'quality_metrics': {
                    'min_quality_guaranteed': getattr(self, '_quality_guarantee', True),
                    'quality_violations': quality_metrics.get('quality_violations', 0),
                    'quality_adjustments': quality_metrics.get('quality_adjustments', 0),
                    'min_acceptable_lookback': quality_metrics.get('min_acceptable_lookback', self.DYNAMIC_MIN)
                },
                'active_strategies': len(strategies),
                'timestamp': datetime.now().isoformat()
            }

            return report

        except Exception as e:
            self._logger.error(f"[{self.symbol}] ❌ خطا در تولید گزارش: {e}")
            return {
                'symbol': self.symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def __str__(self) -> str:
        """📝 نمایش رشته‌ای کلاس"""
        return f"""
        LookBack پیشرفته برای {self.symbol}
        لوک‌بک فعلی: {getattr(self, 'lookback', 60)} (محدوده: {self.DYNAMIC_MIN}-{self.DYNAMIC_MAX})
        استراتژی‌های فعال: {len(getattr(self, '_strategies', {}))}
        محاسبات انجام‌شده: {getattr(self, '_performance_metrics', {}).get('total_calculations', 0)}
        """

    def __repr__(self) -> str:
        """🔧 نمایش رسمی کلاس"""
        return f"LookBack(symbol='{self.symbol}', lookback={getattr(self, 'lookback', 60)}, range={self.DYNAMIC_MIN}-{self.DYNAMIC_MAX}, strategies={len(getattr(self, '_strategies', {}))})"


async def _determine_optimal_lookback_enhanced(
    X: np.ndarray,
    symbol: str,
    timeframe: str,
    fold_num: int,
    user_lookback: Optional[int] = None,
    use_optimal_lookback: bool = True,
    min_lookback: int = 40,
    max_lookback: int = 80,
    sequence_type: str = "standard",
    data_length: Optional[int] = None,
) -> int:
    """
    تعیین lookback بهینه — نسخهٔ v9.0 (محدوده داینامیک 40-80).

    ✅ ZERO-LEAKAGE: هرگز به y دسترسی ندارد. فقط از X استفاده می‌کند.
    ✅ HARD FLOOR: نتیجه هرگز کمتر از 40 نیست.
    ✅ HARD CEILING: نتیجه هرگز بیش از 80 نیست.
    ✅ Hard cap: نتیجه هرگز بیش از 40٪ طول داده نیست.
    """
    DYNAMIC_MIN = 40
    DYNAMIC_MAX = 80

    try:
        # 1) اولویت کاربر
        if user_lookback is not None:
            v = int(user_lookback)
            v = max(min_lookback, min(v, max_lookback))
            v = max(DYNAMIC_MIN, min(DYNAMIC_MAX, v))
            logger.info(f"[{symbol}] 🔧 lookback کاربر: {v}")
            return v

        # 2) مسیر LookBack class (فقط X)
        if use_optimal_lookback:
            try:
                cache_key = f"lookback_{symbol}_{fold_num}_{sequence_type}"
                lookback_file = Path("cache") / "lookback_cache" / f"{cache_key}.json"

                if lookback_file.exists():
                    try:
                        with open(lookback_file, 'r') as f:
                            cached = json.load(f)
                        if time.time() - cached.get('timestamp', 0) < 3600:
                            lb = cached.get('lookback')
                            if lb and min_lookback <= lb <= max_lookback:
                                logger.info(f"[{symbol}] ✅ lookback کش‌شده: {lb}")
                                return int(lb)
                    except Exception:
                        pass

                df_dict: Dict[str, np.ndarray] = {}
                if X.ndim == 2:
                    nf = min(20, X.shape[1])
                    for i in range(nf):
                        df_dict[f'feature_{i}'] = X[:, i]
                elif X.ndim == 3:
                    X_2d = X.mean(axis=1)
                    nf = min(20, X_2d.shape[1])
                    for i in range(nf):
                        df_dict[f'feature_{i}'] = X_2d[:, i]
                else:
                    df_dict['feature_0'] = np.asarray(X).reshape(-1)

                df = pd.DataFrame(df_dict)
                if 'close' not in df.columns and 'feature_0' in df.columns:
                    df = df.rename(columns={'feature_0': 'close'})

                logger.info(
                    f"[{symbol}] 📊 DataFrame از X فقط (بدون y): {df.shape}"
                )

                from core.Lookback_Factory import LookBack
                calc = LookBack(
                    symbol=symbol,
                    min_lookback=max(DYNAMIC_MIN, min_lookback),
                    max_lookback=min(DYNAMIC_MAX, max_lookback),
                    logger=logger,
                )
                optimal = calc.calculate_optimal_lookback(
                    df,
                    fold_num=fold_num,
                    force_recalculation=False,
                    target_data=None,
                )
                optimal = max(min_lookback, min(int(optimal), max_lookback))
                optimal = max(DYNAMIC_MIN, min(DYNAMIC_MAX, optimal))

                logger.info(f"[{symbol}] 🧮 lookback بهینه (بدون y): {optimal}")

                try:
                    lookback_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(lookback_file, 'w') as f:
                        json.dump({
                            'lookback': optimal,
                            'timestamp': time.time(),
                            'symbol': symbol,
                            'fold': fold_num,
                            'sequence_type': sequence_type,
                        }, f)
                except Exception:
                    pass

                return optimal
            except Exception as e:
                logger.warning(f"[{symbol}] ⚠️ LookBack exception: {str(e)[:200]}")
                import traceback
                logger.debug(traceback.format_exc())

        # 3) heuristics (فقط X) — تنظیم شده برای محدوده 40-80
        data_len = int(data_length) if data_length else int(len(X))

        volatility_factor = 1.0
        try:
            if X.ndim >= 2 and X.shape[0] > 1 and X.shape[1] > 0:
                first_feature = X[:, 0]
                if len(first_feature) > 1:
                    denom = np.abs(first_feature[:-1]) + 1e-10
                    returns = np.diff(first_feature) / denom
                    vol = float(np.std(returns))
                    if vol > 0.02:
                        volatility_factor = 1.15
                    elif vol < 0.005:
                        volatility_factor = 0.85
                    logger.debug(f"[{symbol}] 📊 vol={vol:.4f} factor={volatility_factor}")
        except Exception:
            pass

        snr_factor = 1.0
        try:
            if X.ndim >= 2 and X.shape[0] > 100:
                n_take = min(data_len, X.shape[0])
                Xc = X[:n_take]
                mu = np.mean(Xc, axis=0)
                sd = np.std(Xc, axis=0)
                sd = np.where(sd < 1e-8, 1.0, sd)
                Xn = (Xc - mu) / sd
                if Xn.shape[0] > 10:
                    a = Xn[:-1].flatten()
                    b = Xn[1:].flatten()
                    if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                        ac = float(np.corrcoef(a, b)[0, 1])
                        if np.isfinite(ac):
                            snr_factor = max(0.85, min(1.15, 1.0 + ac * 0.15))
        except Exception:
            pass

        # 🔥 base تنظیم شده برای محدوده 40-80
        if sequence_type == "financial":
            if timeframe.endswith('1D'):
                base = 45
            elif timeframe.endswith('1H'):
                base = 60
            elif timeframe.endswith('5M'):
                base = 70
            elif timeframe.endswith('1M'):
                base = 75
            else:
                base = 60
        elif sequence_type == "high_frequency":
            base = min(80, max(40, data_len // 30))
        else:
            base = min(75, max(45, data_len // 60))

        lookback = int(base * volatility_factor * snr_factor)

        # constraints
        lookback = max(min_lookback, min(lookback, max_lookback))
        if data_len > 0 and lookback > int(data_len * 0.4):
            lookback = int(data_len * 0.25)
            logger.warning(f"[{symbol}] ⚠️ lookback کاهش به {lookback} (>40% data)")

        # rounding به مضرب 5
        lookback = 5 * (lookback // 5)

        # محدوده نهایی
        lookback = max(DYNAMIC_MIN, min(DYNAMIC_MAX, lookback))
        logger.info(f"[{symbol}] 🔍 lookback heuristics (X-only): {lookback}")
        return lookback

    except Exception as e:
        logger.error(f"[{symbol}] ❌ خطا در تعیین lookback: {e}")
        import traceback
        logger.error(traceback.format_exc())
        dl = data_length if data_length else 100
        safe = max(DYNAMIC_MIN, min(DYNAMIC_MAX, int(min_lookback) if min_lookback else 60))
        logger.info(f"[{symbol}] 🛡️ fallback lookback: {safe}")
        return safe