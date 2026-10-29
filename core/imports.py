# ====================================================================================
# فایل: core/imports.py
from __future__ import annotations
def _safe_import(module_name: str, import_as: str = None, fallback_value=None, print_error: bool = True):
    """
    ایمپورت امن با پرینت خطا در صورت失敗
    
    Args:
        module_name: نام ماژول برای ایمپورت (مثلاً 'numpy' یا 'pandas')
        import_as: نام متغیر برای ذخیره (اختیاری)
        fallback_value: مقدار جایگزین در صورت خطا
        print_error: آیا خطا پرینت شود
    
    Returns:
        ماژول ایمپورت شده یا fallback_value
    """
    try:
        if import_as:
            module = __import__(module_name, fromlist=[import_as])
            if hasattr(module, import_as):
                return getattr(module, import_as)
            return module
        else:
            return __import__(module_name)
    except ImportError as e:
        if print_error:
            print(f"⚠️ IMPORT WARNING: Could not import '{module_name}' - {str(e)}")
            print(f"   → Using fallback value: {fallback_value if fallback_value is not None else 'None'}")
            print(f"   → Some features may be disabled")
        return fallback_value
    except Exception as e:
        if print_error:
            print(f"❌ IMPORT ERROR: Unexpected error importing '{module_name}' - {str(e)}")
            print(f"   → Using fallback value: {fallback_value if fallback_value is not None else 'None'}")
        return fallback_value


# ───────────────────────────────────────────────────────────────────────────────
# 🐍 CORE PYTHON & SYSTEM (همیشه در دسترس)
# ───────────────────────────────────────────────────────────────────────────────

import os
import sys
import re
import io
import copy
import gc
import glob
import math
import time
import json
import uuid
import queue
import shutil
import socket
import signal
import ast
import hashlib
import pickle
import zlib
import logging
import inspect
import warnings
import sqlite3
import asyncio
import traceback
import tempfile
import random
import zipfile
import argparse
import csv
import platform
import webbrowser
import contextlib
import functools
import threading
import multiprocessing
import subprocess
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import (
    Any, Callable, Dict, Generator, List, Optional, Tuple, Type, Union,
    Awaitable, Set, Protocol, TypeVar, Generic, ClassVar
)
from types import SimpleNamespace
from enum import Enum, auto
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from threading import Lock
from multiprocessing import cpu_count
from functools import lru_cache, partial, wraps
from contextlib import contextmanager, asynccontextmanager
from io import StringIO, BytesIO
from random import randint
from copy import deepcopy
import importlib
import importlib.util
import pkgutil
from dataclasses import dataclass, field, asdict, fields, is_dataclass
import os
import sys
import json
import csv
import pickle
import hashlib
import shutil
import threading
import time
import gzip
import zlib
import base64
from pathlib import Path
from typing import (
    List, Dict, Optional, Any, Union, Tuple, 
    Callable, Iterator, Generator, Set, FrozenSet
)
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, wraps
from contextlib import contextmanager, ExitStack
import logging
import traceback
import asyncio
import aiofiles
import aiofiles.os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import gc
import psutil
import signal
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    fcntl = None
    HAS_FCNTL = False
import tempfile
import uuid
import re
from abc import ABC, abstractmethod


# ───────────────────────────────────────────────────────────────────────────────
# 📦 PACKAGE MANAGEMENT & VERSIONING
# ───────────────────────────────────────────────────────────────────────────────
try:
    from packaging import version
    import semver
    _HAS_PACKAGING = True
    _HAS_SEMVER = True
    print("✅ packaging, semver imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import packaging/semver: {_e}")
    _HAS_PACKAGING = False
    _HAS_SEMVER = False
    version = None
    semver = None


# ───────────────────────────────────────────────────────────────────────────────
# 🔧 PLATFORM-SPECIFIC IMPORTS (fcntl برای لینوکس/مک)
# ───────────────────────────────────────────────────────────────────────────────
if sys.platform != 'win32':
    try:
        import fcntl
        from absl.flags._helpers import fcntl as absl_fcntl
        _HAS_FCNTL = True
        print("✅ fcntl imported successfully (Linux/Mac)")
    except (ImportError, AttributeError) as _e:
        print(f"⚠️ Could not import fcntl/absl_fcntl: {_e}")
        fcntl = None
        absl_fcntl = None
        _HAS_FCNTL = False
else:
    fcntl = None
    absl_fcntl = None
    _HAS_FCNTL = False
    
    try:
        import absl.flags._helpers
        absl.flags._helpers.fcntl = None
        print("✅ absl patched for Windows")
    except (ImportError, AttributeError) as _e:
        print(f"⚠️ Could not patch absl: {_e}")


# ───────────────────────────────────────────────────────────────────────────────
# 🖥️ SYSTEM MONITORING & UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
try:
    import psutil
    _HAS_PSUTIL = True
    print(f"✅ psutil imported successfully (version: {psutil.__version__ if hasattr(psutil, '__version__') else 'unknown'})")
except ImportError as _e:
    print(f"⚠️ Could not import psutil: {_e}")
    psutil = None
    _HAS_PSUTIL = False

try:
    from tqdm import tqdm
    from tqdm.asyncio import tqdm_asyncio
    _HAS_TQDM = True
    print("✅ tqdm imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import tqdm: {_e}")
    tqdm = None
    tqdm_asyncio = None
    _HAS_TQDM = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _HAS_WATCHDOG = True
    print("✅ watchdog imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import watchdog: {_e}")
    Observer = None
    FileSystemEventHandler = None
    _HAS_WATCHDOG = False


# ───────────────────────────────────────────────────────────────────────────────
# 🔒 SECURITY & ENVIRONMENT
# ───────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
    print("✅ python-dotenv imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import dotenv: {_e}")
    load_dotenv = None
    _HAS_DOTENV = False

try:
    from filelock import FileLock
    _HAS_FILELOCK = True
    print("✅ filelock imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import filelock: {_e}")
    FileLock = None
    _HAS_FILELOCK = False


# ───────────────────────────────────────────────────────────────────────────────
# 🔄 RETRY & RESILIENCE
# ───────────────────────────────────────────────────────────────────────────────
try:
    from tenacity import retry, wait_exponential
    _HAS_TENACITY = True
    print("✅ tenacity imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import tenacity: {_e}")
    retry = None
    wait_exponential = None
    _HAS_TENACITY = False


# ───────────────────────────────────────────────────────────────────────────────
# 🧮 DATA & SCIENTIFIC COMPUTING
# ───────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    from numpy.typing import NDArray
    _HAS_NUMPY = True
    print(f"✅ numpy imported successfully (version: {np.__version__})")
except ImportError as _e:
    print(f"⚠️ Could not import numpy: {_e}")
    np = None
    NDArray = None
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
    print(f"✅ pandas imported successfully (version: {pd.__version__})")
except ImportError as _e:
    print(f"⚠️ Could not import pandas: {_e}")
    pd = None
    _HAS_PANDAS = False

try:
    import scipy
    from scipy import stats, signal
    from scipy.stats import (
        anderson, kurtosis, normaltest, shapiro, skew,
        median_abs_deviation, entropy, jarque_bera, zscore,
        wasserstein_distance
    )
    from scipy.interpolate import interp1d, CubicSpline
    from scipy.fft import fft, ifft
    _HAS_SCIPY = True
    print(f"✅ scipy imported successfully (version: {scipy.__version__})")
except ImportError as _e:
    print(f"⚠️ Could not import scipy: {_e}")
    scipy = None
    stats = None
    signal = None
    _HAS_SCIPY = False

try:
    import joblib
    _HAS_JOBLIB = True
    print(f"✅ joblib imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import joblib: {_e}")
    joblib = None
    _HAS_JOBLIB = False

try:
    from statsmodels.tsa.stattools import adfuller as _adfuller, adfuller
    _HAS_STATSMODELS = True
    print(f"✅ statsmodels imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import statsmodels: {_e}")
    _HAS_STATSMODELS = False
    _adfuller = None
    adfuller = None


# ───────────────────────────────────────────────────────────────────────────────
# 🤖 MACHINE LEARNING - SCIKIT-LEARN
# ───────────────────────────────────────────────────────────────────────────────
try:
    from sklearn.base import BaseEstimator, TransformerMixin, is_classifier, is_regressor
    from sklearn.preprocessing import (
        StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler,
        PowerTransformer, QuantileTransformer, KBinsDiscretizer
    )
    from sklearn.model_selection import (
        train_test_split, TimeSeriesSplit, StratifiedKFold, KFold,
        GridSearchCV, cross_val_score
    )
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
        r2_score, explained_variance_score, median_absolute_error,
        mean_squared_log_error, accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, log_loss, hamming_loss,
        classification_report, confusion_matrix
    )
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_selection import SelectKBest, mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.impute import KNNImputer
    from sklearn.dummy import DummyClassifier
    from sklearn.covariance import EllipticEnvelope
    from sklearn.utils.multiclass import type_of_target
    from sklearn.utils.validation import check_array, check_is_fitted
    from sklearn.pipeline import make_pipeline
    from sklearn.exceptions import ConvergenceWarning, NotFittedError
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, r2_score
    _HAS_SKLEARN = True
    print(f"✅ scikit-learn imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import scikit-learn: {_e}")
    _HAS_SKLEARN = False
    # تعریف placeholder برای کلاس‌های اصلی
    BaseEstimator = object
    TransformerMixin = object
    StandardScaler = None
    RobustScaler = None
    MinMaxScaler = None


# ───────────────────────────────────────────────────────────────────────────────
# 🚀 GRADIENT BOOSTING FRAMEWORKS
# ───────────────────────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    _HAS_LIGHTGBM = True
    print(f"✅ lightgbm imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import lightgbm: {_e}")
    lgb = None
    LGBMRegressor = None
    _HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
    print(f"✅ xgboost imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import xgboost: {_e}")
    xgb = None
    XGBRegressor = None
    _HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor
    _HAS_CATBOOST = True
    print(f"✅ catboost imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import catboost: {_e}")
    CatBoostRegressor = None
    _HAS_CATBOOST = False


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 DEEP LEARNING - JAX & FLAX
# ───────────────────────────────────────────────────────────────────────────────
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

JAX_AVAILABLE = False
try:
    # JAX / FLAX / Optax (Optional)
    try:
        import jax
        import jax.numpy as jnp
        import flax
        import optax
        import jax.numpy as jnp
        from jax import jit, vmap, random, pmap, value_and_grad
        from flax import linen as nn
        from flax import struct
        from flax.training import train_state, checkpoints
        import orbax.checkpoint as ocp
        
        jax.config.update('jax_platform_name', 'cpu')
        jax.config.update('jax_debug_nans', False)
        JAX_AVAILABLE = True
        print(f"✅ JAX/FLAX/Optax imported successfully (JAX version: {jax.__version__})")

    except Exception as ex:
        jax = None
        jnp = None
        flax = None
        optax = None
        JAX_AVAILABLE = False

        print(f"⚠️ Could not import JAX/FLAX/Optax: {ex}")

except ImportError as _e:
    print(f"⚠️ Could not import JAX/FLAX/Optax: {_e}")
    print(f"   → JAX deep learning features disabled")
    JAX_AVAILABLE = False

# قبل از import tensorflow
import sys
import types

# ایجاد یک ماژول جعلی برای jax
mock_jax = types.ModuleType('jax')
mock_jax.version = types.ModuleType('version')
mock_jax.version.__version__ = '0.4.0'
sys.modules['jax'] = mock_jax
sys.modules['jax.version'] = mock_jax.version

# ───────────────────────────────────────────────────────────────────────────────
# 🧠 DEEP LEARNING - TENSORFLOW & KERAS
# ───────────────────────────────────────────────────────────────────────────────
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Sequential, Model, load_model as keras_load_model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, Input, BatchNormalization,
        Bidirectional, Conv1D, GaussianNoise, GlobalAveragePooling1D,
        InputLayer, LayerNormalization, MultiHeadAttention, SpatialDropout1D,
        Embedding, Reshape, Concatenate, Lambda, Attention
    )
    from tensorflow.keras.callbacks import (
        Callback, EarlyStopping, LambdaCallback, ModelCheckpoint,
        ReduceLROnPlateau, TensorBoard, TerminateOnNaN,
        LearningRateScheduler, ProgbarLogger, BackupAndRestore,
        CSVLogger
    )
    from tensorflow.keras.losses import Huber, MeanSquaredError, Loss
    from tensorflow.keras.metrics import Metric
    from tensorflow.keras.optimizers import Adam, RMSprop, AdamW, Optimizer
    from tensorflow.keras.optimizers.schedules import CosineDecayRestarts, LearningRateSchedule
    from tensorflow.keras.optimizers.legacy import Adam as LegacyAdam, RMSprop as LegacyRMSprop, SGD as LegacySGD
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    from tensorflow.keras.regularizers import l1_l2, l2
    from tensorflow.keras.initializers import VarianceScaling, Orthogonal
    from tensorflow.keras.constraints import MinMaxNorm
    from tensorflow.keras.utils import Progbar
    from tensorflow.keras import regularizers, constraints, initializers, layers, models
    from sklearn.decomposition import IncrementalPCA
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    
    TF_AVAILABLE = True
    print(f"✅ TensorFlow/Keras imported successfully (version: {tf.__version__})")
except ImportError as _e:
    print(f"⚠️ Could not import TensorFlow/Keras: {_e}")
    print(f"   → TensorFlow deep learning features disabled")
    TF_AVAILABLE = False
    tf = None
    keras = None


# ───────────────────────────────────────────────────────────────────────────────
# 🎯 HYPERPARAMETER TUNING
# ───────────────────────────────────────────────────────────────────────────────
try:
    import keras_tuner
    from keras_tuner import HyperModel, RandomSearch
    from keras_tuner.engine.hyperparameters import HyperParameters
    _HAS_KERAS_TUNER = True
    print("✅ keras-tuner imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import keras-tuner: {_e}")
    keras_tuner = None
    HyperModel = None
    RandomSearch = None
    HyperParameters = None
    _HAS_KERAS_TUNER = False

try:
    import optuna
    import optuna.integration
    from optuna.visualization import plot_optimization_history
    _HAS_OPTUNA = True
    print("✅ optuna imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import optuna: {_e}")
    optuna = None
    _HAS_OPTUNA = False


# ───────────────────────────────────────────────────────────────────────────────
# 📈 TIME SERIES ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────
try:
    from prophet import Prophet
    _HAS_PROPHET = True
    print("✅ prophet imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import prophet: {_e}")
    Prophet = None
    _HAS_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
    from statsmodels.tools.eval_measures import aic, bic
    import statsmodels.api as sm
    from statsmodels.tsa import stattools
    _HAS_STATSMODELS_FULL = True
    print("✅ statsmodels full imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import statsmodels full: {_e}")
    ARIMA = None
    ARIMAResults = None
    aic = None
    bic = None
    sm = None
    stattools = None
    _HAS_STATSMODELS_FULL = False

try:
    from arch import arch_model
    _HAS_ARCH = True
    print("✅ arch imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import arch: {_e}")
    arch_model = None
    _HAS_ARCH = False


# ───────────────────────────────────────────────────────────────────────────────
# 📊 TECHNICAL ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────
try:
    import talib
    _HAS_TALIB = True
    print("✅ TA-Lib imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import TA-Lib: {_e}")
    talib = None
    _HAS_TALIB = False

try:
    import ta
    from ta.momentum import (
        RSIIndicator, StochasticOscillator, WilliamsRIndicator,
        AwesomeOscillatorIndicator
    )
    from ta.trend import (
        MACD, ADXIndicator, CCIIndicator, DPOIndicator,
        PSARIndicator, VortexIndicator, MassIndex, IchimokuIndicator
    )
    from ta.volatility import (
        BollingerBands, KeltnerChannel, DonchianChannel, AverageTrueRange
    )
    from ta.volume import (
        OnBalanceVolumeIndicator, VolumePriceTrendIndicator,
        MFIIndicator, EaseOfMovementIndicator, ChaikinMoneyFlowIndicator
    )
    _HAS_TA = True
    print("✅ ta (technical analysis) imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import ta: {_e}")
    ta = None
    _HAS_TA = False


# ───────────────────────────────────────────────────────────────────────────────
# 📈 VISUALIZATION & PLOTTING
# ───────────────────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, Circle, Polygon
    from matplotlib.lines import Line2D
    from matplotlib.widgets import Button
    _HAS_MATPLOTLIB = True
    print(f"✅ matplotlib imported successfully (version: {matplotlib.__version__})")
except ImportError as _e:
    print(f"⚠️ Could not import matplotlib: {_e}")
    matplotlib = None
    plt = None
    mdates = None
    mticker = None
    _HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    _HAS_SEABORN = True
    print("✅ seaborn imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import seaborn: {_e}")
    sns = None
    _HAS_SEABORN = False

try:
    import networkx as nx
    from pyvis.network import Network
    _HAS_NETWORKX = True
    print("✅ networkx imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import networkx: {_e}")
    nx = None
    Network = None
    _HAS_NETWORKX = False

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    _HAS_IPYWIDGETS = True
    print("✅ ipywidgets imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import ipywidgets: {_e}")
    widgets = None
    display = None
    clear_output = None
    _HAS_IPYWIDGETS = False

try:
    import tabulate
    _HAS_TABULATE = True
    print("✅ tabulate imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import tabulate: {_e}")
    tabulate = None
    _HAS_TABULATE = False


# ───────────────────────────────────────────────────────────────────────────────
# 💹 TRADING & FINANCIAL APIs
# ───────────────────────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    _HAS_MT5 = True
    print("✅ MetaTrader5 imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import MetaTrader5: {_e}")
    print(f"   → MT5 trading features disabled")
    mt5 = None
    _HAS_MT5 = False

try:
    import investpy
    _HAS_INVESTPY = True
    print("✅ investpy imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import investpy: {_e}")
    investpy = None
    _HAS_INVESTPY = False

try:
    import requests
    from bs4 import BeautifulSoup
    _HAS_REQUESTS = True
    print("✅ requests/beautifulsoup imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import requests/beautifulsoup: {_e}")
    requests = None
    BeautifulSoup = None
    _HAS_REQUESTS = False

try:
    from yfinance import Ticker
    _HAS_YFINANCE = True
    print("✅ yfinance imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import yfinance: {_e}")
    Ticker = None
    _HAS_YFINANCE = False


# ───────────────────────────────────────────────────────────────────────────────
# 🖥️ GUI FRAMEWORKS
# ───────────────────────────────────────────────────────────────────────────────
try:
    from tkinter import Tk, filedialog
    _HAS_TKINTER = True
    print("✅ tkinter imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import tkinter: {_e}")
    Tk = None
    filedialog = None
    _HAS_TKINTER = False


# ───────────────────────────────────────────────────────────────────────────────
# 🔍 MODEL INTERPRETABILITY
# ───────────────────────────────────────────────────────────────────────────────
try:
    import shap
    _HAS_SHAP = True
    print("✅ shap imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import shap: {_e}")
    shap = None
    _HAS_SHAP = False

try:
    from graphviz import Digraph
    _HAS_GRAPHVIZ = True
    print("✅ graphviz imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import graphviz: {_e}")
    Digraph = None
    _HAS_GRAPHVIZ = False


# ───────────────────────────────────────────────────────────────────────────────
# 🧪 EXPERIMENT TRACKING
# ───────────────────────────────────────────────────────────────────────────────
try:
    import mlflow
    _HAS_MLFLOW = True
    print("✅ mlflow imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import mlflow: {_e}")
    mlflow = None
    _HAS_MLFLOW = False


# ───────────────────────────────────────────────────────────────────────────────
# 🔧 LOGGING & CONFIGURATION
# ───────────────────────────────────────────────────────────────────────────────
import logging.handlers
from logging import Logger
from logging.handlers import RotatingFileHandler


# ───────────────────────────────────────────────────────────────────────────────
# ⚡ PERFORMANCE & NUMBA
# ───────────────────────────────────────────────────────────────────────────────
try:
    from numba import njit, prange, jit
    _HAS_NUMBA = True
    print("✅ numba imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import numba: {_e}")
    print(f"   → JIT compilation disabled")
    njit = lambda x: x
    prange = range
    jit = lambda x: x
    _HAS_NUMBA = False


# ───────────────────────────────────────────────────────────────────────────────
# 📁 FILE OPERATIONS
# ───────────────────────────────────────────────────────────────────────────────
try:
    import aiofiles
    import aiofiles.os
    _HAS_AIOFILES = True
    print("✅ aiofiles imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import aiofiles: {_e}")
    aiofiles = None
    _HAS_AIOFILES = False


# ───────────────────────────────────────────────────────────────────────────────
# 🔧 ADDITIONAL UTILITIES
# ───────────────────────────────────────────────────────────────────────────────
try:
    from async_timeout import timeout
    _HAS_ASYNC_TIMEOUT = True
    print("✅ async-timeout imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import async-timeout: {_e}")
    timeout = None
    _HAS_ASYNC_TIMEOUT = False

try:
    from typing_extensions import Self
    _HAS_TYPING_EXTENSIONS = True
    print("✅ typing_extensions imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import typing_extensions: {_e}")
    Self = None
    _HAS_TYPING_EXTENSIONS = False

try:
    import threadpoolctl
    _HAS_THREADPOOLCTL = True
    print("✅ threadpoolctl imported successfully")
except ImportError as _e:
    print(f"⚠️ Could not import threadpoolctl: {_e}")
    threadpoolctl = None
    _HAS_THREADPOOLCTL = False


# ───────────────────────────────────────────────────────────────────────────────
# ⚠️ WARNING FILTERS
# ───────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_addons")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ───────────────────────────────────────────────────────────────────────────────
# 📋 FINAL CHECKS & UTILITY FUNCTIONS
# ───────────────────────────────────────────────────────────────────────────────

def check_imports_status():
    """بررسی وضعیت ایمپورت‌های اصلی - با پرینت کامل"""
    print("\n" + "="*60)
    print("📊 IMPORT STATUS REPORT")
    print("="*60)
    
    status = {
        'jax': JAX_AVAILABLE,
        'tensorflow': TF_AVAILABLE,
        'numpy': np is not None,
        'pandas': pd is not None,
        'scikit-learn': _HAS_SKLEARN,
        'pytorch': 'torch' in sys.modules,
        'mt5': _HAS_MT5,
    }
    
    for name, available in status.items():
        icon = "✅" if available else "❌"
        print(f"   {icon} {name}: {'Available' if available else 'NOT Available'}")
    
    print("="*60)
    return status


# Platform information
PLATFORM = sys.platform
IS_WINDOWS = PLATFORM == 'win32'
IS_LINUX = PLATFORM == 'linux'
IS_MAC = PLATFORM == 'darwin'

print(f"\n{'='*60}")
print(f"✅ Platform: {PLATFORM}")
print(f"✅ Imports completed - Summary:")
print(f"   • NumPy: {'✓' if np is not None else '✗'}")
print(f"   • Pandas: {'✓' if pd is not None else '✗'}")
print(f"   • Scikit-learn: {'✓' if _HAS_SKLEARN else '✗'}")
print(f"   • TensorFlow: {'✓' if TF_AVAILABLE else '✗'}")
print(f"   • JAX: {'✓' if JAX_AVAILABLE else '✗'}")
print(f"   • MetaTrader5: {'✓' if _HAS_MT5 else '✗'}")
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════════════════════════
#                               END OF IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════