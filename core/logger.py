"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ADVANCED TRADE LOGGER - v12.0 PRODUCTION GRADE (CONTINUOUS & MULTI-SYSTEM) ║
║  ✅ Unique Machine ID for Multi-System Isolation                            ║
║  ✅ Continuous Log Storage (From Start to Finish)                           ║
║  ✅ Separate Internal System Logger (No mixing with trade logs)             ║
║  ✅ 4-Hour Auto Send Interval (Optimized for large continuous files)        ║
║  ✅ Complete Source Info: file, class, function, line number                ║
║  🆕 Per-Logger Dedicated Files (trade / train / callback / workflow /       ║
║     strategy_mgm / learning_penalty_mgm / runtime)                          ║
║  ✅ Auto Send All Continuous Files → Rubika + Eitaa + Telegram              ║
║  ✅ Auto Database Sender (Every 4 Hours)                                    ║
║  ✅ Smart Large File Handling (Auto-Zip before send if > 15MB)              ║
║  ✅ Separate Error & Warning Continuous Log Files                           ║
║  ✅ Sound Manager for Errors                                                ║
║  ✅ Production Ready - No Crashes                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
# =====================[ PHASE 1: IMPORTS ]=====================
import logging
import logging.handlers
import os
import sys
import inspect
import threading
import json
import hashlib
import atexit
import traceback
import time
import shutil
import tempfile
import builtins
import sqlite3
import zipfile
from datetime import datetime, timedelta
from typing import Any, Callable, Optional, Dict, List, Set, Tuple, Union
from pathlib import Path
from queue import Queue, Empty
from collections import deque

# =====================[ PHASE 2: RUBIKA IMPORTS & MACHINE ID ]=====================
try:
    from core.rubika_logger import (
        send_rubika_message,
        send_latest_log_to_rubika,
        send_log_file_to_rubika,
        send_latest_log_to_eitaa,
        send_log_file_to_eitaa,
        send_checkpoint,
        send_deployment_package,
        RUBIKA_TOKEN,
        RUBIKA_CHAT_ID,
        EITAA_TOKEN,
        EITAA_CHAT_ID,
        get_machine_id,
        MACHINE_ID
    )
    RUBIKA_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Rubika module import error: {e}")
    RUBIKA_AVAILABLE = False
    MACHINE_ID = "UNKNOWN_MACHINE"

    def get_machine_id() -> str:
        return MACHINE_ID

    def send_rubika_message(msg): return False
    def send_latest_log_to_rubika(*args, **kwargs): return False
    def send_log_file_to_rubika(*args, **kwargs): return False
    def send_latest_log_to_eitaa(*args, **kwargs): return False
    def send_log_file_to_eitaa(*args, **kwargs): return False
    def send_checkpoint(*args, **kwargs): return False
    def send_deployment_package(*args, **kwargs): return False

# =====================[ PHASE 2.5: 🆕 LOGGER TYPE MAPPING ]=====================
# 🆕 نقشه جامع انواع لاگر → (ایموجی، عنوان، پیشوند فایل)
# ترتیب مهم است: موارد خاص‌تر ابتدا بررسی می‌شوند.
LOGGER_TYPE_MAPPING: Dict[str, Dict[str, str]] = {
    "learning_penalty_mgm": {
        "emoji": "🧠",
        "title": "Learning Penalty Log",
        "file_prefix": "learning_penalty_mgm",
    },
    "strategy_mgm": {
        "emoji": "🎯",
        "title": "Strategy Management Log",
        "file_prefix": "strategy_mgm",
    },
    "system_logger": {
        "emoji": "⚙️",
        "title": "System Logger",
        "file_prefix": "system_logger",
    },
    "trade": {
        "emoji": "📊",
        "title": "Trade Log",
        "file_prefix": "trade",
    },
    "train": {
        "emoji": "🤖",
        "title": "Training Log",
        "file_prefix": "train",
    },
    "callback": {
        "emoji": "🔄",
        "title": "Callback Log",
        "file_prefix": "callback",
    },
    "workflow": {
        "emoji": "🌊",
        "title": "Workflow Log",
        "file_prefix": "workflow",
    },
    "runtime": {
        "emoji": "📋",
        "title": "Runtime Log",
        "file_prefix": "runtime",
    },
}

def detect_logger_type(logger_name: str) -> str:
    """تشخیص نوع لاگر بر اساس نام آن. اگر مطابقت نداشت → 'runtime'"""
    name_lower = (logger_name or "").lower()
    for key in LOGGER_TYPE_MAPPING.keys():
        if key in name_lower:
            return key
    return "runtime"

def build_continuous_filename(logger_type: str) -> str:
    """ساخت نام فایل پیوسته بر اساس نوع لاگر"""
    info = LOGGER_TYPE_MAPPING.get(logger_type, LOGGER_TYPE_MAPPING["runtime"])
    return f"{info['file_prefix']}_{MACHINE_ID}_continuous.log"

# =====================[ PHASE 3: BASE CONFIGURATION ]=====================
LOG_DIR = os.environ.get("TRADE_LOG_DIR", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Error/Warning log configuration
ERROR_LOG_CONFIG = {
    "enabled": True,
    "error_file": f"error_{MACHINE_ID}_continuous.log",
    "warning_file": f"warning_{MACHINE_ID}_continuous.log",
    "error_dir": os.path.join(LOG_DIR, "errors"),
    "max_bytes": 100 * 1024 * 1024,  # 100 MB safety limit
    "backup_count": 3,
    "daily_rotation": False,  # Disabled for true continuous storage
}

# Continuous Send configuration (4 Hours = 14400 seconds)
CONTINUOUS_SEND_CONFIG = {
    "enabled": True,
    "interval_seconds": 14400,  # 4 hours
    "send_to_rubika": True,
    "send_to_eitaa": True,
    "send_all_log_files": True,
    "prevent_duplicate_send": True,
    "send_on_startup": False,
    "max_file_size_for_direct_send_mb": 15,  # Auto-zip if larger to prevent API failure
}

# Database sender configuration
DATABASE_SENDER_CONFIG = {
    "enabled": True,
    "interval_seconds": 14400,  # 4 hours
    "db_file_path": "TRADE/strategy_stats.db",
    "send_to_rubika": True,
    "send_to_eitaa": True,
    "tracking_file": os.path.join(LOG_DIR, "db_sent_tracking.json"),
    "max_retries": 3,
    "retry_delay": 5,
    "send_on_startup": False,
}

DATE_FMT = "%Y-%m-%d %H:%M:%S"

# =====================[ PHASE 4: SOUND CONFIGURATION ]=====================
SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "sounds")
os.makedirs(SOUNDS_DIR, exist_ok=True)
DEFAULT_ERROR_SOUND = os.path.join(SOUNDS_DIR, "error.wav")
DEFAULT_CRITICAL_SOUND = os.path.join(SOUNDS_DIR, "critical.wav")
SOUND_VOLUME = 0.7
_sound_initialized = False

try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    _sound_initialized = True
except Exception as e:
    print(f"⚠️ Sound card not available: {e}")
    _sound_initialized = False

# =====================[ PHASE 9.5: SOURCE-ROUTING FILE HANDLER ]=====================

class SourceRoutingFileHandler(logging.Handler):
    """
    مسیریابی لاگ به فایل مقصد بر اساس «فایل مبدأ لاگ» (record.pathname).
    این کار مشکل استفاده از لاگرهای مشترک (مثل strategy_mgm) از داخل
    فایل‌های دیگر (مثل Learning_signal_Management.py) را حل می‌کند.
    """

    # نگاشت «الگوی نام فایل مبدأ» → «نوع لاگ مقصد»
    # ترتیب مهم است: موارد خاص‌تر بالاتر بیایند.
    FILE_PATTERNS: List[Tuple[str, str]] = [
        # ---- learning_penalty_mgm (اولویت بالا) ----
        ("learning_signal_management.py", "learning_penalty_mgm"),
        ("learning_penalty",              "learning_penalty_mgm"),
        ("penalty_mgm",                   "learning_penalty_mgm"),
        ("continuouslearningengine",      "learning_penalty_mgm"),  # ✅ NEW
        ("signalfil",                     "learning_penalty_mgm"),  # ✅ NEW (SignalFileManager)
        ("strategyscorebooster",          "learning_penalty_mgm"),  # ✅ NEW
        ("dynamicweighting",              "learning_penalty_mgm"),  # ✅ NEW
        ("signalquality",                 "learning_penalty_mgm"),  # ✅ NEW
        ("textreport",                    "learning_penalty_mgm"),  # ✅ NEW
        ("visualreport",                  "learning_penalty_mgm"),  # ✅ NEW

        # ---- فایل‌هایی که باید به trade بروند ----
        ("forward_test_live.py",          "trade"),
        ("forward_test",                  "trade"),
        ("live_trading",                  "trade"),
        ("trade_executor",                "trade"),
        ("order_manager",                 "trade"),

        # ---- فایل‌های استراتژی ----
        ("strategy_management.py",        "strategy_mgm"),
        ("strategy_manager",              "strategy_mgm"),

        # ---- فایل‌های آموزش/کال‌بک/ورک‌فلو ----
        ("train",                         "train"),
        ("callback",                      "callback"),
        ("workflow",                      "workflow"),
    ]

    def __init__(self, fmt: str, datefmt: str, level: int = logging.DEBUG):
        super().__init__(level=level)
        self._formatter = SourceFormatter(fmt=fmt, datefmt=datefmt)
        self._file_handlers: Dict[str, logging.FileHandler] = {}
        self._fh_lock = threading.Lock()

    # ------------------------------------------------------------
    def _resolve_log_type(self, record: logging.LogRecord) -> str:
        # 1) ابتدا بر اساس نام فایل مبدأ
        try:
            source_file = os.path.basename(record.pathname or "").lower()
        except Exception:
            source_file = ""

        if source_file:
            for pattern, log_type in self.FILE_PATTERNS:
                if pattern in source_file:
                    return log_type

        # 2) در صورت عدم تطابق → بر اساس نام لاگر
        return detect_logger_type(record.name)

    # ------------------------------------------------------------
    def _get_file_handler(self, log_type: str) -> logging.FileHandler:
        existing = self._file_handlers.get(log_type)
        if existing is not None:
            return existing

        with self._fh_lock:
            existing = self._file_handlers.get(log_type)
            if existing is not None:
                return existing

            if log_type not in LOGGER_TYPE_MAPPING:
                log_type = "runtime"

            filename = build_continuous_filename(log_type)
            path = Path(LOG_DIR) / filename
            path.parent.mkdir(parents=True, exist_ok=True)

            if not path.exists():
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(f"# Continuous Log file started at {datetime.now().isoformat()}\n")
                        f.write(f"# Machine ID: {MACHINE_ID}\n")
                        f.write(f"# Routed by: SourceRoutingFileHandler\n")
                        f.write(f"# Log Type: {log_type} ({LOGGER_TYPE_MAPPING[log_type]['title']})\n")
                except Exception:
                    pass

            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            handler.setFormatter(self._formatter)
            self._file_handlers[log_type] = handler
            return handler

    # ------------------------------------------------------------
    def emit(self, record: logging.LogRecord):
        try:
            log_type = self._resolve_log_type(record)
            handler = self._get_file_handler(log_type)
            handler.emit(record)
        except Exception:
            self.handleError(record)

    def close(self):
        with self._fh_lock:
            for h in self._file_handlers.values():
                try:
                    h.close()
                except Exception:
                    pass
            self._file_handlers.clear()
        super().close()
        
# =====================[ PHASE 5: SOUND MANAGER CLASS ]=====================
class SoundManager:
    """Sound management for error playback"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SoundManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            self.enabled = True
            self.sounds = {
                'ERROR': DEFAULT_ERROR_SOUND,
                'CRITICAL': DEFAULT_CRITICAL_SOUND
            }
            for level, sound_path in self.sounds.items():
                if not os.path.exists(sound_path):
                    self._create_default_sound(level, sound_path)
            self._initialized = True

    def _create_default_sound(self, level: str, path: str):
        """Create default sound files"""
        try:
            import numpy as np
            import wave as wave_module
            frequencies = {'ERROR': 440, 'CRITICAL': 220}
            freq = frequencies.get(level, 440)
            duration = 0.5
            sample_rate = 22050
            t = np.linspace(0, duration, int(sample_rate * duration), False)

            if level == 'ERROR':
                wave = 0.5 * np.sin(2 * np.pi * freq * t)
                wave *= np.exp(-5 * t)
            else:
                wave = 0.3 * np.sin(2 * np.pi * freq * t)
                wave *= (1 + 0.3 * np.sin(2 * np.pi * 5 * t))
                wave *= np.exp(-3 * t)

            wave = np.int16(wave * 32767)
            with wave_module.open(path, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave.tobytes())
        except Exception as e:
            print(f"⚠️ Error creating default sound: {e}")
            Path(path).touch()

    def play_sound(self, level: str):
        """Play sound for specified error level"""
        if not self.enabled or not _sound_initialized:
            return
        sound_path = self.sounds.get(level)
        if not sound_path or not os.path.exists(sound_path):
            return

        def _play():
            try:
                sound = pygame.mixer.Sound(sound_path)
                sound.set_volume(SOUND_VOLUME)
                sound.play()
                pygame.time.wait(int(sound.get_length() * 1000))
            except Exception:
                pass

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def set_sound(self, level: str, sound_path: str) -> bool:
        if os.path.exists(sound_path):
            self.sounds[level] = sound_path
            return True
        return False

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def set_volume(self, volume: float):
        global SOUND_VOLUME
        SOUND_VOLUME = max(0.0, min(1.0, volume))

# =====================[ PHASE 6: DATABASE SEND TRACKER ]=====================
class DatabaseSendTracker:
    """Track database file sends to prevent duplicates"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseSendTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            self.tracking_file = DATABASE_SENDER_CONFIG["tracking_file"]
            self.sent_hashes: Set[str] = set()
            self._load_tracking_data()
            self._initialized = True

    def _load_tracking_data(self):
        try:
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.sent_hashes = set(data.get("sent_hashes", []))
        except Exception as e:
            print(f"⚠️ [DB Tracker] Load error: {e}")
            self.sent_hashes = set()

    def _save_tracking_data(self):
        try:
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            data = {
                "sent_hashes": list(self.sent_hashes),
                "last_updated": datetime.now().isoformat()
            }
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ [DB Tracker] Save error: {e}")

    def get_file_hash(self, file_path: str) -> str:
        try:
            if not os.path.exists(file_path):
                return ""
            stat = os.stat(file_path)
            content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()[:16]
        except Exception:
            return hashlib.md5(file_path.encode()).hexdigest()[:16]

    def is_already_sent(self, file_path: str) -> bool:
        file_hash = self.get_file_hash(file_path)
        return file_hash in self.sent_hashes

    def mark_as_sent(self, file_path: str):
        file_hash = self.get_file_hash(file_path)
        if file_hash:
            self.sent_hashes.add(file_hash)
            self._save_tracking_data()

    def cleanup_old_tracking(self, max_entries: int = 100):
        if len(self.sent_hashes) > max_entries:
            hashes_list = list(self.sent_hashes)
            self.sent_hashes = set(hashes_list[-max_entries:])
            self._save_tracking_data()

# =====================[ PHASE 7: DATABASE SENDER ]=====================
class DatabaseSender:
    """Automatic database file sender (Every 4 Hours)"""
    _instance = None
    _lock = threading.Lock()
    _sender_thread = None
    _running = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseSender, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            self.tracker = DatabaseSendTracker()
            self.last_send_time = datetime.now()
            self._db_file_found = False
            self._initialized = True
            self._start_sender()

    def _start_sender(self):
        if self._running:
            return
        self._running = True
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name="DatabaseSender"
        )
        self._sender_thread.start()
        print(f"🔄 [DatabaseSender] Started (sends strategy_stats.db every 4 hours) | Machine: {MACHINE_ID}")

    def _sender_loop(self):
        first_check_done = False
        if DATABASE_SENDER_CONFIG.get("send_on_startup", True):
            time.sleep(2)
            db_path = self._get_db_file_path()
            if db_path:
                print(f"✅ [DatabaseSender] Database file found on startup: {db_path}")
                self._send_database()
                self.last_send_time = datetime.now()
            else:
                print("⏳ [DatabaseSender] Database file not found on startup, waiting...")

        while self._running:
            try:
                now = datetime.now()
                if not self._db_file_found:
                    db_path = self._get_db_file_path()
                    if db_path:
                        self._db_file_found = True
                        print(f"✅ [DatabaseSender] Database file found: {db_path}")
                    else:
                        if not first_check_done:
                            print("⏳ [DatabaseSender] Waiting for database file to be created...")
                            first_check_done = True
                        time.sleep(30)
                        continue

                time_since_send = (now - self.last_send_time).total_seconds()
                if time_since_send >= DATABASE_SENDER_CONFIG["interval_seconds"]:
                    db_path = self._get_db_file_path()
                    if db_path:
                        self._db_file_found = True
                        self._send_database()
                        self.last_send_time = now
                    else:
                        self._db_file_found = False
                        print("⚠️ [DatabaseSender] Database file lost, waiting for re-creation...")
                        continue
                time.sleep(60)
            except Exception as e:
                print(f"⚠️ [DatabaseSender] Loop error: {e}")
                traceback.print_exc()
                time.sleep(30)

    def _get_db_file_path(self) -> Optional[Path]:
        db_file = DATABASE_SENDER_CONFIG["db_file_path"]
        possible_paths = [
            Path(db_file),
            Path.cwd() / db_file,
            Path.cwd() / "TRADE" / db_file,
            Path.cwd() / "project-new" / "TRADE" / db_file,
            Path.cwd().parent / "TRADE" / db_file,
            Path.cwd().parent.parent / "TRADE" / db_file,
            Path.cwd() / "data" / db_file,
            Path.cwd() / "database" / db_file,
            Path.cwd() / "logs" / db_file,
        ]
        if os.path.isabs(db_file):
            possible_paths.append(Path(db_file))

        for path in possible_paths:
            if path and path.exists() and path.is_file():
                return path

        try:
            for root, dirs, files in os.walk(Path.cwd()):
                if db_file in files:
                    return Path(root) / db_file
        except Exception:
            pass
        return None

    def _send_database(self):
        if not DATABASE_SENDER_CONFIG.get("enabled", True):
            return
        print(f"📤 [DatabaseSender] Sending strategy_stats.db...")
        db_path = self._get_db_file_path()

        if not db_path:
            print(f"⏳ [DatabaseSender] Database file not found yet, will retry later...")
            self._db_file_found = False
            return

        self._db_file_found = True
        if self.tracker.is_already_sent(str(db_path)):
            print(f"⏭️ [DatabaseSender] Already sent: {db_path.name} (unchanged)")
            return

        try:
            rubika_success = False
            if DATABASE_SENDER_CONFIG.get("send_to_rubika", True) and RUBIKA_AVAILABLE:
                try:
                    rubika_success = send_log_file_to_rubika(str(db_path))
                    if rubika_success:
                        print(f"✅ [DatabaseSender] Rubika sent: {db_path.name}")
                except Exception as e:
                    print(f"❌ [DatabaseSender] Rubika error: {e}")

            eitaa_success = False
            if DATABASE_SENDER_CONFIG.get("send_to_eitaa", True) and RUBIKA_AVAILABLE:
                try:
                    db_size = db_path.stat().st_size / 1024
                    caption = f"📊 *Strategy Stats DB*\n🖥️ Machine: {MACHINE_ID}\n📁 {db_path.name}\n📦 {db_size:.2f} KB"
                    eitaa_success = send_log_file_to_eitaa(str(db_path), caption=caption)
                    if eitaa_success:
                        print(f"✅ [DatabaseSender] Eitaa sent: {db_path.name}")
                except Exception as e:
                    print(f"❌ [DatabaseSender] Eitaa error: {e}")

            if rubika_success or eitaa_success:
                self.tracker.mark_as_sent(str(db_path))
                print(f"✅ [DatabaseSender] Database sent successfully")
        except Exception as e:
            print(f"❌ [DatabaseSender] Error: {e}")
            traceback.print_exc()

    def force_send_database(self) -> bool:
        self._send_database()
        return True

    def get_status(self) -> Dict[str, Any]:
        db_path = self._get_db_file_path()
        return {
            "enabled": DATABASE_SENDER_CONFIG.get("enabled", True),
            "running": self._running,
            "machine_id": MACHINE_ID,
            "db_file_path": str(db_path) if db_path else None,
            "db_file_found": self._db_file_found,
            "last_send_time": self.last_send_time.isoformat(),
            "interval_seconds": DATABASE_SENDER_CONFIG.get("interval_seconds", 14400),
            "sent_count": len(self.tracker.sent_hashes),
        }

    def stop(self):
        self._running = False
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5)

# =====================[ PHASE 8: ERROR WARNING LOGGER ]=====================
class ErrorWarningLogger:
    """Separate error and warning logging (Continuous, Machine-Specific)"""
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ErrorWarningLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            self.error_logger = None
            self.warning_logger = None
            self.error_dir = ERROR_LOG_CONFIG["error_dir"]
            self._setup_loggers()
            self._initialized = True

    def _setup_loggers(self):
        try:
            os.makedirs(self.error_dir, exist_ok=True)
            error_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-25s | %(source)s | %(message)s",
                datefmt=DATE_FMT
            )

            self.error_logger = logging.getLogger("error_logger")
            self.error_logger.setLevel(logging.ERROR)
            self.error_logger.handlers.clear()
            self.error_logger.propagate = False

            self.warning_logger = logging.getLogger("warning_logger")
            self.warning_logger.setLevel(logging.WARNING)
            self.warning_logger.handlers.clear()
            self.warning_logger.propagate = False

            # Continuous Error File
            error_file_path = os.path.join(self.error_dir, ERROR_LOG_CONFIG["error_file"])
            error_handler = logging.FileHandler(error_file_path, mode='a', encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(error_formatter)
            self.error_logger.addHandler(error_handler)

            # Continuous Warning File
            warning_file_path = os.path.join(self.error_dir, ERROR_LOG_CONFIG["warning_file"])
            warning_handler = logging.FileHandler(warning_file_path, mode='a', encoding='utf-8')
            warning_handler.setLevel(logging.WARNING)
            warning_handler.setFormatter(error_formatter)
            self.warning_logger.addHandler(warning_handler)

        except Exception as e:
            print(f"❌ [ErrorWarningLogger] Setup error: {e}")
            traceback.print_exc()

    def log_error(self, record: logging.LogRecord):
        if not self.error_logger or not ERROR_LOG_CONFIG["enabled"]:
            return
        try:
            if not hasattr(record, 'source') or not record.source:
                record.source = self._extract_source(record)
            self.error_logger.handle(record)
        except Exception as e:
            print(f"⚠️ Error logging error: {e}")

    def log_warning(self, record: logging.LogRecord):
        if not self.warning_logger or not ERROR_LOG_CONFIG["enabled"]:
            return
        try:
            if not hasattr(record, 'source') or not record.source:
                record.source = self._extract_source(record)
            self.warning_logger.handle(record)
        except Exception as e:
            print(f"⚠️ Error logging warning: {e}")

    def _extract_source(self, record: logging.LogRecord) -> str:
        try:
            if hasattr(record, 'source') and record.source and str(record.source).strip() != "unknown":
                return str(record.source)

            frames = inspect.getouterframes(inspect.currentframe())
            logger_methods = {'debug', 'info', 'warning', 'error', 'critical', 'exception', 'trade', 'log', '_get_source_info', 'emit', 'handle', 'callHandlers', 'makeRecord'}

            def is_internal(filename, func_name):
                if func_name in logger_methods:
                    return True
                filename_lower = filename.lower()
                if 'site-packages' in filename_lower or 'lib/python' in filename_lower:
                    return True
                if '/logging/' in filename_lower or '\\logging\\' in filename_lower:
                    return True
                base = os.path.basename(filename_lower)
                if base in {'threading.py', 'inspect.py', 'traceback.py', 'runpy.py', 'code.py', 'contextlib.py', 'abc.py', 'logger.py'}:
                    return True
                return False

            for frame_info in frames:
                frame = frame_info.frame
                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_name
                lineno = frame.f_lineno
                if not filename or not func_name:
                    continue
                if is_internal(filename, func_name):
                    continue

                file_name = os.path.basename(filename)
                class_name = ""
                if 'self' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['self'].__class__.__name__
                    except Exception:
                        pass
                elif 'cls' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['cls'].__name__
                    except Exception:
                        pass

                if class_name:
                    return f"{class_name}.{func_name}() | {file_name}:{lineno}"
                elif func_name and func_name != "<module>":
                    return f"{func_name}() | {file_name}:{lineno}"
                else:
                    return f"{file_name}:{lineno}"

            if record.funcName and record.filename:
                return f"{record.funcName}() | {os.path.basename(record.filename)}:{record.lineno}"
            return "unknown"
        except Exception:
            if record.funcName and record.filename:
                return f"{record.funcName}() | {os.path.basename(record.filename)}:{record.lineno}"
            return "unknown"

    def get_error_file_path(self) -> str:
        return os.path.join(self.error_dir, ERROR_LOG_CONFIG["error_file"])

    def get_warning_file_path(self) -> str:
        return os.path.join(self.error_dir, ERROR_LOG_CONFIG["warning_file"])

    def get_error_count(self) -> int:
        try:
            file_path = self.get_error_file_path()
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f)
        except Exception:
            pass
        return 0

    def get_warning_count(self) -> int:
        try:
            file_path = self.get_warning_file_path()
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return sum(1 for _ in f)
        except Exception:
            pass
        return 0

    def clear_error_log(self) -> bool:
        try:
            file_path = self.get_error_file_path()
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception:
            pass
        return False

    def clear_warning_log(self) -> bool:
        try:
            file_path = self.get_warning_file_path()
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception:
            pass
        return False

# =====================[ PHASE 9: CONTINUOUS LOG MANAGER ]=====================
class ContinuousLogManager:
    """
    Manages continuous log files (No rotation, append-only).
    Sends every 4 hours. Auto-zips large files before sending to prevent API failure.
    🆕 Automatically picks up ALL *_continuous.log files including
       strategy_mgm & learning_penalty_mgm.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ContinuousLogManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            self._loggers: List[logging.Logger] = []
            self._handlers: List[logging.Handler] = []
            self._last_send_time = datetime.now()
            self._running = False
            self._sender_thread: Optional[threading.Thread] = None
            self._initialized = True
            self._start_sender_thread()

    def _start_sender_thread(self):
        if self._running:
            return
        self._running = True
        self._sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True,
            name="ContinuousSendThread"
        )
        self._sender_thread.start()
        print(f"🔄 [ContinuousLogManager] 4-Hour sender started | Machine: {MACHINE_ID}")

    def _sender_loop(self):
        if CONTINUOUS_SEND_CONFIG.get("send_on_startup", True):
            time.sleep(3)
            print("📤 [ContinuousLogManager] Initial send on startup...")
            self._send_all_continuous_logs()
            self._last_send_time = datetime.now()

        while self._running:
            try:
                now = datetime.now()
                time_since_send = (now - self._last_send_time).total_seconds()

                if time_since_send >= CONTINUOUS_SEND_CONFIG["interval_seconds"]:
                    print(f"🔄 [ContinuousLogManager] 4-Hour send triggered at {now.strftime('%H:%M:%S')}")
                    self._send_all_continuous_logs()
                    self._last_send_time = now

                time.sleep(60)
            except Exception as e:
                print(f"⚠️ [ContinuousLogManager] Loop error: {e}")
                traceback.print_exc()
                time.sleep(60)

    def _send_all_continuous_logs(self):
        if not CONTINUOUS_SEND_CONFIG["send_all_log_files"]:
            return

        try:
            # Flush all handlers before sending
            for logger in self._loggers:
                for handler in logger.handlers:
                    try:
                        handler.flush()
                    except Exception:
                        pass

            files_to_send = self._get_all_continuous_log_files()
            if not files_to_send:
                print("⚠️ [ContinuousLogManager] No log files found to send")
                return

            print(f"📤 [ContinuousLogManager] Found {len(files_to_send)} continuous log file(s) to send")
            for f in files_to_send:
                print(f"   ├─ {f.name}")

            for file_path in files_to_send:
                self._send_single_file_smart(file_path)

            print(f"✅ [ContinuousLogManager] Send cycle completed")

        except Exception as e:
            print(f"❌ [ContinuousLogManager] Error sending files: {e}")
            traceback.print_exc()

    def _get_all_continuous_log_files(self) -> List[Path]:
        """
        🆕 تمام فایل‌های *_continuous.log را پیدا می‌کند، شامل:
        trade / train / callback / workflow / strategy_mgm / learning_penalty_mgm / runtime / system_logger / error / warning
        """
        try:
            log_files = []

            # Main logs (تمام فایل‌های پیوسته در پوشه logs)
            for f in Path(LOG_DIR).iterdir():
                if f.is_file() and f.suffix == ".log" and "continuous" in f.name:
                    try:
                        if f.stat().st_size > 0:
                            log_files.append(f)
                    except Exception:
                        pass

            # Error/Warning logs
            error_dir = Path(ERROR_LOG_CONFIG["error_dir"])
            if error_dir.exists():
                for f in error_dir.iterdir():
                    if f.is_file() and f.suffix == ".log" and "continuous" in f.name:
                        try:
                            if f.stat().st_size > 0:
                                log_files.append(f)
                        except Exception:
                            pass

            # حذف تکراری‌ها
            unique: Dict[str, Path] = {}
            for f in log_files:
                unique[str(f.resolve())] = f

            result = list(unique.values())
            result.sort(key=lambda x: x.name)
            return result
        except Exception as e:
            print(f"⚠️ [ContinuousLogManager] Error getting files: {e}")
            return []

    def _send_single_file_smart(self, file_path: Path):
        """Sends file, auto-zipping if > 15MB to prevent API timeout/failure"""
        try:
            file_str = str(file_path)
            file_name = file_path.name
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            caption = self._get_file_caption(file_name)

            temp_zip_path = None

            # Smart Large File Handling
            if file_size_mb > CONTINUOUS_SEND_CONFIG.get("max_file_size_for_direct_send_mb", 15):
                print(f"📦 [ContinuousLogManager] File > 15MB ({file_size_mb:.1f}MB). Zipping for safe transmission...")
                temp_zip_path = file_path.with_suffix('.zip')
                with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(file_path, arcname=file_name)
                file_str = str(temp_zip_path)
                file_name = temp_zip_path.name
                print(f"✅ [ContinuousLogManager] Zipped to: {file_name}")

            # Send to Rubika
            if CONTINUOUS_SEND_CONFIG["send_to_rubika"] and RUBIKA_AVAILABLE:
                try:
                    result = send_log_file_to_rubika(file_str)
                    if result:
                        print(f"✅ [ContinuousLogManager] Rubika sent: {file_name}")
                    else:
                        print(f"⚠️ [ContinuousLogManager] Rubika send failed for {file_name}")
                except Exception as e:
                    print(f"❌ [ContinuousLogManager] Rubika error for {file_name}: {e}")

            # Send to Eitaa
            if CONTINUOUS_SEND_CONFIG["send_to_eitaa"] and RUBIKA_AVAILABLE:
                try:
                    result = send_log_file_to_eitaa(file_str, caption=caption)
                    if result:
                        print(f"✅ [ContinuousLogManager] Eitaa sent: {file_name}")
                    else:
                        print(f"⚠️ [ContinuousLogManager] Eitaa send failed for {file_name}")
                except Exception as e:
                    print(f"❌ [ContinuousLogManager] Eitaa error for {file_name}: {e}")

            # Cleanup temp zip
            if temp_zip_path and temp_zip_path.exists():
                try:
                    temp_zip_path.unlink()
                except Exception:
                    pass

        except Exception as e:
            print(f"❌ [ContinuousLogManager] Send error for {file_path.name}: {e}")

    def _get_file_caption(self, file_name: str) -> str:
        """
        🆕 کپشن هوشمند برای هر نوع فایل لاگ بر اساس LOGGER_TYPE_MAPPING
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        name_lower = file_name.lower()

        # defaults
        emoji, title = "📋", "Continuous Log"

        # جستجو بر اساس نقشه
        for log_type, info in LOGGER_TYPE_MAPPING.items():
            if log_type in name_lower:
                emoji = info["emoji"]
                title = info["title"]
                break
        else:
            # fallbacks خاص
            if "error" in name_lower:
                emoji, title = "❌", "Error Log"
            elif "warning" in name_lower:
                emoji, title = "⚠️", "Warning Log"

        return f"{emoji} *{title} - 4H Update*\n🖥️ Machine: {MACHINE_ID}\n📁 {file_name}\n🕐 {now}"

    def register_logger(self, logger: logging.Logger):
        if logger not in self._loggers:
            self._loggers.append(logger)

    def register_handler(self, handler: logging.Handler):
        if handler not in self._handlers:
            self._handlers.append(handler)

    def force_send_all_files(self) -> int:
        self._send_all_continuous_logs()
        return len(self._get_all_continuous_log_files())

    def stop(self):
        self._running = False
        if self._sender_thread and self._sender_thread.is_alive():
            self._sender_thread.join(timeout=5)

# =====================[ PHASE 10: SYSTEM INTERNAL LOGGER ]=====================
class SystemInternalLogger:
    """Dedicated logger for the logging system itself to prevent mixing with trade logs"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SystemInternalLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            self.logger = logging.getLogger("SystemInternalLogger")
            self.logger.setLevel(logging.INFO)
            self.logger.handlers.clear()
            self.logger.propagate = False

            log_file = Path(LOG_DIR) / f"system_logger_{MACHINE_ID}_continuous.log"
            handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt=DATE_FMT
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            self._initialized = True

    def info(self, msg: str):
        self.logger.info(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

# =====================[ PHASE 11: SOURCE FORMATTER ]=====================
class SourceFormatter(logging.Formatter):
    """Professional formatter with complete source info"""
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    SPECIAL_COLORS = {
        "train": "\033[38;5;46m",
        "callback": "\033[38;5;10m",
        "workflow": "\033[38;5;39m",
        "trade": "\033[38;5;51m",
        "strategy_mgm": "\033[38;5;208m",             # 🆕 نارنجی
        "learning_penalty_mgm": "\033[38;5;135m",     # 🆕 بنفش
    }
    RESET = "\033[0m"

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None, style: str = "%"):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(source)s | %(message)s"
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, 'source') or not record.source or str(record.source).strip() == "unknown":
            source_info = self._extract_source_info(record)
            if source_info:
                record.source = source_info
            else:
                if record.funcName and record.filename:
                    record.source = f"{record.funcName}() | {os.path.basename(record.filename)}:{record.lineno}"
                else:
                    record.source = "unknown"

        color = self._get_color(record)
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"

    def _extract_source_info(self, record: logging.LogRecord) -> str:
        try:
            if hasattr(record, 'source') and record.source and str(record.source).strip() != "unknown":
                return str(record.source)

            frames = inspect.getouterframes(inspect.currentframe())
            logger_methods = {
                'debug', 'info', 'warning', 'error', 'critical',
                'exception', 'trade', 'log', '_get_source_info', 'emit',
                'handle', 'callHandlers', 'makeRecord', 'print_to_logger',
                'redirect_prints_to_logger'
            }

            def is_internal(filename, func_name):
                if func_name in logger_methods:
                    return True
                filename_lower = filename.lower()
                if 'site-packages' in filename_lower or 'lib/python' in filename_lower:
                    return True
                if '/logging/' in filename_lower or '\\logging\\' in filename_lower:
                    return True
                base = os.path.basename(filename_lower)
                if base in {'threading.py', 'inspect.py', 'traceback.py', 'runpy.py', 'code.py', 'contextlib.py', 'abc.py', 'logger.py'}:
                    return True
                return False

            for frame_info in frames:
                frame = frame_info.frame
                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_name
                lineno = frame.f_lineno
                if not filename or not func_name:
                    continue
                if func_name == 'print_to_logger':
                    continue
                if is_internal(filename, func_name):
                    continue

                file_name = os.path.basename(filename)
                class_name = ""
                if 'self' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['self'].__class__.__name__
                    except Exception:
                        pass
                elif 'cls' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['cls'].__name__
                    except Exception:
                        pass

                if not class_name:
                    for var_name, var_value in frame.f_locals.items():
                        if var_name in ('self', 'cls'):
                            continue
                        try:
                            if hasattr(var_value, '__class__') and not var_name.startswith('_'):
                                if hasattr(var_value, func_name):
                                    class_name = var_value.__class__.__name__
                                    break
                        except Exception:
                            pass

                if class_name:
                    return f"{class_name}.{func_name}() | {file_name}:{lineno}"
                elif func_name and func_name != "<module>":
                    return f"{func_name}() | {file_name}:{lineno}"
                else:
                    return f"{file_name}:{lineno}"

            if record.filename and record.funcName:
                return f"{record.funcName}() | {os.path.basename(record.filename)}:{record.lineno}"
            return "unknown"
        except Exception:
            if record.filename and record.funcName:
                return f"{record.funcName}() | {os.path.basename(record.filename)}:{record.lineno}"
            return "unknown"

    def _get_color(self, record: logging.LogRecord) -> str:
        source_file = os.path.basename(record.pathname or "").lower()
        # اول از فایل مبدأ، بعد از نام لاگر
        for pattern, log_type in SourceRoutingFileHandler.FILE_PATTERNS:
            if pattern in source_file:
                return self.SPECIAL_COLORS.get(log_type, "")
        logger_name = record.name.lower()
        for log_type in ("learning_penalty_mgm", "strategy_mgm", "train", "trade", "callback", "workflow"):
            if log_type in logger_name:
                return self.SPECIAL_COLORS.get(log_type, "")
        return self.COLORS.get(record.levelname, "")
# =====================[ PHASE 9.6: GLOBAL ROUTING HANDLER ]=====================
_global_routing_handler: Optional[SourceRoutingFileHandler] = None
_routing_handler_lock = threading.Lock()

def get_global_routing_handler(fmt: str, datefmt: str) -> SourceRoutingFileHandler:
    global _global_routing_handler
    if _global_routing_handler is None:
        with _routing_handler_lock:
            if _global_routing_handler is None:
                _global_routing_handler = SourceRoutingFileHandler(fmt=fmt, datefmt=datefmt)
    return _global_routing_handler
    
# =====================[ PHASE 12: MAIN ADVANCED TRADE LOGGER ]=====================
class AdvancedTradeLogger:
    """Advanced logger with continuous storage and 4-hour auto-send"""
    _manager = None
    _sound_manager = None
    _db_sender = None
    _error_warning_logger = None
    _system_internal_logger = None
    _configured_loggers = set()
    _initialized = False
    _init_lock = threading.Lock()

    def __init__(self, name: str, enable_telegram: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.enable_telegram = enable_telegram
        self._internal_logger = logging.getLogger(f"_internal_{name}")
        self._initialize_system()

        try:
            if name not in AdvancedTradeLogger._configured_loggers:
                self._configure_logger()
                AdvancedTradeLogger._configured_loggers.add(name)
        except Exception as e:
            self._internal_logger.error(f"Init error: {e}")
            raise

    @classmethod
    def _initialize_system(cls):
        with cls._init_lock:
            if cls._initialized:
                return
            try:
                if cls._sound_manager is None:
                    cls._sound_manager = SoundManager()
                if cls._manager is None:
                    cls._manager = ContinuousLogManager()
                if cls._db_sender is None:
                    cls._db_sender = DatabaseSender()
                if cls._error_warning_logger is None:
                    cls._error_warning_logger = ErrorWarningLogger()
                if cls._system_internal_logger is None:
                    cls._system_internal_logger = SystemInternalLogger()

                cls._system_internal_logger.info(f"System Logger Initialized | Machine: {MACHINE_ID}")
                cls._initialized = True
            except Exception as e:
                print(f"⚠️ System init error: {e}")
                cls._initialized = True

    @classmethod
    def get_manager(cls) -> Optional[ContinuousLogManager]:
        cls._initialize_system()
        return cls._manager

    @classmethod
    def get_db_sender(cls) -> Optional[DatabaseSender]:
        cls._initialize_system()
        return cls._db_sender

    @classmethod
    def get_error_warning_logger(cls) -> Optional[ErrorWarningLogger]:
        cls._initialize_system()
        return cls._error_warning_logger

    def _get_source_info(self) -> str:
        try:
            all_frames = inspect.stack()
            logger_methods = {
                'debug', 'info', 'warning', 'error', 'critical',
                'exception', 'trade', 'log', '_get_source_info', 'emit',
                'print_to_logger', 'redirect_prints_to_logger', 'setup_logger'
            }

            def is_internal(filename, func_name):
                if func_name in logger_methods:
                    return True
                filename_lower = filename.lower()
                if 'site-packages' in filename_lower or 'lib/python' in filename_lower:
                    return True
                if '/logging/' in filename_lower or '\\logging\\' in filename_lower:
                    return True
                base = os.path.basename(filename_lower)
                if base in {'threading.py', 'inspect.py', 'traceback.py', 'runpy.py', 'code.py', 'contextlib.py', 'abc.py', 'logger.py'}:
                    return True
                if func_name == 'print_to_logger':
                    return True
                return False

            found_frame = None
            for frame_info in all_frames:
                frame = frame_info.frame
                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_name
                lineno = frame.f_lineno
                if not filename or not func_name:
                    continue
                if func_name == 'print_to_logger':
                    continue
                if os.path.basename(filename) == 'logger.py':
                    continue
                if is_internal(filename, func_name):
                    continue

                found_frame = frame_info
                break

            if found_frame:
                frame = found_frame.frame
                filename = frame.f_code.co_filename
                func_name = frame.f_code.co_name
                lineno = frame.f_lineno
                file_basename = os.path.basename(filename)

                class_name = ""
                if 'self' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['self'].__class__.__name__
                    except Exception:
                        pass
                if not class_name and 'cls' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['cls'].__name__
                    except Exception:
                        pass

                if not class_name:
                    for var_name, var_value in frame.f_locals.items():
                        if var_name in ('self', 'cls'):
                            continue
                        try:
                            if hasattr(var_value, '__class__') and not var_name.startswith('_'):
                                if hasattr(var_value, func_name):
                                    class_name = var_value.__class__.__name__
                                    break
                        except Exception:
                            pass

                if class_name:
                    return f"{class_name}.{func_name}() | {file_basename}:{lineno}"
                elif func_name and func_name != "<module>":
                    return f"{func_name}() | {file_basename}:{lineno}"
                else:
                    return f"{file_basename}:{lineno}"

            return "unknown"
        except Exception as e:
            return f"error: {str(e)}"

    def _configure_logger(self):
        """
        🆕 پیکربندی لاگر و ساخت فایل اختصاصی برای هر نوع:
        trade / train / callback / workflow / strategy_mgm / learning_penalty_mgm / runtime
        """
        try:
            self.logger.setLevel(logging.DEBUG)
            self.logger.handlers.clear()
            self.logger.propagate = False

            fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(source)s | %(message)s"
            formatter = SourceFormatter(fmt=fmt, datefmt=DATE_FMT)

            # Console handler
            try:
                sh = logging.StreamHandler(sys.stdout)
                sh.setLevel(logging.INFO)
                sh.setFormatter(formatter)
                self.logger.addHandler(sh)
            except Exception as e:
                self._internal_logger.error(f"Console handler error: {e}")

            # 🆕 File handler با مسیریابی بر اساس فایل مبدأ
            try:
                manager = self.get_manager()

                # ✅ به‌جای FileHandler اختصاصی برای هر Logger، از هندلر مسیریاب مشترک استفاده می‌کنیم
                routing_handler = get_global_routing_handler(fmt=fmt, datefmt=DATE_FMT)
                self.logger.addHandler(routing_handler)

                # برای اطلاع ContinuousLogManager که این لاگر زنده است
                if manager:
                    manager.register_logger(self.logger)
                    manager.register_handler(routing_handler)

                self._internal_logger.info(
                    f"[LoggerConfig] '{self.name}' → routed by source-file "
                    f"(initial type: {detect_logger_type(self.name)})"
                )
            except Exception as e:
                self._internal_logger.error(f"Routing handler error: {e}")
            # Sound handler
            try:
                class SoundErrorHandler(logging.Handler):
                    def emit(self, record):
                        if record.levelno >= logging.ERROR:
                            try:
                                if AdvancedTradeLogger._sound_manager:
                                    level = 'ERROR' if record.levelno == logging.ERROR else 'CRITICAL'
                                    AdvancedTradeLogger._sound_manager.play_sound(level)
                            except Exception:
                                pass

                sound_handler = SoundErrorHandler()
                sound_handler.setLevel(logging.ERROR)
                self.logger.addHandler(sound_handler)
            except Exception as e:
                self._internal_logger.error(f"Sound handler error: {e}")

            # Error/Warning handler
            try:
                class ErrorWarningHandler(logging.Handler):
                    def emit(self, record):
                        try:
                            ew_logger = AdvancedTradeLogger._error_warning_logger
                            if ew_logger:
                                if not hasattr(record, 'source') or not record.source:
                                    record.source = self._extract_source(record)
                                if record.levelno >= logging.ERROR:
                                    ew_logger.log_error(record)
                                elif record.levelno >= logging.WARNING:
                                    ew_logger.log_warning(record)
                        except Exception:
                            pass

                    def _extract_source(self, record):
                        try:
                            if record.funcName and record.filename:
                                return f"{record.funcName}() | {os.path.basename(record.filename)}:{record.lineno}"
                            return "unknown"
                        except Exception:
                            return "unknown"

                ew_handler = ErrorWarningHandler()
                ew_handler.setLevel(logging.WARNING)
                self.logger.addHandler(ew_handler)
            except Exception as e:
                self._internal_logger.error(f"Error/Warning handler error: {e}")

        except Exception as e:
            self._internal_logger.error(f"CRITICAL config error: {e}")
            raise

    def debug(self, msg: str, *args, **kwargs):
        try:
            source = self._get_source_info()
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['source'] = source
            self.logger.debug(msg, *args, **kwargs)
        except Exception as e:
            self._internal_logger.error(f"Debug failed: {e}")

    def info(self, msg: str, *args, **kwargs):
        try:
            source = self._get_source_info()
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['source'] = source
            self.logger.info(msg, *args, **kwargs)
        except Exception as e:
            self._internal_logger.error(f"Info failed: {e}")

    def warning(self, msg: str, *args, **kwargs):
        try:
            source = self._get_source_info()
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['source'] = source
            self.logger.warning(msg, *args, **kwargs)
        except Exception as e:
            self._internal_logger.error(f"Warning failed: {e}")

    def error(self, msg: str, *args, **kwargs):
        try:
            source = self._get_source_info()
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['source'] = source
            self.logger.error(msg, *args, **kwargs)
            if self._sound_manager:
                self._sound_manager.play_sound('ERROR')
        except Exception as e:
            self._internal_logger.error(f"Error failed: {e}")

    def critical(self, msg: str, *args, **kwargs):
        try:
            source = self._get_source_info()
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['source'] = source
            self.logger.critical(msg, *args, **kwargs)
            if self._sound_manager:
                self._sound_manager.play_sound('CRITICAL')
        except Exception as e:
            self._internal_logger.error(f"Critical failed: {e}")

    def exception(self, msg: str, *args, **kwargs):
        try:
            source = self._get_source_info()
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['source'] = source
            self.logger.exception(msg, *args, **kwargs)
            if self._sound_manager:
                self._sound_manager.play_sound('ERROR')
        except Exception as e:
            self._internal_logger.error(f"Exception failed: {e}")

    def trade(self, msg: str, trade_data: Dict[str, Any] = None, *args, **kwargs):
        try:
            source = self._get_source_info()
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['source'] = source
            self.logger.info(f"📊 TRADE: {msg}", *args, **kwargs)
            if self.enable_telegram and RUBIKA_AVAILABLE:
                try:
                    if trade_data:
                        send_rubika_message(f"📊 *Trade Alert*\n🖥️ Machine: {MACHINE_ID}\n{msg}\n{json.dumps(trade_data, indent=2)}")
                    else:
                        send_rubika_message(f"📊 *Trade Alert*\n🖥️ Machine: {MACHINE_ID}\n{msg}")
                except Exception:
                    pass
        except Exception as e:
            self._internal_logger.error(f"Trade failed: {e}")

    def send_log_file(self, file_path: Optional[str] = None) -> bool:
        """
        🆕 اگر file_path=None باشد، فایل continuous مربوط به همین لاگر را می‌فرستد
        (بر اساس نوع لاگر: trade/train/strategy_mgm/learning_penalty_mgm/...)
        """
        if not self.enable_telegram or not RUBIKA_AVAILABLE:
            return False
        if file_path is None:
            logger_type = detect_logger_type(self.name)
            filename = build_continuous_filename(logger_type)
            file_path = str(Path(LOG_DIR) / filename)

        if os.path.exists(file_path):
            return send_log_file_to_rubika(file_path)
        return False

    def send_all_log_files(self) -> int:
        manager = self.get_manager()
        if manager:
            return manager.force_send_all_files()
        return 0

    def send_database(self) -> bool:
        db_sender = self.get_db_sender()
        if db_sender:
            return db_sender.force_send_database()
        return False

    def get_db_status(self) -> Dict[str, Any]:
        db_sender = self.get_db_sender()
        if db_sender:
            return db_sender.get_status()
        return {"error": "Database sender not initialized"}

    def get_error_log_path(self) -> str:
        ew_logger = self.get_error_warning_logger()
        if ew_logger:
            return ew_logger.get_error_file_path()
        return ""

    def get_warning_log_path(self) -> str:
        ew_logger = self.get_error_warning_logger()
        if ew_logger:
            return ew_logger.get_warning_file_path()
        return ""

    def get_error_count(self) -> int:
        ew_logger = self.get_error_warning_logger()
        if ew_logger:
            return ew_logger.get_error_count()
        return 0

    def get_warning_count(self) -> int:
        ew_logger = self.get_error_warning_logger()
        if ew_logger:
            return ew_logger.get_warning_count()
        return 0

    def clear_error_log(self) -> bool:
        ew_logger = self.get_error_warning_logger()
        if ew_logger:
            return ew_logger.clear_error_log()
        return False

    def clear_warning_log(self) -> bool:
        ew_logger = self.get_error_warning_logger()
        if ew_logger:
            return ew_logger.clear_warning_log()
        return False

    def send_checkpoint(self, symbol: str, timeframe: str, fold, **kwargs) -> bool:
        if not self.enable_telegram or not RUBIKA_AVAILABLE:
            return False
        return send_checkpoint(symbol, timeframe, fold, **kwargs)

    def send_deployment_package(self, symbol: str, timeframe: str, fold, **kwargs) -> bool:
        if not self.enable_telegram or not RUBIKA_AVAILABLE:
            return False
        return send_deployment_package(symbol, timeframe, fold, **kwargs)

# =====================[ PHASE 13: HELPER FUNCTIONS ]=====================
def setup_logger(name: str = "runtime", enable_telegram: bool = True) -> AdvancedTradeLogger:
    """Setup and return logger instance"""
    return AdvancedTradeLogger(name, enable_telegram)

def get_manager() -> Optional[ContinuousLogManager]:
    """Get global continuous log manager instance"""
    return AdvancedTradeLogger.get_manager()

def get_db_sender() -> Optional[DatabaseSender]:
    """Get global database sender instance"""
    return AdvancedTradeLogger.get_db_sender()

def get_error_warning_logger() -> Optional[ErrorWarningLogger]:
    """Get global error/warning logger instance"""
    return AdvancedTradeLogger.get_error_warning_logger()

def force_send_all_logs() -> int:
    """Force send all log files (including strategy_mgm & learning_penalty_mgm)"""
    manager = get_manager()
    if manager:
        return manager.force_send_all_files()
    return 0

def force_send_database() -> bool:
    """Force send database"""
    db_sender = get_db_sender()
    if db_sender:
        return db_sender.force_send_database()
    return False

def get_db_status() -> Dict[str, Any]:
    """Get database status"""
    db_sender = get_db_sender()
    if db_sender:
        return db_sender.get_status()
    return {"error": "Database sender not initialized"}

def get_log_file_path(log_type: str) -> Optional[Path]:
    """
    🆕 مسیر فایل لاگ پیوسته را برای یک نوع مشخص برمی‌گرداند.
    مثال: get_log_file_path("strategy_mgm") → logs/strategy_mgm_<MACHINE>_continuous.log
    """
    log_type = (log_type or "").lower()
    if log_type not in LOGGER_TYPE_MAPPING:
        log_type = "runtime"
    filename = build_continuous_filename(log_type)
    path = Path(LOG_DIR) / filename
    if path.exists():
        return path
    return None

def list_all_log_files() -> List[Path]:
    """🆕 لیست کامل تمام فایل‌های لاگ پیوسته موجود"""
    manager = get_manager()
    if manager:
        return manager._get_all_continuous_log_files()
    return []

def get_current_log_file() -> Optional[Path]:
    """Get current continuous runtime log file path"""
    return get_log_file_path("runtime")

def get_rotation_status() -> Dict[str, Any]:
    """Get continuous send status"""
    manager = get_manager()
    if manager:
        return {
            "machine_id": MACHINE_ID,
            "running": manager._running,
            "last_send_time": manager._last_send_time.isoformat() if manager._last_send_time else None,
            "interval_seconds": CONTINUOUS_SEND_CONFIG["interval_seconds"],
            "send_to_rubika": CONTINUOUS_SEND_CONFIG["send_to_rubika"],
            "send_to_eitaa": CONTINUOUS_SEND_CONFIG["send_to_eitaa"],
            "send_all_files": CONTINUOUS_SEND_CONFIG["send_all_log_files"],
            "supported_log_types": list(LOGGER_TYPE_MAPPING.keys()),
        }
    return {"error": "Manager not initialized"}

def get_error_file_path() -> str:
    """Get error log file path"""
    ew_logger = get_error_warning_logger()
    if ew_logger:
        return ew_logger.get_error_file_path()
    return ""

def get_warning_file_path() -> str:
    """Get warning log file path"""
    ew_logger = get_error_warning_logger()
    if ew_logger:
        return ew_logger.get_warning_file_path()
    return ""

def get_error_count() -> int:
    """Get error count"""
    ew_logger = get_error_warning_logger()
    if ew_logger:
        return ew_logger.get_error_count()
    return 0

def get_warning_count() -> int:
    """Get warning count"""
    ew_logger = get_error_warning_logger()
    if ew_logger:
        return ew_logger.get_warning_count()
    return 0

def clear_error_log() -> bool:
    """Clear error log"""
    ew_logger = get_error_warning_logger()
    if ew_logger:
        return ew_logger.clear_error_log()
    return False

def clear_warning_log() -> bool:
    """Clear warning log"""
    ew_logger = get_error_warning_logger()
    if ew_logger:
        return ew_logger.clear_warning_log()
    return False

# =====================[ PHASE 14: PRINT REDIRECT ]=====================
_original_print = builtins.print

def redirect_prints_to_logger(logger_name: str = "runtime", max_rate_per_second: int = 100):
    """Redirect print statements to logger"""
    message_history = deque(maxlen=100)
    last_message_count = {}
    logger_obj = AdvancedTradeLogger(logger_name)
    runtime_logger = logger_obj.logger

    def print_to_logger(*args, **kwargs):
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        file = kwargs.get('file', None)
        message = sep.join(str(arg) for arg in args) + end

        if file is not None:
            return _original_print(*args, **kwargs)

        current_time = time.time()
        message_hash = hash(message[:100])
        if message_hash in last_message_count:
            if current_time - last_message_count[message_hash][0] < 1.0:
                last_message_count[message_hash][1] += 1
                if last_message_count[message_hash][1] > max_rate_per_second:
                    return
            else:
                last_message_count[message_hash] = [current_time, 1]
        else:
            last_message_count[message_hash] = [current_time, 1]

        try:
            all_frames = inspect.stack()
            real_frame = None
            for frame_info in all_frames:
                filename = frame_info.filename
                func_name = frame_info.function
                if func_name == 'print_to_logger' or func_name == 'redirect_prints_to_logger':
                    continue
                if os.path.basename(filename) == 'logger.py':
                    continue
                if func_name in {'debug', 'info', 'warning', 'error', 'critical', 'exception', 'log', '_get_source_info'}:
                    continue
                real_frame = frame_info
                break

            if real_frame:
                frame = real_frame.frame
                filename = os.path.basename(real_frame.filename)
                func_name = real_frame.function
                lineno = real_frame.lineno

                class_name = ""
                if 'self' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['self'].__class__.__name__
                    except Exception:
                        pass
                if not class_name and 'cls' in frame.f_locals:
                    try:
                        class_name = frame.f_locals['cls'].__name__
                    except Exception:
                        pass

                if class_name:
                    source = f"{class_name}.{func_name}() | {filename}:{lineno}"
                elif func_name and func_name != "<module>":
                    source = f"{func_name}() | {filename}:{lineno}"
                else:
                    source = f"{filename}:{lineno}"
            else:
                source = "unknown"
        except Exception:
            source = "unknown"

        runtime_logger.info(f"[PRINT] {message.rstrip()}", extra={'source': source})
        _original_print(*args, **kwargs)

    builtins.print = print_to_logger
    return runtime_logger

def restore_original_print():
    """Restore original print function"""
    builtins.print = _original_print
    print("🔄 Print restored")

# =====================[ PHASE 15: CLEANUP ]=====================
def cleanup():
    """Cleanup on exit"""
    try:
        manager = get_manager()
        if manager:
            manager.stop()
            print("🔄 [Cleanup] Continuous Log Manager stopped")

        db_sender = get_db_sender()
        if db_sender:
            db_sender.stop()
            print("🔄 [Cleanup] Database sender stopped")
    except Exception:
        pass

atexit.register(cleanup)

# =====================[ PHASE 16: INITIALIZATION ]=====================
logger = AdvancedTradeLogger("runtime", enable_telegram=True)

try:
    redirect_prints_to_logger("runtime", max_rate_per_second=50)
except Exception as e:
    print(f"⚠️ Print redirect error: {e}")

logger.info("🚀 Logging system initialized")
logger.info(f"🖥️ Machine ID: {MACHINE_ID}")
logger.info(f"⏰ Continuous Send Interval: {CONTINUOUS_SEND_CONFIG['interval_seconds'] // 3600} hours")
logger.info(f"📋 Rubika: {CONTINUOUS_SEND_CONFIG['send_to_rubika']}, Eitaa: {CONTINUOUS_SEND_CONFIG['send_to_eitaa']}")
logger.info(f"📁 Send all continuous files: {CONTINUOUS_SEND_CONFIG['send_all_log_files']}")
logger.info(f"🗄️ Database sender: enabled, interval {DATABASE_SENDER_CONFIG['interval_seconds'] // 3600} hours")
logger.info(f"📁 Error log: {get_error_file_path()}")
logger.info(f"📁 Warning log: {get_warning_file_path()}")
logger.info(f"🧩 Supported log types: {list(LOGGER_TYPE_MAPPING.keys())}")

print("✅ Core logger loaded successfully")
print(f"🖥️ Machine ID: {MACHINE_ID}")
print(f"📁 Current continuous log file: {get_current_log_file()}")
print(f"🗄️ Database sender status: {get_db_status()}")
print(f"❌ Error log file: {get_error_file_path()}")
print(f"⚠️ Warning log file: {get_warning_file_path()}")
print(f"🧩 Supported log types: {list(LOGGER_TYPE_MAPPING.keys())}")

# =====================[ PHASE 17: TEST EXECUTION ]=====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(f"🧪 TEST: Advanced Trade Logger v12.0 | Machine: {MACHINE_ID}")
    print("=" * 80)

    test_logger = setup_logger("test_complete", enable_telegram=False)

    def test_function():
        test_logger.info("✅ This is a test info message")
        test_logger.warning("⚠️ This is a test warning message")
        test_logger.error("❌ This is a test error message")
        test_logger.debug("🐛 This is a test debug message")
        test_logger.critical("🔥 This is a test critical message")

        def nested_function():
            test_logger.info("📌 This is from nested function")
            test_logger.warning("⚠️ This is a warning from nested function")
            test_logger.error("❌ This is an error from nested function")
        nested_function()

    print("\n📝 Writing test logs with full source info...")
    print("📋 Expected format: timestamp | level | logger | Class.method() | file.py:line | message")
    print("-" * 80)
    test_function()

    class TestClass:
        def __init__(self):
            self.logger = setup_logger("test_class", enable_telegram=False)

        def test_method(self):
            self.logger.info("📌 This is from TestClass.test_method()")
            self.logger.warning("⚠️ Warning from TestClass")
            self.logger.error("❌ Error from TestClass")

            def inner_method():
                self.logger.info("📌 This is from inner method inside TestClass")
            inner_method()

    test_obj = TestClass()
    test_obj.test_method()

    # 🆕 تست لاگرهای جدید
    print("\n" + "-" * 80)
    print("🎯 Testing NEW loggers: strategy_mgm & learning_penalty_mgm")
    print("-" * 80)
    strategy_logger = setup_logger("strategy_mgm", enable_telegram=False)
    strategy_logger.info("🎯 Strategy Manager initialized")
    strategy_logger.info("🎯 Loading strategies from DB...")
    strategy_logger.warning("⚠️ Strategy 'X' deprecated")
    strategy_logger.error("❌ Strategy 'Y' failed to load")

    penalty_logger = setup_logger("learning_penalty_mgm", enable_telegram=False)
    penalty_logger.info("🧠 Learning Penalty Manager initialized")
    penalty_logger.info("🧠 Applying penalty to model #42")
    penalty_logger.warning("⚠️ Penalty threshold exceeded for model #7")
    penalty_logger.error("❌ Failed to persist penalty state")

    print("\n" + "-" * 80)
    print("📊 Status:")
    status = get_rotation_status()
    print(f"   Machine ID: {status.get('machine_id', 'N/A')}")
    print(f"   Running: {status.get('running', False)}")
    print(f"   Send Interval: {status.get('interval_seconds', 0) // 3600} hours")
    print(f"   Supported Log Types: {status.get('supported_log_types', [])}")
    print(f"   Error count: {get_error_count()}")
    print(f"   Warning count: {get_warning_count()}")
    print("\n📁 Files in logs/:")
    for f in list_all_log_files():
        print(f"   ├─ {f.name}")
    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print(f"📁 Check log files in 'logs/' directory for continuous storage (Machine: {MACHINE_ID})")
    print("=" * 80)