# ============================================================================
# SECTION 1: IMPORTS
# ============================================================================

import asyncio
import sys
import os
import time
import json
import traceback
import argparse
import signal
import glob
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core imports
from core.imports import *
from core.logger import setup_logger, AdvancedTradeLogger, redirect_prints_to_logger
tel_log = setup_logger("trade") #ارسال گزارشهای متنی به روبیکا
logger = AdvancedTradeLogger("trade")
redirect_prints_to_logger("trade", max_rate_per_second=50)
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv(override=True)
# ============================================================================
# SECTION 2: DEFAULT CREDENTIALS (HARDCODED IN MAIN ONLY)
# ============================================================================
MT5_LOGIN_RAW = os.getenv("MT5_LOGIN")
MT5_LOGIN = None
if MT5_LOGIN_RAW:
    try:
        MT5_LOGIN = int(MT5_LOGIN_RAW)
    except (ValueError, TypeError) as e:
        logger.error(f"❌ MT5_LOGIN is not a valid integer: {MT5_LOGIN_RAW} | {e}")
else:
    logger.warning("⚠️ MT5_LOGIN not found in .env — will rely on cached credentials")

MT5_PASSWORD = os.getenv("MT5_PASSWORD")
if not MT5_PASSWORD:
    logger.warning("⚠️ MT5_PASSWORD not found in .env — will rely on cached credentials")

MT5_SERVERS_RAW = os.getenv("MT5_SERVERS")
MT5_SERVERS = []
if MT5_SERVERS_RAW:
    MT5_SERVERS = [s.strip() for s in MT5_SERVERS_RAW.split(",") if s.strip()]
else:
    logger.warning("⚠️ MT5_SERVERS not found in .env — will rely on cached credentials")

DEFAULT_MT5_LOGIN = MT5_LOGIN
DEFAULT_MT5_PASSWORD = MT5_PASSWORD
DEFAULT_MT5_SERVERS = MT5_SERVERS
DEFAULT_MT5_PATH = None  # Will be auto-detected

# Training configuration
TOTAL_FOLDS = 5
NUM_FOLDS_TO_PROCESS = 1
folds = TOTAL_FOLDS
sum_folds = TOTAL_FOLDS

# Symbol configuration
DEFAULT_SYMBOL = "XAUUSDs" #XAUUSDs, GOLD vs USD
# ============================================================================
# SECTION 3: MT5 PATH AUTO-DETECTION
# ============================================================================

def find_mt5_path() -> Optional[str]:
    """
    پیدا کردن خودکار مسیر نصب متاتریدر و در صورت نیاز، اجرای آن
    ✅ بدون هاردکد - جستجو در مسیرهای استاندارد و استفاده از .env
    ✅ ذخیره مسیر پیدا شده در فایل cache
    ✅ FIXED: برگرداندن مسیر کامل executable به صورت string
    """
    import platform
    import glob
    import subprocess
    from dotenv import load_dotenv
    load_dotenv()

    phase = "FIND_MT5_PATH"

    # ⭐ اولویت اول: خواندن از محیط (.env)
    mt5_path_env = os.getenv("MT5_PATH")
    if mt5_path_env and isinstance(mt5_path_env, str) and os.path.exists(mt5_path_env):
        logger.info(f"✅ MT5 path loaded from .env: {mt5_path_env}")
        return mt5_path_env

    # ⭐ بررسی فایل کش برای مسیر ذخیره شده
    cache_path_file = Path("TRADE/mt5_path.json")
    if cache_path_file.exists():
        try:
            with open(cache_path_file, 'r') as f:
                cached_data = json.load(f)
                cached_path = cached_data.get('mt5_path')
                if cached_path and isinstance(cached_path, str) and os.path.exists(cached_path):
                    logger.info(f"✅ MT5 path loaded from cache: {cached_path}")
                    return cached_path
        except Exception as e:
            logger.debug(f"Could not load cached MT5 path: {e}")

    system = platform.system()
    mt5_path = None

    try:
        if system == "Windows":
            # ========== روش 1: جستجو در مسیرهای استاندارد ==========
            possible_paths = [
                "C:\\Program Files\\MetaTrader 5\\terminal64.exe",
                "C:\\Program Files\\MetaTrader 5\\terminal.exe",
                "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
                "C:\\Program Files (x86)\\MetaTrader 5\\terminal.exe",
                "D:\\met\\MetaTrader 5\\terminal64.exe",
                "D:\\MetaTrader 5\\terminal64.exe",
                "J:\\MetaTrader 5\\terminal64.exe",
                os.path.expandvars("%PROGRAMFILES%\\MetaTrader 5\\terminal64.exe"),
                os.path.expandvars("%PROGRAMFILES(X86)%\\MetaTrader 5\\terminal64.exe"),
            ]

            for path in possible_paths:
                try:
                    if '*' in path:
                        matches = glob.glob(path)
                        if matches:
                            mt5_path = matches[0]
                            if os.path.exists(mt5_path):
                                break
                    elif os.path.exists(path):
                        mt5_path = path
                        break
                except Exception as e:
                    logger.debug(f"Error checking path {path}: {e}")
                    continue

            # ========== روش 2: جستجو در Registry ==========
            if mt5_path is None:
                try:
                    import winreg
                    keys = [
                        r"SOFTWARE\MetaQuotes\MetaTrader 5",
                        r"SOFTWARE\WOW6432Node\MetaQuotes\MetaTrader 5",
                    ]
                    for key_path in keys:
                        try:
                            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path)
                            install_path = winreg.QueryValueEx(key, "InstallPath")[0]
                            if install_path:
                                mt5_path = os.path.join(install_path, "terminal64.exe")
                                if os.path.exists(mt5_path):
                                    break
                                mt5_path = os.path.join(install_path, "terminal.exe")
                                if os.path.exists(mt5_path):
                                    break
                        except Exception:
                            pass
                except Exception:
                    pass

        elif system == "Darwin":  # macOS
            possible_paths = [
                "/Applications/MetaTrader 5.app/Contents/MacOS/MetaTrader 5",
                "/Applications/MT5.app/Contents/MacOS/MT5",
                os.path.expanduser("~/Applications/MetaTrader 5.app/Contents/MacOS/MetaTrader 5"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    mt5_path = path
                    break

        elif system == "Linux":
            possible_paths = [
                "/opt/metatrader5/terminal",
                "/opt/mt5/terminal",
                "~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe",
            ]
            for path in possible_paths:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    mt5_path = expanded
                    break

    except Exception as e:
        logger.error(f"[{phase}] ❌ Error detecting MT5 path: {e}")
        mt5_path = None

    # ========== اعتبارسنجی و ذخیره مسیر ==========
    final_path = None
    if mt5_path and isinstance(mt5_path, str) and os.path.exists(mt5_path):
        final_path = mt5_path
        logger.info(f"✅ MT5 path auto-detected: {final_path}")

        # ذخیره در کش
        try:
            cache_path_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path_file, 'w') as f:
                json.dump({'mt5_path': final_path, 'detected_at': datetime.now().isoformat()}, f, indent=2)
            logger.info(f"✅ MT5 path saved to cache: {final_path}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save MT5 path to cache: {e}")
    else:
        logger.warning("⚠️ Could not auto-detect MT5 path")

    return final_path

# ============================================================================
# SECTION 4: LOGGER SETUP
# ============================================================================

tel_log = setup_logger("trade")
redirect_prints_to_logger()
logger = AdvancedTradeLogger(__name__)


class AutoMT5:
    _last_status = False
    _running = False
    _lock = threading.Lock()
    _thread: Optional[threading.Thread] = None
    _connection_counter = 0
    _last_connection_time = None
    _consecutive_failures = 0
    _max_consecutive_failures = 10
    _reconnect_delay = 30
    _max_reconnect_delay = 300
    _credential_file = Path("TRADE/mt5_credentials.json")
    _offline_file = Path("cache/offline_status.json")
    _is_offline_announced = False

    _RAW_DATA_DIR = Path("data/raw/mt5")

    @classmethod
    def _has_raw_data(cls, symbol: str = None) -> bool:
        """بررسی وجود داده خام در دیسک"""
        try:
            if not cls._RAW_DATA_DIR.exists():
                return False

            if symbol:
                pattern = f"{symbol}_*_raw.h5"
                matches = list(cls._RAW_DATA_DIR.glob(pattern))
                if matches:
                    logger.debug(f"📁 Found {len(matches)} raw data files for {symbol}")
                    return True
            else:
                matches = list(cls._RAW_DATA_DIR.glob("*_raw.h5"))
                if matches:
                    logger.debug(f"📁 Found {len(matches)} raw data files in total")
                    return True

            return False

        except Exception as e:
            logger.debug(f"Error checking raw data: {e}")
            return False

    @classmethod
    def _load_credentials_from_file(cls) -> Optional[Dict[str, Any]]:
        """بارگذاری credentials از فایل کش"""
        phase = "MT5_MGR_LOAD_CREDS"
        try:
            if not cls._credential_file.exists():
                logger.debug(f"[{phase}] Credentials file not found: {cls._credential_file}")
                return None

            with open(cls._credential_file, 'r') as f:
                cred_data = json.load(f)

            login = cred_data.get('login')
            password = cred_data.get('password')
            server = cred_data.get('server')
            mt5_path = cred_data.get('mt5_path')
            connected_at = cred_data.get('connected_at')

            if login and password and server:
                logger.debug(f"[{phase}] Loaded credentials: login={login}, server={server}")
                return {
                    'login': str(login),
                    'password': password,
                    'server': server,
                    'mt5_path': mt5_path
                }

            return None

        except json.JSONDecodeError as e:
            logger.debug(f"[{phase}] Invalid JSON: {e}")
            return None
        except Exception as e:
            logger.debug(f"[{phase}] Failed to load: {e}")
            return None

    @classmethod
    def _save_credentials_to_file(cls, login: str, password: str, server: str, mt5_path: str = None) -> bool:
        """ذخیره credentials در فایل کش"""
        phase = "MT5_MGR_SAVE_CREDS"
        try:
            cls._credential_file.parent.mkdir(parents=True, exist_ok=True)

            cred_data = {
                'login': login,
                'password': password,
                'server': server,
                'mt5_path': mt5_path,
                'connected_at': datetime.now(timezone.utc).isoformat()
            }

            with open(cls._credential_file, 'w') as f:
                json.dump(cred_data, f, indent=2)

            logger.debug(f"[{phase}] Credentials saved to {cls._credential_file}")
            return True
        except Exception as e:
            logger.warning(f"[{phase}] Could not save credentials: {e}")
            return False

    @classmethod
    def _clear_offline_status(cls):
        """پاک کردن وضعیت آفلاین"""
        try:
            if cls._offline_file.exists():
                cls._offline_file.unlink()
                logger.debug("AutoMT5: Offline status cleared")
        except Exception as e:
            logger.debug(f"AutoMT5: Could not clear offline status: {e}")

    @classmethod
    def _save_offline_status(cls, reason: str):
        """ذخیره وضعیت آفلاین"""
        try:
            offline_data = {
                "offline_mode": True,
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "last_online": datetime.now(timezone.utc).isoformat()
            }
            cls._offline_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cls._offline_file, 'w') as f:
                json.dump(offline_data, f, indent=2)
            logger.debug(f"AutoMT5: Offline status saved: {reason}")
        except Exception as e:
            logger.debug(f"AutoMT5: Could not save offline status: {e}")

    @classmethod
    def _get_available_symbols_from_raw(cls) -> List[str]:
        """دریافت لیست نمادهای موجود در داده خام"""
        symbols = set()
        try:
            if cls._RAW_DATA_DIR.exists():
                for file in cls._RAW_DATA_DIR.glob("*_raw.h5"):
                    name_parts = file.stem.split('_')
                    if name_parts:
                        symbols.add(name_parts[0])
            return list(symbols)
        except Exception as e:
            logger.debug(f"Error getting symbols from raw: {e}")
            return []

    @classmethod
    def start(cls, interval: int = 30, symbol: str = None) -> None:
        """شروع حلقه خودترمیم MT5"""
        with cls._lock:
            if cls._running:
                return

            # بررسی وجود داده خام
            has_raw = cls._has_raw_data(symbol)

            if has_raw:
                logger.info("📁 Raw data files found on disk - MT5 connection NOT needed")
                logger.info("💡 System will run in CACHE-ONLY mode (no MT5 checks)")
                cls._running = False
                return

            logger.info("🔌 No raw data found - starting MT5 self-healing loop")
            cls._running = True
            cls._connection_counter = 0
            cls._consecutive_failures = 0
            cls._reconnect_delay = 30
            cls._is_offline_announced = False
            cls._thread = threading.Thread(
                target=cls._loop, args=(interval, symbol), daemon=True, name="AutoMT5"
            )
            cls._thread.start()
            logger.info("🔄 AutoMT5 self-healing loop started (%ss)", interval)

    @classmethod
    def stop(cls) -> None:
        """متوقف کردن حلقه"""
        with cls._lock:
            if not cls._running:
                return
            cls._running = False
            if cls._thread and cls._thread.is_alive():
                cls._thread.join(timeout=5)
            logger.info("🛑 AutoMT5 loop stopped")

    @classmethod
    def _loop(cls, interval: int, symbol: str = None) -> None:
        """حلقه self-healing - با مدیریت خطا و لاگ محدود"""
        logger.info("AutoMT5 loop thread started")

        # بررسی داده خام در شروع
        if cls._has_raw_data(symbol):
            logger.info("📁 Raw data detected at startup - stopping MT5 checks")
            cls._running = False
            return

        cached_login = None
        cached_password = None
        cached_server = None
        cached_path = None

        # ✅ FIXED: بارگذاری اعتبارنامه‌ها از فایل .env در اولین اجرا
        from dotenv import load_dotenv
        load_dotenv()
        
        env_login = os.getenv("MT5_LOGIN")
        env_password = os.getenv("MT5_PASSWORD")
        env_servers_raw = os.getenv("MT5_SERVERS")
        env_servers = [s.strip() for s in env_servers_raw.split(",") if s.strip()] if env_servers_raw else []
        env_path = os.getenv("MT5_PATH")
        
        if env_login and env_password and env_servers:
            logger.info(f"📋 Using credentials from .env: login={env_login}, server={env_servers[0]}")
            cached_login = env_login
            cached_password = env_password
            cached_server = env_servers[0]
            cached_path = env_path
        
        # اگر از .env بارگذاری نشد، از فایل کش استفاده کن
        if cached_login is None:
            creds = cls._load_credentials_from_file()
            if creds:
                cached_login = creds.get('login')
                cached_password = creds.get('password')
                cached_server = creds.get('server')
                cached_path = creds.get('mt5_path')
                logger.info(f"📋 Initial credentials loaded from file: login={cached_login}")

        try:
            while cls._running:
                # بررسی داده خام در هر سیکل
                if cls._has_raw_data(symbol):
                    logger.info("📁 Raw data files detected! MT5 connection no longer needed.")
                    logger.info("💡 Stopping MT5 heartbeat and reconnection attempts.")
                    cls._clear_offline_status()
                    cls._running = False
                    break

                try:
                    try:
                        import MetaTrader5 as mt5

                        MT5_AVAILABLE = True

                    except ImportError:

                        mt5 = None

                        MT5_AVAILABLE = False

                    terminal_info = None
                    if MT5_AVAILABLE:
                        try:
                            terminal_info = mt5.terminal_info()
                        except Exception:
                            pass

                    else:
                        logger.info("Running in Offline Mode")

                    if terminal_info is None or not terminal_info.connected:
                        cls._consecutive_failures += 1

                        current_delay = min(cls._reconnect_delay * (2 ** (min(cls._consecutive_failures, 5) - 1)),
                                            cls._max_reconnect_delay)

                        if cls._consecutive_failures % 5 == 1 or cls._consecutive_failures == 1:
                            logger.warning(
                                f"⚠️ MT5 disconnected – reconnecting... (attempt #{cls._consecutive_failures})")

                        if cls._consecutive_failures >= cls._max_consecutive_failures and not cls._is_offline_announced:
                            cls._is_offline_announced = True
                            logger.warning("📴 MT5 connection failed - entering offline mode")
                            cls._save_offline_status("connection lost")

                        if cls._consecutive_failures % 3 == 1:
                            fresh_creds = cls._load_credentials_from_file()
                            if fresh_creds:
                                cached_login = fresh_creds.get('login')
                                cached_password = fresh_creds.get('password')
                                cached_server = fresh_creds.get('server')
                                cached_path = fresh_creds.get('mt5_path')

                        if cached_login and cached_password and cached_server:
                            try:
                                mt5.shutdown()
                                time.sleep(0.5)

                                init_params = {
                                    "login": int(cached_login),
                                    "password": cached_password,
                                    "server": cached_server,
                                    "timeout": 10000
                                }
                                if cached_path and os.path.exists(cached_path):
                                    init_params["path"] = cached_path

                                ok = mt5.initialize(**init_params)

                                if ok:
                                    account_info = mt5.account_info()
                                    if account_info is not None:
                                        cls._last_status = True
                                        cls._connection_counter += 1
                                        cls._last_connection_time = datetime.now()
                                        cls._consecutive_failures = 0
                                        cls._reconnect_delay = 30
                                        cls._clear_offline_status()
                                        cls._is_offline_announced = False
                                        logger.info(f"✅ MT5 reconnected (total: {cls._connection_counter})")
                                    else:
                                        cls._last_status = False
                                else:
                                    cls._last_status = False
                            except Exception as e:
                                cls._last_status = False
                                logger.debug(f"AutoMT5 connection error: {e}")
                        else:
                            cls._last_status = False
                    else:
                        if cls._consecutive_failures > 0:
                            logger.info("✅ MT5 connection restored")
                        cls._consecutive_failures = 0
                        cls._reconnect_delay = 30
                        cls._last_status = True
                        cls._clear_offline_status()
                        cls._is_offline_announced = False

                except ImportError as e:
                    logger.debug(f"AutoMT5 ImportError: {e}")
                    cls._last_status = False
                except Exception as e:
                    if cls._consecutive_failures % 10 == 0:
                        logger.debug(f"AutoMT5 minor error: {e}")
                    cls._last_status = False

                if cls._last_status:
                    sleep_time = interval
                else:
                    sleep_time = min(cls._reconnect_delay * (2 ** (min(cls._consecutive_failures, 5) - 1)),
                                     cls._max_reconnect_delay)

                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"❌ AutoMT5 loop crashed: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            cls._running = False
            logger.info("🛑 AutoMT5 loop stopped")
            
    @classmethod
    def status(cls) -> bool:
        """بررسی وضعیت - اگر داده خام دارد True برگردان"""
        if cls._has_raw_data():
            return True
        return cls._last_status

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """دریافت آمار وضعیت"""
        return {
            "connected": cls.status(),
            "has_raw_data": cls._has_raw_data(),
            "raw_symbols": cls._get_available_symbols_from_raw(),
            "reconnect_count": cls._connection_counter,
            "last_connection": cls._last_connection_time.isoformat() if cls._last_connection_time else None,
            "consecutive_failures": cls._consecutive_failures,
            "running": cls._running
        }
        
class MT5ConnectionManager:
    """مدیریت اتصال داینامیک به MT5 - FULLY AUTOMATIC, NO USER INPUT"""
    from dotenv import load_dotenv
    import os
    from pathlib import Path

    def __init__(self):
        self.is_online = False
        self.offline_mode = False
        self.connection_retries = 0
        self.max_retries = 3
        self.offline_data = {}
        self._last_heartbeat = None
        self._heartbeat_interval = 60
        self._reconnect_lock = threading.Lock()
        self._is_reconnecting = False
        self._heartbeat_running = False
        self._heartbeat_thread = None

        # مسیر دایرکتوری داده خام
        self._RAW_DATA_DIR = Path("data/raw/mt5")

        # مسیر فایل credentials
        self._credential_file = Path("TRADE/mt5_credentials.json")

        # متغیرهای credentials
        self._login = None
        self._password = None
        self._server = None
        self._mt5_path = None

        # ============================================================
        # ✅ CRITICAL FIX: Load .env with absolute path
        # ============================================================

        # ✅ Find .env file in project root
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            logger.info(f"✅ Loaded .env from {env_path}")
        else:
            # Fallback: search in current directory
            load_dotenv(override=True)
            logger.info(f"✅ Loaded .env from current directory")

        # ✅ NOW read credentials (will always have values)
        env_login = os.getenv("MT5_LOGIN")
        env_password = os.getenv("MT5_PASSWORD")
        env_servers_raw = os.getenv("MT5_SERVERS")
        env_servers = [s.strip() for s in env_servers_raw.split(",") if s.strip()] if env_servers_raw else []
        env_path = os.getenv("MT5_PATH")

        # ✅ If .env values are still None, raise explicit error
        if not env_login or not env_password or not env_servers:
            logger.error("❌ CRITICAL: MT5 credentials not found in .env! Cannot continue.")
            raise RuntimeError("MT5 credentials not found in .env file")

        self._login = str(env_login)
        self._password = env_password
        self._server = env_servers[0]
        self._mt5_path = env_path

        # ذخیره credentials در کش
        self._save_credentials_to_file()

        # فقط اگر داده خام ندارد، heartbeat را استارت بزن
        if not self._has_raw_data():
            self._start_heartbeat()
        else:
            logger.info("📁 Raw data exists - MT5 heartbeat NOT started")
            self.offline_mode = True
            self.is_online = False

    def _has_raw_data(self, symbol: str = None) -> bool:
        """بررسی وجود داده خام در دیسک"""
        try:
            if not self._RAW_DATA_DIR.exists():
                return False

            if symbol:
                pattern = f"{symbol}_*_raw.h5"
                matches = list(self._RAW_DATA_DIR.glob(pattern))
                return len(matches) > 0
            else:
                matches = list(self._RAW_DATA_DIR.glob("*_raw.h5"))
                return len(matches) > 0
        except Exception:
            return False

    def _load_credentials_from_file(self) -> bool:
        """بارگذاری credentials از فایل کش"""
        phase = "MT5_MGR_LOAD_CREDS"
        try:
            if not self._credential_file.exists():
                logger.debug(f"[{phase}] Credentials file not found: {self._credential_file}")
                return False

            with open(self._credential_file, 'r') as f:
                cred_data = json.load(f)

            login = cred_data.get('login')
            password = cred_data.get('password')
            server = cred_data.get('server')
            mt5_path = cred_data.get('mt5_path')

            if login and password and server:
                self._login = str(login)
                self._password = password
                self._server = server
                self._mt5_path = mt5_path
                logger.debug(f"[{phase}] Loaded credentials: login={self._login}, server={self._server}")
                return True

            return False

        except json.JSONDecodeError as e:
            logger.debug(f"[{phase}] Invalid JSON: {e}")
            return False
        except Exception as e:
            logger.debug(f"[{phase}] Failed to load: {e}")
            return False

    def _save_credentials_to_file(self) -> bool:
        """ذخیره credentials در فایل برای استفاده بعدی"""
        phase = "MT5_MGR_SAVE_CREDS"
        try:
            self._credential_file.parent.mkdir(parents=True, exist_ok=True)

            cred_data = {
                'login': self._login,
                'password': self._password,
                'server': self._server,
                'mt5_path': self._mt5_path,
                'connected_at': datetime.now().isoformat()
            }

            with open(self._credential_file, 'w') as f:
                json.dump(cred_data, f, indent=2)

            logger.debug(f"[{phase}] Credentials saved to {self._credential_file}")
            return True
        except Exception as e:
            logger.warning(f"[{phase}] Could not save credentials: {e}")
            return False

    def _start_heartbeat(self):
        """شروع heartbeat - با interval بالاتر و لاگ محدود"""
        phase = "HEARTBEAT_START"
        try:
            if self._heartbeat_running:
                return

            self._heartbeat_running = True
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="MT5Heartbeat")
            self._heartbeat_thread.start()
            logger.debug("💓 MT5 Heartbeat thread started")
        except Exception as e:
            logger.error(f"[{phase}] Failed to start heartbeat: {e}")
            self._heartbeat_running = False

    def _heartbeat_loop(self):
        """حفظ اتصال - نسخه غیرمزاحم با لاگ محدود"""
        consecutive_failures = 0

        try:
            while self._heartbeat_running:
                try:
                    if self.is_online:
                        try:
                            import MetaTrader5 as mt5
                            account_info = mt5.account_info()
                        except ImportError:
                            mt5 = None
                            account_info = None
                        
                        if account_info is None:
                            consecutive_failures += 1
                            if consecutive_failures >= 10:
                                self.is_online = False
                                consecutive_failures = 0
                        else:
                            self._last_heartbeat = datetime.now()
                            consecutive_failures = 0
                    time.sleep(self._heartbeat_interval)
                except Exception:
                    time.sleep(5)
        except Exception as e:
            logger.error(f"Heartbeat loop crashed: {e}")
        finally:
            logger.debug("Heartbeat loop stopped")
            self._heartbeat_running = False

    def check_connection(self, timeout: int = 30, ask_user: bool = False) -> bool:
        """بررسی اتصال به MT5 - FULLY AUTOMATIC, NO USER INPUT"""

        # ✅ If raw data exists, return True (no MT5 needed)
        if self._has_raw_data():
            if not self.offline_mode:
                logger.info("📁 Raw data available - using cached data mode (no MT5 needed)")
                self.offline_mode = True
                self.is_online = False
            return True

        # ✅ If offline mode, silently return
        if self.offline_mode:
            return True

        # ✅ If already online, quick test
        if self.is_online:
            try:
                import MetaTrader5 as mt5
                account_info = mt5.account_info()
                if account_info is not None:
                    return True
                else:
                    self.is_online = False
            except:
                self.is_online = False

        with self._reconnect_lock:
            if self._is_reconnecting:
                return self.is_online
            self._is_reconnecting = True

        try:
            for attempt in range(self.max_retries):
                try:
                    import MetaTrader5 as mt5
                    
                    if attempt == 0:
                        logger.debug(f"Checking MT5 connection...")
                    
                    mt5.shutdown()
                    time.sleep(0.5)

                    # ✅ Build init params with credentials (NO hardcoded values)
                    init_params = {
                        "login": int(self._login),      # ✅ From .env
                        "password": self._password,     # ✅ From .env
                        "server": self._server,         # ✅ From .env
                        "timeout": timeout * 1000
                    }

                    if self._mt5_path and os.path.exists(self._mt5_path):
                        init_params["path"] = self._mt5_path

                    # ✅ Initialize MT5 with credentials (NO user input)
                    if mt5.initialize(**init_params):
                        account_info = mt5.account_info()
                        if account_info is not None:
                            self.is_online = True
                            self.offline_mode = False
                            self.connection_retries = 0
                            self._last_heartbeat = datetime.now()
                            
                            if attempt > 0:
                                logger.info(f"✅ MT5 reconnected")
                            else:
                                logger.debug(f"✅ MT5 connection OK")
                            
                            self._save_credentials_to_file()
                            return True
                    else:
                        if attempt == 0:
                            logger.debug(f"MT5 connection failed, will retry silently")

                except Exception as e:
                    if attempt == 0:
                        logger.debug(f"MT5 connection error: {e}")

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        finally:
            with self._reconnect_lock:
                self._is_reconnecting = False

        self.is_online = False
        return False

    def set_credentials(self, login: str, password: str, server: str):
        """تنظیم credentials برای اتصال (override)"""
        phase = "SET_CREDENTIALS"
        try:
            with self._reconnect_lock:
                if login:
                    self._login = str(login)
                if password:
                    self._password = password
                if server:
                    self._server = server
                logger.info(f"[{phase}] MT5 credentials updated: login={self._login}")

                self._save_credentials_to_file()
        except Exception as e:
            logger.error(f"[{phase}] Failed to set credentials: {e}")

    def get_credentials(self) -> Tuple[str, str, str]:
        """دریافت credentials فعلی"""
        return self._login, self._password, self._server

    def get_status(self) -> Dict[str, Any]:
        """دریافت وضعیت فعلی اتصال - با اطلاعات داده خام"""
        return {
            "is_online": self.is_online,
            "offline_mode": self.offline_mode,
            "has_raw_data": self._has_raw_data(),
            "connection_retries": self.connection_retries,
            "login": self._login,
            "server": self._server,
            "mt5_path": self._mt5_path
        }

    def force_reconnect(self) -> bool:
        """بازگرداندن اجباری اتصال - فقط در صورت نیاز"""
        logger.info("Force reconnecting to MT5...")
        self.is_online = False
        return self.check_connection(timeout=60, ask_user=False)

    def stop_heartbeat(self):
        """متوقف کردن heartbeat thread"""
        try:
            self._heartbeat_running = False
            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=5)
            logger.debug("Heartbeat thread stopped")
        except Exception:
            pass

mt5_connection_manager = MT5ConnectionManager()


# ============================================================================
# SECTION 7: BOT HEALTH MONITOR
# ============================================================================

class BotHealthMonitor:
    def __init__(self, idle_threshold: int = 600, check_interval: int = 600,
                 mt5_check_interval: int = 10):

        logger.info("🔧 Initializing BotHealthMonitor")

        self.last_activity = datetime.now()
        self.active = False
        self._thread = None
        self._idle_threshold = idle_threshold
        self._check_interval = check_interval
        self._mt5_check_interval = mt5_check_interval
        self._started_once = False
        self._lock = threading.RLock()

    def start(self):
        import threading
        with self._lock:
            if self.active or self._started_once:
                return
            self._started_once = True
            self.active = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="HealthMonitor")
            self._thread.start()
            logger.info("🔄 Health monitor started")
            try:
                AutoMT5.start(self._mt5_check_interval)
            except Exception as e:
                logger.warning(f"⚠️ AutoMT5 start error: {e}")

    def _monitor_loop(self):
        loop_counter = 0
        try:
            while self.active:
                loop_counter += 1
                idle_time = (datetime.now() - self.last_activity).total_seconds()
                if idle_time > self._idle_threshold:
                    logger.warning(f"⚠️ Idle for {int(idle_time)}s")
                time.sleep(self._check_interval)
        except Exception as e:
            logger.error(f"❌ Monitor loop error: {e}")
        finally:
            self.active = False

    def stop(self):
        try:
            self.active = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5)
            try:
                AutoMT5.stop()
            except:
                pass
            logger.info("✅ Health monitor stopped")
        except Exception as e:
            logger.error(f"❌ Stop error: {e}")

    def mark(self, stage: Optional[str] = None):
        self.last_activity = datetime.now()


health = BotHealthMonitor()


# ============================================================================
# SECTION 8: GET OS TYPE
# ============================================================================

def get_os_type() -> str:
    try:
        if os.name == 'nt':
            return 'windows'
        elif os.name == 'posix':
            if sys.platform == 'darwin':
                return 'mac'
            elif 'linux' in sys.platform.lower():
                return 'linux'
            else:
                return 'posix'
        else:
            return 'unknown'
    except:
        if sys.platform.startswith('win'):
            return 'windows'
        elif sys.platform.startswith('linux'):
            return 'linux'
        elif sys.platform.startswith('darwin'):
            return 'mac'
        else:
            return 'unknown'


os_type = get_os_type()
training_types = ['custom', 'simple', 'optuna', 'jax'] if os_type != 'windows' else ['simple']


# ============================================================================
# SECTION 9: MAIN ASYNC FUNCTION
# ============================================================================

async def main():
    """تابع اصلی برنامه - با بررسی داده خام قبل از اتصال MT5"""

    try:
        parser = argparse.ArgumentParser(
            description="Financial Data Forensic Engine v6.0 - Wall Street Quant Edition"
        )
        parser.add_argument("--symbol", type=str, default=DEFAULT_SYMBOL,
                            help=f"Trading symbol (default: {DEFAULT_SYMBOL})")
        parser.add_argument("--timeframe", type=str, default="None",
                            help="Timeframe (default: None)")
        parser.add_argument("--fold", type=int, default=None,
                            help="Fold number (default: auto-detect)")
        parser.add_argument("--verbose", action="store_true",
                            help="Enable verbose logging")
        parser.add_argument("--save", type=str, default="audit_result.json",
                            help="Save path for audit results")
        parser.add_argument("--skip-tests", action="store_true",
                            help="Skip static and dynamic tests")
        parser.add_argument("--skip-training", action="store_true",
                            help="Skip model training")
        parser.add_argument("--offline", action="store_true",
                            help="Force offline mode")
        parser.add_argument("--mt5-login", type=str, default=None,
                            help="Override MT5 login")
        parser.add_argument("--mt5-password", type=str, default=None,
                            help="Override MT5 password")
        parser.add_argument("--mt5-server", type=str, default=None,
                            help="Override MT5 server")

        args = parser.parse_args()

        # ====================================================================
        # ⭐⭐⭐⭐⭐ CRITICAL FIX: ابتدا بررسی کن داده خام موجود است یا نه
        # ====================================================================
        RAW_DATA_DIR = Path("data/raw/mt5")
        has_raw_data = False
        raw_files = []
        mt5_connected = False
        mt5_connect_attempts = 0
        max_mt5_attempts = 2  # ⭐ کاهش به 2 تلاش        
        if RAW_DATA_DIR.exists():
            raw_files = list(RAW_DATA_DIR.glob(f"{args.symbol}*_raw.h5"))
            has_raw_data = len(raw_files) > 0

        if has_raw_data:
            logger.info("=" * 80)
            logger.info(f"📁 RAW DATA FOUND ON DISK - {len(raw_files)} file(s)")
            logger.info("=" * 80)
            for rf in raw_files:
                logger.info(f"   📄 {rf.name}")
            logger.info("")
            logger.info("💡 MT5 connection is NOT required. System will run in CACHE-ONLY mode.")
            logger.info("=" * 80)

            # ⭐ آفلاین کردن مستقیم connection manager
            mt5_connection_manager.offline_mode = True
            mt5_connection_manager.is_online = False

        else:
            logger.info("=" * 80)
            logger.info("🔌 INITIALIZING MT5 CONNECTION (NO RAW DATA FOUND)")
            logger.info("=" * 80)

            # Update credentials from command line if provided
            if args.mt5_login or args.mt5_password or args.mt5_server:
                mt5_connection_manager.set_credentials(
                    args.mt5_login, args.mt5_password, args.mt5_server
                )

            # Force offline mode if requested
            if args.offline:
                mt5_connection_manager.offline_mode = True
                logger.info("📴 Forced offline mode activated")

            # فقط اگر داده خام ندارد، سعی کن به MT5 وصل شوی

            while not mt5_connected and mt5_connect_attempts < max_mt5_attempts:
                mt5_connect_attempts += 1
                logger.info(f"🔄 MT5 connection attempt {mt5_connect_attempts}/{max_mt5_attempts}")

                try:
                    try:
                        import MetaTrader5 as mt5

                        MT5_AVAILABLE = True

                    except ImportError:

                        mt5 = None

                        MT5_AVAILABLE = False

                    login, password, server = mt5_connection_manager.get_credentials()
                    mt5_path = mt5_connection_manager._mt5_path
                    if MT5_AVAILABLE:
                        mt5.shutdown()
                    else:
                        logger.info("Running in Offline Mode")
                    time.sleep(0.5)

                    init_params = {
                        "login": int(login),
                        "password": password,
                        "server": server
                    }
                    if mt5_path and os.path.exists(mt5_path):
                        init_params["path"] = mt5_path

                    if mt5.initialize(**init_params):
                        account_info = mt5.account_info()
                        if account_info is not None:
                            mt5_connected = True
                            logger.info(f"✅ MT5 connected! Account: {account_info.login}")
                            mt5_connection_manager._save_credentials_to_file()
                            break
                    else:
                        error = mt5.last_error()
                        logger.warning(f"❌ MT5 connection failed: {error}")

                except Exception as e:
                    logger.error(f"❌ MT5 connection error: {e}")

                if mt5_connect_attempts < max_mt5_attempts:
                    wait_time = 5 * mt5_connect_attempts
                    logger.info(f"⏳ Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)

            if not mt5_connected:
                logger.warning("⚠️ Could not connect to MT5. System will run in offline mode if data exists.")
                mt5_connection_manager.offline_mode = True

        # ====================================================================
        # PHASE 2: Log startup info
        # ====================================================================
        login, password, server = mt5_connection_manager.get_credentials()

        logger.info("=" * 80)
        logger.info("🚀 SYSTEM STARTUP - WALL STREET QUANT EDITION v6.0")
        logger.info("=" * 80)
        logger.info(f"   Symbol: {args.symbol}")
        logger.info(f"   Timeframe: {args.timeframe}")
        logger.info(f"   OS: {os_type}")
        logger.info(f"   MT5 Login: {login}")
        logger.info(f"   MT5 Server: {server}")
        logger.info(f"   MT5 Path: {mt5_connection_manager._mt5_path or 'Auto-detect'}")
        logger.info(f"   MT5 Connected: {mt5_connected}")
        logger.info("=" * 80)

        tel_log.trade(f"""
╔══════════════════════════════════════════════════════════════╗
║              🚀 SYSTEM STARTED - WALL STREET EDITION         ║
╠══════════════════════════════════════════════════════════════╣
║  Symbol: {args.symbol}
║  Timeframe: {args.timeframe}
║  MT5 Login: {login}
║  MT5 Connected: {mt5_connected}
║  OS: {os_type}
║  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
╚══════════════════════════════════════════════════════════════╝
        """)

        # ====================================================================
        # PHASE 3: Start health monitor (AutoMT5 now uses file credentials)
        # ====================================================================
        health.start()
        # ====================================================================
        # PHASE 5: Run workflow manager
        # ====================================================================
        tel_log.trade("📊 PHASE 2: Running Workflow Manager")
        try:
            logger.info("📊 PHASE 1: Initializing DataFetcher")
            from core.data_fetcher import DataFetcher
            data_fetcher = DataFetcher()
            workflow_result = await data_fetcher.workflow_manager(
                symbol=args.symbol,
                force_fresh_start=True
            )

            if not workflow_result or not workflow_result.get("results", {}).get("workflow_completed", False):
                tel_log.trade("❌ WORKFLOW MANAGER FAILED - Check logs")
            else:
                tel_log.trade("✅ Workflow manager completed")

        except ImportError:
            tel_log.trade("❌ WORKFLOW MANAGER FAILED")

        logger.send_log_file("runtime")

        # ====================================================================
        # PHASE 6: Run static tests
        # ====================================================================
        if not args.skip_tests:
            logger.info("📊 PHASE 3: Running Static Data Tests")
            tel_log.trade("🔍 Starting static data verification...")

            try:
                from core.DATA_VERIFIER import verify_date_column_in_all_splits
                from core.global_method import get_active_symbol_timeframe
                active_symbol, active_tf = get_active_symbol_timeframe()
                if active_tf is None:
                    raise RuntimeError("Cannot run forensic audit without active timeframe")

                Static_test = await verify_date_column_in_all_splits(
                    symbol=args.symbol,
                    timeframe=active_tf,
                    fold=args.fold,
                    verbose=args.verbose,
                    save_path=args.save
                )

                logger.send_log_file("runtime")

                if Static_test:
                    tel_log.trade("✅ Static data verification passed")
                else:
                    tel_log.trade("⚠️ Static data verification had warnings")

            except ImportError:
                tel_log.trade("⚠️ Static tests skipped - module not available")
            except Exception as e:
                tel_log.trade(f"❌ Static tests failed: {str(e)[:100]}")

        # ====================================================================
        # PHASE 7: Run dynamic tests
        # ====================================================================
        if not args.skip_tests:
            tel_log.trade("🔍 Starting dynamic data verification...")

            try:
                from DYNAMIC_TESTS import DynamicLearnabilitySuite

                dynamic_suite = DynamicLearnabilitySuite()
                Dynamic_test = await dynamic_suite.run_tests()

                logger.send_log_file("runtime")

                # ═══════════════════════════════════════════════════════════
                # ✅ FIX (audit #17): verdict is based on `is_true_edge`,
                # not on truthiness of the result dict.
                # ═══════════════════════════════════════════════════════════
                if not isinstance(Dynamic_test, dict):
                    tel_log.trade(
                        f"⚠️ Dynamic verification returned non-dict "
                        f"({type(Dynamic_test).__name__}) — unknown status"
                    )
                elif 'error' in Dynamic_test:
                    tel_log.trade(
                        f"❌ Dynamic verification FAILED: "
                        f"{str(Dynamic_test['error'])[:120]}"
                    )
                else:
                    _is_true_edge = bool(Dynamic_test.get('is_true_edge', False))
                    _test_metrics = Dynamic_test.get('test_metrics') or {}
                    _sharpe = float(_test_metrics.get('sharpe_ratio', 0.0) or 0.0)
                    _wr = float(_test_metrics.get('win_rate', 0.0) or 0.0)
                    _trades = int(_test_metrics.get('total_trades', 0) or 0)
                    _best_ep = Dynamic_test.get('best_epoch', '?')
                    _space = str(_test_metrics.get('return_space', 'unknown'))
                    _ruined = bool(_test_metrics.get('ruined', False))
                    _suspect = bool(_test_metrics.get('sharpe_suspect', False))

                    if _is_true_edge:
                        tel_log.trade(
                            f"✅ Dynamic verification passed — TRUE EDGE confirmed\n"
                            f"   Sharpe={_sharpe:.3f} | WR={_wr:.2%} | "
                            f"Trades={_trades} | BestEpoch={_best_ep} | "
                            f"Space={_space}"
                        )
                    else:
                        tel_log.trade(
                            f"⚠️ Dynamic verification completed — NO TRUE EDGE\n"
                            f"   Sharpe={_sharpe:.3f} | WR={_wr:.2%} | "
                            f"Trades={_trades} | BestEpoch={_best_ep} | "
                            f"Space={_space} | Ruined={_ruined} | "
                            f"Suspect={_suspect}\n"
                            f"   Model trained but edge rejected — "
                            f"do NOT deploy to live trading."
                        )

            except ImportError:
                tel_log.trade("⚠️ Dynamic tests skipped - module not available")
            except Exception as e:
                tel_log.trade(f"❌ Dynamic tests failed: {str(e)[:100]}")

        # ====================================================================
        # PHASE 9: Shutdown
        # ====================================================================
        tel_log.trade(f"""
╔══════════════════════════════════════════════════════════════╗
║              ✅ SYSTEM SHUTDOWN COMPLETE                      ║
╠══════════════════════════════════════════════════════════════╣
║  Symbol: {args.symbol}
║  Timeframe: {args.timeframe}
║  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
╚══════════════════════════════════════════════════════════════╝
        """)

    except KeyboardInterrupt:
        tel_log.trade("🛑 System shutdown by user")
    except Exception as e:
        tel_log.trade(f"🔴 Fatal error: {str(e)[:200]}")
        raise
    finally:
        health.stop()


# ============================================================================
# SECTION 10: ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown by user")
    except Exception as e:
        print(f"\n🔴 Fatal error: {e}")
        traceback.print_exc()
        exit(3)