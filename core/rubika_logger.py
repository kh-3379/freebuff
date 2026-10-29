"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  RUBIKA, EITAA & TELEGRAM LOGGER - v8.3 PRODUCTION GRADE                     ║
║  ✅ Backward Compatible: send_latest_log_to_rubika & send_latest_log_to_eitaa║
║  ✅ Send logs to Rubika, Eitaa AND Telegram (Simultaneously)                 ║
║  ✅ Send Checkpoint (with split for all platforms)                           ║
║  ✅ Send Deployment Package                                                  ║
║  ✅ NO PROXY overhead for Telegram (Direct Server Connection)                ║
║  ✅ AUTO-DISABLE Rubika/Eitaa on Linux/Mac Servers                           ║
║  🆕 Full support for strategy_mgm & learning_penalty_mgm log types          ║
║  🆕 Robust file-pattern matching for all continuous log types                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =====================[ فاز 1: ایمپورت‌ها ]#=====================
import json
import requests
import logging
import threading
import time
import sys
import atexit
import os
import traceback
import base64
import zipfile
import shutil
import tempfile
import re
import platform
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from queue import Queue, Empty

# =====================[ فاز 1.5: شناسه سخت‌افزاری یکتا و تشخیص پلتفرم ]#=====================
def get_machine_id() -> str:
    try:
        node = platform.node() or "UNKNOWN_NODE"
        mac = uuid.getnode()
        mac_hash = hashlib.md5(str(mac).encode()).hexdigest()[:6].upper()
        return f"{node}{mac_hash}"
    except Exception:
        return f"SYSTEM{uuid.uuid4().hex[:6].upper()}"

MACHINE_ID = get_machine_id()

# ✅ تشخیص خودکار پلتفرم برای غیرفعال کردن ربات‌های ایرانی روی سرور
SYSTEM_OS = platform.system().lower()
IS_LINUX_OR_MAC = SYSTEM_OS in ['linux', 'darwin']  # darwin is macOS

if IS_LINUX_OR_MAC:
    print(f"🖥️ [System Info] Detected {SYSTEM_OS}. Disabling Rubika/Eitaa connections automatically.")

# =====================[ فاز 2: تنظیمات پایه ]#=====================
RUBIKA_TOKEN = "ICFIA0SNICDVTWRNRXFJPMQTFOCCLZIBUYIHMQPLZSNKKFTDUSBTGJMSCVEHYPMD"
RUBIKA_CHAT_ID = "@kh_3379trade"
RUBIKA_URL = f"https://botapi.rubika.ir/v3/{RUBIKA_TOKEN}/sendMessage"

EITAA_TOKEN = "bot348385:860e8146-8823-4168-bc27-121209840fab"
EITAA_CHAT_ID = "11130509"
EITAA_API_URL = f"https://eitaayar.ir/api/{EITAA_TOKEN}/sendFile"

TELEGRAM_BOT_TOKEN = "8157755095:AAH2VSeFfEb8gjf8znFZ10QZenx04Osah-E"
TELEGRAM_CHAT_ID = "95046517"
TELEGRAM_MSG_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
TELEGRAM_DOC_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"

BASE_DATA_DIR = Path.cwd() / "data"
BASE_MODELS_DIR = Path.cwd() / "MODELS"
BASE_CHECKPOINTS_DIR = BASE_MODELS_DIR / "checkpoints"
LOG_DIR = Path.cwd() / "logs"
SEND_TIMEOUT = 30

# =====================[ فاز 3: لاگر داخلی ]#=====================
_rubika_logger = logging.getLogger("unified_logger")

def _safe_print(msg: str, level: str = "INFO", exc_info: bool = False):
    try:
        if msg is None:
            msg = "None"
        elif not isinstance(msg, str):
            try:
                msg = str(msg)
            except Exception:
                msg = "Unprintable object"

        level = level.strip().upper()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🐛"}.get(level, "📝")
        formatted_msg = f"{timestamp} {emoji} [{level}] [{MACHINE_ID}] {msg}"

        if exc_info:
            formatted_msg += f"\n{traceback.format_exc()}"

        try:
            print(formatted_msg, flush=True)
        except UnicodeEncodeError:
            print(formatted_msg.encode('ascii', errors='replace').decode('ascii'), flush=True)

        # لاگ به فایل لوکال همیشه انجام می‌شود
        try:
            logger = logging.getLogger("unified_logger")
            if level == "ERROR":
                logger.error(msg)
            elif level == "WARNING":
                logger.warning(msg)
            else:
                logger.info(msg)
        except Exception:
            pass

    except Exception:
        print(str(msg), flush=True)

# =====================[ فاز 4: کلاس SmartRubikaManager ]#=====================
class SmartRubikaManager:
    _circuit_lock = threading.Lock()
    _consecutive_failures = 0
    _circuit_open_until = 0.0
    CIRCUIT_FAILURE_THRESHOLD = 3
    CIRCUIT_COOLDOWN_SECONDS = 300

    def __init__(self):
        self.session = requests.Session()
        self.timeout = 20
        self.max_retries = 1

    @classmethod
    def _is_circuit_open(cls) -> bool:
        with cls._circuit_lock:
            return time.time() < cls._circuit_open_until

    @classmethod
    def _record_failure(cls):
        with cls._circuit_lock:
            cls._consecutive_failures += 1
            if cls._consecutive_failures >= cls.CIRCUIT_FAILURE_THRESHOLD:
                cls._circuit_open_until = time.time() + cls.CIRCUIT_COOLDOWN_SECONDS
                _safe_print(f"🔌 [CircuitBreaker] OPEN for {cls.CIRCUIT_COOLDOWN_SECONDS}s", "WARNING")

    @classmethod
    def _record_success(cls):
        with cls._circuit_lock:
            cls._consecutive_failures = 0
            cls._circuit_open_until = 0.0

    def send(self, payload: Dict[str, Any]) -> bool:
        # ✅ اگر لینوکس/مک هستیم، هیچ درخواستی به روبیکا نرو
        if IS_LINUX_OR_MAC:
            return True  # موفقیت کاذب برای جلوگیری از خطاهای زنجیره ای

        if "chat_id" not in payload or "text" not in payload:
            return False

        if self._is_circuit_open():
            return False

        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(RUBIKA_URL, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "OK":
                        self._record_success()
                        return True
                self._record_failure()
            except Exception:
                self._record_failure()
            time.sleep(0.5)
        return False

# =====================[ فاز 5: کلاس SmartTelegramManager ]#=====================
class SmartTelegramManager:
    def __init__(self):
        self.session = requests.Session()
        self.timeout = 20
        self.max_retries = 1

    def send_text(self, text: str, parse_mode: str = "Markdown") -> bool:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text[:4096],
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(TELEGRAM_MSG_URL, json=payload, timeout=self.timeout)
                if response.status_code == 200 and response.json().get("ok"):
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

    def send_document(self, file_path: str, caption: str) -> bool:
        path = Path(file_path)
        if not path.exists():
            return False
        for attempt in range(self.max_retries + 1):
            try:
                with open(path, 'rb') as f:
                    files = {'document': (path.name, f, 'application/octet-stream')}
                    data = {
                        'chat_id': TELEGRAM_CHAT_ID,
                        'caption': caption[:1024],
                        'parse_mode': 'Markdown'
                    }
                    response = self.session.post(TELEGRAM_DOC_URL, data=data, files=files, timeout=self.timeout)
                    if response.status_code == 200 and response.json().get("ok"):
                        return True
            except Exception:
                pass
            time.sleep(0.5)
        return False

# =====================[ فاز 6: صف‌ها و کارگران ]#=====================
_rubika_manager = SmartRubikaManager()
_telegram_manager = SmartTelegramManager()
_message_queue = Queue()
_telegram_queue = Queue()
_queue_thread = None
_telegram_queue_thread = None
_queue_running = False

def _queue_worker():
    global _queue_running
    while _queue_running:
        try:
            item = _message_queue.get(timeout=1)
            if item is None:
                continue
            _rubika_manager.send(item)
            _message_queue.task_done()
        except Empty:
            continue
        except Exception:
            pass

def _telegram_queue_worker():
    global _queue_running
    while _queue_running:
        try:
            item = _telegram_queue.get(timeout=1)
            if item is None:
                continue
            msg_type = item.get("type", "text")
            if msg_type == "text":
                _telegram_manager.send_text(item["text"], item.get("parse_mode", "Markdown"))
            elif msg_type == "document":
                _telegram_manager.send_document(item["file_path"], item.get("caption", ""))
            _telegram_queue.task_done()
        except Empty:
            continue
        except Exception:
            pass

def start_queue_workers():
    global _queue_thread, _telegram_queue_thread, _queue_running
    if not _queue_running:
        _queue_running = True
        _queue_thread = threading.Thread(target=_queue_worker, daemon=True, name="RubikaQueueWorker")
        _queue_thread.start()
        _telegram_queue_thread = threading.Thread(target=_telegram_queue_worker, daemon=True, name="TelegramQueueWorker")
        _telegram_queue_thread.start()

# ✅ سازگاری با نام قدیمی
start_queue_worker = start_queue_workers

def stop_queue_workers():
    global _queue_running
    if _queue_running:
        _queue_running = False
        if _queue_thread and _queue_thread.is_alive():
            _queue_thread.join(timeout=2)
        if _telegram_queue_thread and _telegram_queue_thread.is_alive():
            _telegram_queue_thread.join(timeout=2)

stop_queue_worker = stop_queue_workers

def send_rubika_message(message: str, chat_id: Optional[str] = None) -> bool:
    if not message:
        return False

    target_chat_id = chat_id if chat_id else RUBIKA_CHAT_ID
    payload = {"chat_id": target_chat_id, "text": message[:4096]}

    start_queue_workers()
    _message_queue.put(payload)

    # تلگرام همیشه فعال است
    _telegram_queue.put({"type": "text", "text": message, "parse_mode": "Markdown"})

    return True

# =====================[ فاز 7: Notifier ]#=====================
class UnifiedTradeNotifier:
    def __init__(self):
        self.enabled = True
        self._sent_messages = set()

    def _get_message_hash(self, message: str) -> str:
        return str(hash(message))[:20]

    def notify_trade(self, trade_data: Dict[str, Any]):
        if not self.enabled:
            return

        action = trade_data.get('action', 'UNKNOWN')
        symbol = trade_data.get('symbol', 'N/A')
        quantity = trade_data.get('quantity', 0)
        price = trade_data.get('price', 0)
        pnl = trade_data.get('pnl', 0)

        action_emoji = {'BUY': '🟢', 'SELL': '🔴', 'CLOSE': '⚫', 'OPEN': '🟡'}.get(action, '⚪')

        message = (
            f"{action_emoji} *TRADE EXECUTED*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"*Action*: {action}\n"
            f"*Symbol*: {symbol}\n"
            f"*Quantity*: {quantity}\n"
            f"*Price*: ${price:.2f}\n"
        )

        if pnl is not None:
            pnl_emoji = '📈' if pnl > 0 else '📉' if pnl < 0 else ''
            message += f"*P&L*: {pnl_emoji} ${pnl:.2f}\n"

        message += f"*Time*: {datetime.now().strftime('%H:%M:%S')}"

        msg_hash = self._get_message_hash(message)
        if msg_hash not in self._sent_messages:
            self._sent_messages.add(msg_hash)
            send_rubika_message(message)
            if len(self._sent_messages) > 1000:
                self._sent_messages.clear()

    def notify_startup(self, bot_name: str = "Trading Bot"):
        message = (
            f"🤖 *{bot_name} STARTED*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"✅ *Status*: Running\n"
            f"🖥️ *Machine*: {MACHINE_ID}\n"
            f"⏰ *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"━━━━━━━━━━━━━━━━"
        )
        send_rubika_message(message)

    def notify_error(self, error_msg: str):
        message = (
            f"🔥 *ERROR*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"🖥️ *Machine*: {MACHINE_ID}\n"
            f"*Message*: {error_msg[:200]}\n"
            f"*Time*: {datetime.now().strftime('%H:%M:%S')}"
        )
        send_rubika_message(message)

# ✅ سازگاری با نام‌های قدیمی
RubikaTradeNotifier = UnifiedTradeNotifier
TelegramTradeNotifier = UnifiedTradeNotifier
_unified_notifier = UnifiedTradeNotifier()

def get_unified_notifier() -> UnifiedTradeNotifier:
    return _unified_notifier

# ✅ سازگاری با نام‌های قدیمی
get_rubika_notifier = get_unified_notifier
get_telegram_notifier = get_unified_notifier

# =====================[ فاز 8: ارسال فایل ]#=====================
def send_log_file_to_rubika(log_file_path: str) -> bool:
    log_path = Path(log_file_path)
    if not log_path.exists():
        return False

    # ✅ اگر لینوکس/مک هستیم، ارسال مستقیم به روبیکا را رد کن
    if IS_LINUX_OR_MAC:
        return True  # بازگشت موفقیت کاذب

    if SmartRubikaManager._is_circuit_open():
        return False

    caption = f"📄 Log File: {log_path.name} | Machine: {MACHINE_ID}"

    # تلگرام همیشه دریافت می‌کند
    _telegram_queue.put({"type": "document", "file_path": str(log_path), "caption": caption})

    try:
        req_url = f"https://botapi.rubika.ir/v3/{RUBIKA_TOKEN}/requestSendFile"
        r = requests.post(req_url, json={"type": "Document", "chat_id": RUBIKA_CHAT_ID}, timeout=20)
        if r.status_code != 200 or "result" not in r.json():
            SmartRubikaManager._record_failure()
            return False

        result = r.json()["result"]
        upload_url, file_id = result.get("upload_url"), result.get("file_id")

        with open(log_path, "rb") as f:
            r2 = requests.post(upload_url, files={"file": (log_path.name, f, "application/octet-stream")}, timeout=20)

        if r2.status_code == 200 and r2.json().get("status") == "OK":
            final_file_id = r2.json().get("result", {}).get("file_id") or file_id
            r3 = requests.post(
                f"https://botapi.rubika.ir/v3/{RUBIKA_TOKEN}/sendFile",
                json={"chat_id": RUBIKA_CHAT_ID, "file_id": final_file_id, "caption": caption},
                timeout=20
            )
            if r3.status_code == 200 and r3.json().get("status") == "OK":
                SmartRubikaManager._record_success()
                return True

        SmartRubikaManager._record_failure()
        return False
    except Exception:
        SmartRubikaManager._record_failure()
        return False

def send_log_file_to_eitaa(log_file_path: str, caption: str = "📊 Log Report") -> bool:
    log_path = Path(log_file_path)
    if not log_path.exists():
        return False

    # ✅ اگر لینوکس/مک هستیم، ارسال به ایتا را رد کن
    if IS_LINUX_OR_MAC:
        return True  # بازگشت موفقیت کاذب

    if SmartRubikaManager._is_circuit_open():
        return False

    try:
        with open(log_path, 'rb') as f:
            files = {'file': (log_path.name, f, 'text/plain')}
            data = {
                'chat_id': EITAA_CHAT_ID,
                'title': f"Log - {log_path.name[:30]}",
                'caption': f"{caption} | Machine: {MACHINE_ID}"[:200],
            }
            response = requests.post(EITAA_API_URL, data=data, files=files, verify=False, timeout=20)
            if response.status_code == 200 and response.json().get('ok'):
                SmartRubikaManager._record_success()
                return True
            SmartRubikaManager._record_failure()
            return False
    except Exception:
        SmartRubikaManager._record_failure()
        return False

# =====================[ فاز 9: توابع Backward-Compatible (دلیگیت) ]#=====================
# 🆕 انواع لاگ پشتیبانی‌شده (برای مطابقت الگو)
SUPPORTED_LOG_TYPES = (
    "trade", "train", "callback", "workflow", "runtime",
    "strategy_mgm", "learning_penalty_mgm", "system_logger",
    "error", "warning"
)

def _find_latest_log_file(log_type: str) -> Optional[Path]:
    """
    جستجو برای جدیدترین فایل لاگ بر اساس نوع
    پشتیبانی از هر دو الگوی نام‌گذاری:
      - continuous: {log_type}_{MACHINE_ID}_continuous.log
      - legacy:     {log_type}_{YYYYMMDD}.log
    """
    LOG_BASE_PATH = Path("logs")
    if not LOG_BASE_PATH.exists():
        return None

    today = datetime.now().strftime('%Y%m%d')
    candidates: List[Path] = []

    search_dirs = [LOG_BASE_PATH, LOG_BASE_PATH / "errors"]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for f in search_dir.iterdir():
            if not (f.is_file() and f.suffix == ".log"):
                continue
            name_lower = f.name.lower()
            type_lower = log_type.lower()
            # match continuous pattern OR dated pattern OR substring match
            if (type_lower in name_lower) or (f"{log_type}_{today}" in f.name):
                candidates.append(f)

    if not candidates:
        return None

    # جدیدترین بر اساس زمان تغییر
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

def send_latest_log_to_rubika(log_type: str = "runtime", also_send_to_eitaa: bool = True) -> bool:
    """
    ✅ BACKWARD COMPATIBLE
    ارسال آخرین لاگ به روبیکا + ایتا + تلگرام (همزمان)
    از انواع: trade, train, callback, workflow, runtime,
             strategy_mgm, learning_penalty_mgm, system_logger پشتیبانی می‌کند.
    """
    log_file_path = _find_latest_log_file(log_type)
    if not log_file_path:
        _safe_print(f"❌ Log file '{log_type}' not found", "ERROR")
        return False

    _safe_print(f"📤 Sending latest '{log_type}' log: {log_file_path.name}")

    # 1. ارسال به روبیکا (+ تلگرام به‌صورت خودکار درون این تابع)
    rubika_success = send_log_file_to_rubika(str(log_file_path))

    # 2. ارسال به ایتا
    eitaa_success = False
    if also_send_to_eitaa:
        caption = f"📊 {log_type.upper()} Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        eitaa_success = send_log_file_to_eitaa(str(log_file_path), caption=caption)

    return rubika_success or eitaa_success

def send_latest_log_to_eitaa(log_type: str = "runtime", caption: Optional[str] = None) -> bool:
    """
    ✅ BACKWARD COMPATIBLE
    ارسال آخرین لاگ به ایتا + تلگرام (همزمان)
    """
    log_file_path = _find_latest_log_file(log_type)
    if not log_file_path:
        _safe_print(f"❌ Log file '{log_type}' not found", "ERROR")
        return False

    if caption is None:
        caption = f"📊 {log_type.upper()} Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # ارسال به ایتا
    eitaa_success = send_log_file_to_eitaa(str(log_file_path), caption=caption)

    # ارسال به تلگرام (همزمان)
    _telegram_queue.put({"type": "document", "file_path": str(log_file_path), "caption": caption})

    return eitaa_success

def send_telegram_message(message: str, parse_mode: str = "Markdown") -> bool:
    """
    ✅ BACKWARD COMPATIBLE
    اگر جایی مستقیماً send_telegram_message صدا زده شده باشد.
    """
    try:
        start_queue_workers()
        _telegram_queue.put({"type": "text", "text": message, "parse_mode": parse_mode})
        return True
    except Exception:
        return False

# =====================[ فاز 10: چک‌پوینت‌ها ]#=====================
def find_checkpoint_files(
    symbol: str, timeframe: str, fold: str,
    model_architecture: Optional[str] = None,
    specific_epoch: Optional[int] = None,
    include_best_model: bool = True
) -> Tuple[List[Path], Optional[Path], Optional[Path], Dict[str, Any]]:
    metadata = {
        "last_epoch": None, "model_architecture": model_architecture,
        "full_path": None, "found_models": [], "found_metadata": None,
        "error": None, "best_model_path": None
    }
    fold_str = str(fold) if not isinstance(fold, str) else fold
    if not fold_str.startswith("fold_"):
        fold_str = f"fold_{fold_str}"

    try:
        base_checkpoint_path = BASE_CHECKPOINTS_DIR / symbol / timeframe / fold_str
        if not base_checkpoint_path.exists():
            metadata["error"] = f"Path not found: {base_checkpoint_path}"
            return [], None, None, metadata

        checkpoint_path = base_checkpoint_path / model_architecture if model_architecture else base_checkpoint_path
        if not checkpoint_path.exists():
            possible_paths = [s for s in base_checkpoint_path.iterdir() if s.is_dir() and (any(s.glob("epoch_*.keras")) or any(s.glob("best_model.keras")))]
            if not possible_paths:
                metadata["error"] = f"No checkpoint found"
                return [], None, None, metadata
            possible_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            checkpoint_path = possible_paths[0]
            metadata["model_architecture"] = checkpoint_path.name
            metadata["full_path"] = str(checkpoint_path)

        model_files = []
        epoch_pattern = re.compile(r"epoch_(\d+).keras$")
        best_model_path = checkpoint_path / "best_model.keras"

        if best_model_path.exists() and include_best_model:
            model_files.append(best_model_path)
            metadata["best_model_path"] = str(best_model_path)

        for file_path in checkpoint_path.glob("epoch_*.keras"):
            match = epoch_pattern.match(file_path.name)
            if match and (specific_epoch is None or int(match.group(1)) == specific_epoch):
                model_files.append(file_path)

        if not model_files:
            metadata["error"] = "No model files found"
            return [], None, None, metadata

        model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for f in model_files:
            match = re.search(r"epoch_(\d+)", f.name)
            if match:
                epoch_num = int(match.group(1))
                if metadata["last_epoch"] is None or epoch_num > metadata["last_epoch"]:
                    metadata["last_epoch"] = epoch_num

        metadata["found_models"] = [str(p) for p in model_files]

        metadata_file = None
        for candidate in [checkpoint_path / "FULL_METADATA.json", checkpoint_path / f"epoch_{metadata['last_epoch']:04d}_FULL_METADATA.json"]:
            if candidate.exists():
                metadata_file = candidate
                break

        if metadata_file:
            metadata["found_metadata"] = str(metadata_file)

        return model_files, checkpoint_path / "last_epoch.txt", metadata_file, metadata

    except Exception as e:
        metadata["error"] = str(e)
        return [], None, None, metadata

def create_full_checkpoint_zip(model_file: Path, metadata_file: Optional[Path], symbol: str, timeframe: str, fold: str, epoch: int) -> Optional[Path]:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("logs") / "checkpoint_packages"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_zip_path = output_dir / f"checkpoint_{symbol}_{timeframe}_{fold}_epoch_{epoch:04d}_{timestamp}.zip"

        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if model_file and model_file.exists():
                zipf.write(model_file, arcname=model_file.name)
            if metadata_file and metadata_file.exists():
                zipf.write(metadata_file, arcname=metadata_file.name)

            readme = f"CHECKPOINT\nSymbol: {symbol}\nTF: {timeframe}\nFold: {fold}\nEpoch: {epoch:04d}\nMachine: {MACHINE_ID}\nCreated: {datetime.now()}"
            zipf.writestr("README.txt", readme)

        return output_zip_path
    except Exception:
        return None

def split_file_to_parts_base64(file_path: Path, metadata_file_path: Optional[Path] = None, part_size_mb: float = 2.5) -> List[Path]:
    try:
        output_dir = file_path.parent / f"{file_path.stem}_parts_base64"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'rb') as f:
            file_data = f.read()

        base64_data = base64.b64encode(file_data).decode('ascii')
        original_size, base64_size = len(file_data), len(base64_data)
        part_size_bytes = int(part_size_mb * 1024 * 1024)
        total_parts = (base64_size + part_size_bytes - 1) // part_size_bytes

        parts = []
        for i in range(total_parts):
            start, end = i * part_size_bytes, min((i + 1) * part_size_bytes, base64_size)
            part_path = output_dir / f"{file_path.stem}_part_{i+1:03d}_of_{total_parts:03d}.txt"
            with open(part_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "part": i + 1,
                    "total_parts": total_parts,
                    "original_filename": file_path.name,
                    "original_size": original_size,
                    "data": base64_data[start:end]
                }, f)
            parts.append(part_path)

        if metadata_file_path and metadata_file_path.exists():
            with open(metadata_file_path, 'rb') as f:
                metadata_base64 = base64.b64encode(f.read()).decode('ascii')
            metadata_part_path = output_dir / f"{file_path.stem}_METADATA_part.txt"
            with open(metadata_part_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "type": "FULL_METADATA",
                    "original_filename": metadata_file_path.name,
                    "data": metadata_base64
                }, f)
            parts.append(metadata_part_path)

        return parts
    except Exception:
        return []

def send_checkpoint_parts_to_rubika_base64(parts: List[Path], symbol: str, epoch: int, total_parts: int) -> bool:
    data_parts = [p for p in parts if 'METADATA_part' not in p.name and 'reconstruct_metadata' not in p.name]
    metadata_part = next((p for p in parts if 'METADATA_part' in p.name), None)
    actual_total = len(data_parts)

    if actual_total == 0:
        return False

    success_count = 0
    PART_TIMEOUT = 20

    for i, part_path in enumerate(data_parts, 1):
        caption = f"Checkpoint Part {i}/{actual_total} - {symbol} - Epoch {epoch:04d} | Machine: {MACHINE_ID}"

        # تلگرام همیشه ارسال می‌کند
        _telegram_queue.put({"type": "document", "file_path": str(part_path), "caption": caption})

        # اگر لینوکس/مک نیست، به روبیکا ارسال کن
        if not IS_LINUX_OR_MAC:
            try:
                r = requests.post(f"https://botapi.rubika.ir/v3/{RUBIKA_TOKEN}/requestSendFile", json={"type": "Document", "chat_id": RUBIKA_CHAT_ID}, timeout=PART_TIMEOUT)
                if r.status_code != 200 or r.json().get("status") != "OK":
                    SmartRubikaManager._record_failure()
                    return False

                res = r.json()["result"]
                upload_url, file_id = res.get("upload_url"), res.get("file_id")

                with open(part_path, 'rb') as f:
                    r2 = requests.post(upload_url, files={"file": (part_path.name, f, "text/plain")}, timeout=PART_TIMEOUT)

                if r2.status_code == 200 and r2.json().get("status") == "OK":
                    final_id = r2.json().get("result", {}).get("file_id") or file_id
                    r3 = requests.post(f"https://botapi.rubika.ir/v3/{RUBIKA_TOKEN}/sendFile", json={"chat_id": RUBIKA_CHAT_ID, "file_id": final_id, "text": caption[:1024]}, timeout=PART_TIMEOUT)
                    if r3.status_code == 200 and r3.json().get("status") == "OK":
                        success_count += 1
                    else:
                        return False
                else:
                    return False
            except Exception:
                SmartRubikaManager._record_failure()
                return False
        else:
            success_count += 1

    if success_count == actual_total:
        if not IS_LINUX_OR_MAC:
            SmartRubikaManager._record_success()

        if metadata_part and success_count == actual_total:
            _telegram_queue.put({"type": "document", "file_path": str(metadata_part), "caption": f"📋 FULL_METADATA {symbol} E{epoch:04d}"})

            if not IS_LINUX_OR_MAC:
                try:
                    r = requests.post(f"https://botapi.rubika.ir/v3/{RUBIKA_TOKEN}/requestSendFile", json={"type": "Document", "chat_id": RUBIKA_CHAT_ID}, timeout=PART_TIMEOUT)
                    if r.status_code == 200:
                        res = r.json()["result"]
                        upload_url, file_id = res.get("upload_url"), res.get("file_id")
                        with open(metadata_part, 'rb') as f:
                            r2 = requests.post(upload_url, files={"file": (metadata_part.name, f, "text/plain")}, timeout=PART_TIMEOUT)
                        if r2.status_code == 200:
                            final_id = r2.json().get("result", {}).get("file_id") or file_id
                            requests.post(f"https://botapi.rubika.ir/v3/{RUBIKA_TOKEN}/sendFile", json={"chat_id": RUBIKA_CHAT_ID, "file_id": final_id, "text": f"📋 METADATA {symbol}"}, timeout=PART_TIMEOUT)
                except Exception:
                    pass

    return success_count == actual_total

def send_checkpoint_parts_to_eitaa_base64(parts: List[Path], symbol: str, epoch: int, total_parts: int) -> bool:
    if IS_LINUX_OR_MAC:
        return True

    data_parts = [p for p in parts if 'METADATA_part' not in p.name and 'reconstruct_metadata' not in p.name]
    metadata_part = next((p for p in parts if 'METADATA_part' in p.name), None)
    actual_total = len(data_parts)

    if actual_total == 0 or SmartRubikaManager._is_circuit_open():
        return False

    success_count = 0
    PART_TIMEOUT = 20

    for i, part_path in enumerate(data_parts, 1):
        caption = f"Checkpoint Part {i}/{actual_total} - {symbol} - Epoch {epoch:04d} | Machine: {MACHINE_ID}"
        try:
            with open(part_path, 'rb') as f:
                files = {'file': (part_path.name, f, 'text/plain')}
                res = requests.post(EITAA_API_URL, data={'chat_id': EITAA_CHAT_ID, 'title': f"Part{i:03d}", 'caption': caption[:200]}, files=files, timeout=PART_TIMEOUT, verify=False)
                if res.status_code == 200 and res.json().get('ok'):
                    success_count += 1
                else:
                    return False
        except Exception:
            return False

    if success_count == actual_total:
        SmartRubikaManager._record_success()
        if metadata_part and success_count == actual_total:
            try:
                with open(metadata_part, 'rb') as f:
                    requests.post(EITAA_API_URL, data={'chat_id': EITAA_CHAT_ID, 'title': 'METADATA', 'caption': f'📋 {symbol} E{epoch:04d}'}, files={'file': (metadata_part.name, f, 'application/json')}, timeout=PART_TIMEOUT, verify=False)
            except Exception:
                pass
    return success_count == actual_total

def send_checkpoint(
    symbol: str, timeframe: str, fold,
    model_architecture: Optional[str] = None,
    specific_epoch: Optional[int] = None,
    also_send_to_eitaa: bool = True,
    delete_after_send: bool = True,
    include_additional_files: bool = True
) -> bool:
    """ارسال چک‌پوینت به صورت غیرمسدودکننده (Background Thread) به همه پلتفرم‌ها"""
    phase = "SEND_CHECKPOINT_V8"
    fold_str = str(fold) if not isinstance(fold, str) else fold
    if not fold_str.startswith("fold_"):
        fold_str = f"fold_{fold_str}"

    def _worker():
        try:
            _safe_print(f"\n{'=' * 60}")
            _safe_print(f"🔄 [{phase}] (async) Sending checkpoint")
            _safe_print(f"   Symbol: {symbol}, TF: {timeframe}, Fold: {fold_str}")

            model_files, last_epoch_file, metadata_file, metadata = find_checkpoint_files(
                symbol=symbol, timeframe=timeframe, fold=fold_str,
                model_architecture=model_architecture,
                specific_epoch=specific_epoch, include_best_model=True
            )

            if not model_files:
                _safe_print(f"❌ [{phase}] No model files", "ERROR")
                return

            best_model_file = None
            for f in model_files:
                if "best_model" in f.name.lower():
                    best_model_file = f
                    break

            if best_model_file is None:
                best_model_file = model_files[0]

            match = re.search(r"epoch_(\d+)", best_model_file.name)
            best_epoch = int(match.group(1)) if match else (metadata.get("last_epoch") or 0)

            zip_path = create_full_checkpoint_zip(
                model_file=best_model_file, metadata_file=metadata_file,
                symbol=symbol, timeframe=timeframe, fold=fold_str, epoch=best_epoch
            )

            if zip_path is None or not zip_path.exists():
                _safe_print(f"❌ [{phase}] Zip failed", "ERROR")
                return

            parts = split_file_to_parts_base64(
                file_path=zip_path, metadata_file_path=metadata_file, part_size_mb=2.5
            )

            if not parts:
                _safe_print(f"❌ [{phase}] Split failed", "ERROR")
                return

            total_parts = len([
                p for p in parts
                if "METADATA" not in p.name and "reconstruct_metadata" not in p.name
            ])

            rubika_success = False
            eitaa_success = False
            telegram_queued = False

            try:
                rubika_success = send_checkpoint_parts_to_rubika_base64(
                    parts, symbol, best_epoch, total_parts
                )
                telegram_queued = True
            except Exception as e:
                _safe_print(f"❌ [{phase}] Rubika/Tel Queue error: {e}", "ERROR")

            if also_send_to_eitaa and not SmartRubikaManager._is_circuit_open():
                try:
                    eitaa_success = send_checkpoint_parts_to_eitaa_base64(
                        parts, symbol, best_epoch, total_parts
                    )
                except Exception as e:
                    _safe_print(f"❌ [{phase}] Eitaa error: {e}", "ERROR")

            if delete_after_send:
                try:
                    zip_path.unlink(missing_ok=True)
                    parts_dir = parts[0].parent
                    for part in parts:
                        try:
                            part.unlink()
                        except Exception:
                            pass
                    try:
                        parts_dir.rmdir()
                    except Exception:
                        pass
                except Exception:
                    pass

            t_status = "Queued" if telegram_queued else "Failed"
            _safe_print(
                f"✅ [{phase}] done (R={rubika_success}, E={eitaa_success}, T={t_status})"
            )
            _safe_print(f"{'=' * 60}\n")

        except Exception as e:
            _safe_print(f"❌ [{phase}] worker error: {e}", "ERROR", exc_info=True)

    t = threading.Thread(target=_worker, daemon=True, name=f"SendCheckpoint_E{specific_epoch or 0}")
    t.start()
    return True

# =====================[ فاز 11: دپلویمنت ]#=====================
def create_deployment_zip(symbol: str, timeframe: str, fold: str, output_zip_path: Optional[Path] = None) -> Optional[Path]:
    try:
        models_base = BASE_MODELS_DIR / "edge_models" / symbol / timeframe / fold
        if not models_base.exists():
            return None

        model_files = list(models_base.rglob("*.keras")) + list(models_base.rglob("*.h5"))
        if not model_files:
            return None

        if output_zip_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("logs") / "deployment_packages"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_zip_path = out_dir / f"deploy_{symbol}_{timeframe}_{fold}_{timestamp}.zip"

        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for mf in model_files:
                try:
                    zipf.write(mf, arcname=str(mf.relative_to(BASE_MODELS_DIR)))
                except ValueError:
                    zipf.write(mf, arcname=f"MODELS/{mf.name}")

        return output_zip_path
    except Exception:
        return None

def send_deployment_package(
    symbol: str, timeframe: str, fold: str,
    also_send_to_eitaa: bool = True,
    delete_after_send: bool = True
) -> bool:
    try:
        zip_path = create_deployment_zip(symbol, timeframe, fold)
        if not zip_path or not zip_path.exists():
            return False

        caption = f"📦 Deployment - {symbol} {timeframe} | Machine: {MACHINE_ID}"

        _telegram_queue.put({"type": "document", "file_path": str(zip_path), "caption": caption})

        if not IS_LINUX_OR_MAC:
            send_log_file_to_rubika(str(zip_path))
            if also_send_to_eitaa:
                send_log_file_to_eitaa(str(zip_path), caption=caption)

        if delete_after_send:
            zip_path.unlink(missing_ok=True)

        return True
    except Exception:
        return False

# =====================[ فاز 12: Log Handler ]#=====================
class RubikaLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.enabled = False

    def emit(self, record):
        pass

TelegramLogHandler = RubikaLogHandler

# =====================[ فاز 13: پاکسازی ]#=====================
def cleanup():
    stop_queue_workers()

atexit.register(cleanup)

# =====================[ فاز 14: تست ]#=====================
def test_rubika_logger():
    print("\n" + "=" * 70)
    print(f"🧪 TEST: Unified Logger v8.3 | Machine: {MACHINE_ID} | OS: {SYSTEM_OS} | Disable IR Bots: {IS_LINUX_OR_MAC}")
    print("=" * 70)

    print("\n📝 [1] send_rubika_message...")
    send_rubika_message(f"🧪 Test v8.3 | Machine: {MACHINE_ID}")

    print("\n📝 [2] send_latest_log_to_rubika (backward compat)...")
    send_latest_log_to_rubika("runtime", also_send_to_eitaa=True)

    print("\n📊 [3] notifier...")
    get_rubika_notifier().notify_trade({'action': 'BUY', 'symbol': 'BTCUSDT', 'quantity': 0.1, 'price': 45000, 'pnl': 150})

    print("\n✅ Test completed — Check Telegram (and Rubika/Eitaa only if Windows)")
    print("=" * 70)

if __name__ == "__main__":
    test_rubika_logger()