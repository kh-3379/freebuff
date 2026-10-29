"""
================================================================================
ECONOMIC_CALENDAR.py - Economic Calendar Monitor for Trading System
================================================================================
Version: 6.0.0 - SELF-CONTAINED + ASYNC SAFE + BACKGROUND LOOP + CACHE SAFE
          ✅ Robust event normalization, adaptive concurrency, safer parsers
Author: Senior Quantitative Architect
================================================================================
"""
from __future__ import annotations

import asyncio
import html as html_lib
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except Exception:  # pragma: no cover - optional dependency
    aiohttp = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency

    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

try:
    from core.logger import setup_logger  # type: ignore
except Exception:  # pragma: no cover - fallback logger for standalone use

    def setup_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger


logger = setup_logger("economic_calendar")


# ============================================================
# GLOBAL HELPERS
# ============================================================

_NUMERIC_SUFFIXES = {
    "k": 1_000.0,
    "m": 1_000_000.0,
    "b": 1_000_000_000.0,
    "t": 1_000_000_000_000.0,
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: Any) -> Optional[datetime]:
    """Convert common datetime representations to timezone-aware UTC datetime."""
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        timestamp = float(value)
        if timestamp > 1_000_000_000_000:  # milliseconds
            timestamp /= 1000.0
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    if hasattr(value, "to_pydatetime"):
        with suppress(Exception):
            dt = value.to_pydatetime()
            if isinstance(dt, datetime):
                return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    iso_candidates = [
        text,
        text.replace("T", " "),
    ]
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]

    for candidate in iso_candidates:
        try:
            dt = datetime.fromisoformat(candidate)
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    for fmt in formats:
        with suppress(ValueError):
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)

    return None


def _coerce_float(value: Any) -> Optional[float]:
    """Parse numeric values including common suffixes such as K/M/B/T."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = (
        text.replace(",", "")
        .replace("%", "")
        .replace("−", "-")
        .replace("—", "-")
        .replace("–", "-")
        .lower()
    )
    if normalized in {"-", "--", "n/a", "na", "none", "null"}:
        return None

    suffix = normalized[-1]
    multiplier = _NUMERIC_SUFFIXES.get(suffix, 1.0)
    if multiplier != 1.0:
        normalized = normalized[:-1].strip()

    try:
        number = float(normalized)
        if not math.isfinite(number):
            return None
        return number * multiplier
    except ValueError:
        return None


def _impact_from_value(value: Any) -> "EventImpact":
    if isinstance(value, EventImpact):
        return value

    if value is None:
        return EventImpact.LOW

    text = str(value).strip().lower()
    if text in {"high", "3", "3.0", "strong"}:
        return EventImpact.HIGH
    if text in {"medium", "2", "2.0", "moderate", "mid"}:
        return EventImpact.MEDIUM
    if text in {"holiday", "0", "0.0"}:
        return EventImpact.HOLIDAY
    return EventImpact.LOW


# ============================================================
# PROVIDER DATA CLASS
# ============================================================


@dataclass
class Provider:
    name: str
    url: str
    priority: int
    region: str  # "IR" or "GLOBAL"
    parser: str = "generic_json"
    timeout: int = 10
    enabled: bool = True
    requires_api_key: bool = False
    api_key_env: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    score: float = 1.0


# ============================================================
# IRAN PROVIDERS
# ============================================================

IRAN_PROVIDERS = [
    Provider(
        name="streamdata",
        url="https://streamdata.ir/markets/economic-calendar",
        priority=100,
        region="IR",
        parser="streamdata",
        timeout=8,
        score=1.00,
    ),
    Provider(
        name="tgju",
        url="https://www.tgju.org/economic-calendar",
        priority=90,
        region="IR",
        parser="html_tgju",
        timeout=8,
        score=0.95,
    ),
    Provider(
        name="iranbourse",
        url="https://www.iranbourse.com/api/economic-events",
        priority=80,
        region="IR",
        parser="generic_json",
        timeout=8,
        score=0.80,
    ),
]


# ============================================================
# FOREIGN PROVIDERS
# ============================================================

FOREIGN_PROVIDERS = [
    Provider(
        name="forexfactory",
        url="https://nfs.faireconomy.media/ff_calendar_thisweek.json",
        priority=50,
        region="GLOBAL",
        parser="forexfactory",
        timeout=10,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        },
        score=0.95,
    ),
    Provider(
        name="tradingeconomics",
        url="https://api.tradingeconomics.com/calendar",
        priority=50,
        region="GLOBAL",
        parser="tradingeconomics",
        timeout=10,
        requires_api_key=True,
        api_key_env="TRADING_ECONOMICS_API_KEY",
        headers={"Accept": "application/json"},
        params={"importance": 1, "lang": "fa", "f": "json"},
        score=1.00,
    ),
    Provider(
        name="myfxbook",
        url="https://www.myfxbook.com/api/get-economic-calendar.json",
        priority=45,
        region="GLOBAL",
        parser="myfxbook",
        timeout=10,
        headers={"Accept": "application/json"},
        score=0.85,
    ),
    Provider(
        name="fxstreet",
        url="https://calendar-api.fxstreet.com/en/api/v1/eventDates",
        priority=45,
        region="GLOBAL",
        parser="fxstreet",
        timeout=10,
        headers={"Accept": "application/json"},
        score=0.90,
    ),
    Provider(
        name="tradingview_api",
        url="https://api.tradingviewapi.com/api/calendar/economic",
        priority=40,
        region="GLOBAL",
        parser="tradingview",
        timeout=10,
        requires_api_key=True,
        api_key_env="TRADINGVIEW_API_KEY",
        headers={"Accept": "application/json"},
        score=0.90,
    ),
]

ALL_PROVIDERS = IRAN_PROVIDERS + FOREIGN_PROVIDERS


# ============================================================
# EVENT IMPACT ENUM
# ============================================================


class EventImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    HOLIDAY = "holiday"

    @classmethod
    def from_value(cls, value: Any) -> "EventImpact":
        return _impact_from_value(value)


# ============================================================
# ECONOMIC EVENT DATA CLASS
# ============================================================


@dataclass
class EconomicEvent:
    id: str
    title: str
    country: str
    impact: EventImpact
    datetime: datetime
    previous: Optional[float] = None
    forecast: Optional[float] = None
    actual: Optional[float] = None
    currency: Optional[str] = None
    description: str = ""
    source: str = "unknown"

    def __post_init__(self) -> None:
        try:
            if not self.id or self.id == "ff_" or len(str(self.id)) < 3:
                self.id = f"evt_{uuid.uuid4().hex[:12]}"
            else:
                self.id = str(self.id)

            self.title = str(self.title or "Unknown Event").strip() or "Unknown Event"
            self.country = str(self.country or "US").strip() or "US"
            self.currency = str(self.currency or "USD").strip() or "USD"
            self.description = str(self.description or "")
            self.source = str(self.source or "unknown")
            self.previous = _coerce_float(self.previous)
            self.forecast = _coerce_float(self.forecast)
            self.actual = _coerce_float(self.actual)
            self.impact = _impact_from_value(self.impact)
            dt = _coerce_datetime(self.datetime)
            self.datetime = dt if dt is not None else (_utc_now() - timedelta(days=365))
        except Exception as exc:
            logger.error(f"EconomicEvent post-init error: {exc}", exc_info=True)
            self.id = f"evt_{uuid.uuid4().hex[:12]}"
            self.title = str(self.title or "Unknown Event")
            self.country = str(self.country or "US")
            self.impact = _impact_from_value(self.impact)
            self.datetime = _utc_now() - timedelta(days=365)
            self.currency = str(self.currency or "USD")
            self.description = str(self.description or "")
            self.source = str(self.source or "unknown")

    @property
    def minutes_until(self) -> int:
        try:
            event_dt = _coerce_datetime(self.datetime)
            if event_dt is None:
                return 9999
            delta = event_dt - _utc_now()
            return max(0, int(delta.total_seconds() / 60))
        except Exception:
            return 9999

    @property
    def is_upcoming(self) -> bool:
        try:
            event_dt = _coerce_datetime(self.datetime)
            return bool(event_dt and event_dt > _utc_now())
        except Exception:
            return False

    def is_recent(self, minutes: int = 30) -> bool:
        try:
            event_dt = _coerce_datetime(self.datetime)
            if event_dt is None:
                return False
            elapsed = (_utc_now() - event_dt).total_seconds() / 60.0
            return 0 <= elapsed <= float(minutes)
        except Exception:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "country": self.country,
            "impact": self.impact.value,
            "datetime": self.datetime.astimezone(timezone.utc).isoformat(),
            "previous": self.previous,
            "forecast": self.forecast,
            "actual": self.actual,
            "currency": self.currency,
            "description": self.description,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_source: str = "unknown") -> "EconomicEvent":
        dt_value = data.get("datetime")
        dt = _coerce_datetime(dt_value)
        if dt is None:
            raise ValueError("Invalid or missing datetime")

        return cls(
            id=str(data.get("id") or f"evt_{uuid.uuid4().hex[:12]}"),
            title=str(data.get("title") or "Unknown Event"),
            country=str(data.get("country") or "US"),
            impact=_impact_from_value(data.get("impact")),
            datetime=dt,
            previous=_coerce_float(data.get("previous")),
            forecast=_coerce_float(data.get("forecast")),
            actual=_coerce_float(data.get("actual")),
            currency=str(data.get("currency") or "USD"),
            description=str(data.get("description") or ""),
            source=str(data.get("source") or default_source),
        )


# ============================================================
# ECONOMIC CALENDAR MONITOR
# ============================================================


class EconomicCalendarMonitor:
    """
    Economic calendar monitor with multi-source fetch, cache persistence,
    fail-safe trading checks, and adaptive runtime behavior.

    V6.1.0 - Audited & Bug-fixed:
      ✅ update_events returns bool and distinguishes success/cache/failure
      ✅ Circuit-breaker + retry coordination fixed
      ✅ Thread-safe iteration over high_impact_events via snapshot
      ✅ _recover_event_loop truly restarts a dead background loop
      ✅ Dead code removed, API-key miss logged
    """

    def __init__(
        self,
        cache_file: str = "TRADE/economic_events.json",
        update_interval_minutes: int = None,
        auto_update: bool = True,
        auto_validate: bool = True,
        max_retries: int = 3,
        retry_delay: int = 2,
        use_thread_pool: bool = True,
    ):
        phase = "ECONOMIC_CALENDAR_INIT_V6"
        try:
            start_time = time.time()
            load_dotenv()

            self.cache_file = cache_file
            self.update_interval_minutes = update_interval_minutes or int(
                os.getenv("ECON_CALENDAR_UPDATE_INTERVAL_MINUTES", "30")
            )
            self.auto_update = auto_update
            self.auto_validate = auto_validate
            self.max_retries = max(1, int(max_retries))
            self.retry_delay = max(0, int(retry_delay))
            self.use_thread_pool = use_thread_pool

            self.providers = [p for p in ALL_PROVIDERS if p.enabled]
            self.providers.sort(
                key=lambda provider: (
                    provider.region != "IR",
                    -provider.priority,
                    provider.name,
                )
            )
            logger.info(f"[{phase}] Active providers: {len(self.providers)}")

            self._state_lock = threading.RLock()
            self._source_failure_tracking: Dict[str, Dict[str, Any]] = {}
            self._source_cooldown_seconds = 300
            self._max_parallel_fetches = self._resolve_parallelism()

            self.REDUCE_CONFIDENCE_MINUTES = self._safe_int_env("ECON_REDUCE_CONFIDENCE_MINUTES", 30)
            self.INCREASE_SPREAD_MINUTES = self._safe_int_env("ECON_INCREASE_SPREAD_MINUTES", 15)
            self.BLOCK_TRADING_MINUTES = self._safe_int_env("ECON_BLOCK_TRADING_MINUTES", 10)

            self.CONFIDENCE_REDUCTION_FACTOR = {
                EventImpact.HIGH: self._safe_float_env("ECON_CONF_REDUCTION_HIGH", 0.95),
                EventImpact.MEDIUM: self._safe_float_env("ECON_CONF_REDUCTION_MEDIUM", 0.97),
                EventImpact.LOW: self._safe_float_env("ECON_CONF_REDUCTION_LOW", 0.99),
                EventImpact.HOLIDAY: self._safe_float_env("ECON_CONF_REDUCTION_HOLIDAY", 0.98),
            }
            self.SPREAD_MULTIPLIER = {
                EventImpact.HIGH: self._safe_float_env("ECON_SPREAD_MULTIPLIER_HIGH", 3.0),
                EventImpact.MEDIUM: self._safe_float_env("ECON_SPREAD_MULTIPLIER_MEDIUM", 1.5),
                EventImpact.LOW: self._safe_float_env("ECON_SPREAD_MULTIPLIER_LOW", 1.1),
                EventImpact.HOLIDAY: self._safe_float_env("ECON_SPREAD_MULTIPLIER_HOLIDAY", 1.2),
            }
            self._min_confidence = self._safe_float_env("ECON_MIN_CONFIDENCE", 0.35)
            self._max_confidence = self._safe_float_env("ECON_MAX_CONFIDENCE", 0.75)

            self.events: List[EconomicEvent] = []
            self.high_impact_events: List[EconomicEvent] = []
            self._update_task: Optional[asyncio.Task[Any]] = None
            self._background_thread: Optional[threading.Thread] = None
            self._loop: Optional[asyncio.AbstractEventLoop] = None
            self._running = False
            self._last_update_time: Optional[datetime] = None
            self._last_successful_update: Optional[datetime] = None
            self._consecutive_failures = 0
            self._max_consecutive_failures = 10

            self._thread_pool = None
            if use_thread_pool:
                from concurrent.futures import ThreadPoolExecutor

                self._thread_pool = ThreadPoolExecutor(
                    max_workers=min(4, max(1, os.cpu_count() or 1))
                )

            self._stats = {
                "total_updates": 0,
                "successful_updates": 0,
                "failed_updates": 0,
                "high_impact_detected": 0,
                "trading_blocks": 0,
                "confidence_reductions": 0,
                "source_usage": {},
                "last_error": None,
                "last_error_time": None,
            }

            self._api_failure_count = 0
            self._api_failure_threshold = 3
            self._api_circuit_breaker_open = False
            self._api_circuit_breaker_reset_time = None
            self._api_circuit_breaker_timeout_seconds = 300

            self._fallback_mode_active = False
            self._fallback_mode_start_time = None
            self._fallback_mode_duration_seconds = 600

            self._load_from_cache()

            if auto_update:
                self.start()

            elapsed_ms = (time.time() - start_time) * 1000.0
            logger.info(f"[{phase}] ✅ EconomicCalendarMonitor initialized")
            logger.info(f"[{phase}]    cache_file: {self.cache_file}")
            logger.info(f"[{phase}]    update_interval: {self.update_interval_minutes} minutes")
            logger.info(f"[{phase}]    max_retries: {self.max_retries}")
            logger.info(f"[{phase}]    active_providers: {len(self.providers)}")
            logger.info(f"[{phase}]    init_time: {elapsed_ms:.2f}ms")
        except Exception as exc:
            logger.error(f"[{phase}] ❌ Init error: {exc}", exc_info=True)
            raise

    # ========================================================
    # INTERNAL CONFIG HELPERS
    # ========================================================

    @staticmethod
    def _safe_int_env(name: str, default: int) -> int:
        with suppress(Exception):
            return int(os.getenv(name, str(default)))
        return int(default)

    @staticmethod
    def _safe_float_env(name: str, default: float) -> float:
        with suppress(Exception):
            return float(os.getenv(name, str(default)))
        return float(default)

    def _resolve_parallelism(self) -> int:
        env_value = os.getenv("ECON_MAX_PARALLEL_FETCHES")
        if env_value:
            with suppress(Exception):
                return max(1, int(env_value))
        cpu_count = os.cpu_count() or 2
        return max(2, min(8, cpu_count * 2))

    # ========================================================
    # EVENT LOOP MANAGEMENT
    # ========================================================

    def _get_new_event_loop(self) -> asyncio.AbstractEventLoop:
        return asyncio.new_event_loop()

    def _ensure_event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_running_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
            return loop
        except RuntimeError:
            if self._loop and not self._loop.is_closed():
                return self._loop
            loop = asyncio.new_event_loop()
            self._loop = loop
            return loop

    def _background_loop_runner(self) -> None:
        phase = "BACKGROUND_LOOP"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        task: Optional[asyncio.Task[Any]] = None
        try:
            task = loop.create_task(self._update_loop())
            self._update_task = task
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            logger.info(f"[{phase}] Background loop cancelled")
        except Exception as exc:
            logger.error(f"[{phase}] Error: {exc}", exc_info=True)
        finally:
            pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                with suppress(Exception):
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            with suppress(Exception):
                loop.close()
            self._loop = None
            self._update_task = None

    def _recover_event_loop(self) -> None:
        """
        بازیابی حلقهٔ بروزرسانی وقتی ترد پس‌زمینه مرده یا حلقه بسته شده است.

        رفع باگ V6.0.0: قبلاً `self.start()` صدا زده می‌شد ولی چون
        `_running=True` بود، `start()` فوراً return می‌کرد و بازیابی
        هرگز رخ نمی‌داد. اکنون ابتدا `_running` را صفر کرده و با یک
        ترد جدید، حلقه را از نو راه می‌اندازیم.
        """
        phase = "RECOVER_LOOP"
        try:
            loop_alive = self._loop is not None and not self._loop.is_closed()
            thread_alive = self._background_thread is not None and self._background_thread.is_alive()

            if loop_alive and thread_alive:
                return

            if not self._running:
                # اگر حلقه در حال اجرا نیست ولی ما انتظار داریم زنده باشد،
                # در حالت عادی این متد فراخوانی نمی‌شود.
                return

            logger.warning(f"[{phase}] Background loop is dead; restarting")
            # به‌طور موقت _running را خاموش می‌کنیم تا start() بتواند عبور کند
            self._running = False
            with suppress(Exception):
                self.start()
        except Exception as exc:
            logger.error(f"[{phase}] Failed to recover event loop: {exc}", exc_info=True)

    # ========================================================
    # CACHE MANAGEMENT
    # ========================================================

    def _load_from_cache(self) -> None:
        phase = "LOAD_CACHE"
        cache_path = Path(self.cache_file)
        try:
            if not cache_path.exists():
                backup_dir = cache_path.parent if cache_path.parent != Path("") else Path(".")
                pattern = f"{cache_path.stem}*.backup"
                backup_files = list(backup_dir.glob(pattern)) if backup_dir.exists() else []
                if backup_files:
                    latest_backup = max(backup_files, key=lambda item: item.stat().st_mtime)
                    logger.info(f"[{phase}] Restoring backup: {latest_backup}")
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(latest_backup, cache_path)
                    except Exception as exc:
                        logger.error(f"[{phase}] Failed to restore backup: {exc}", exc_info=True)
                return

            with cache_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)

            events_data = data.get("events", []) if isinstance(data, dict) else []
            if not isinstance(events_data, list) or not events_data:
                logger.warning(f"[{phase}] No events in cache file")
                return

            loaded_events: List[EconomicEvent] = []
            for event_data in events_data:
                if not isinstance(event_data, dict):
                    continue
                try:
                    loaded_events.append(EconomicEvent.from_dict(event_data, default_source="cache"))
                except Exception as exc:
                    logger.debug(f"[{phase}] Error parsing cached event: {exc}")
                    continue

            week_ago = _utc_now() - timedelta(days=7)
            loaded_events = [event for event in loaded_events if event.datetime > week_ago]

            with self._state_lock:
                self.events = loaded_events
                self._update_high_impact_events()

            logger.info(f"[{phase}] ✅ Loaded {len(self.events)} events from cache")
        except Exception as exc:
            logger.error(f"[{phase}] Failed to load cache: {exc}", exc_info=True)
            backup_path = f"{self.cache_file}.corrupted_{_utc_now().strftime('%Y%m%d_%H%M%S')}"
            with suppress(Exception):
                shutil.copy2(self.cache_file, backup_path)
                logger.info(f"[{phase}] Corrupted cache backed up to: {backup_path}")

    def _save_to_cache(self) -> None:
        phase = "SAVE_CACHE"
        cache_path = Path(self.cache_file)
        temp_path = Path(f"{self.cache_file}.tmp")
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            with self._state_lock:
                events_snapshot = [event.to_dict() for event in self.events]
                last_update = self._last_update_time.isoformat() if self._last_update_time else None
                last_success = (
                    self._last_successful_update.isoformat()
                    if self._last_successful_update
                    else None
                )
                failure_count = self._consecutive_failures

            payload = {
                "events": events_snapshot,
                "last_update": last_update,
                "last_successful_update": last_success,
                "total_events": len(events_snapshot),
                "consecutive_failures": failure_count,
            }

            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)

            os.replace(temp_path, cache_path)
            backup_file = Path(f"{self.cache_file}.backup")
            with suppress(Exception):
                shutil.copy2(cache_path, backup_file)

            logger.debug(f"[{phase}] Saved {len(events_snapshot)} events to cache")
        except Exception as exc:
            logger.error(f"[{phase}] Failed to save cache: {exc}", exc_info=True)
            with suppress(Exception):
                if temp_path.exists():
                    temp_path.unlink()

    # ========================================================
    # INTERNAL STATE MANAGEMENT
    # ========================================================

    def _update_high_impact_events(self) -> None:
        """⚠️ باید تنها در حالی که `_state_lock` گرفته شده فراخوانی شود."""
        high_impact = [
            event
            for event in self.events
            if event.impact in {EventImpact.HIGH, EventImpact.MEDIUM}
        ]
        high_impact.sort(key=lambda item: item.datetime)
        self.high_impact_events = high_impact

    def _record_source_failure(self, source_name: str) -> None:
        with self._state_lock:
            tracking = self._source_failure_tracking.setdefault(
                source_name, {"consecutive_failures": 0, "last_failed": None}
            )
            tracking["consecutive_failures"] = int(tracking.get("consecutive_failures", 0)) + 1
            tracking["last_failed"] = _utc_now()
            count = tracking["consecutive_failures"]
        if count >= 3:
            logger.warning(
                f"[{source_name}] Disabling source temporarily after 3 consecutive failures"
            )

    def _reset_source_failure(self, source_name: str) -> None:
        with self._state_lock:
            if source_name in self._source_failure_tracking:
                self._source_failure_tracking[source_name]["consecutive_failures"] = 0
                self._source_failure_tracking[source_name]["last_failed"] = None

    def _is_source_in_cooldown(self, source_name: str) -> bool:
        with self._state_lock:
            tracking = self._source_failure_tracking.get(source_name)
            if not tracking:
                return False
            consecutive = int(tracking.get("consecutive_failures", 0) or 0)
            last_failed = tracking.get("last_failed")
            if consecutive < 3 or not isinstance(last_failed, datetime):
                return False
            elapsed = (_utc_now() - last_failed).total_seconds()
            if elapsed < self._source_cooldown_seconds:
                return True
        self._reset_source_failure(source_name)
        return False

    # ========================================================
    # EVENT NORMALIZATION HELPERS
    # ========================================================

    @staticmethod
    def _parse_float(value: Any) -> Optional[float]:
        return _coerce_float(value)

    @staticmethod
    def _event_key(event: Dict[str, Any]) -> str:
        event_id = str(event.get("id") or "").strip()
        if event_id:
            return event_id

        dt = _coerce_datetime(event.get("datetime"))
        parts = [
            dt.isoformat() if dt else "",
            str(event.get("country") or "").strip().lower(),
            str(event.get("currency") or "").strip().lower(),
            str(event.get("title") or "").strip().lower(),
        ]
        raw = "|".join(parts)
        return uuid.uuid5(uuid.NAMESPACE_URL, raw).hex

    def _build_event_payload(
        self,
        *,
        event_id: str,
        title: Any,
        country: Any,
        impact: Any,
        datetime_value: Any,
        previous: Any = None,
        forecast: Any = None,
        actual: Any = None,
        currency: Any = None,
        description: Any = "",
        source: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        dt = _coerce_datetime(datetime_value)
        if dt is None:
            return None

        clean_id = str(event_id or "").strip()
        if not clean_id or clean_id == "ff_" or len(clean_id) < 3:
            clean_id = f"evt_{uuid.uuid4().hex[:12]}"

        return {
            "id": clean_id,
            "title": str(title or "Unknown Event").strip() or "Unknown Event",
            "country": str(country or "US").strip() or "US",
            "impact": _impact_from_value(impact),
            "datetime": dt,
            "previous": _coerce_float(previous),
            "forecast": _coerce_float(forecast),
            "actual": _coerce_float(actual),
            "currency": str(currency or "USD").strip() or "USD",
            "description": str(description or ""),
            "source": str(source or "unknown"),
        }

    def _extract_html_rows(self, html_text: str) -> List[str]:
        if not html_text:
            return []
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, flags=re.IGNORECASE | re.DOTALL)
        if rows:
            return rows
        blocks = re.findall(
            r"<div[^>]*class=\"[^\"]*(?:event|calendar)[^\"]*\"[^>]*>(.*?)</div>",
            html_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return blocks

    def _extract_title_from_text(self, text: str) -> str:
        cleaned = re.sub(
            r"\s+", " ", html_lib.unescape(re.sub(r"<[^>]+>", " ", text))
        ).strip()
        if not cleaned:
            return "Unknown Event"
        parts = cleaned.split(" ")
        return cleaned if len(parts) <= 6 else " ".join(parts[:6])

    # ========================================================
    # PARSERS
    # ========================================================

    def _parse_forexfactory(self, data: Any, source_name: str) -> List[Dict[str, Any]]:
        items = data.get("events") if isinstance(data, dict) else data
        if isinstance(data, dict) and items is None:
            items = data.get("data") or data.get("results") or data.get("items") or []

        if isinstance(items, list):
            events: List[Dict[str, Any]] = []
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                dt_value = item.get("date") or item.get("datetime") or item.get("time")
                payload = self._build_event_payload(
                    event_id=str(item.get("id") or f"ff_{idx}_{uuid.uuid4().hex[:8]}"),
                    title=item.get("title") or item.get("event") or item.get("name") or "Unknown Event",
                    country=item.get("country") or "US",
                    impact=item.get("impact") or item.get("importance") or "low",
                    datetime_value=dt_value,
                    previous=item.get("previous"),
                    forecast=item.get("forecast"),
                    actual=item.get("actual"),
                    currency=item.get("currency") or "USD",
                    description=item.get("description") or "",
                    source=source_name,
                )
                if payload:
                    events.append(payload)
            return events

        if isinstance(data, dict) and "raw" in data and isinstance(data["raw"], str):
            return self._parse_html_tgju(data["raw"], source_name)

        return []

    def _parse_tradingeconomics(
        self, data: Dict[str, Any], source_name: str
    ) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            data = data.get("data") or data.get("results") or data.get("events") or []
        if not isinstance(data, list):
            return []

        events: List[Dict[str, Any]] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            payload = self._build_event_payload(
                event_id=str(item.get("id") or f"te_{idx}_{uuid.uuid4().hex[:8]}"),
                title=item.get("Event") or item.get("title") or item.get("name") or "Unknown Event",
                country=item.get("Country") or item.get("country") or "US",
                impact=item.get("Importance") or item.get("importance") or item.get("Impact") or "low",
                datetime_value=item.get("Date") or item.get("datetime") or item.get("date"),
                previous=item.get("Previous") or item.get("previous"),
                forecast=item.get("Forecast") or item.get("forecast"),
                actual=item.get("Actual") or item.get("actual"),
                currency=item.get("Currency") or item.get("currency") or "USD",
                description=item.get("Description") or item.get("description") or "",
                source=source_name,
            )
            if payload:
                events.append(payload)
        return events

    def _parse_myfxbook(self, data: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
        items = []
        if isinstance(data, dict):
            items = data.get("economicEvents") or data.get("events") or data.get("data") or []
        if not isinstance(items, list):
            return []

        events: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            payload = self._build_event_payload(
                event_id=str(item.get("id") or f"mfx_{idx}_{uuid.uuid4().hex[:8]}"),
                title=item.get("title") or item.get("name") or item.get("event") or "Unknown Event",
                country=item.get("country") or "US",
                impact=item.get("impact") or item.get("Importance") or "low",
                datetime_value=item.get("date") or item.get("time") or item.get("datetime"),
                previous=item.get("previous"),
                forecast=item.get("forecast"),
                actual=item.get("actual"),
                currency=item.get("currency") or "USD",
                description=item.get("description") or "",
                source=source_name,
            )
            if payload:
                events.append(payload)
        return events

    def _parse_fxstreet(self, data: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
        items = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("result") or data.get("events") or []
        if not isinstance(items, list):
            return []

        events: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            payload = self._build_event_payload(
                event_id=str(item.get("id") or f"fxs_{idx}_{uuid.uuid4().hex[:8]}"),
                title=item.get("title") or item.get("name") or item.get("event") or "Unknown Event",
                country=item.get("country") or "US",
                impact=item.get("impact") or item.get("importance") or "low",
                datetime_value=item.get("date") or item.get("datetime") or item.get("eventDate"),
                previous=item.get("previous"),
                forecast=item.get("forecast"),
                actual=item.get("actual"),
                currency=item.get("currency") or "USD",
                description=item.get("description") or "",
                source=source_name,
            )
            if payload:
                events.append(payload)
        return events

    def _parse_tradingview(self, data: Dict[str, Any], source_name: str) -> List[Dict[str, Any]]:
        if isinstance(data, dict):
            wrapper = data.get("data") if isinstance(data.get("data"), dict) else data
            items = wrapper.get("result") if isinstance(wrapper, dict) else []
        else:
            items = []
        if not isinstance(items, list):
            return []

        events: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            payload = self._build_event_payload(
                event_id=str(item.get("id") or f"tv_{idx}_{uuid.uuid4().hex[:8]}"),
                title=item.get("event") or item.get("title") or item.get("name") or "Unknown Event",
                country=item.get("country") or "US",
                impact=item.get("importance") or item.get("impact") or "low",
                datetime_value=item.get("eventDate") or item.get("date") or item.get("datetime"),
                previous=item.get("previous"),
                forecast=item.get("forecast"),
                actual=item.get("actual"),
                currency=item.get("currency") or "USD",
                description=item.get("description") or "",
                source=source_name,
            )
            if payload:
                events.append(payload)
        return events

    def _parse_streamdata(self, data: Any, source_name: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        if isinstance(data, dict):
            items = (
                data.get("data")
                or data.get("events")
                or data.get("results")
                or data.get("items")
                or []
            )
            if isinstance(items, list):
                for idx, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue
                    payload = self._build_event_payload(
                        event_id=str(item.get("id") or f"sd_{idx}_{uuid.uuid4().hex[:8]}"),
                        title=item.get("title") or item.get("event") or item.get("name") or "Unknown Event",
                        country=item.get("country") or "IR",
                        impact=item.get("importance") or item.get("impact") or "low",
                        datetime_value=item.get("date") or item.get("datetime") or item.get("time"),
                        previous=item.get("previous"),
                        forecast=item.get("forecast"),
                        actual=item.get("actual"),
                        currency=item.get("currency") or "IRR",
                        description=item.get("description") or "",
                        source=source_name,
                    )
                    if payload:
                        events.append(payload)
                if events:
                    return events

        html_text = ""
        if isinstance(data, str):
            html_text = data
        elif isinstance(data, dict):
            html_text = str(data.get("raw") or "")

        if not html_text:
            return events

        for idx, block in enumerate(self._extract_html_rows(html_text)):
            try:
                cleaned = html_lib.unescape(
                    re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", block))
                ).strip()
                if not cleaned:
                    continue

                dt = None
                for pattern in (
                    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
                    r"(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
                ):
                    match = re.search(pattern, cleaned)
                    if match:
                        dt = _coerce_datetime(match.group(1))
                        break
                if dt is None:
                    continue

                title = self._extract_title_from_text(block)
                impact = EventImpact.MEDIUM
                lower_block = block.lower()
                if "high" in lower_block or "بالا" in block:
                    impact = EventImpact.HIGH
                elif "low" in lower_block or "پایین" in block:
                    impact = EventImpact.LOW

                payload = self._build_event_payload(
                    event_id=f"sd_html_{idx}_{uuid.uuid4().hex[:8]}",
                    title=title,
                    country="IR",
                    impact=impact,
                    datetime_value=dt,
                    previous=None,
                    forecast=None,
                    actual=None,
                    currency="IRR",
                    description="",
                    source=source_name,
                )
                if payload:
                    events.append(payload)
            except Exception as exc:
                logger.debug(f"Error parsing streamdata HTML item {idx}: {exc}")
                continue

        return events

    def _parse_html_tgju(self, data: Any, source_name: str) -> List[Dict[str, Any]]:
        """Best-effort HTML parser for tgju-style pages."""
        if not isinstance(data, str):
            if isinstance(data, dict):
                data = str(data.get("raw") or "")
            else:
                data = str(data or "")

        html_text = data.strip()
        if not html_text:
            return []

        events: List[Dict[str, Any]] = []
        for idx, block in enumerate(self._extract_html_rows(html_text)):
            try:
                cleaned = html_lib.unescape(
                    re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", block))
                ).strip()
                if not cleaned:
                    continue

                dt = None
                for pattern in (
                    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
                    r"(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}(?::\d{2})?)",
                ):
                    match = re.search(pattern, cleaned)
                    if match:
                        dt = _coerce_datetime(match.group(1))
                        break
                if dt is None:
                    continue

                lower_block = block.lower()
                impact = EventImpact.MEDIUM
                if "high" in lower_block or "بالا" in block:
                    impact = EventImpact.HIGH
                elif "low" in lower_block or "پایین" in block:
                    impact = EventImpact.LOW

                payload = self._build_event_payload(
                    event_id=f"tgju_{idx}_{uuid.uuid4().hex[:8]}",
                    title=self._extract_title_from_text(block),
                    country="IR",
                    impact=impact,
                    datetime_value=dt,
                    previous=None,
                    forecast=None,
                    actual=None,
                    currency="IRR",
                    description="",
                    source=source_name,
                )
                if payload:
                    events.append(payload)
            except Exception as exc:
                logger.debug(f"Error parsing tgju HTML item {idx}: {exc}")
                continue

        if not events:
            logger.warning(f"[{source_name}] No tgju events extracted from HTML")
        return events

    def _parse_generic(self, data: Any, source_name: str) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = []
            for key in ("data", "events", "results", "items"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
        else:
            items = []

        if not isinstance(items, list):
            return []

        events: List[Dict[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            payload = self._build_event_payload(
                event_id=str(item.get("id") or f"gen_{idx}_{uuid.uuid4().hex[:8]}"),
                title=item.get("title") or item.get("name") or item.get("event") or "Unknown Event",
                country=item.get("country") or "US",
                impact=item.get("impact") or item.get("importance") or "low",
                datetime_value=item.get("datetime") or item.get("date") or item.get("time"),
                previous=item.get("previous"),
                forecast=item.get("forecast"),
                actual=item.get("actual"),
                currency=item.get("currency") or "USD",
                description=item.get("description") or "",
                source=source_name,
            )
            if payload:
                events.append(payload)
        return events

    # ========================================================
    # NETWORK FETCHING
    # ========================================================

    async def _fetch_from_provider(
        self,
        provider: Provider,
        session: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        source_name = provider.name
        if self._is_source_in_cooldown(source_name):
            logger.debug(f"[{source_name}] In cooldown")
            return None

        if aiohttp is None:
            logger.error(f"[{source_name}] aiohttp is not available")
            self._record_source_failure(source_name)
            return None

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7",
                "Connection": "keep-alive",
            }
            if provider.headers:
                headers.update(provider.headers)

            params = dict(provider.params or {})
            if provider.requires_api_key and provider.api_key_env:
                api_key = os.getenv(provider.api_key_env)
                if not api_key:
                    logger.debug(
                        f"[{source_name}] Missing API key env: {provider.api_key_env}"
                    )
                    self._record_source_failure(source_name)
                    return None
                if provider.name == "tradingeconomics":
                    params["c"] = api_key
                else:
                    headers["Authorization"] = f"Bearer {api_key}"

            timeout = aiohttp.ClientTimeout(
                total=max(1, provider.timeout),
                connect=max(1, min(provider.timeout, 5)),
                sock_read=max(1, provider.timeout),
            )

            owns_session = session is None
            if owns_session:
                session = aiohttp.ClientSession()

            try:
                async with session.get(
                    provider.url, headers=headers, params=params, timeout=timeout
                ) as response:
                    if response.status != 200:
                        logger.warning(f"[{source_name}] HTTP {response.status}")
                        self._record_source_failure(source_name)
                        return None

                    text = await response.text(errors="ignore")
                    if not text or not text.strip():
                        logger.warning(f"[{source_name}] Empty response")
                        self._record_source_failure(source_name)
                        return None

                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        data = {"raw": text}

                    parser_map = {
                        "forexfactory": self._parse_forexfactory,
                        "tradingeconomics": self._parse_tradingeconomics,
                        "myfxbook": self._parse_myfxbook,
                        "fxstreet": self._parse_fxstreet,
                        "tradingview": self._parse_tradingview,
                        "streamdata": self._parse_streamdata,
                        "html_tgju": self._parse_html_tgju,
                    }
                    parser_func = parser_map.get(provider.parser, self._parse_generic)
                    events = parser_func(data, source_name)

                    if not events:
                        logger.warning(f"[{source_name}] No events parsed")
                        self._record_source_failure(source_name)
                        return None

                    logger.info(f"[{source_name}] ✅ {len(events)} events")
                    self._reset_source_failure(source_name)
                    with self._state_lock:
                        source_usage = self._stats["source_usage"]
                        source_usage[source_name] = int(source_usage.get(source_name, 0)) + 1

                    return {
                        "provider": source_name,
                        "region": provider.region,
                        "score": provider.score,
                        "events": events,
                    }
            finally:
                if owns_session and session is not None:
                    await session.close()

        except asyncio.TimeoutError:
            logger.debug(f"[{source_name}] Timeout")
            self._record_source_failure(source_name)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"[{source_name}] Fetch error: {exc}")
            self._record_source_failure(source_name)
            return None

    async def _fetch_all_providers(self) -> List[Dict[str, Any]]:
        if not self.providers:
            return []

        results: List[Dict[str, Any]] = []
        iran = [p for p in self.providers if p.region == "IR" and p.enabled]
        foreign = [p for p in self.providers if p.region != "IR" and p.enabled]
        all_groups = [("Iranian", iran), ("foreign", foreign)]

        if aiohttp is None:
            logger.error("aiohttp is not installed; cannot fetch providers")
            return []

        connector = aiohttp.TCPConnector(
            limit=self._max_parallel_fetches, ttl_dns_cache=300
        )
        timeout = aiohttp.ClientTimeout(
            total=max(p.timeout for p in self.providers) + 2
        )
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for label, group in all_groups:
                if not group:
                    continue
                logger.info(f"Fetching {len(group)} {label} providers...")
                semaphore = asyncio.Semaphore(self._max_parallel_fetches)

                async def bound_fetch(
                    provider: Provider, _sem=semaphore, _session=session
                ) -> Optional[Dict[str, Any]]:
                    async with _sem:
                        return await self._fetch_from_provider(provider, session=_session)

                tasks = [bound_fetch(provider) for provider in group]
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in group_results:
                    if isinstance(result, dict):
                        results.append(result)
                    elif isinstance(result, Exception):
                        logger.debug(f"Provider fetch error: {result}")

        logger.info(f"Successful providers: {len(results)}")
        return results

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    def _deduplicate_events(
        self, provider_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        database: Dict[str, Dict[str, Any]] = {}

        for result in provider_results:
            provider = str(result.get("provider") or "unknown")
            score = float(result.get("score") or 0.0)
            events = result.get("events") or []
            if not isinstance(events, list):
                continue

            for event in events:
                if not isinstance(event, dict):
                    continue

                normalized = dict(event)
                normalized.setdefault("source", provider)
                normalized["sources"] = list(
                    dict.fromkeys([provider] + list(event.get("sources") or []))
                )
                normalized["source_score"] = score
                key = self._event_key(normalized)

                existing = database.get(key)
                if existing is None:
                    database[key] = normalized
                    continue

                existing_score = float(existing.get("source_score") or 0.0)
                merged_sources = list(
                    dict.fromkeys(list(existing.get("sources") or []) + [provider])
                )
                if score > existing_score:
                    normalized["sources"] = merged_sources
                    normalized["source_score"] = max(score, existing_score)
                    database[key] = normalized
                else:
                    existing["sources"] = merged_sources
                    existing["source_score"] = max(existing_score, score)

        merged_events = list(database.values())
        merged_events.sort(
            key=lambda event: _coerce_datetime(event.get("datetime")) or _utc_now()
        )
        return merged_events

    # ========================================================
    # PUBLIC UPDATE API
    # ========================================================

    async def update_events(self, force: bool = False) -> bool:
        """
        Returns:
            True  -> داده تازه fetch شد، یا کش معتبر (<24h) استفاده شد،
                     یا به‌دلیل interval هنوز نیازی به update نبود.
            False -> واقعاً داده‌ای دریافت نشد و کش هم قابل‌اعتماد نبود.
        """
        phase = "UPDATE_EVENTS_V6"
        now = _utc_now()

        # --- skip due to interval ---
        if not force and self._last_update_time:
            elapsed_minutes = (now - self._last_update_time).total_seconds() / 60.0
            if elapsed_minutes < self.update_interval_minutes:
                logger.debug(
                    f"[{phase}] Skipping update "
                    f"({elapsed_minutes:.1f} minutes since last update)"
                )
                return True  # هیچ شکستی رخ نداده

        # --- circuit breaker ---
        if self._api_circuit_breaker_open and not force:
            if self._api_circuit_breaker_reset_time and now < self._api_circuit_breaker_reset_time:
                logger.warning(f"[{phase}] Circuit breaker open; skipping fetch")
                return False
            # زمان بازیابی رسیده
            self._api_circuit_breaker_open = False
            self._api_circuit_breaker_reset_time = None
            self._api_failure_count = 0

        try:
            logger.info(f"[{phase}] Fetching economic events from {len(self.providers)} providers...")
            provider_results = await self._fetch_all_providers()

            # ---------- no results ----------
            if not provider_results:
                with self._state_lock:
                    has_events = bool(self.events)
                    last_success = self._last_successful_update

                if has_events and last_success is not None:
                    cache_age_hours = (now - last_success).total_seconds() / 3600.0
                    if cache_age_hours < 24:
                        logger.info(
                            f"[{phase}] Using cached events (age: {cache_age_hours:.1f}h)"
                        )
                        with self._state_lock:
                            self._update_high_impact_events()
                            self._last_update_time = now
                            self._stats["total_updates"] += 1
                            self._stats["successful_updates"] += 1
                        return True

                # شکست واقعی
                logger.warning(
                    f"[{phase}] No provider returned data and cache is stale/empty"
                )
                with self._state_lock:
                    self._stats["total_updates"] += 1
                    self._stats["failed_updates"] += 1
                    self._stats["last_error"] = (
                        "No provider returned data and cache is stale/empty"
                    )
                    self._stats["last_error_time"] = now.isoformat()
                    self._consecutive_failures += 1
                    self._api_failure_count += 1
                    opened = self._api_failure_count >= self._api_failure_threshold
                    if opened:
                        self._api_circuit_breaker_open = True
                        self._api_circuit_breaker_reset_time = now + timedelta(
                            seconds=self._api_circuit_breaker_timeout_seconds
                        )

                if opened:
                    logger.warning(
                        f"[{phase}] Circuit breaker opened after "
                        f"{self._api_failure_count} failures"
                    )

                if not has_events:
                    self._load_sample_events()

                with self._state_lock:
                    self._last_update_time = now
                return False

            # ---------- merge & persist ----------
            merged_events = self._deduplicate_events(provider_results)
            new_events: List[EconomicEvent] = []
            for event_data in merged_events:
                try:
                    new_events.append(EconomicEvent.from_dict(event_data, default_source="unknown"))
                except Exception as exc:
                    logger.debug(f"[{phase}] Error creating event: {exc}")
                    continue

            week_ago = now - timedelta(days=7)
            with self._state_lock:
                retained = [event for event in self.events if event.datetime > week_ago]
                existing_by_id = {event.id: event for event in retained}

                for event in new_events:
                    previous = existing_by_id.get(event.id)
                    if previous is None or event.datetime >= previous.datetime:
                        existing_by_id[event.id] = event

                self.events = sorted(existing_by_id.values(), key=lambda item: item.datetime)
                self._update_high_impact_events()
                self._last_update_time = now
                self._last_successful_update = now
                self._consecutive_failures = 0
                self._api_failure_count = 0
                self._stats["total_updates"] += 1
                self._stats["successful_updates"] += 1
                self._stats["high_impact_detected"] = len(
                    [e for e in self.high_impact_events if e.is_upcoming]
                )

            self._save_to_cache()

            if self._api_circuit_breaker_open:
                self._api_circuit_breaker_open = False
                self._api_circuit_breaker_reset_time = None
                self._api_failure_count = 0
                logger.info(f"[{phase}] Circuit breaker reset")

            logger.info(
                f"[{phase}] ✅ Updated: {len(self.events)} total, "
                f"{len(self.high_impact_events)} important, "
                f"from {len(provider_results)} sources"
            )
            return True

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[{phase}] Error updating events: {exc}", exc_info=True)
            with self._state_lock:
                self._stats["total_updates"] += 1
                self._stats["failed_updates"] += 1
                self._stats["last_error"] = str(exc)
                self._stats["last_error_time"] = _utc_now().isoformat()
                self._consecutive_failures += 1
                self._api_failure_count += 1
                if self._api_failure_count >= self._api_failure_threshold:
                    self._api_circuit_breaker_open = True
                    self._api_circuit_breaker_reset_time = _utc_now() + timedelta(
                        seconds=self._api_circuit_breaker_timeout_seconds
                    )
                    logger.warning(
                        f"[{phase}] Circuit breaker opened after "
                        f"{self._api_failure_count} failures"
                    )
            return False

    async def update_events_with_retry(self, force: bool = False) -> bool:
        phase = "UPDATE_WITH_RETRY"
        for attempt in range(self.max_retries):
            try:
                logger.info(f"[{phase}] Attempt {attempt + 1}/{self.max_retries}")
                success = await self.update_events(force=force)
                if success:
                    return True

                #logger.warning(
                #    f"[{phase}] Attempt {attempt + 1} did not yield usable data"
                #)
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.info(f"[{phase}] Waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[{phase}] Attempt {attempt + 1} failed: {exc}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    logger.info(f"[{phase}] Waiting {wait_time}s before retry")
                    await asyncio.sleep(wait_time)

        logger.error(f"[{phase}] All {self.max_retries} attempts failed")
        if self._consecutive_failures >= 3:
            self._recover_event_loop()
        return False

    # ========================================================
    # SAMPLE DATA
    # ========================================================

    def _load_sample_events(self) -> None:
        phase = "LOAD_SAMPLE"
        try:
            now_aware = _utc_now()
            sample_events = [
                EconomicEvent(
                    id=f"sample_{uuid.uuid4().hex[:8]}",
                    title="Sample: US Non-Farm Payrolls (Past)",
                    country="US",
                    impact=EventImpact.HIGH,
                    datetime=now_aware - timedelta(hours=2),
                    previous=150.0,
                    forecast=180.0,
                    actual=None,
                    currency="USD",
                    description="Sample event for testing (Non-blocking)",
                    source="sample",
                )
            ]
            with self._state_lock:
                self.events.extend(sample_events)
                self._update_high_impact_events()
            self._save_to_cache()
            logger.info(f"[{phase}] Loaded {len(sample_events)} sample events")
        except Exception as exc:
            logger.error(f"[{phase}] Failed to load sample events: {exc}", exc_info=True)

    # ========================================================
    # TRADING UTILITIES
    # ========================================================

    def _snapshot_high_impact(self) -> List[EconomicEvent]:
        with self._state_lock:
            return list(self.high_impact_events)

    def is_trading_allowed(self) -> Tuple[bool, str]:
        """
        ╔══════════════════════════════════════════════════════════════════════════╗
        ║   IS TRADING ALLOWED — V7 (NEVER BLOCKS, INFORMATIONAL ONLY)            ║
        ║                                                                          ║
        ║  🔴 تغییر اساسی نسبت به V6:                                             ║
        ║    • این متد هرگز `False` برنمی‌گرداند مگر با صراحت env درخواست شود.    ║
        ║    • به‌جای بلاک، فقط اطلاعات را برای کاهش جزئی اعتماد ارائه می‌دهد.    ║
        ║    • برای فعال‌کردن حالت بلاک (رفتار قدیمی V6):                         ║
        ║         ECON_HARD_BLOCK_ENABLED=true                                     ║
        ║                                                                          ║
        ║  نتیجه: در حالت پیش‌فرض، PHASE 6 در FORWARD_TEST_LIVE.py هرگز           ║
        ║         early-return نمی‌کند و کل معامله بلاک نمی‌شود.                 ║
        ╚══════════════════════════════════════════════════════════════════════════╝
        """
        phase = "TRADING_CHECK_V7"
        try:
            # ─── اگر بلاک صریحاً فعال نشده، همیشه allow ──────────────────────
            hard_block_env = os.getenv("ECON_HARD_BLOCK_ENABLED", "false")
            hard_block_enabled = str(hard_block_env).strip().lower() in (
                "1", "true", "yes", "on"
            )

            if not hard_block_enabled:
                # ─── اطلاعات‌رسانی بدون بلاک ─────────────────────────────────
                try:
                    snapshot = self._snapshot_high_impact()
                except Exception:
                    snapshot = []

                imminent: Optional[EconomicEvent] = None
                for event in snapshot:
                    try:
                        if (event.is_upcoming
                                and event.minutes_until <= self.BLOCK_TRADING_MINUTES):
                            imminent = event
                            break
                    except Exception:
                        continue

                if imminent is not None:
                    return True, (
                        f"Trading allowed (info-only mode): "
                        f"{imminent.title} in {imminent.minutes_until} min "
                        f"→ confidence reduced via multiplier"
                    )
                return True, "Trading allowed"

            # ─── حالت بلاک صریح (رفتار legacy V6) ───────────────────────────
            now_aware = _utc_now()

            if self._api_circuit_breaker_open:
                if (self._api_circuit_breaker_reset_time
                        and now_aware >= self._api_circuit_breaker_reset_time):
                    with self._state_lock:
                        self._api_circuit_breaker_open = False
                        self._api_failure_count = 0
                        self._api_circuit_breaker_reset_time = None
                else:
                    return True, "Circuit breaker open - fallback allow"

            try:
                snapshot = self._snapshot_high_impact()
            except Exception:
                snapshot = []

            for event in snapshot:
                try:
                    if (event.is_upcoming
                            and event.minutes_until <= self.BLOCK_TRADING_MINUTES):
                        with self._state_lock:
                            self._stats["trading_blocks"] = int(
                                self._stats.get("trading_blocks", 0)
                            ) + 1
                        return False, (
                            f"Trading blocked: {event.title} in "
                            f"{event.minutes_until} minutes"
                        )
                    if event.is_recent(self.BLOCK_TRADING_MINUTES):
                        with self._state_lock:
                            self._stats["trading_blocks"] = int(
                                self._stats.get("trading_blocks", 0)
                            ) + 1
                        minutes_ago = int(
                            (now_aware - event.datetime).total_seconds() / 60
                        )
                        return False, (
                            f"Trading blocked: {event.title} released "
                            f"{minutes_ago} min ago"
                        )
                except Exception:
                    continue

            return True, "Trading allowed"

        except Exception as exc:
            logger.error(
                f"[{phase}] ❌ error, allowing by fail-safe: {exc}", exc_info=True
            )
            return True, "Error in check, allowing by default (fail-safe)"
            
    def get_spread_multiplier(self) -> float:
        try:
            relevant: List[EconomicEvent] = []
            for event in self._snapshot_high_impact():
                try:
                    if event.is_upcoming and event.minutes_until <= self.INCREASE_SPREAD_MINUTES:
                        relevant.append(event)
                    elif event.is_recent(15):
                        relevant.append(event)
                except Exception:
                    continue
            if not relevant:
                return 1.0
            return max(self.SPREAD_MULTIPLIER.get(event.impact, 1.0) for event in relevant)
        except Exception:
            return 1.0

    def get_confidence_multiplier(self) -> float:
        """
        ╔══════════════════════════════════════════════════════════════════════════╗
        ║   GET CONFIDENCE MULTIPLIER — V7 (HARD 5% CAP, ENV-DRIVEN)              ║
        ║                                                                          ║
        ║  قواعد:                                                                  ║
        ║    • کف مطلق ضریب = max(ECON_MIN_MULTIPLIER, 0.95)  → حداکثر ۵٪ افت      ║
        ║    • اگر تقویم در دسترس نبود (هیچ update موفق در ۲۴ ساعت اخیر)          ║
        ║      → ضریب = ECON_UNAVAILABLE_PENALTY (پیش‌فرض 0.95)                  ║
        ║    • اگر رویدادهای نزدیک وجود داشتند → min(ضریب‌ها) ولی هرگز زیر 0.95   ║
        ║    • اگر هیچ رویداد نزدیکی نبود → 1.0                                    ║
        ║                                                                          ║
        ║  همه‌ی مقادیر قابل تنظیم از .env:                                        ║
        ║    ECON_MIN_MULTIPLIER         = 0.95  (کف سخت)                         ║
        ║    ECON_UNAVAILABLE_PENALTY    = 0.95  (جریمه نبود داده)               ║
        ║    ECON_CONF_REDUCTION_HIGH    = 0.95                                   ║
        ║    ECON_CONF_REDUCTION_MEDIUM  = 0.97                                   ║
        ║    ECON_CONF_REDUCTION_LOW     = 0.99                                   ║
        ║    ECON_CONF_REDUCTION_HOLIDAY = 0.98                                   ║
        ║    ECON_CALENDAR_STALE_HOURS   = 24                                     ║
        ╚══════════════════════════════════════════════════════════════════════════╝
        """
        phase = "GET_CONFIDENCE_MULTIPLIER_V7"

        # ─── HARD FLOOR: هرگز کمتر از این مقدار برنگردان ─────────────────────
        hard_floor = self._safe_float_env("ECON_MIN_MULTIPLIER", 0.95)
        hard_floor = float(max(0.90, min(1.00, hard_floor)))  # clip ایمنی

        try:
            # ─── CASE 1: No successful update ever → penalty + floor ────────
            last_success = self._last_successful_update
            if last_success is None:
                unavailable_penalty = self._safe_float_env(
                    "ECON_UNAVAILABLE_PENALTY", 0.95
                )
                result = max(hard_floor, min(1.00, float(unavailable_penalty)))
                logger.debug(
                    f"[{phase}] ℹ️ Calendar unavailable (no successful update) "
                    f"→ multiplier={result:.4f} (floor={hard_floor:.4f})"
                )
                return result

            # ─── CASE 2: Stale calendar (>N hours) → penalty + floor ────────
            try:
                stale_hours = self._safe_int_env(
                    "ECON_CALENDAR_STALE_HOURS", 24
                )
                age_hours = (_utc_now() - last_success).total_seconds() / 3600.0
                if age_hours > float(stale_hours):
                    unavailable_penalty = self._safe_float_env(
                        "ECON_UNAVAILABLE_PENALTY", 0.95
                    )
                    result = max(
                        hard_floor, min(1.00, float(unavailable_penalty))
                    )
                    logger.debug(
                        f"[{phase}] ℹ️ Calendar stale "
                        f"({age_hours:.1f}h > {stale_hours}h) "
                        f"→ multiplier={result:.4f}"
                    )
                    return result
            except Exception as stale_err:
                logger.debug(
                    f"[{phase}] stale-check failed, continuing: {stale_err}"
                )

            # ─── CASE 3: Upcoming events → min(factors) but NOT below floor ─
            per_impact_factor: Dict[EventImpact, float] = {
                EventImpact.HIGH: self._safe_float_env(
                    "ECON_CONF_REDUCTION_HIGH", 0.95
                ),
                EventImpact.MEDIUM: self._safe_float_env(
                    "ECON_CONF_REDUCTION_MEDIUM", 0.97
                ),
                EventImpact.LOW: self._safe_float_env(
                    "ECON_CONF_REDUCTION_LOW", 0.99
                ),
                EventImpact.HOLIDAY: self._safe_float_env(
                    "ECON_CONF_REDUCTION_HOLIDAY", 0.98
                ),
            }
            # clip تمامی ضرایب به بازه [hard_floor, 1.0]
            for impact_key in list(per_impact_factor.keys()):
                raw = float(per_impact_factor[impact_key])
                per_impact_factor[impact_key] = float(
                    max(hard_floor, min(1.00, raw))
                )

            relevant: List[EconomicEvent] = []
            try:
                snapshot = self._snapshot_high_impact()
            except Exception:
                snapshot = []

            for event in snapshot:
                try:
                    if (event.is_upcoming
                            and event.minutes_until <= self.REDUCE_CONFIDENCE_MINUTES):
                        relevant.append(event)
                        continue
                    if event.is_recent(self.REDUCE_CONFIDENCE_MINUTES):
                        relevant.append(event)
                except Exception:
                    continue

            if not relevant:
                logger.debug(
                    f"[{phase}] No relevant events → multiplier=1.0"
                )
                return 1.0

            try:
                multiplier = min(
                    per_impact_factor.get(event.impact, 1.0)
                    for event in relevant
                )
            except ValueError:
                multiplier = 1.0

            # ─── اعمال کف سخت ────────────────────────────────────────────────
            multiplier = float(max(hard_floor, min(1.00, multiplier)))

            try:
                with self._state_lock:
                    self._stats["confidence_reductions"] = int(
                        self._stats.get("confidence_reductions", 0)
                    ) + 1
            except Exception:
                pass

            logger.debug(
                f"[{phase}] relevant_events={len(relevant)} "
                f"→ multiplier={multiplier:.4f} (floor={hard_floor:.4f})"
            )
            return multiplier

        except Exception as exc:
            logger.error(
                f"[{phase}] ❌ fallback to hard_floor: {exc}", exc_info=True
            )
            return float(hard_floor)
            
    def get_adjusted_confidence(self, base_confidence: float) -> float:
        """
        ╔══════════════════════════════════════════════════════════════════════════╗
        ║   GET ADJUSTED CONFIDENCE — V7 (HARD 5% CAP)                            ║
        ║                                                                          ║
        ║  • حداکثر کاهش = ۵٪ (0.95 multiplier)                                   ║
        ║  • کف مطلق = base_confidence × 0.95                                     ║
        ║  • سقف مطلق = min(base_confidence, max_confidence)                      ║
        ║  • از اضافه‌شدن اختیاری `_min_confidence` روی سیگنال‌های خوب جلوگیری    ║
        ║    می‌کند (سیگنال 0.90 هرگز به 0.35 کاهش نمی‌یابد).                    ║
        ╚══════════════════════════════════════════════════════════════════════════╝
        """
        phase = "ADJUSTED_CONFIDENCE_V7"
        try:
            try:
                base = float(base_confidence)
            except (TypeError, ValueError):
                return 0.0

            if not math.isfinite(base):
                return 0.0

            base = float(max(0.0, min(1.0, base)))

            hard_floor_multiplier = self._safe_float_env(
                "ECON_MIN_MULTIPLIER", 0.95
            )
            hard_floor_multiplier = float(
                max(0.90, min(1.00, hard_floor_multiplier))
            )

            multiplier = self.get_confidence_multiplier()
            multiplier = float(max(hard_floor_multiplier, min(1.00, multiplier)))

            adjusted = base * multiplier

            # ─── کف مطلق: هرگز کمتر از 95٪ مقدار اولیه ─────────────────────
            absolute_floor = base * hard_floor_multiplier
            adjusted = float(max(adjusted, absolute_floor))

            # ─── سقف = base اصلی (اقتصاد نباید اعتماد را بالا ببرد) ──────────
            adjusted = float(min(adjusted, base))

            adjusted = float(max(0.0, min(1.0, adjusted)))

            logger.debug(
                f"[{phase}] base={base:.4f} × mult={multiplier:.4f} "
                f"→ adjusted={adjusted:.4f} "
                f"(hard_floor={absolute_floor:.4f})"
            )
            return adjusted

        except Exception as exc:
            logger.error(
                f"[{phase}] ❌ fallback to base: {exc}", exc_info=True
            )
            try:
                return float(max(0.0, min(1.0, float(base_confidence))))
            except Exception:
                return 0.0

    def get_adjusted_spread(self, base_spread: float) -> float:
        return float(base_spread) * self.get_spread_multiplier()

    # ========================================================
    # AUTO-UPDATE LOOP
    # ========================================================

    async def _sleep_with_stop(self, seconds: float, chunk: float = 5.0) -> None:
        remaining = max(0.0, float(seconds))
        while self._running and remaining > 0:
            delay = min(chunk, remaining)
            await asyncio.sleep(delay)
            remaining -= delay

    async def _update_loop(self) -> None:
        phase = "UPDATE_LOOP_V6"
        logger.info(
            f"[{phase}] Starting auto-update loop "
            f"(interval={self.update_interval_minutes} minutes)"
        )
        consecutive_failures = 0
        max_failures_before_cooldown = 3

        while self._running:
            try:
                success = await self.update_events_with_retry()
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(
                        f"[{phase}] Update failed. Consecutive failures: "
                        f"{consecutive_failures}"
                    )

                if consecutive_failures >= max_failures_before_cooldown:
                    cooldown_seconds = 600
                    logger.warning(
                        f"[{phase}] Too many failures. Cooling down for "
                        f"{cooldown_seconds}s"
                    )
                    await self._sleep_with_stop(cooldown_seconds, chunk=10)
                    consecutive_failures = 0
                else:
                    await self._sleep_with_stop(
                        self.update_interval_minutes * 60, chunk=10
                    )
            except asyncio.CancelledError:
                logger.info(f"[{phase}] Update loop cancelled")
                break
            except Exception as exc:
                logger.error(f"[{phase}] Update loop error: {exc}", exc_info=True)
                consecutive_failures += 1
                await self._sleep_with_stop(60, chunk=10)

        logger.info(f"[{phase}] Auto-update loop stopped")

    def start(self) -> None:
        if self._running:
            logger.warning("EconomicCalendarMonitor already running")
            return

        self._running = True
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None and running_loop.is_running():
            self._loop = running_loop
            self._update_task = running_loop.create_task(self._update_loop())
            self._background_thread = None
            logger.info("✅ EconomicCalendarMonitor started in running event loop")
            return

        self._background_thread = threading.Thread(
            target=self._background_loop_runner,
            name="EconomicCalendarMonitorLoop",
            daemon=True,
        )
        self._background_thread.start()
        logger.info("✅ EconomicCalendarMonitor started in background thread")

    def stop(self) -> None:
        self._running = False
        task = self._update_task
        loop = self._loop

        if task is not None and not task.done():
            if loop is not None and loop.is_running():
                with suppress(Exception):
                    loop.call_soon_threadsafe(task.cancel)
            else:
                with suppress(Exception):
                    task.cancel()

        thread = self._background_thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)

        self._background_thread = None
        logger.info("🛑 EconomicCalendarMonitor stopped")

    def force_update_sync(self) -> bool:
        phase = "FORCE_UPDATE_SYNC"
        loop = self._get_new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return bool(loop.run_until_complete(self.update_events_with_retry(force=True)))
        except Exception as exc:
            logger.error(f"[{phase}] Sync update failed: {exc}", exc_info=True)
            return False
        finally:
            with suppress(Exception):
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for pending_task in pending:
                    pending_task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            with suppress(Exception):
                loop.close()
            with suppress(Exception):
                asyncio.set_event_loop(None)

    def get_stats(self) -> Dict[str, Any]:
        with self._state_lock:
            stats = dict(self._stats)
            stats.update(
                {
                    "events_count": len(self.events),
                    "high_impact_count": len(self.high_impact_events),
                    "is_running": self._running,
                    "last_update": (
                        self._last_update_time.isoformat()
                        if self._last_update_time
                        else None
                    ),
                    "last_successful_update": (
                        self._last_successful_update.isoformat()
                        if self._last_successful_update
                        else None
                    ),
                    "consecutive_failures": self._consecutive_failures,
                    "update_interval_minutes": self.update_interval_minutes,
                    "active_providers": len(self.providers),
                    "max_parallel_fetches": self._max_parallel_fetches,
                    "api_circuit_breaker_open": self._api_circuit_breaker_open,
                }
            )
            return stats

    @property
    def is_running(self) -> bool:
        return self._running

    def close(self) -> None:
        self.stop()
        if self._thread_pool:
            with suppress(Exception):
                self._thread_pool.shutdown(wait=False, cancel_futures=True)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

if __name__ == "__main__":
    monitor = EconomicCalendarMonitor(auto_update=False)
    ok = monitor.force_update_sync()
    print(json.dumps({"success": ok, "stats": monitor.get_stats()}, ensure_ascii=False, indent=2))
