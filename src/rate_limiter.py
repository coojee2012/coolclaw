"""Async token-bucket rate limiter with RPM + RPD tracking.

Per-provider isolation, configurable RPM/RPD/burst/queue, asyncio-only.
Daily quota resets at midnight Pacific Time (Google's convention).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

PACIFIC = timezone(timedelta(hours=-7))


class RateLimitExceeded(Exception):
    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"Rate limit exceeded for {provider}: {reason}")


@dataclass
class ProviderStats:
    processed: int = 0
    rejected: int = 0
    waiting: int = 0
    total_wait_ms: float = 0.0
    daily_count: int = 0
    daily_limit: int = 0
    daily_reset_pst: str = ""


@dataclass
class _Bucket:
    capacity: int
    tokens: float
    refill_rate: float
    max_queue: int
    timeout: float
    rpd: int = 0
    cooldown_seconds: float = 60.0

    queue: asyncio.Queue = field(init=False)
    lock: asyncio.Lock = field(init=False)
    stats: ProviderStats = field(default_factory=ProviderStats, init=False)
    _day_key: str = field(default="", init=False)
    _last_429: float = field(default=0.0, init=False)
    _429_count_today: int = field(default=0, init=False)
    _last_refill: float = field(default=0.0, init=False)
    _rpd_exhausted: bool = field(default=False, init=False)
    def __post_init__(self):
        self.queue = asyncio.Queue(maxsize=self.max_queue)
        self.lock = asyncio.Lock()
        self._last_refill = time.monotonic()
        self._refresh_day()

    def _refresh_day(self):
        now = datetime.now(PACIFIC)
        key = now.strftime("%Y-%m-%d")
        if key != self._day_key:
            self._day_key = key
            self.stats.daily_count = 0
            self._429_count_today = 0
            self._last_429 = 0.0
            self._rpd_exhausted = False
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            self.stats.daily_reset_pst = tomorrow.strftime("%H:%M PT")
        self.stats.daily_limit = self.rpd


class RateLimiter:
    def __init__(self):
        self._buckets: dict[str, _Bucket] = {}
        self._active: dict[str, int] = {}

    def add_provider(
        self,
        name: str,
        rpm: int = 10,
        burst: int | None = None,
        queue_size: int = 50,
        timeout: float = 30,
        rpd: int = 0,
        cooldown_seconds: float = 60.0,
    ):
        cap = burst if burst is not None else rpm
        bucket = _Bucket(
            capacity=cap,
            tokens=float(cap),
            refill_rate=rpm / 60.0,
            max_queue=queue_size,
            timeout=timeout,
            rpd=rpd,
            cooldown_seconds=cooldown_seconds,
        )
        self._buckets[name] = bucket
        self._active[name] = 0
        rpd_info = f", {rpd} RPD" if rpd else ""
        logger.info(
            f"[RATE] {name}: {rpm} RPM, burst={cap}, "
            f"queue={queue_size}{rpd_info}, cooldown={cooldown_seconds}s"
        )

    async def acquire(self, provider: str) -> bool:
        bucket = self._buckets.get(provider)
        if bucket is None:
            return True

        bucket._refresh_day()

        if bucket.rpd > 0 and bucket.stats.daily_count >= bucket.rpd:
            bucket.stats.rejected += 1
            raise RateLimitExceeded(
                provider,
                f"daily limit {bucket.rpd} reached "
                f"({bucket.stats.daily_count}/{bucket.rpd}), "
                f"resets at {bucket.stats.daily_reset_pst}",
            )

        try:
            bucket.queue.put_nowait(True)
        except asyncio.QueueFull:
            bucket.stats.rejected += 1
            raise RateLimitExceeded(
                provider,
                f"queue full ({bucket.queue.qsize()}/{bucket.max_queue})",
            )

        bucket.stats.waiting = bucket.queue.qsize()
        wait_start = time.monotonic()

        try:
            await asyncio.wait_for(
                self._wait_for_token(bucket), timeout=bucket.timeout
            )
            elapsed_ms = (time.monotonic() - wait_start) * 1000
            bucket.stats.total_wait_ms += elapsed_ms
            bucket.stats.processed += 1
            bucket.stats.daily_count += 1
            self._active[provider] = self._active.get(provider, 0) + 1

            remaining = (bucket.rpd - bucket.stats.daily_count) if bucket.rpd else -1
            if bucket.rpd and remaining <= 50:
                logger.warning(
                    f"[RATE] {provider}: {remaining} daily requests left "
                    f"({bucket.stats.daily_count}/{bucket.rpd})"
                )
            elif elapsed_ms > 100:
                logger.debug(
                    f"[RATE] {provider}: waited {elapsed_ms:.0f}ms"
                )
            return True
        except asyncio.TimeoutError:
            try:
                bucket.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            bucket.stats.rejected += 1
            bucket.stats.waiting = bucket.queue.qsize()
            raise RateLimitExceeded(
                provider,
                f"wait timeout ({bucket.timeout}s)",
            )

    def release(self, provider: str):
        bucket = self._buckets.get(provider)
        if bucket is None:
            return

        self._active[provider] = max(0, self._active.get(provider, 1) - 1)

        try:
            bucket.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        bucket.stats.waiting = bucket.queue.qsize()

    async def _wait_for_token(self, bucket: _Bucket):
        while True:
            async with bucket.lock:
                now = time.monotonic()
                elapsed = now - bucket._last_refill
                bucket.tokens = min(
                    float(bucket.capacity),
                    bucket.tokens + elapsed * bucket.refill_rate,
                )
                bucket._last_refill = now
                if bucket.tokens >= 1.0:
                    bucket.tokens -= 1.0
                    return
                wait_time = (1.0 - bucket.tokens) / bucket.refill_rate
            await asyncio.sleep(wait_time)

    def get_stats(self) -> dict[str, dict]:
        result = {}
        for name, bucket in self._buckets.items():
            bucket._refresh_day()
            rpd_used = bucket.stats.daily_count
            rpd_limit = bucket.rpd
            result[name] = {
                "rpm": int(bucket.refill_rate * 60),
                "processed": bucket.stats.processed,
                "rejected": bucket.stats.rejected,
                "active": self._active.get(name, 0),
                "avg_wait_ms": (
                    bucket.stats.total_wait_ms / bucket.stats.processed
                    if bucket.stats.processed > 0
                    else 0
                ),
                "rpd_used": rpd_used,
                "rpd_limit": rpd_limit,
                "rpd_remaining": (rpd_limit - rpd_used) if rpd_limit else -1,
                "rpd_resets": bucket.stats.daily_reset_pst,
            }
        return result

    def get_active_count(self, provider: str) -> int:
        return self._active.get(provider, 0)

    def mark_exhausted(self, provider: str):
        bucket = self._buckets.get(provider)
        if bucket:
            bucket._last_429 = time.monotonic()
            bucket._429_count_today += 1
            logger.warning(
                f"[RATE] {provider}: cooldown {bucket.cooldown_seconds}s "
                f"(429 #{bucket._429_count_today})"
            )

    def mark_rpd_exhausted(self, provider: str):
        bucket = self._buckets.get(provider)
        if bucket:
            bucket._rpd_exhausted = True
            bucket.stats.daily_count = bucket.rpd + 1
            logger.warning(
                f"[RATE] {provider}: marked RPD exhausted for today"
            )

    def is_exhausted(self, provider: str) -> bool:
        bucket = self._buckets.get(provider)
        if not bucket:
            return False
        bucket._refresh_day()
        if bucket._rpd_exhausted or bucket.stats.daily_count >= bucket.rpd:
            return True
        if bucket._last_429 > 0:
            elapsed = time.monotonic() - bucket._last_429
            if elapsed < bucket.cooldown_seconds:
                return True
        return False

    def get_cooldown_remaining(self, provider: str) -> float:
        bucket = self._buckets.get(provider)
        if not bucket or bucket._last_429 <= 0:
            return 0.0
        elapsed = time.monotonic() - bucket._last_429
        remaining = bucket.cooldown_seconds - elapsed
        return max(0.0, remaining)

    def shutdown(self):
        for bucket in self._buckets.values():
            pass


rate_limiter = RateLimiter()
