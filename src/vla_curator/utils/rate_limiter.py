"""
Rate limiting and retry utilities for model backend API calls.

``RateLimiter`` implements a token-bucket algorithm that enforces a
requests-per-minute cap without busy-waiting.

``RetryWithBackoff`` is a callable wrapper that retries on specified
exceptions with exponential back-off.  It is provided as a standalone
utility; the backend classes also use tenacity decorators internally.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token-bucket rate limiter.

    Usage
    -----
    limiter = RateLimiter(requests_per_minute=60)

    for item in items:
        limiter.acquire()      # blocks until a token is available
        result = api_call(item)
    """

    def __init__(self, requests_per_minute: float) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive.")
        self.interval = 60.0 / requests_per_minute
        self._last_request: float = 0.0

    def acquire(self) -> None:
        """Block until the rate limit allows the next request."""
        now = time.monotonic()
        elapsed = now - self._last_request
        wait = self.interval - elapsed
        if wait > 0:
            logger.debug("Rate limiter: sleeping %.2fs.", wait)
            time.sleep(wait)
        self._last_request = time.monotonic()

    @property
    def requests_per_minute(self) -> float:
        return 60.0 / self.interval


class RetryWithBackoff:
    """
    Retry a callable on specified exceptions with exponential back-off.

    Usage
    -----
    retry = RetryWithBackoff(
        max_attempts=5,
        base_delay=1.0,
        max_delay=60.0,
        exceptions=(RateLimitError, TimeoutError),
    )

    result = retry(api_call, args=(prompt,))
    """

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        jitter: bool = True,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exceptions = exceptions
        self.jitter = jitter

    def __call__(self, fn: Callable, args=(), kwargs=None) -> object:
        if kwargs is None:
            kwargs = {}
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except self.exceptions as exc:
                last_exc = exc
                if attempt == self.max_attempts:
                    break
                delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
                if self.jitter:
                    import random
                    delay *= (0.5 + random.random())
                logger.warning(
                    "Attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt,
                    self.max_attempts,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
        raise last_exc  # type: ignore[misc]
