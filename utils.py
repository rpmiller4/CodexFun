from __future__ import annotations
import time
from typing import Any, Callable, Dict, Tuple

MIN_SIGMA = 0.12  # 12 vol hard floor


def capped_sigma(raw_sigma: float, floor: float = MIN_SIGMA) -> float:
    """Return sigma respecting the minimum floor."""
    return max(raw_sigma, floor)


_fetch_cache: Dict[Tuple[str, Tuple[Any, ...], Tuple[Tuple[str, Any], ...]], Any] = {}


def fetch_with_retry(func: Callable[..., Any], *args: Any, retries: int = 3, delay: float = 1.0, **kwargs: Any) -> Any:
    """Call ``func`` with retries and simple caching."""
    key = (getattr(func, "__qualname__", repr(func)), args, tuple(sorted(kwargs.items())))
    if key in _fetch_cache:
        return _fetch_cache[key]
    last_exc: Exception | None = None
    for _ in range(retries):
        try:
            result = func(*args, **kwargs)
            _fetch_cache[key] = result
            return result
        except Exception as exc:  # pragma: no cover - network errors hard to simulate
            last_exc = exc
            time.sleep(delay)
    if last_exc:
        raise last_exc
    raise RuntimeError("fetch_with_retry failed")
