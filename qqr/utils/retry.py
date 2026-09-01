import asyncio
import inspect
import logging
import reprlib
import time
from functools import wraps
from typing import Any, Callable

from .envs import RETRY_STOP_AFTER_ATTEMPT, RETRY_WAIT_FIXED

__all__ = ["retry"]

logger = logging.getLogger(__name__)


def _should_retry(
    result: Any,
    retry_msg: list,
    retry_if_result: Callable,
    retry_if_not_result: Callable,
) -> bool:
    if retry_if_result is not None and retry_if_result(result):
        retry_msg.append("Result met `retry_if_result` condition")
        return True

    if retry_if_not_result is not None and not retry_if_not_result(result):
        retry_msg.append("Result met `retry_if_not_result` condition")
        return True

    return False


def _async_wrapper(
    func: Callable,
    stop_after_attempt: int,
    wait_fixed: float,
    retry_if_result: Callable,
    retry_if_not_result: Callable,
    return_on_failure: Any,
):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        for attempt in range(stop_after_attempt):
            retry_msg = []
            try:
                retry_msg.append(
                    f"[Retry {attempt + 1}/{stop_after_attempt}] '{func.__name__}' failed"
                )
                if args:
                    retry_msg.append(f"Args: {reprlib.repr(args)}")
                if kwargs:
                    retry_msg.append(f"Kwargs: {reprlib.repr(kwargs)}")

                result = await func(*args, **kwargs)

                retry_msg.append(f"Result: {reprlib.repr(result)}")

                if _should_retry(
                    result, retry_msg, retry_if_result, retry_if_not_result
                ):
                    pass
                else:
                    return result

            except Exception as e:
                retry_msg.append(f"{type(e).__name__}: {e}")

            logger.warning(" | ".join(retry_msg))
            if attempt < stop_after_attempt - 1:
                await asyncio.sleep(wait_fixed)

        return return_on_failure

    return async_wrapper


def _sync_wrapper(
    func: Callable,
    stop_after_attempt: int,
    wait_fixed: float,
    retry_if_result: Callable,
    retry_if_not_result: Callable,
    return_on_failure: Any,
):
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        for attempt in range(stop_after_attempt):
            retry_msg = []
            try:
                retry_msg.append(
                    f"[Retry {attempt + 1}/{stop_after_attempt}] '{func.__name__}' failed"
                )
                if args:
                    retry_msg.append(f"Args: {reprlib.repr(args)}")
                if kwargs:
                    retry_msg.append(f"Kwargs: {reprlib.repr(kwargs)}")

                result = func(*args, **kwargs)

                retry_msg.append(f"Result: {reprlib.repr(result)}")

                if _should_retry(
                    result, retry_msg, retry_if_result, retry_if_not_result
                ):
                    pass
                else:
                    return result

            except Exception as e:
                retry_msg.append(f"{type(e).__name__}: {e}")

            logger.warning(" | ".join(retry_msg))
            if attempt < stop_after_attempt - 1:
                time.sleep(wait_fixed)

        return return_on_failure

    return sync_wrapper


def retry(
    stop_after_attempt: int = RETRY_STOP_AFTER_ATTEMPT,
    wait_fixed: float = RETRY_WAIT_FIXED,
    retry_if_result: Callable[[Any], bool] | None = None,
    retry_if_not_result: Callable[[Any], bool] | None = None,
    return_on_failure: Any | None = None,
):
    """
    Decorator that retries a function (Sync or Async) until a condition is met.

    Args:
        stop_after_attempt: Maximum number of attempts (default: 3).
        wait_fixed: Delay in seconds between attempts (default: 1.0).
        retry_if_result: A callable that takes the function's result and retries if the condition is met.
        retry_if_not_result: A callable that takes the function's result and retries if the condition is not met.
        return_on_failure: The value to return if all retries fail (default: None).
    """

    def decorator(func):
        wrapper_args = (
            func,
            stop_after_attempt,
            wait_fixed,
            retry_if_result,
            retry_if_not_result,
            return_on_failure,
        )

        if inspect.iscoroutinefunction(func):
            return _async_wrapper(*wrapper_args)
        else:
            return _sync_wrapper(*wrapper_args)

    return decorator
