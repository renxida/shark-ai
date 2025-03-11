import functools
import logging
import time
import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Optional, Generator, AsyncGenerator, Callable, Union

# Configure logger
logger = logging.getLogger("shortfin-llm.tracing")

# Flag to track initialization status
_INITIALIZED = False

# Global configuration
class TracingConfig:
    enabled: bool = True
    backend: str = "log"  # 'log' or 'tracy'
    detail_level: int = 1  # 0=minimal, 1=standard, 2=verbose
    app_name: str = "ShortfinLLM"

    @classmethod
    def is_enabled(cls) -> bool:
        return cls.enabled

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        cls.enabled = enabled
        _ensure_initialized()

    @classmethod
    def set_backend(cls, backend: str) -> None:
        if backend not in ["log", "tracy"]:
            raise ValueError(f"Unsupported tracing backend: {backend}")
        cls.backend = backend
        _ensure_initialized()

    @classmethod
    def set_detail_level(cls, level: int) -> None:
        if level not in [0, 1, 2]:
            raise ValueError(f"Unsupported detail level: {level}")
        cls.detail_level = level

    @classmethod
    def set_app_name(cls, app_name: str) -> None:
        cls.app_name = app_name
        _ensure_initialized()


def _ensure_initialized() -> None:
    """Ensure that Tracy is initialized if it's being used."""
    global _INITIALIZED

    if _INITIALIZED:
        return

    if not TracingConfig.is_enabled():
        return

    if TracingConfig.backend == "tracy":
        raise NotImplementedError("Tracy tracing not yet implemented")

    _INITIALIZED = True


# Core tracing functions
def frame_enter(
    task: str, attributes: Optional[Dict[str, Any]] = None, detail_level: int = 1
) -> Optional[Dict[str, Any]]:
    """Enter a tracing frame for a task with attributes."""
    if not TracingConfig.is_enabled() or TracingConfig.detail_level < detail_level:
        return None

    _ensure_initialized()
    frame_data = {
        "task": task,
        "attributes": attributes or {},
        "start_time": time.time(),
        "detail_level": detail_level,
        "zone": None,
    }

    # If using tracy backend, create a zone
    if TracingConfig.backend == "tracy":
        raise NotImplementedError("Tracy tracing not yet implemented")

    # Log entry event if detail level high enough
    if detail_level >= 2:
        attrs_str = ""
        if attributes:
            attrs_list = [f"{k}={v}" for k, v in (attributes or {}).items()]
            attrs_str = f" [{', '.join(attrs_list)}]"

        msg = f"ENTER: {task}{attrs_str}"
        logger.info(msg)

    return frame_data


def frame_exit(
    frame_data: Optional[Dict[str, Any]], exc_info: tuple = (None, None, None)
) -> None:
    """Exit a tracing frame and log duration."""
    if frame_data is None:
        return

    if not TracingConfig.is_enabled() or TracingConfig.detail_level < frame_data.get(
        "detail_level", 0
    ):
        return

    duration = time.time() - frame_data["start_time"]
    task = frame_data["task"]
    attributes = frame_data["attributes"]

    # If using tracy backend, end the zone
    if TracingConfig.backend == "tracy" and frame_data.get("zone") is not None:
        raise NotImplementedError("Tracy tracing not yet implemented")

    # Log the duration
    duration_ms = round(duration * 1e3)
    attrs_str = ""

    if attributes:
        attrs_list = [f"{k}={v}" for k, v in attributes.items()]
        attrs_str = f" [{', '.join(attrs_list)}]"

    msg = f"EXIT: {task} in {duration_ms}ms{attrs_str}"

    if TracingConfig.backend == "log":
        logger.info(msg)
    elif TracingConfig.backend == "tracy":
        raise NotImplementedError("Tracy tracing not yet implemented")


# Context managers for manual tracing
@contextmanager
def trace_context(
    task: str, attributes: Optional[Dict[str, Any]] = None, detail_level: int = 1
) -> Generator[None, None, None]:
    """Context manager for manual tracing of code blocks."""
    frame_data = frame_enter(task, attributes, detail_level)
    try:
        yield
    finally:
        frame_exit(frame_data, exc_info=(None, None, None))


@asynccontextmanager
async def async_trace_context(
    task: str, attributes: Optional[Dict[str, Any]] = None, detail_level: int = 1
) -> AsyncGenerator[None, None]:
    """Async context manager for manual tracing of code blocks."""
    frame_data = frame_enter(task, attributes, detail_level)
    try:
        yield
    finally:
        frame_exit(frame_data, exc_info=(None, None, None))


# Decorator for function tracing
def trace_fn(task: Optional[str] = None, detail_level: int = 1):
    """
    Decorator for tracing function execution.

    Args:
        task: Description of the task being executed (defaults to function name)
        detail_level: Minimum detail level required to capture this trace
    """

    def _decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped_fn_async(*args: Any, **kwargs: Any) -> Any:
            if (
                not TracingConfig.is_enabled()
                or TracingConfig.detail_level < detail_level
            ):
                return await func(*args, **kwargs)

            # Extract function name if task not provided
            fn_task = task if task is not None else func.__name__

            # Extract request ID if available
            attributes = {}
            if len(args) > 0:
                if hasattr(args[0], "request_id"):
                    attributes["request_id"] = getattr(args[0], "request_id")
                elif (
                    hasattr(args[0], "exec_requests")
                    and len(getattr(args[0], "exec_requests", [])) > 0
                ):
                    first_req = getattr(args[0], "exec_requests")[0]
                    if hasattr(first_req, "request_id"):
                        attributes["request_id"] = getattr(first_req, "request_id")

            # Create tracing frame
            frame_data = frame_enter(fn_task, attributes, detail_level)

            try:
                # Execute the function
                ret = await func(*args, **kwargs)
                return ret
            finally:
                # End tracing frame
                frame_exit(frame_data)

        @functools.wraps(func)
        def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
            if (
                not TracingConfig.is_enabled()
                or TracingConfig.detail_level < detail_level
            ):
                return func(*args, **kwargs)

            # Extract function name if task not provided
            fn_task = task if task is not None else func.__name__

            # Extract request ID if available
            attributes = {}
            if len(args) > 0:
                if hasattr(args[0], "request_id"):
                    attributes["request_id"] = getattr(args[0], "request_id")
                elif (
                    hasattr(args[0], "exec_requests")
                    and len(getattr(args[0], "exec_requests", [])) > 0
                ):
                    first_req = getattr(args[0], "exec_requests")[0]
                    if hasattr(first_req, "request_id"):
                        attributes["request_id"] = getattr(first_req, "request_id")

            # Create tracing frame
            frame_data = frame_enter(fn_task, attributes, detail_level)

            try:
                # Execute the function
                ret = func(*args, **kwargs)
                return ret
            finally:
                # End tracing frame
                frame_exit(frame_data)

        # Return async or sync wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return wrapped_fn_async
        return wrapped_fn

    return _decorator
