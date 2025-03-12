# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
import time
import asyncio
import logging
from unittest.mock import patch, MagicMock

from shortfin_apps.llm.components.tracing import (
    LoggerTracingBackend,
    TracingConfig,
    trace_context,
    async_trace_context,
    trace_fn,
)


class TestLoggerTracingBackend:
    """Tests for the LoggerTracingBackend class."""

    def test_frame_tracking(self):
        """Test that frames are tracked correctly."""
        backend = LoggerTracingBackend()
        backend.init("TestApp")

        # Enter a frame
        backend.frame_enter("test_frame", "task123")
        assert len(backend._frames) == 1
        assert ("test_frame", "task123") in backend._frames

        # Exit the frame
        with patch("logging.Logger.info") as mock_info:
            backend.frame_exit("test_frame", "task123")
            mock_info.assert_called_once()
            assert (
                "TRACE: test_frame [task=task123] completed in"
                in mock_info.call_args[0][0]
            )

        # Check that the frame was removed
        assert len(backend._frames) == 0

    def test_frame_exit_without_enter(self):
        """Test exiting a frame that was never entered."""
        backend = LoggerTracingBackend()
        backend.init("TestApp")

        # Try to exit a frame that wasn't entered
        with patch("logging.Logger.warning") as mock_warning:
            backend.frame_exit("unknown_frame", "task123")
            mock_warning.assert_called_once()
            assert (
                "TRACE: Exit without matching enter for unknown_frame"
                in mock_warning.call_args[0][0]
            )


class TestTracingConfig:
    """Tests for the TracingConfig class."""

    def setup_method(self):
        """Reset the TracingConfig before each test."""
        TracingConfig.enabled = True
        TracingConfig.app_name = "TestApp"
        TracingConfig.backend = LoggerTracingBackend()
        TracingConfig._initialized = False

    def test_disable_tracing(self):
        """Test disabling tracing."""
        assert TracingConfig.is_enabled() is True
        TracingConfig.set_enabled(False)
        assert TracingConfig.is_enabled() is False

    def test_set_backend(self):
        """Test setting the backend to log."""
        with patch.object(LoggerTracingBackend, "init") as mock_init:
            TracingConfig.set_backend("log")
            assert isinstance(TracingConfig.backend, LoggerTracingBackend)
            mock_init.assert_called_once_with("TestApp")

    def test_set_invalid_backend(self):
        """Test setting an invalid backend."""
        with pytest.raises(ValueError, match="Unsupported tracing backend"):
            TracingConfig.set_backend("invalid")


class TestTraceContext:
    """Tests for the trace_context context manager."""

    def setup_method(self):
        """Set up the test environment."""
        TracingConfig.enabled = True
        TracingConfig.app_name = "TestApp"
        TracingConfig.backend = LoggerTracingBackend()
        TracingConfig._initialized = False

    def test_trace_context(self):
        """Test that trace_context calls frame_enter and frame_exit."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        with trace_context("test_operation", "task123"):
            # Check that frame_enter was called
            backend.frame_enter.assert_called_once_with("test_operation", "task123")
            backend.frame_exit.assert_not_called()

        # Check that frame_exit was called
        backend.frame_exit.assert_called_once_with("test_operation", "task123")

    def test_trace_context_with_exception(self):
        """Test that trace_context calls frame_exit even when an exception occurs."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        try:
            with trace_context("test_operation", "task123"):
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Check that both methods were called
        backend.frame_enter.assert_called_once_with("test_operation", "task123")
        backend.frame_exit.assert_called_once_with("test_operation", "task123")

    def test_trace_context_disabled(self):
        """Test that trace_context does nothing when tracing is disabled."""
        TracingConfig.set_enabled(False)
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        with trace_context("test_operation", "task123"):
            pass

        # Check that neither method was called
        backend.frame_enter.assert_not_called()
        backend.frame_exit.assert_not_called()


# Create a custom event loop for async tests instead of using pytest-asyncio
@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestAsyncTraceContext:
    """Tests for the async_trace_context async context manager."""

    def setup_method(self):
        """Set up the test environment."""
        TracingConfig.enabled = True
        TracingConfig.app_name = "TestApp"
        TracingConfig.backend = LoggerTracingBackend()
        TracingConfig._initialized = False

    def test_async_trace_context(self, event_loop):
        """Test that async_trace_context calls frame_enter and frame_exit."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        async def run_test():
            async with async_trace_context("async_operation", "task456"):
                # Check that frame_enter was called
                backend.frame_enter.assert_called_once_with(
                    "async_operation", "task456"
                )
                backend.frame_exit.assert_not_called()

            # Check that frame_exit was called
            backend.frame_exit.assert_called_once_with("async_operation", "task456")

        event_loop.run_until_complete(run_test())

    def test_async_trace_context_with_exception(self, event_loop):
        """Test that async_trace_context calls frame_exit even when an exception occurs."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        async def run_test():
            try:
                async with async_trace_context("async_operation", "task456"):
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Check that both methods were called
            backend.frame_enter.assert_called_once_with("async_operation", "task456")
            backend.frame_exit.assert_called_once_with("async_operation", "task456")

        event_loop.run_until_complete(run_test())


class TestTraceFn:
    """Tests for the trace_fn decorator."""

    def setup_method(self):
        """Set up the test environment."""
        TracingConfig.enabled = True
        TracingConfig.app_name = "TestApp"
        TracingConfig.backend = LoggerTracingBackend()
        TracingConfig._initialized = False

    def test_sync_function_tracing(self):
        """Test that trace_fn correctly traces a synchronous function."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        @trace_fn("custom_operation")
        def test_function():
            return "result"

        result = test_function()

        assert result == "result"
        backend.frame_enter.assert_called_once_with("custom_operation", "unknown")
        backend.frame_exit.assert_called_once_with("custom_operation", "unknown")

    def test_sync_function_default_frame_name(self):
        """Test that trace_fn uses the function name if no frame name is provided."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        @trace_fn()
        def test_default_name():
            return "result"

        result = test_default_name()

        assert result == "result"
        backend.frame_enter.assert_called_once_with("test_default_name", "unknown")
        backend.frame_exit.assert_called_once_with("test_default_name", "unknown")

    def test_async_function_tracing(self, event_loop):
        """Test that trace_fn correctly traces an asynchronous function."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        @trace_fn("async_operation")
        async def test_async_function():
            await asyncio.sleep(0.01)
            return "async result"

        result = event_loop.run_until_complete(test_async_function())

        assert result == "async result"
        backend.frame_enter.assert_called_once_with("async_operation", "unknown")
        backend.frame_exit.assert_called_once_with("async_operation", "unknown")

    def test_request_id_extraction(self):
        """Test that trace_fn extracts the request_id from the first argument."""
        backend = MagicMock(spec=LoggerTracingBackend)
        TracingConfig.backend = backend

        class DummyRequest:
            def __init__(self, request_id):
                self.request_id = request_id

        @trace_fn()
        def function_with_request(request):
            return f"Processing {request.request_id}"

        result = function_with_request(DummyRequest("req789"))

        assert result == "Processing req789"
        backend.frame_enter.assert_called_once_with("function_with_request", "req789")
        backend.frame_exit.assert_called_once_with("function_with_request", "req789")
