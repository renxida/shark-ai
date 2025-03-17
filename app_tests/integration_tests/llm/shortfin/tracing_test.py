"""
Integration test to ensure tracing functionality works correctly with the Shortfin LLM server.
This tests that the server can generate a valid tracy capture file without crashing.
"""

import logging
import os
import pytest
import requests
import time
import uuid
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

from ..model_management import AccuracyValidationException
from ..server_management import ServerInstance, ServerConfig

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        ["tinystories_llama2_25m", {"prefix_sharing": "none"}],
    ],
    indirect=True,
)

# Same golden values as in tinystories_llama2_25m_test.py
GOLDEN_PROMPT = "Once upon a time"
GOLDEN_RESPONSE = ", there was a little girl named Lily."  # this assumes purely deterministic greedy search


class TestTracingFunctionality:
    """Test suite for tracing functionality with the LLM server."""

    def test_tracing_generation(self, server: Tuple[Any, int], tmp_path: Path) -> None:
        """Tests that tracing works while running text generation.

        Args:
            server: Tuple of (process, port) from server fixture
            tmp_path: Temporary directory for storing the trace file
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        # Determine output directory for artifacts (use environment variable if set, otherwise tmp_path)
        output_dir = os.environ.get("TEST_OUTPUT_DIR")
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = tmp_path

        # Start tracy capture
        capture_file = output_dir / "capture.tracy"
        logger.info(f"Tracy capture will be saved to: {capture_file}")

        tracy_process = subprocess.Popen(
            ["iree-tracy-capture", "-o", str(capture_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            # Wait a moment for tracy capture to initialize
            time.sleep(3)

            # Run generation with tracing enabled
            prompt = GOLDEN_PROMPT
            expected_prefix = GOLDEN_RESPONSE

            # Set environment variable to enable tracing for the request if not already set
            tracy_env_was_set = "SHORTFIN_PY_RUNTIME" in os.environ
            if not tracy_env_was_set:
                os.environ["SHORTFIN_PY_RUNTIME"] = "tracy"

            # Make generation request
            response = self._generate(prompt, port)

            # Verify the response is correct to ensure basic functionality works
            if not expected_prefix in response:
                raise AccuracyValidationException(
                    expected=f"{expected_prefix}...",
                    actual=response,
                    message=f"Generation did not match expected pattern.\nExpected to start with: {expected_prefix}\nActual response: {response}",
                )

            # Clean up environment variable if we set it
            if not tracy_env_was_set:
                os.environ.pop("SHORTFIN_PY_RUNTIME", None)

            # Wait a moment for trace data to be collected
            time.sleep(5)

            # Terminate tracy capture process
            tracy_process.terminate()
            tracy_process.wait(timeout=10)

            # Verify that the trace file exists and has content
            assert capture_file.exists(), "Tracy capture file was not created"
            assert capture_file.stat().st_size > 0, "Tracy capture file is empty"

            logger.info(f"Successfully generated tracy capture at {capture_file}")

        finally:
            # Ensure the tracy process is terminated
            if tracy_process.poll() is None:
                tracy_process.terminate()
                try:
                    tracy_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    tracy_process.kill()

    def _generate(self, prompt: str, port: int) -> str:
        """Helper method to make generation request to server.

        Args:
            prompt: Input text prompt
            port: Server port number

        Returns:
            Generated text response

        Raises:
            requests.exceptions.RequestException: If request fails
            AccuracyValidationException: If response format is invalid
        """
        payload = {
            "sampling_params": {"max_completion_tokens": 15, "temperature": 0.7},
            "rid": uuid.uuid4().hex,
            "stream": False,
            "text": prompt,
        }

        response = requests.post(
            f"http://localhost:{port}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        # Parse and validate streaming response format
        data = response.text
        if not data.startswith("data: "):
            raise AccuracyValidationException(
                expected="Response starting with 'data: '",
                actual=data,
                message=f"Invalid response format.\nExpected format starting with 'data: '\nActual response: {data}",
            )

        return data[6:].rstrip("\n")
