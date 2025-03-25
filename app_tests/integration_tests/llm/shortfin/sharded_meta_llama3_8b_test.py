"""
Test for sharded Meta LLaMa 3.1 8B model with tensor parallelism.

This test validates tensor parallelism with a production-grade model (Meta LLaMa 3.1 8B).
It's an extension of the basic sharded model test but uses a larger, more complex model
to validate that the tensor parallelism implementation scales correctly to real-world models.

The test can run on either CPU or GPU infrastructure and tests both single requests
and concurrent workloads to validate the server's ability to handle parallel processing
with a sharded LLaMa 3.1 8B model.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import logging
import pytest
import requests
import uuid

logger = logging.getLogger(__name__)

from ..model_management import (
    AccuracyValidationException,
    ModelConfig,
    ModelSource,
    ModelProcessor,
)
import os
import pytest
from ..device_settings import get_device_settings_by_name

# Define tensor parallelism parameter that can be easily changed in a single place
TENSOR_PARALLELISM_SIZE = 2  # Change this to 2, 4, 8, etc. as needed

# Create a base model for testing - using create_shardable_model to apply tensor parallelism
from ..model_management import create_shardable_model

# Register a custom model config for this test
MODEL_NAME = "llama3.1_8b_tp2"

# Customize test parameters
BASE_MODEL = "llama3.1_8b"  # Use the predefined LLaMa 3.1 8B model

# Determine device type based on pytest test_device fixture
def is_gpu_device(device_name):
    """Determine if the device is a GPU based on name"""
    device_name = device_name.lower()
    return "gfx" in device_name or "hip" in device_name or "gpu" in device_name


# Add a fixture to determine device type
@pytest.fixture(scope="function")
def device_type(test_device):
    """Determine if we're running on CPU or GPU based on test_device"""
    return "gpu" if is_gpu_device(test_device) else "cpu"


# Check for sufficient compute resources for parallel testing
import multiprocessing


def check_available_resources(device_type):
    """Check available compute resources based on device type"""
    if device_type == "cpu":
        try:
            # Get logical CPU count using multiprocessing
            cpu_count = multiprocessing.cpu_count()
            return cpu_count, "CPU cores"
        except Exception as e:
            logger.warning(f"Failed to check CPU count: {e}")
            return 4, "CPU cores"  # Assume 4 CPU cores if check fails
    else:
        # Check GPU devices - in a real environment, we would query actual GPU count
        # For now, we'll assume 4 GPUs if using GPU device type
        return 4, "GPU devices"  # Default assumption for testing


# Initialize the custom model configuration before tests run
@pytest.fixture(scope="module", autouse=True)
def setup_custom_model():
    """Setup custom model configuration for LLaMa 3.1 8B with sharding."""
    from ..model_management import TEST_MODELS

    # Create the shardable model based on the base LLaMa 3.1 8B model
    TEST_MODELS[MODEL_NAME] = create_shardable_model(
        model_name=BASE_MODEL,
        tp_size=TENSOR_PARALLELISM_SIZE,
        batch_sizes=(1, 2),  # Reduced batch sizes for larger model
    )

    # Log the model configuration
    logger.info(
        f"Registered custom model {MODEL_NAME} with tensor parallelism size {TENSOR_PARALLELISM_SIZE}"
    )

    yield

    # Cleanup (optional)
    if MODEL_NAME in TEST_MODELS:
        del TEST_MODELS[MODEL_NAME]


# Setup the model configuration for tests
# Note: The actual device settings are automatically applied by the model_artifacts fixture
pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        [MODEL_NAME, {"prefix_sharing": "none"}],
    ],
    indirect=True,
)

# Test prompts appropriate for LLaMa 3.1
PROMPT = "What is the capital of the United States?"
EXPECTED_PATTERN = "The capital of the United States is Washington, D.C. Washington, D.C."  # Most explanations will contain this word


class TestShardedLlama31Server:
    """Test suite for sharded LLaMa 3.1 8B model server functionality on both CPU and GPU."""

    def test_non_concurrent_request_sharded(
        self, server: tuple[Any, int], test_device, device_type
    ) -> None:
        """Tests single request generation with a sharded LLaMa 3.1 8B model.

        Sends a single request to test the server's ability to handle
        a basic request with the sharded LLaMa 3.1 8B model using tensor parallelism.
        This test validates that a production-grade model can be correctly sharded
        and served for inference.

        Args:
            server: Tuple of (process, port) from server fixture
            test_device: The device being tested (from pytest fixture)
            device_type: Simplified device type (cpu/gpu) from fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        available_resources, resource_type = check_available_resources(device_type)
        minimum_resources_needed = TENSOR_PARALLELISM_SIZE
        if available_resources < minimum_resources_needed:
            logger.warning(
                f"Available {resource_type} ({available_resources}) may be insufficient "
                f"for tensor parallelism size {TENSOR_PARALLELISM_SIZE}. "
                f"Recommended minimum: {minimum_resources_needed} {resource_type}."
            )

        logger.info(
            f"Testing with {test_device} (type: {device_type}), tensor parallelism: {TENSOR_PARALLELISM_SIZE}, "
            f"single request"
        )

        # Send a single request
        response = self._generate(PROMPT, port)
        if EXPECTED_PATTERN not in response:
            raise AccuracyValidationException(
                expected=f"...{EXPECTED_PATTERN}...",
                actual=response,
                message=f"Generation did not contain expected pattern.\nExpected to contain: {EXPECTED_PATTERN}\nActual response: {response}",
            )

    @pytest.mark.xfail(
        reason="Memory access fault by GPU node-3 (Agent handle: 0x555c24e83f80) on address 0x7fc28a1e4000. Reason: Unknown."
    )
    def test_concurrent_generation_sharded(
        self, server: tuple[Any, int], test_device, device_type
    ) -> None:
        """Tests concurrent text generation with a sharded LLaMa 3.1 8B model.

        Uses multiple concurrent requests to test the server's ability to handle
        parallel workloads with the sharded LLaMa 3.1 8B model using tensor parallelism.
        This tests the production readiness of the tensor parallelism implementation
        with a larger, more complex model under concurrent load.

        Args:
            server: Tuple of (process, port) from server fixture
            test_device: The device being tested (from pytest fixture)
            device_type: Simplified device type (cpu/gpu) from fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        available_resources, resource_type = check_available_resources(device_type)
        minimum_resources_needed = TENSOR_PARALLELISM_SIZE
        if available_resources < minimum_resources_needed:
            logger.warning(
                f"Available {resource_type} ({available_resources}) may be insufficient "
                f"for tensor parallelism size {TENSOR_PARALLELISM_SIZE}. "
                f"Recommended minimum: {minimum_resources_needed} {resource_type}."
            )

        concurrent_requests = 3

        logger.info(
            f"Testing with {test_device} (type: {device_type}), tensor parallelism: {TENSOR_PARALLELISM_SIZE}, "
            f"concurrent_requests: {concurrent_requests}"
        )

        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [
                executor.submit(self._generate, PROMPT, port)
                for _ in range(concurrent_requests)
            ]

            for future in as_completed(futures):
                response = future.result()
                if EXPECTED_PATTERN not in response:
                    raise AccuracyValidationException(
                        expected=f"...{EXPECTED_PATTERN}...",
                        actual=response,
                        message=f"Generation did not contain expected pattern.\nExpected to contain: {EXPECTED_PATTERN}\nActual response: {response}",
                    )

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
            "text": prompt,
            "sampling_params": {
                "max_completion_tokens": 15,
                "temperature": 0.0,
            },
            "rid": uuid.uuid4().hex,
            "stream": False,
        }

        response = requests.post(
            f"http://localhost:{port}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.text
        if not data.startswith("data: "):
            raise AccuracyValidationException(
                expected="Response starting with 'data: '",
                actual=data,
                message=f"Invalid response format.\nExpected format starting with 'data: '\nActual response: {data}",
            )

        return data[6:].rstrip("\n")
