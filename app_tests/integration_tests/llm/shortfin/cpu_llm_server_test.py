"""Main test module for LLM server functionality."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import pytest
import requests
from typing import Dict, Any
import uuid

logger = logging.getLogger(__name__)


# TODO: move this one level up and share this with sglang tests
class AccuracyValidationException(RuntimeError):
    """Custom exception for accuracy validation failures."""

    def __init__(
        self,
        expected: str = "[[expected generation output not provided]]",
        actual: str = "[[actual generation output not provided]]",
        message: str = None,
    ):
        self.expected = expected
        self.actual = actual
        self.message = (
            message
            or f"Output validation failed.\nExpected: {expected}\nActually: {actual}"
        )
        super().__init__(self.message)


class TestLLMServer:
    """Test suite for LLM server functionality."""

    @pytest.mark.parametrize(
        "model_artifacts,server",
        [
            ("open_llama_3b", {"model": "open_llama_3b", "prefix_sharing": "none"}),
            ("open_llama_3b", {"model": "open_llama_3b", "prefix_sharing": "trie"}),
            pytest.param(
                "llama3.1_8b",
                {"model": "llama3.1_8b", "prefix_sharing": "none"},
            ),
            pytest.param(
                "llama3.1_8b",
                {"model": "llama3.1_8b", "prefix_sharing": "trie"},
            ),
        ],
        ids=[
            "open_llama_3b_none",
            "open_llama_3b_trie",
            "llama31_8b_none",
            "llama31_8b_trie",
        ],
        indirect=True,
    )
    def test_basic_generation(self, server: tuple[Any, int]) -> None:
        """Tests basic text generation capabilities.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        response = self._generate("1 2 3 4 5 ", port)
        expected_prefix = "6 7 8"
        if not response.startswith(expected_prefix):
            raise AccuracyValidationException(
                expected=f"{expected_prefix}...",
                actual=response,
                message=f"Generation did not match expected pattern.\nExpected to start with: {expected_prefix}\nActual response: {response}",
            )

    @pytest.mark.parametrize(
        "model_artifacts,server,encoded_prompt",
        [
            (
                "open_llama_3b",
                {"model": "open_llama_3b", "prefix_sharing": "none"},
                "0 1 2 3 4 5 ",
            ),
            (
                "open_llama_3b",
                {"model": "open_llama_3b", "prefix_sharing": "trie"},
                "0 1 2 3 4 5 ",
            ),
            pytest.param(
                "llama3.1_8b",
                {"model": "llama3.1_8b", "prefix_sharing": "none"},
                "0 1 2 3 4 5 ",
            ),
            pytest.param(
                "llama3.1_8b",
                {"model": "llama3.1_8b", "prefix_sharing": "trie"},
                "0 1 2 3 4 5 ",
            ),
        ],
        ids=[
            "open_llama_3b_none_input_ids",
            "open_llama_3b_trie_input_ids",
            "llama31_8b_none_input_ids",
            "llama31_8b_trie_input_ids",
        ],
        indirect=True,
    )
    def test_basic_generation_input_ids(
        self, server: tuple[Any, int], encoded_prompt
    ) -> None:
        """Tests basic text generation capabilities.

        Args:
            server: Tuple of (process, port) from server fixture
        """
        process, port = server
        assert process.poll() is None, "Server process terminated unexpectedly"

        response = self._generate(encoded_prompt, port, input_ids=True)
        expected_prefix = "6 7 8"
        if not response.startswith(expected_prefix):
            raise AccuracyValidationException(
                expected=f"{expected_prefix}...",
                actual=response,
                message=f"Generation did not match expected pattern.\nExpected to start with: {expected_prefix}\nActual response: {response}",
            )
