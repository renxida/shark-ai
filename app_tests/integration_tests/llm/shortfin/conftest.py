"""Test fixtures and configurations."""

import hashlib
import pytest
from pathlib import Path
from tokenizers import Tokenizer, Encoding

from ..model_management import (
    ModelProcessor,
    ModelArtifacts,
    TEST_MODELS,
)
from ..server_management import ServerInstance, ServerConfig


@pytest.fixture(scope="module")
def model_artifacts(tmp_path_factory, request):
    """Prepares model artifacts in a cached directory."""
    model_config = TEST_MODELS[request.param]
    cache_key = hashlib.md5(str(model_config).encode()).hexdigest()

    cache_dir = tmp_path_factory.mktemp("model_cache")
    model_dir = cache_dir / cache_key

    # Return cached artifacts if available
    if model_dir.exists():
        return ModelArtifacts(
            weights_path=model_dir / model_config.model_file,
            tokenizer_path=model_dir / "tokenizer.json",
            mlir_path=model_dir / "model.mlir",
            vmfb_path=model_dir / "model.vmfb",
            config_path=model_dir / "config.json",
        )

    # Process model and create artifacts
    processor = ModelProcessor(cache_dir)
    return processor.process_model(model_config)


@pytest.fixture(scope="module")
def server(model_artifacts, request):
    """Starts and manages the test server."""
    model_config = model_artifacts.model_config

    server_config = ServerConfig(
        artifacts=model_artifacts,
        device_settings=model_config.device_settings,
        prefix_sharing_algorithm=request.param.get("prefix_sharing", "none"),
    )

    server_instance = ServerInstance(server_config)
    server_instance.start()
    process, port = server_instance.process, server_instance.port
    yield process, port

    process.terminate()
    process.wait()


@pytest.fixture(scope="module")
def encoded_prompt(model_artifacts: ModelArtifacts, request) -> list[int]:
    tokenizer = Tokenizer.from_file(str(model_artifacts.tokenizer_path))
    return tokenizer.encode(request.param).ids
