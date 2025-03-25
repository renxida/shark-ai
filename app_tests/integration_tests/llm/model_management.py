"""Module for managing model artifacts through various processing stages."""
import logging
import tempfile
import zipfile
import urllib.request
from pathlib import Path
import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum, auto

from sharktank.utils.hf_datasets import Dataset, RemoteFile, get_dataset

from . import device_settings

logger = logging.getLogger(__name__)


def get_llama_cpp_path() -> Path:
    """Downloads and extracts llama.cpp if needed, returns path to installation."""
    # Use system temp directory as base
    temp_base = Path(tempfile.gettempdir()) / "sharktank_llamacpp"
    llama_cpp_dir = temp_base / "llama.cpp-b4696"

    # Only download and extract if not already present
    if not llama_cpp_dir.exists():
        temp_base.mkdir(parents=True, exist_ok=True)
        zip_path = temp_base / "llama.cpp.zip"

        # Download zip file
        logger.info("Downloading llama.cpp...")
        urllib.request.urlretrieve(
            "https://github.com/ggerganov/llama.cpp/archive/refs/tags/b4696.zip",
            zip_path,
        )

        # Extract zip file
        logger.info("Extracting llama.cpp...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_base)

        # Clean up zip file
        zip_path.unlink()

        logger.info(f"llama.cpp installed at {llama_cpp_dir}")

    return llama_cpp_dir


class AccuracyValidationException(RuntimeError):
    """Custom exception for accuracy validation failures."""

    def __init__(
        self,
        message: str = None,
        expected: str = "[[expected generation output not provided]]",
        actual: str = "[[actual generation output not provided]]",
    ):
        self.expected = expected
        self.actual = actual
        self.message = (
            message
            or f"Output validation failed.\nExpected: {expected}\nActually: {actual}"
        )
        super().__init__(self.message)


class ModelSource(Enum):
    HUGGINGFACE_FROM_GGUF = auto()
    LOCAL = auto()
    AZURE = auto()
    HUGGINGFACE_FROM_SAFETENSORS = auto()


@dataclass
class AzureConfig:
    """Configuration for Azure blob storage downloads."""

    account_name: str
    container_name: str
    blob_path: str
    auth_mode: str = "key"


@dataclass
class ModelConfig:
    """Configuration for model source and settings."""

    model_file: str
    tokenizer_id: str
    batch_sizes: Tuple[int, ...]
    device_settings: "DeviceSettings"
    source: ModelSource
    dataset_name: Optional[str] = None  # Name of the dataset in hf_datasets.py
    repo_id: Optional[str] = None
    local_path: Optional[Path] = None
    azure_config: Optional[AzureConfig] = None
    tensor_parallelism_size: Optional[
        int
    ] = None  # Number of shards for tensor parallelism

    def __post_init__(self):
        if self.source == ModelSource.HUGGINGFACE_FROM_GGUF:
            if not (self.dataset_name or self.repo_id):
                raise ValueError(
                    "Either dataset_name or repo_id required for HuggingFace models"
                )
        elif self.source == ModelSource.LOCAL and not self.local_path:
            raise ValueError("local_path required for local models")
        elif self.source == ModelSource.AZURE and not self.azure_config:
            raise ValueError("azure_config required for Azure models")
        elif self.source == ModelSource.HUGGINGFACE_FROM_SAFETENSORS:
            if not self.dataset_name:
                raise ValueError(
                    "dataset_name required for HUGGINGFACE_FROM_SAFETENSORS models"
                )


@dataclass
class ModelArtifacts:
    """Container for all paths related to model artifacts."""

    weights_path: Path  # Main weights file (unranked for sharded models)
    tokenizer_path: Path
    mlir_path: Path
    vmfb_path: Path
    config_path: Path
    model_config: ModelConfig  # config that was originally used to generate these artifacts
    shard_paths: Optional[list[Path]] = None  # Paths to sharded weight files (rank0-N)


class ModelStageManager:
    """Manages different stages of model processing with caching behavior."""

    def __init__(self, base_dir: Path, config: ModelConfig):
        self.base_dir = base_dir
        self.config = config
        self.model_dir = self._get_model_dir()
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_dir(self) -> Path:
        """Creates and returns appropriate model directory based on source."""
        if self.config.source == ModelSource.HUGGINGFACE_FROM_GGUF:
            if self.config.dataset_name:
                return self.base_dir / self.config.dataset_name.replace("/", "_")
            return self.base_dir / self.config.repo_id.replace("/", "_")
        elif self.config.source == ModelSource.LOCAL:
            return self.base_dir / "local" / self.config.local_path.stem
        elif self.config.source == ModelSource.AZURE:
            return (
                self.base_dir
                / "azure"
                / self.config.azure_config.blob_path.replace("/", "_")
            )
        elif self.config.source == ModelSource.HUGGINGFACE_FROM_SAFETENSORS:
            return self.base_dir / self.config.dataset_name.replace("/", "_")
        raise ValueError(f"Unsupported model source: {self.config.source}")

    def _download_from_huggingface(self) -> Path:
        """Downloads model from HuggingFace using hf_datasets.py."""
        model_path = self.model_dir / self.config.model_file
        if not model_path.exists():
            if self.config.dataset_name:
                logger.info(
                    f"Downloading model {self.config.dataset_name} using hf_datasets"
                )
                dataset = get_dataset(self.config.dataset_name)
                downloaded_files = dataset.download(local_dir=self.model_dir)

                # Find the model file in downloaded files
                for file_id, paths in downloaded_files.items():
                    for path in paths:
                        if path.name == self.config.model_file:
                            return path

                raise ValueError(
                    f"Model file {self.config.model_file} not found in dataset {self.config.dataset_name}"
                )
            else:
                logger.info(f"Downloading model {self.config.repo_id} from HuggingFace")
                # Create a temporary dataset for direct repo downloads
                remote_file = RemoteFile(
                    file_id="model",
                    repo_id=self.config.repo_id,
                    filename=self.config.model_file,
                )
                downloaded_paths = remote_file.download(local_dir=self.model_dir)
                return downloaded_paths[0]

        return model_path

    def _download_and_convert_from_huggingface(self) -> Path:
        """Downloads model from HuggingFace and converts through GGUF to IRPA."""
        irpa_path = self.model_dir / "model.irpa"

        if not irpa_path.exists():
            logger.info(
                f"Processing model `{self.config.dataset_name}` from HuggingFace through GGUF to IRPA"
            )

            # Step 1: Download from HuggingFace
            hf_model_path = self.model_dir / "model_hf_repo_clone"
            if not hf_model_path.exists():
                logger.info(
                    f"Downloading model from HuggingFace: `{self.config.dataset_name}`"
                )
                dataset = get_dataset(self.config.dataset_name)
                downloaded_files = dataset.download(local_dir=self.model_dir)

            # Step 2: Convert to GGUF
            gguf_path = self.model_dir / "model.gguf"
            if not gguf_path.exists():
                logger.info("Converting model to GGUF format")
                subprocess.run(
                    [
                        "python",
                        get_llama_cpp_path() / "convert_hf_to_gguf.py",
                        self.model_dir,
                        "--outfile",
                        str(gguf_path),
                        "--outtype",
                        "f32",
                    ],
                    check=True,
                )

            # Step 3: Convert to IRPA
            logger.info("Converting GGUF to IRPA format")
            subprocess.run(
                [
                    "python",
                    "-m",
                    "sharktank.tools.dump_gguf",
                    f"--gguf-file={gguf_path}",
                    "--save",
                    str(irpa_path),
                ],
                check=True,
            )

            # Cleanup intermediate files if desired
            # shutil.rmtree(hf_model_path)
            # gguf_path.unlink()

        return irpa_path

    def _copy_from_local(self) -> Path:
        """Copies model from local filesystem."""
        model_path = self.model_dir / self.config.model_file
        if not model_path.exists():
            import shutil

            logger.info(f"Copying local model from {self.config.local_path}")
            shutil.copy2(self.config.local_path, model_path)
        return model_path

    def _download_from_azure(self) -> Path:
        """Downloads model from Azure blob storage."""
        model_path = self.model_dir / self.config.model_file
        if not model_path.exists():
            logger.info(
                f"Downloading model from Azure blob storage: {self.config.azure_config.blob_path}"
            )
            subprocess.run(
                [
                    "az",
                    "storage",
                    "blob",
                    "download",
                    "--account-name",
                    self.config.azure_config.account_name,
                    "--container-name",
                    self.config.azure_config.container_name,
                    "--name",
                    self.config.azure_config.blob_path,
                    "--file",
                    str(model_path),
                    "--auth-mode",
                    self.config.azure_config.auth_mode,
                ],
                check=True,
            )
        return model_path

    def prepare_tokenizer(self) -> Path:
        """Downloads and prepares tokenizer using hf_datasets.py when possible."""
        tokenizer_path = self.model_dir / "tokenizer.json"

        if not tokenizer_path.exists():
            # First try to get tokenizer from dataset if available
            if self.config.dataset_name:
                dataset = get_dataset(self.config.dataset_name)
                downloaded_files = dataset.download(local_dir=self.model_dir)

                # Look for tokenizer files in downloaded files
                for file_id, paths in downloaded_files.items():
                    for path in paths:
                        if path.name == "tokenizer.json":
                            return path

            # Fall back to downloading from transformers if not found in dataset
            logger.info(
                f"Downloading tokenizer {self.config.tokenizer_id} using transformers"
            )
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_id)
            tokenizer.save_pretrained(self.model_dir)

        return tokenizer_path

    def shard_model(self, weights_path: Path) -> Tuple[Path, list[Path]]:
        """Shards model using tensor parallelism if configured."""
        if not self.config.tensor_parallelism_size:
            return weights_path, None

        # Determine device type from compile flags
        device_type = "cpu"  # Default to CPU
        compile_flags = self.config.device_settings.compile_flags
        for flag in compile_flags:
            if "hip" in flag.lower():
                device_type = "hip"  # Use "hip" for AMD GPU device
                break

        logger.info(
            f"Sharding model with tensor parallelism size {self.config.tensor_parallelism_size} "
            f"for device type: {device_type}"
        )

        # Determine output paths
        base_name = weights_path.stem
        output_base = self.model_dir / f"{base_name}.sharded"
        output_irpa = output_base.with_suffix(".irpa")

        # Build sharding command
        shard_cmd = [
            "python",
            "-m",
            "sharktank.examples.sharding.shard_llm_dataset",
            f"--{weights_path.suffix.strip('.')}-file={weights_path}",
            f"--output-irpa={output_irpa}",
            f"--tensor-parallelism-size={self.config.tensor_parallelism_size}",
        ]

        logger.info(f"Running sharding command: {' '.join(shard_cmd)}")

        try:
            result = subprocess.run(
                shard_cmd, check=True, capture_output=True, text=True
            )
            logger.info(f"Sharding succeeded")
        except subprocess.CalledProcessError as e:
            logger.error(f"Sharding failed with code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

        # Collect paths to all shards
        shard_paths = [
            output_base.with_suffix(f".rank{i}.irpa")
            for i in range(self.config.tensor_parallelism_size)
        ]

        logger.info(f"Model successfully sharded into {len(shard_paths)} shards")
        return output_irpa, shard_paths

    def export_model(self, weights_path: Path) -> Tuple[Path, Path]:
        """Exports model to MLIR format."""
        bs_string = ",".join(map(str, self.config.batch_sizes))
        mlir_path = self.model_dir / "model.mlir"
        config_path = self.model_dir / "config.json"
        logger.info(
            "Exporting model with following settings:\n"
            f"  MLIR Path: {mlir_path}\n"
            f"  Config Path: {config_path}\n"
            f"  Batch Sizes: {bs_string}"
        )

        # For sharded models, we use the unranked irpa file
        if self.config.tensor_parallelism_size:
            weights_path = weights_path.with_suffix(".irpa")

        # Build command
        export_cmd = [
            "python",
            "-m",
            "sharktank.examples.export_paged_llm_v1",
            "--use-attention-mask",
            "--block-seq-stride=16",
            f"--{weights_path.suffix.strip('.')}-file={weights_path}",
            f"--output-mlir={mlir_path}",
            f"--output-config={config_path}",
            f"--bs-prefill={bs_string}",
            f"--bs-decode={bs_string}",
        ]

        if self.config.tensor_parallelism_size:
            export_cmd.append(
                f"--tensor-parallelism-size={self.config.tensor_parallelism_size}"
            )

        logger.info(f"Running export command: {' '.join(export_cmd)}")

        try:
            result = subprocess.run(
                export_cmd, check=True, capture_output=True, text=True
            )
            logger.info(f"Export succeeded.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Export failed with code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

        logger.info(f"Model successfully exported to {mlir_path}")
        return mlir_path, config_path

    def compile_model(self, mlir_path: Path) -> Path:
        """Compiles model to VMFB format."""
        vmfb_path = self.model_dir / "model.vmfb"
        logger.info(f"Compiling model to {vmfb_path}")

        compile_command = [
            "iree-compile",
            str(mlir_path),
            "-o",
            str(vmfb_path),
        ]

        compile_command.extend(self.config.device_settings.compile_flags)

        # Add debug output to see what's happening
        logger.info(f"Running compiler command: {' '.join(compile_command)}")
        try:
            result = subprocess.run(
                compile_command, check=True, capture_output=True, text=True
            )
            logger.info(f"Compilation succeeded")
        except subprocess.CalledProcessError as e:
            logger.error(f"Compilation failed with code {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

        logger.info(f"Model successfully compiled to {vmfb_path}")
        return vmfb_path


class ModelProcessor:
    """Main interface for processing models through all stages."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def process_model(self, config: ModelConfig) -> ModelArtifacts:
        """Process model through all stages and return paths to all artifacts."""
        manager = ModelStageManager(self.base_dir, config)

        # Stage 1: Download weights and tokenizer (cached)
        if config.source == ModelSource.HUGGINGFACE_FROM_GGUF:
            weights_path = manager._download_from_huggingface()
        elif config.source == ModelSource.LOCAL:
            weights_path = manager._copy_from_local()
        elif config.source == ModelSource.AZURE:
            weights_path = manager._download_from_azure()
        elif config.source == ModelSource.HUGGINGFACE_FROM_SAFETENSORS:
            weights_path = manager._download_and_convert_from_huggingface()
        else:
            raise ValueError(f"Unsupported model source: {config.source}")

        tokenizer_path = manager.prepare_tokenizer()

        # Stage 1.5: Shard model if tensor parallelism is configured
        shard_paths = None
        if config.tensor_parallelism_size:
            weights_path, shard_paths = manager.shard_model(weights_path)

        # Stage 2: Export model (fresh every time)
        mlir_path, config_path = manager.export_model(weights_path)

        # Stage 3: Compile model (fresh every time)
        vmfb_path = manager.compile_model(mlir_path)

        return ModelArtifacts(
            weights_path=weights_path,
            tokenizer_path=tokenizer_path,
            mlir_path=mlir_path,
            vmfb_path=vmfb_path,
            config_path=config_path,
            model_config=config,
            shard_paths=shard_paths,
        )


TEST_MODELS = {}

TEST_MODELS["open_llama_3b"] = ModelConfig(
    source=ModelSource.HUGGINGFACE_FROM_GGUF,
    repo_id="SlyEcho/open_llama_3b_v2_gguf",
    model_file="open-llama-3b-v2-f16.gguf",
    tokenizer_id="openlm-research/open_llama_3b_v2",
    batch_sizes=(1, 4),
    device_settings=None,
)

TEST_MODELS["llama3.1_8b"] = ModelConfig(
    source=ModelSource.HUGGINGFACE_FROM_GGUF,
    repo_id="SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
    model_file="meta-llama-3.1-8b-instruct.f16.gguf",
    tokenizer_id="NousResearch/Meta-Llama-3.1-8B",
    batch_sizes=(1, 4),
    device_settings=None,
)
TEST_MODELS[
    "azure_llama"
] = ModelConfig(  # This model is currently unused. When you use it, check to make sure the irpa indeed still exist and remove this comment.
    source=ModelSource.AZURE,
    azure_config=AzureConfig(
        account_name="sharkblobs",
        container_name="halo-models",
        blob_path="llm-dev/llama3_8b/8b_f16.irpa",
    ),
    model_file="azure-llama.irpa",
    tokenizer_id="openlm-research/open_llama_3b_v2",
    batch_sizes=(1, 4),
    device_settings=None,
)

# TODO: upstream this to sharktank
Dataset(
    "Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
    files=[
        RemoteFile(
            file_id="model.safetensors",
            filename="model.safetensors",
            repo_id="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
            extra_filenames=(
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        ),
    ],
)

TEST_MODELS["tinystories_llama2_25m"] = ModelConfig(
    source=ModelSource.HUGGINGFACE_FROM_SAFETENSORS,
    dataset_name="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
    model_file="model.irpa",  # This will be the final converted file name
    tokenizer_id="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
    batch_sizes=(1, 4),
    device_settings=None,
)

# Create a model with configurable tensor parallelism
def create_shardable_model(
    model_name,
    tp_size=None,
    batch_sizes=(1, 4),
    source=None,
    repo_id=None,
    dataset_name=None,
    model_file=None,
    tokenizer_id=None,
    local_path=None,
    azure_config=None,
):
    """Create a model config with configurable tensor parallelism.

    Args:
        model_name: Name of a predefined model in TEST_MODELS, or None if providing custom params
        tp_size: Number of shards for tensor parallelism
        batch_sizes: Tuple of batch sizes to support
        source: Model source type, required if model_name is None
        repo_id: HuggingFace repo ID for GGUF models
        dataset_name: Dataset name for safetensors models
        model_file: Name of the model file
        tokenizer_id: Tokenizer ID
        local_path: Path to local model file
        azure_config: Azure configuration for Azure models

    Returns:
        ModelConfig: Configured model with tensor parallelism
    """
    if model_name and model_name in TEST_MODELS:
        # Copy the existing model config
        base_config = TEST_MODELS[model_name]
        # Create a new config with the same parameters but updated tp_size and batch_sizes
        return ModelConfig(
            source=base_config.source,
            repo_id=base_config.repo_id,
            dataset_name=base_config.dataset_name,
            model_file=base_config.model_file,
            tokenizer_id=base_config.tokenizer_id,
            batch_sizes=batch_sizes,
            device_settings=base_config.device_settings,
            local_path=base_config.local_path,
            azure_config=base_config.azure_config,
            tensor_parallelism_size=tp_size,
        )

    # Create a new config from scratch
    return ModelConfig(
        source=source,
        repo_id=repo_id,
        dataset_name=dataset_name,
        model_file=model_file,
        tokenizer_id=tokenizer_id,
        batch_sizes=batch_sizes,
        device_settings=None,
        local_path=local_path,
        azure_config=azure_config,
        tensor_parallelism_size=tp_size,
    )


# Base function to create TinyStories models with configurable tensor parallelism
def create_tinystories_model(tp_size=None, batch_sizes=(1, 4)):
    """Create a TinyStories model config with configurable tensor parallelism."""
    return create_shardable_model(
        model_name="tinystories_llama2_25m", tp_size=tp_size, batch_sizes=batch_sizes
    )


TEST_MODELS["llama3.1_8b_tp2"] = create_shardable_model(
    model_name="llama3.1_8b",
    tp_size=2,
    batch_sizes=tuple([4]),
)

# Predefined sharded versions of tinystories for common TP sizes
# These are convenience configurations that can be referenced directly in tests
TEST_MODELS["tinystories_tp2"] = create_tinystories_model(
    tp_size=2, batch_sizes=(4,)  # Fixed batch size of 4 for testing
)

TEST_MODELS["tinystories_tp4"] = create_tinystories_model(
    tp_size=4, batch_sizes=(4,)  # Fixed batch size of 4 for testing
)

TEST_MODELS["tinystories_tp8"] = create_tinystories_model(
    tp_size=8, batch_sizes=(8,)  # Fixed batch size of 8 for testing
)

# Example of a sharded model configuration
TEST_MODELS["llama3.1_405b"] = ModelConfig(
    source=ModelSource.HUGGINGFACE_FROM_GGUF,
    repo_id="meta-llama/Llama-3.1-405B",  # Note: This is a placeholder, actual repo may differ
    model_file="llama3.1-405b.f16.gguf",
    tokenizer_id="meta-llama/Llama-3.1-405B",
    batch_sizes=(1, 4),
    device_settings=None,
    tensor_parallelism_size=8,  # Required for 405B model
)

if __name__ == "__main__":
    import argparse
    import sys

    # Configure logging for script mode
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process a model and generate artifacts"
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-name", help="Name of a predefined model in TEST_MODELS"
    )
    model_group.add_argument(
        "--gguf-model", help="HuggingFace repo ID for a GGUF model"
    )
    model_group.add_argument(
        "--safetensors-model", help="HuggingFace dataset name for a safetensors model"
    )
    model_group.add_argument("--local-model", help="Path to a local model file")

    # Required params when not using a predefined model
    parser.add_argument(
        "--model-file", help="Name of the model file (required for custom models)"
    )
    parser.add_argument(
        "--tokenizer-id", help="Tokenizer ID (required for custom models)"
    )

    # Output directory
    parser.add_argument(
        "--output-dir", required=True, help="Directory to store model artifacts"
    )

    # Optional params
    parser.add_argument(
        "--tp-size", type=int, help="Tensor parallelism size (number of shards)"
    )
    parser.add_argument(
        "--batch-sizes", help="Comma-separated list of batch sizes, e.g. '1,4,8'"
    )

    # Device settings
    device_group = parser.add_argument_group("Device settings")
    device_group.add_argument(
        "--cpu", action="store_true", help="Compile for CPU (default)"
    )
    device_group.add_argument("--gpu", action="store_true", help="Compile for GPU")

    args = parser.parse_args()

    # Process batch sizes
    if args.batch_sizes:
        batch_sizes = tuple(int(bs) for bs in args.batch_sizes.split(","))
    else:
        batch_sizes = (1, 4)  # Default

    # Get device settings
    if args.gpu:
        device_settings_name = "gpu"
    else:
        device_settings_name = "cpu"

    device_settings_config = getattr(device_settings, device_settings_name)

    # Create model config based on arguments
    if args.model_name:
        # Use a predefined model with optional overrides
        model_config = create_shardable_model(
            model_name=args.model_name, tp_size=args.tp_size, batch_sizes=batch_sizes
        )
    elif args.gguf_model:
        # Custom GGUF model from HuggingFace
        if not args.model_file or not args.tokenizer_id:
            parser.error(
                "--model-file and --tokenizer-id are required for custom GGUF models"
            )

        model_config = create_shardable_model(
            model_name=None,
            source=ModelSource.HUGGINGFACE_FROM_GGUF,
            repo_id=args.gguf_model,
            model_file=args.model_file,
            tokenizer_id=args.tokenizer_id,
            tp_size=args.tp_size,
            batch_sizes=batch_sizes,
        )
    elif args.safetensors_model:
        # Custom safetensors model from HuggingFace
        if not args.tokenizer_id:
            parser.error("--tokenizer-id is required for custom safetensors models")

        model_config = create_shardable_model(
            model_name=None,
            source=ModelSource.HUGGINGFACE_FROM_SAFETENSORS,
            dataset_name=args.safetensors_model,
            model_file="model.irpa",  # Fixed output name for safetensors -> irpa conversion
            tokenizer_id=args.tokenizer_id,
            tp_size=args.tp_size,
            batch_sizes=batch_sizes,
        )
    elif args.local_model:
        # Local model file
        if not args.model_file or not args.tokenizer_id:
            parser.error(
                "--model-file and --tokenizer-id are required for local models"
            )

        model_config = create_shardable_model(
            model_name=None,
            source=ModelSource.LOCAL,
            local_path=Path(args.local_model),
            model_file=args.model_file,
            tokenizer_id=args.tokenizer_id,
            tp_size=args.tp_size,
            batch_sizes=batch_sizes,
        )

    # Set device settings on the model config
    model_config.device_settings = device_settings_config

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process the model
    logger.info(f"Processing model with the following configuration:")
    logger.info(f"  Model source: {model_config.source}")
    if model_config.repo_id:
        logger.info(f"  Repo ID: {model_config.repo_id}")
    if model_config.dataset_name:
        logger.info(f"  Dataset name: {model_config.dataset_name}")
    if model_config.local_path:
        logger.info(f"  Local path: {model_config.local_path}")
    logger.info(f"  Model file: {model_config.model_file}")
    logger.info(f"  Tokenizer ID: {model_config.tokenizer_id}")
    logger.info(f"  Batch sizes: {model_config.batch_sizes}")
    logger.info(f"  Tensor parallelism size: {model_config.tensor_parallelism_size}")
    logger.info(f"  Output directory: {output_dir}")

    # Create processor and process the model
    processor = ModelProcessor(output_dir)
    artifacts = processor.process_model(model_config)

    # Print summary of artifacts
    logger.info("Model processing completed. Artifacts generated:")
    logger.info(f"  Weights: {artifacts.weights_path}")
    logger.info(f"  Tokenizer: {artifacts.tokenizer_path}")
    logger.info(f"  MLIR: {artifacts.mlir_path}")
    logger.info(f"  VMFB: {artifacts.vmfb_path}")
    logger.info(f"  Config: {artifacts.config_path}")

    if artifacts.shard_paths:
        logger.info(f"  Shards ({len(artifacts.shard_paths)}):")
        for i, shard_path in enumerate(artifacts.shard_paths):
            logger.info(f"    Shard {i}: {shard_path}")
