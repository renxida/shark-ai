#!/bin/bash

# Exit on error
set -e

# Initialize pyenv
eval "$(pyenv init -)"
pyenv activate shark-env

# Set up variables
ARTIFACTS_DIR="$HOME/debug_artifacts/SlyEcho_open_llama_3b_v2_gguf"
MODEL_REPO="SlyEcho/open_llama_3b_v2_gguf"
MODEL_FILE="open-llama-3b-v2-f16.gguf"
TOKENIZER_REPO="openlm-research/open_llama_3b_v2"

# Create artifacts directory
echo "Creating artifacts directory..."
mkdir -p "$ARTIFACTS_DIR"
cd "$ARTIFACTS_DIR"

# Download model if not exists
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading model from HuggingFace..."
    huggingface-cli download --local-dir . "$MODEL_REPO" "$MODEL_FILE"
fi

# Download tokenizer if not exists
if [ ! -f "tokenizer.json" ]; then
    echo "Downloading tokenizer..."
    python3 -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('$TOKENIZER_REPO')
tokenizer.save_pretrained('.')
"
fi

# Export model if not exists
if [ ! -f "model.mlir" ]; then
    echo "Exporting model..."
    python3 -m sharktank.examples.export_paged_llm_v1 \
        --block-seq-stride=16 \
        --gguf-file="$ARTIFACTS_DIR/$MODEL_FILE" \
        --output-mlir="$ARTIFACTS_DIR/model.mlir" \
        --output-config="$ARTIFACTS_DIR/config.json" \
        --bs=1,4
fi

# Compile model if not exists
if [ ! -f "model.vmfb" ]; then
    echo "Compiling model..."
    iree-compile "$ARTIFACTS_DIR/model.mlir" \
        -o "$ARTIFACTS_DIR/model.vmfb" \
        --iree-hal-target-backends=llvm-cpu
fi

# Launch server
echo "Starting server..."
exec python3 -m shortfin_apps.llm.server \
    --tokenizer_json="$ARTIFACTS_DIR/tokenizer.json" \
    --model_config="$ARTIFACTS_DIR/config.json" \
    --vmfb="$ARTIFACTS_DIR/model.vmfb" \
    --parameters="$ARTIFACTS_DIR/$MODEL_FILE" \
    --port=8080 \
    --device=local-task
