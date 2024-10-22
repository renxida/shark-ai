import sys
import os
from pathlib import Path

# Add necessary paths to sys.path
sys.path.append(os.path.abspath("./sharktank"))
sys.path.append(os.path.abspath("./shortfin/python"))

# Sharktank imports
from sharktank.utils.tokenizer import InferenceTokenizer
from sharktank.utils import cli
import argparse

# Shortfin imports
from shortfin_apps.llm.components.tokenizer import Tokenizer


def test_sharktank_tokenizer(text):
    print("Testing Sharktank Tokenizer:")
    tokenizer_config_path = Path("/tmp/sharktank/llama/tokenizer_config.json")
    tokenizer = cli.get_tokenizer(
        argparse.Namespace(
            tokenizer_config_json=tokenizer_config_path, tokenizer_type=None
        )
    )
    encoded, lengths = tokenizer.encode([text])
    print(f"Encoded (token IDs): {encoded[0]}")
    print(f"Lengths: {lengths}")
    decoded = tokenizer.decode(encoded, lengths)
    print(f"Decoded: {decoded}")
    print()


def test_shortfin_tokenizer(text):
    print("Testing Shortfin Tokenizer:")
    tokenizer = Tokenizer.from_tokenizer_json_file(
        "/tmp/sharktank/llama/tokenizer.json"
    )
    encoded = tokenizer.encode([text])
    print(f"Encoded (token IDs): {encoded[0].ids}")
    decoded = tokenizer.decode([e.ids for e in encoded])
    print(f"Decoded: {decoded}")
    print()


if __name__ == "__main__":
    test_text = "Hello, world! This is a test of the tokenizer implementations."

    test_sharktank_tokenizer(test_text)
    test_shortfin_tokenizer(test_text)
