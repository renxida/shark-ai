#!/usr/bin/env python3

import json
from pathlib import Path
import sys
from typing import List, Dict
from transformers import AutoTokenizer
from tabulate import tabulate


def load_tokenizer():
    # Using Llama2 tokenizer as default - adjust as needed
    return AutoTokenizer.from_pretrained("openlm-research/open_llama_3b_v2")


def process_dump_directory(dump_path: str, tokenizer) -> List[Dict]:
    dump_dir = Path(dump_path)
    rows = []

    # Sort directories numerically (0, 1, 2, etc.)
    subdirs = sorted(
        [d for d in dump_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name)
    )

    for subdir in subdirs:
        info_file = subdir / "info.json"
        if not info_file.exists():
            continue

        with open(info_file) as f:
            info = json.load(f)

        # Extract relevant information
        phase_type = "decode" if info["metadata"]["is_decode"] else "prefill"
        batch_size = info["batch_info"]["request_batch_size"]

        # Get cache pages for each request
        cache_pages = [req["cache_pages"] for req in info["requests"]]

        # Convert token IDs to tokens for each request
        token_sequences = []
        for req in info["requests"]:
            token_ids = req["input_token_ids"]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            token_sequences.append(tokens)

        rows.append(
            {
                "type": phase_type,
                "batch_size": batch_size,
                "cache_pages": cache_pages,
                "token_sequences": token_sequences,
            }
        )

    return rows


def format_token_sequences(token_sequences: List[List[str]]) -> str:
    if not token_sequences or not token_sequences[0]:
        return ""

    # Find the maximum length sequence and the maximum token width
    max_seq_len = max(len(seq) for seq in token_sequences)
    max_token_width = max(max(len(token) for token in seq) for seq in token_sequences)

    # Create the token table
    token_table = []

    # Add header row with positions
    header = ["Position"] + [str(i) for i in range(max_seq_len)]
    token_table.append(header)

    # Add token sequences
    for i, seq in enumerate(token_sequences):
        row = [f"Req {i}"]
        row.extend(seq + [""] * (max_seq_len - len(seq)))  # Pad with empty strings
        token_table.append(row)

    return tabulate(
        token_table,
        headers="firstrow",
        tablefmt="grid",
        maxcolwidths=[None] + [max_token_width] * max_seq_len,
    )


def format_table(rows: List[Dict]) -> str:
    table_rows = []

    for row in rows:
        # Format cache pages
        cache_pages_str = "\n".join(
            [f"Req {i}: {pages}" for i, pages in enumerate(row["cache_pages"])]
        )

        # Format token sequences using the new function
        token_seqs_str = format_token_sequences(row["token_sequences"])

        table_rows.append(
            [row["type"], row["batch_size"], cache_pages_str, token_seqs_str]
        )

    return tabulate(
        table_rows,
        headers=["Type", "Batch Size", "Cache Pages", "Token Sequences"],
        tablefmt="grid",
    )


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dump_directory>")
        sys.exit(1)

    dump_path = sys.argv[1]
    tokenizer = load_tokenizer()
    rows = process_dump_directory(dump_path, tokenizer)
    print(format_table(rows))


if __name__ == "__main__":
    main()
