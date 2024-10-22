# Shortfin Tokenizer Implementation

## Relevant Files
- shortfin/python/shortfin_apps/llm/server.py
- shortfin/python/shortfin_apps/llm/components/tokenizer.py

## Key Code Snippets
```python
# From server.py
from .components.tokenizer import Tokenizer

tokenizer = Tokenizer.from_tokenizer_json_file(args.tokenizer_json)

# From tokenizer.py
class Tokenizer:
    @staticmethod
    def from_tokenizer_json_file(json_path: Path | str):
        return Tokenizer(tokenizers.Tokenizer.from_file(str(json_path)))

    def encode(self, texts: list[str]) -> list[tokenizers.Encoding]:
        return self._raw.encode_batch(texts)

    def decode(self, sequences) -> list[str]:
        return self._raw.decode_batch(sequences)
```

## Goal
Trace the chain of command from the server.py entrypoint to the IREE integration for tokenizer operations.
