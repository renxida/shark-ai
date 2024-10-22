# Sharktank Tokenizer Implementation

## Relevant Files
- ./sharktank/sharktank/examples/paged_llm_v1.py
- ./sharktank/sharktank/utils/tokenizer.py

## Key Code Snippets
```python
# From paged_llm_v1.py
from ..utils.tokenizer import InferenceTokenizer

tokenizer = cli.get_tokenizer(args)

# From tokenizer.py
class InferenceTokenizer(ABC):
    def encode(self, texts: list[str], pad_to_multiple_of: int = 1, add_start_token: bool = True):
        raw_rows = self._encode(texts, add_start_token)
        raw_rows, lengths = self.pad_tokens(
            token_ids=raw_rows, pad_to_multiple_of=pad_to_multiple_of
        )
        return raw_rows, lengths

    def decode(self, tokens: Union[list[list[int]]], lens: Optional[list[int]] = None):
        if lens is not None:
            tokens = list(tokens)
            for i, row_length in enumerate(lens):
                tokens[i] = tokens[i][0:row_length]
        return self._decode(tokens)
```

## Goal
Trace the chain of command from the paged_llm_v1.py entrypoint to the IREE integration for tokenizer operations.
