# Sharktank KV Cache Implementation

## Relevant Files
- ./sharktank/sharktank/examples/paged_llm_v1.py
- ./sharktank/sharktank/layers/kv_cache.py

## Key Code Snippets
```python
# From paged_llm_v1.py
class TorchGenerator:
    def __init__(self, model: PagedLlamaModelV1, tokenizer: InferenceTokenizer, ...):
        self.model = model
        if model.cache.is_paged:
            self.shared_cache_state = model.cache.paged.allocate(page_cache_size)
        else:
            self.shared_cache_state = None

# From kv_cache.py
class PagedKVCache(BaseKVCache):
    def allocate(self, page_count: int) -> list[Union[torch.Tensor, SplitPrimitiveTensor]]:
        if self.shard_count == 1:
            return [
                torch.empty(
                    [page_count, self.page_slab_flat_dim],
                    dtype=self.dtype,
                    device=self.device,
                )
            ]
        else:
            shards = [
                torch.empty(
                    [page_count, self.page_slab_flat_dim],
                    dtype=self.dtype,
                    device=self.device,
                )
                for _ in range(self.shard_count)
            ]
            return [SplitPrimitiveTensor(ts=shards, shard_dim=1)]
```

## Goal
Trace the chain of command from the paged_llm_v1.py entrypoint to the IREE integration for KV cache operations.
