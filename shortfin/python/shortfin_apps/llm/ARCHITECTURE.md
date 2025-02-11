# Shortfin LLM Inference - Architecture Doc

Roughly, this is the lifecycle of a request coming from a client.

`server.py` attaches a fastapi client to `components/service.py`

`service.py` invokes the program loaded from VMFB at the `await fn` line (ctrl+f `await fn` to find it).

Everything in between is documented in this diagram below:

```mermaid
sequenceDiagram
    box generate.py
    participant Client
    participant ClientBatch as ClientGenerateBatchProcess<br/>(Handles tokenization & streaming)
    participant GenItem as GenerateItemProcess<br/>(Manages generation sequence)
    participant Batcher as BatcherProcess<br/>(Batches requests & cache mgmt)
    end

    box service.py
    participant Executor as InferenceExecutorProcess<br/>(Runs model inference)
    participant Service as GenerateService<br/>(Top-level model service)
    end
    box kvcache/trie_attention_cache.py
    participant Cache as TriePagedAttentionCache<br/>(KV cache with prefix sharing)
    end

    Client->>ClientBatch: Create generation request
    Note over ClientBatch: Tokenize input if needed
    ClientBatch->>GenItem: Launch generation process

    rect rgb(55, 123, 111)
        Note over GenItem,Cache: Prefill Phase
        GenItem->>Batcher: Submit prefill request
        Batcher->>Cache: acquire_pages_for_tokens()<br/>Get cache pages for input sequence
        Batcher->>Executor: Create executor for batch
        Note over Executor: Load program from VMFB<br/>Select prefill function for batch size
        Executor->>Service: await fn(*args, fiber=self.fiber)<br/>Execute prefill inference
        Executor-->>GenItem: Return logits
        GenItem->>GenItem: Generate first token
    end

    GenItem->>ClientBatch: Stream results (if enabled)

    rect rgb(123, 55, 112)
        Note over GenItem,Cache: Decode Loop
        loop Until max tokens or EOS
            GenItem->>Batcher: Submit decode request
            Batcher->>Cache: extend_allocation()<br/>Extend cache for new token
            Batcher->>Executor: Create executor for batch
            Note over Executor: Select decode function<br/>for batch size
            Executor->>Service: await fn(*args, fiber=self.fiber)<br/>Execute decode inference
            Executor->>Cache: publish_allocated_pages()<br/>Publish completed cache pages
            Executor-->>GenItem: Return logits
            GenItem->>GenItem: Generate next token
        end
    end

    GenItem-->>ClientBatch: Complete generation
    Note over GenItem: free_cache_pages()
    ClientBatch-->>Client: Return final response
```
