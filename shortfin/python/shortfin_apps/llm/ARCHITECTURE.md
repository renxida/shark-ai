# Shortfin LLM Inference - Architecture Doc

Roughly, this is the lifecycle of a request coming from a client.

[server.py](https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/server.py) attaches a fastapi client to [components/service.py](https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/service.py)

`service.py` invokes the program loaded from VMFB at the `await fn` line (see [service.py#L463](https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/service.py#L463)).

Everything in between is documented in this diagram below:

<!-- Reviewer: see a render of this diagram at https://mermaid.live/edit#pako:eNrNV1tv2zYU_iuECgzO4FiSL_IFW7D1umJpFzRZHwYDBiUd2UQkUiWpxGmQ_95Dibo4VtYC20PyEEjU4Xc-fudG3zuRiMFZOQq-FMAjeM3oVtJszQn-hWJPtsBBUg2j_K5azKnULGI55Zq8Shlw_dT6S6qjHaHKvr6zSOXyhRQRKPVLKN2zwR-UxykoosU1cPaVaiY4-YkoLYFmjG9Pjj0g2HsNmUGvcc37AewHyukWYe0RDGh9zB7EkhZIg2gfD8CqNUWkQVBaIb-I4grJtpm2cMDjNW-lUyBvWNSv3Js9RIUWpbv3PAFpWNWLB44_FVyRDKOUElYb9tC_rJx1BbFLFcyVyE9TuEGUCsuS6zKveV_flCdztWSwoVpj7FC7TbnYnwalEOj5CndcoObx7_Wu8lPF4M_PVrFbpnckl5Aw1GhHZRvhjoAp49fdPFqRS1FIPOBvZKd1rlauu0WcIhxFInO5iE8pcw3atXkIUxG602jp0WACs2QWw3yezCazIAQ_8JZB6HtTWEA0jePQw21C6oRxN7_TO8Gb9w3Nc-Wmaeaij1xw5KLcTkG8OF96HbY2J58n03HQYWoT_NkwbSvlxbk_m3SY1iXxLKlOvGWHqi2358m0G_2yJp8Nze92m4p51QpOz84OesIrnBAauh3e9udq00eBH8UNdvWDXVfVnAFsqHmhCUsIB4gh7noqTdFdU9TntOA4zjqu8qpN1wwlRJrIbTiYzYbEH0_wn-_bxnZIxmIObSAuTCtMU3Kxowpae2uFHNpyLcKM6ap34oaDs5o_a2hUqqBp9KVgEja5GYSbRMhNOWPV4KTsye9A255cGhA0sJrUg7IPu61JKz_Uw8zsD41d36nbbeeCxkY9c9MgiRQZ-fzh7cuS0SWkRsb6hAlqXmrdIBOFgWvha1Ck1dQfvaUoUsIHP1O5VUOSsBDkrwrSZFQ-VmevdkLjqhmuPeDdPPgEupCcpALLRfVFq7Gs5zASkEpXt5ujOdfuOxx25d0HI6yKFC8bA8xR4DRMIT45yrcy18qk88c_km-vwVz6MAoib61TfCN_Y-3hBYHuK7KKoOpv_rpsrf49L-MK-Cgte1MT9ljq8YamqYjKehrUcTHrNi1N2DncdrX7j_n4VE7axLNHqPOuZPRU7v0_-Wc99qTfI3yrW16EKVO7WjiIq-K26l1UX4lpsCng1259PwH9A9n9vQznGM3HQWrulD3ZftTHLd1Oe33cwRuniQSoxkN98uO-3eA3J8LxQ03HVDh3TJd1hk4GMqMsxl8_9wZh7egdZLB2VvgY4-BbO2v-gHYUVbq845Gz0rKAoSNFsd05q4SmCt-KPEYF7O-mZhXvxf8IkdVb8NVZ3Tt7Z3UazEeLWeCPPR9n5tRb-kPnzlktFqPJ3BsHU2-ymE2X0-Bh6HwtAcYjf-H5S388X0xn82DqBw_fABb3mgc -->

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

    link ClientBatch: Source @ https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/generate.py#L90
    link GenItem: Source @ https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/generate.py#L26
    link Batcher: Source @ https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/service.py#L153
    link Executor: Source @ https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/service.py#L309
    link Service: Source @ https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/service.py#L36
    link Cache: Source @ https://github.com/nod-ai/shark-ai/blob/4c90a63e5f5de77f5356be16096b104e8ec4ddb0/shortfin/python/shortfin_apps/llm/components/kvcache/trie_attention_cache.py

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
