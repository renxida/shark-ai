# Changes Made for KV Cache Debugging

1. Created a new file: test_kvcache.py
   - This file contains a test script to exercise the KV cache functionality and output debugging information.

2. Modified shortfin/python/shortfin_apps/llm/components/cache.py
   - Added a new method 'dump_cache_contents' to AttnPageCache class for debugging purposes.
   - Added logging in 'acquire_free_pages' and 'release_pages' methods.

3. Modified shortfin/python/shortfin_apps/llm/components/service.py
   - Added logging in InferenceExecutorProcess's 'run' method.
   - Added logging for logits shape, dtype, min, max, and mean values.
   - Added logging in BatcherProcess's 'board_flights' method.

These changes are intended to provide more detailed logging and debugging information for the KV cache system.
