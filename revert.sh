#!/bin/bash

# Remove the test file
rm -f test_kvcache.py

# Revert changes in cache.py
sed -i '/def dump_cache_contents/,/EOL/d' shortfin/python/shortfin_apps/llm/components/cache.py
sed -i '/logger.info(f"Attempting to acquire {count} pages. Available: {len(self.attn_page_free)}")/d' shortfin/python/shortfin_apps/llm/components/cache.py
sed -i '/logger.info(f"Releasing {len(pages)} pages")/d' shortfin/python/shortfin_apps/llm/components/cache.py

# Revert changes in service.py
sed -i '/logger.info("Starting InferenceExecutorProcess run")/d' shortfin/python/shortfin_apps/llm/components/service.py
sed -i '/logger.info(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")/d' shortfin/python/shortfin_apps/llm/components/service.py
sed -i '/logger.info(f"Logits min: {logits.min()}, max: {logits.max()}, mean: {logits.mean()}")/d' shortfin/python/shortfin_apps/llm/components/service.py
sed -i '/logger.info(f"Boarding flights. Prefills: {len(self.pending_prefills)}, Decodes: {len(self.pending_decodes)}")/d' shortfin/python/shortfin_apps/llm/components/service.py

echo "All changes have been reverted."
