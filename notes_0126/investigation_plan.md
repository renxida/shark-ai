# Trie Cache Concurrency Investigation Plan

## Test Files Created

1. **Page Allocation Test**
   - Location: `shortfin/tests/apps/llm/components/kvcache/trie_allocation_test.py`
   - Purpose: Track and verify page allocation/deallocation
   - Detects:
     - Duplicate page allocations across threads
     - Invalid deallocations
     - Reference counting issues
     - Page reuse problems

2. **Trie Structure Test**
   - Location: `shortfin/tests/apps/llm/components/kvcache/trie_structure_test.py`
   - Purpose: Monitor trie modifications and reference counting
   - Detects:
     - Race conditions in node creation/deletion
     - Reference count mismatches
     - Parent-child relationship corruption
     - Concurrent modification issues

3. **Test Runner**
   - Location: `shortfin/tests/apps/llm/components/kvcache/run_concurrency_tests.py`
   - Purpose: Execute tests with detailed logging
   - Features:
     - Multiple iterations of allocation test
     - Detailed logging of operations
     - Result aggregation
     - Timestamped log files

## How to Run Tests

1. Activate environment:
   ```bash
   pyenv activate shark-env
   ```

2. Install package in development mode:
   ```bash
   cd shortfin
   pip install -e .
   ```

3. Run tests:
   ```bash
   python -m tests.apps.llm.components.kvcache.run_concurrency_tests
   ```

## Investigation Strategy

1. **Page Allocation Analysis**
   - Run allocation tests to identify if pages are being:
     - Double-allocated to different requests
     - Freed while still in use
     - Reused without proper reference counting

2. **Trie Structure Analysis**
   - Run structure tests to verify:
     - Node creation/deletion synchronization
     - Reference count accuracy
     - Trie consistency under concurrent modifications

3. **Result Analysis**
   - Check logs for:
     - Duplicate page allocations
     - Reference count mismatches
     - Trie structure corruption
     - Race conditions

## Expected Outcomes

The tests will help identify whether the token generation issues are caused by:
1. Page allocation giving same pages to different requests
2. Reference counting allowing premature page reuse
3. Trie structure becoming corrupted under concurrency

This targeted testing approach will provide concrete evidence of where synchronization is needed, rather than adding locks everywhere.

## Next Steps

1. Fix test imports and get tests running
2. Collect and analyze test results
3. Based on findings, implement specific synchronization fixes
4. Validate fixes with both these tests and original failing tests
