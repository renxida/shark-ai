import pytest
import threading
import queue
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Dict
from collections import defaultdict

from shortfin_apps.llm.components.kvcache.trie_attention_cache import (
    TriePagedAttentionCache,
    CacheAllocationFailure,
)
from shortfin_apps.llm.components.kvcache.page_pool import PagePool, PageInfo

TEST_PAGE_SIZE = 16
TEST_POOL_CAPACITY = 32


class MockPagePool(PagePool):
    def __init__(self, total_pages: int):
        self.available_pages = []
        for i in range(total_pages):
            page = PageInfo(index=i, pool=self)
            self.available_pages.append(page)

    def acquire_free_pages(self, count: int) -> List[PageInfo]:
        # If we don't have enough pages to satisfy the request, return None
        if len(self.available_pages) < count:
            return None

        # Pop from the front of the list 'count' times
        return [self.available_pages.pop(0) for _ in range(count)]

    def free_pages(self, pages: List[PageInfo]):
        # Simply append freed pages to the list of available pages
        for page in pages:
            self.available_pages.append(page)


@pytest.fixture
def page_pool():
    return MockPagePool(total_pages=TEST_POOL_CAPACITY)


@pytest.fixture
def cache(page_pool):
    return TriePagedAttentionCache(page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE)


def test_concurrent_allocation_and_publishing():
    """Test concurrent allocation and publishing with shared prefixes.

    This test simulates multiple threads trying to allocate and publish pages
    for sequences that share common prefixes, similar to the real-world scenario
    where multiple requests generate continuations of the same prefix.
    """
    page_pool = MockPagePool(TEST_POOL_CAPACITY)
    cache = TriePagedAttentionCache(page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE)

    # Common prefix all threads will share
    common_prefix = [1, 2, 3, 4]

    def worker(thread_id: int):
        # Each thread extends common prefix differently
        unique_suffix = [10 + thread_id, 20 + thread_id, 30 + thread_id]
        tokens = common_prefix + unique_suffix

        try:
            # First allocate
            allocation = cache.acquire_pages_for_tokens(tokens)

            # Simulate some work
            time.sleep(random.uniform(0.001, 0.005))

            # Then publish in stages
            for i in range(len(tokens)):
                publish_tokens = tokens[: i + 1]
                allocation.publish_pages_for_tokens(publish_tokens)
                time.sleep(random.uniform(0.001, 0.003))

            return allocation
        except Exception as e:
            return e

    # Run multiple threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        results = [f.result() for f in futures]

    # Verify results
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"Workers encountered errors: {errors}"


def test_interleaved_allocation_publishing():
    """Test interleaved allocation and publishing operations.

    This test creates a scenario where allocations and publishing operations
    are interleaved between threads, checking for race conditions in the
    trie structure modifications.
    """
    page_pool = MockPagePool(TEST_POOL_CAPACITY)
    cache = TriePagedAttentionCache(page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE)

    results = queue.Queue()

    def publisher_thread(allocation, tokens):
        try:
            time.sleep(random.uniform(0.001, 0.003))
            allocation.publish_pages_for_tokens(tokens)
            results.put(("publish", tokens, None))
        except Exception as e:
            results.put(("publish", tokens, e))

    def allocator_thread(tokens):
        try:
            allocation = cache.acquire_pages_for_tokens(tokens)
            results.put(("allocate", tokens, allocation))
            return allocation
        except Exception as e:
            results.put(("allocate", tokens, e))
            raise e

    # Create sequences that will compete for the same trie paths
    sequences = [
        list(range(32)),
        list(range(16)) + list(range(16)),
        list(range(64, 96, 2)),
        [1, 2, 3, 5, 8],
    ]

    threads = []
    allocations = []

    # Start allocation threads
    for seq in sequences:
        t = threading.Thread(target=allocator_thread, args=(seq,))
        threads.append(t)
        t.start()

    # Wait for allocations and collect results
    for t in threads:
        t.join()

    while not results.empty():
        op, tokens, result = results.get()
        if op == "allocate" and not isinstance(result, Exception):
            allocations.append(result)

    # Start publish threads
    threads.clear()
    for alloc, seq in zip(allocations, sequences):
        t = threading.Thread(target=publisher_thread, args=(alloc, seq))
        threads.append(t)
        t.start()

    # Wait for publishing to complete
    for t in threads:
        t.join()

    # Check results
    errors = []
    while not results.empty():
        op, tokens, result = results.get()
        if isinstance(result, Exception):
            errors.append((op, tokens, result))

    assert not errors, f"Operations encountered errors: {errors}"


def test_concurrent_extension_and_publishing():
    """Test concurrent allocation extension and publishing.

    This test simulates the real-world scenario where multiple threads are
    extending their allocations while others are publishing, which happens
    during batch processing of decode requests.
    """
    page_pool = MockPagePool(TEST_POOL_CAPACITY)
    cache = TriePagedAttentionCache(page_pool=page_pool, tokens_per_page=TEST_PAGE_SIZE)

    def worker(thread_id: int):
        # Initial allocation
        initial_tokens = [1, 2, 3, thread_id]
        tokens = initial_tokens
        try:
            allocation = cache.acquire_pages_for_tokens(tokens)

            # Publish initial tokens
            allocation.publish_pages_for_tokens(tokens)

            # Extend and publish in stages
            for i in range(3):
                # Extend with new tokens ensuring unique sequences
                tokens += [100 + thread_id * 10 + i, 200 + thread_id * 10 + i]
                allocation.extend_allocation(tokens)

                # Simulate some work
                time.sleep(random.uniform(0.001, 0.003))

                # Publish new tokens
                allocation.publish_pages_for_tokens(tokens)

            return None
        except Exception as e:
            return e

    # Run multiple threads
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, i) for i in range(4)]
        results = [f.result() for f in futures]

    # Verify results
    errors = [r for r in results if r is not None]
    assert not errors, f"Workers encountered errors: {errors}"


def test_publish_under_memory_pressure():
    """Test publishing behavior when memory is constrained.

    This test verifies that publishing operations work correctly even when
    the page pool is near capacity and multiple threads are competing for
    resources.
    """
    # Use a small pool to create memory pressure
    page_pool = MockPagePool(total_pages=8)
    cache = TriePagedAttentionCache(page_pool=page_pool, tokens_per_page=4)

    def worker(thread_id: int):
        tokens = [thread_id, thread_id + 1, thread_id + 2]
        try:
            # Allocate initial pages
            allocation = cache.acquire_pages_for_tokens(tokens)

            # Publish initial tokens
            allocation.publish_pages_for_tokens(tokens)

            # Try to extend under memory pressure
            extended_tokens = tokens + [thread_id + 3, thread_id + 4]
            allocation.extend_allocation(extended_tokens)

            # Attempt to publish extended tokens
            allocation.publish_pages_for_tokens(extended_tokens)

            return allocation
        except Exception as e:
            return e

    # Run multiple threads
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(worker(i))) for i in range(5)
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Count different types of outcomes
    successes = [r for r in results if not isinstance(r, Exception)]
    allocation_failures = [r for r in results if isinstance(r, CacheAllocationFailure)]
    other_errors = [
        r
        for r in results
        if isinstance(r, Exception) and not isinstance(r, CacheAllocationFailure)
    ]

    # We expect some allocation failures due to memory pressure, but no other errors
    assert len(other_errors) == 0, f"Unexpected errors occurred: {other_errors}"
    assert len(successes) > 0, "Expected some allocations to succeed"
