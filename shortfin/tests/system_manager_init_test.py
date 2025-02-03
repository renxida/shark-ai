import pytest
import multiprocessing


def run_system_manager_test(with_sleep=False):
    """Helper function to run in a separate process."""
    # copying the contents of this into a separate file replicates the issue
    # with_sleep = False # you need to uncomment this to replicate in a separate file
    from shortfin_apps.llm.components.manager import SystemManager
    import time

    sysman = SystemManager(device="local-task")
    sysman.start()

    try:
        if with_sleep:
            time.sleep(1)
        worker = sysman.ls.create_worker("test-worker")
        assert worker is not None

        fiber = sysman.ls.create_fiber(worker)
        assert fiber is not None
        assert len(fiber.devices_dict) > 0

    finally:
        sysman.shutdown()


def test_system_manager_init_with_sleep():
    """Test that creating a worker after SystemManager start works with sleep."""
    process = multiprocessing.Process(target=run_system_manager_test, args=(True,))
    process.start()
    process.join(timeout=5)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Test timed out after 5 seconds")

    assert process.exitcode == 0, "Test failed"


@pytest.mark.xfail(
    reason="Race condition: Worker creation before system fully initialized"
)
def test_system_manager_init_without_sleep():
    """Test that creating a worker immediately after SystemManager start hangs."""
    process = multiprocessing.Process(target=run_system_manager_test, args=(False,))
    process.start()
    process.join(timeout=5)

    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("Test timed out after 5 seconds")

    assert process.exitcode == 0, "Test failed as expected due to race condition"
