# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test tracing functionality with a MobileNet model.

This test verifies that the tracy profiling functionality works correctly
by running a MobileNet model and generating a tracy capture file.
"""

import logging
import os
import pytest
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# skip if iree-tracy-capture is not in $PATH
def is_tracy_capture_available():
    """Check if iree-tracy-capture is available on the system."""
    return shutil.which("iree-tracy-capture") is not None


tracy_not_available = not is_tracy_capture_available()


@pytest.mark.skipif(
    tracy_not_available, reason="iree-tracy-capture is not available on this system"
)
def test_tracing_mobilenet(mobilenet_compiled_path, tmp_path):
    """Test that tracing works with a simple MobileNet model.

    This test:
    1. Starts the tracy capture tool
    2. Runs the MobileNet model with tracing enabled
    3. Verifies a trace file was created

    Args:
        mobilenet_compiled_path: Path to the compiled MobileNet model
        tmp_path: Temporary directory for storing the trace file
    """
    output_dir = tmp_path

    # Define output files
    capture_file = output_dir / "capture.tracy"
    tracy_log = output_dir / "tracy.log"

    # Step 1: Start tracy capture
    logger.info(f"Tracy capture will be saved to: {capture_file}")

    tracy_process = None
    try:
        # Start tracy capture with output redirection and prevent it from exiting
        # when the first client disconnects
        env = os.environ.copy()
        env["TRACY_NO_EXIT"] = "1"

        with open(tracy_log, "w") as tracy_stdout:
            tracy_process = subprocess.Popen(
                ["iree-tracy-capture", "-o", str(capture_file)],
                stdout=tracy_stdout,
                stderr=subprocess.STDOUT,
                env=env,
            )

        # Step 2: Run MobileNet model with tracing enabled via subprocess
        logger.info("Running MobileNet model with tracing enabled via subprocess")

        # Use the dedicated runner script in the same directory
        runner_script = Path(__file__).parent / "mobilenet_runner_for_tracing.py"

        # Run the script in a subprocess with tracing enabled
        env = os.environ.copy()
        env["SHORTFIN_PY_RUNTIME"] = "tracy"
        env["TRACY_ENABLE"] = "1"

        logger.info(f"Launching subprocess to run MobileNet model")
        try:
            result = subprocess.run(
                [sys.executable, str(runner_script), str(mobilenet_compiled_path)],
                env=env,
                check=False,  # Don't fail on error
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Log the output from the subprocess
            logger.info(f"Subprocess output:\n{result.stdout}")

            # Manually check if it was successful
            if result.returncode != 0:
                logger.error(f"Subprocess failed with exit code {result.returncode}")
                logger.error(f"Command: {' '.join(result.args)}")
                logger.error(f"Output: {result.stdout}")
        except Exception as e:
            logger.error(f"Error running subprocess: {e}")
    finally:
        # shut down tracy
        logger.info("Waiting for tracy to finish collecting data...")
        time.sleep(5)
        # Ensure the tracy process is terminated
        if tracy_process and tracy_process.poll() is None:
            logger.info("Terminating tracy process...")
            tracy_process.terminate()
            try:
                tracy_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Tracy process didn't terminate, killing it...")
                tracy_process.kill()

    assert (
        capture_file.exists()
    ), f"Tracy capture file was not created at {capture_file}"
    assert (
        capture_file.stat().st_size > 0
    ), f"Tracy capture file at {capture_file} is empty"
    logger.info(
        f"Tracy capture should be at: {capture_file} (exists: {capture_file.exists()})"
    )
    if capture_file.exists():
        logger.info(f"Tracy capture file size: {capture_file.stat().st_size} bytes")
