# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test tracing functionality with a MobileNet model.

This test verifies that the tracy profiling functionality works correctly
by running a MobileNet model and generating a tracy capture file.
"""

import array
import logging
import os
import pytest
import shutil
import subprocess
import sys
import time
from pathlib import Path

import shortfin as sf
import shortfin.array as sfnp

logger = logging.getLogger(__name__)

# skip if iree-tracy-capture is not in $PATH
def is_tracy_capture_available():
    """Check if iree-tracy-capture is available on the system."""
    return shutil.which("iree-tracy-capture") is not None


# Check if shortfin was built with tracing enabled
def is_tracing_enabled():
    """Check if shortfin was built with tracing enabled."""
    import importlib

    try:
        importlib.import_module("_shortfin_tracy")
        return True
    except ModuleNotFoundError:
        # Only logged at debug level since we'll have a warning later
        logger.debug("_shortfin_tracy module not found")
        return False


def run_mobilenet(model_path):
    """Run MobileNet model with tracing enabled.

    Args:
        model_path: Path to the compiled VMFB file
    """
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Running MobileNet model from {model_path}")

    # Create system and fiber with CPU backend
    sc = sf.host.CPUSystemBuilder()
    system = sc.create_system()
    fiber = system.create_fiber()
    device = fiber.device(0)

    # Load model
    program_module = system.load_module(model_path)
    program = sf.Program([program_module], devices=system.devices)
    main_function = program["module.torch-jit-export"]

    # Create input tensor
    dummy_data = array.array(
        "f", ([0.2] * (224 * 224)) + ([0.4] * (224 * 224)) + ([-0.2] * (224 * 224))
    )
    device_input = sfnp.device_array(device, [1, 3, 224, 224], sfnp.float32)
    staging_input = device_input.for_transfer()
    with staging_input.map(discard=True) as m:
        m.fill(dummy_data)
    device_input.copy_from(staging_input)

    # Run model
    async def run_model():
        logger.info("Running MobileNet inference")
        (device_output,) = await main_function(device_input, fiber=fiber)
        # Transfer output back to host
        host_output = device_output.for_transfer()
        host_output.copy_from(device_output)
        await device
        # Clean up
        del device_output
        del host_output
        logger.info("MobileNet inference completed")

    system.run(run_model())

    # Clean up
    system.shutdown()
    logger.info("MobileNet model execution finished")


def run_mobilenet_subprocess(model_path):
    """Execute MobileNet model with tracy profiling in a subprocess.

    Args:
        model_path: Path to the compiled VMFB file

    Returns:
        subprocess.CompletedProcess object with the result of the subprocess execution
    """
    # Run this same file in a subprocess with tracing enabled
    env = os.environ.copy()
    env["SHORTFIN_PY_RUNTIME"] = "tracy"
    env["TRACY_ENABLE"] = "1"

    logger.info(f"Launching subprocess to run MobileNet model")

    # Use this same file as the runner script, but pass a special flag to indicate
    # we want to run the model directly rather than run the test
    return subprocess.run(
        [sys.executable, __file__, str(model_path), "--run-mobilenet-as-subprocess"],
        env=env,
        check=False,  # Don't fail on error
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


tracing_not_enabled = not is_tracing_enabled()
tracy_capture_not_available = not is_tracy_capture_available()
if tracy_capture_not_available:
    logger.warning(
        "iree-tracy-capture tool is not available in PATH. "
        "To fix this, build IREE with Tracy support enabled: "
        "cmake -DIREE_ENABLE_RUNTIME_TRACING=ON -DTRACY_ENABLE=ON"
    )
if tracing_not_enabled:
    logger.warning(
        "Shortfin was not built with tracing enabled. "
        "To enable tracing, rebuild shortfin with: "
        "cmake -DSHORTFIN_ENABLE_TRACING=ON"
    )


@pytest.mark.skipif(
    tracy_capture_not_available or tracing_not_enabled,
    reason="iree-tracy-capture is not available or shortfin was not built with tracing enabled",
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

        result = run_mobilenet_subprocess(mobilenet_compiled_path)

        # Log the output from the subprocess
        logger.info(f"Subprocess output:\n{result.stdout}")

        # Manually check if it was successful
        if result.returncode != 0:
            logger.error(f"Subprocess failed with exit code {result.returncode}")
            logger.error(f"Command: {' '.join(result.args)}")
            logger.error(f"Output: {result.stdout}")

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


if __name__ == "__main__":
    # Check if we're being run directly to execute the model (from the subprocess)
    if len(sys.argv) >= 3 and sys.argv[2] == "--run-mobilenet-as-subprocess":
        model_path = sys.argv[1]
        run_mobilenet(model_path)
        sys.exit(0)
