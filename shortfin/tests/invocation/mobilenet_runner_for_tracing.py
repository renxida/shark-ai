# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runner script to execute MobileNet model with tracy profiling.

This script is called as a subprocess from trace_mobilenet_test.py to ensure
that the tracy profiling captures the execution in a separate process.
"""

import array
import logging
import sys
from pathlib import Path

import shortfin as sf
import shortfin.array as sfnp

logger = logging.getLogger(__name__)


def run_mobilenet_model(model_path):
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_mobilenet_vmfb>")
        sys.exit(1)

    model_path = sys.argv[1]
    run_mobilenet_model(model_path)
