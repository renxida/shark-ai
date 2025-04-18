# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(systems)

shortfin_cc_component(
  NAME
    shortfin_local
  HDRS
    async.h
    device.h
    fiber.h
    messaging.h
    process.h
    program.h
    worker.h
    scheduler.h
    system.h
  SRCS
    async.cc
    device.cc
    fiber.cc
    messaging.cc
    process.cc
    program.cc
    worker.cc
    scheduler.cc
    system.cc
  COMPONENTS
    shortfin_support
  DEPS
    iree_base_base
    iree_base_loop_sync
    iree_hal_hal
    iree_hal_utils_allocators
    iree_io_formats_parser_registry
    iree_modules_io_parameters_parameters
    iree_modules_hal_hal
    iree_vm_vm
    iree_vm_bytecode_module
)
