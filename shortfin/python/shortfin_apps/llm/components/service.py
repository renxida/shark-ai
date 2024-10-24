# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
from pathlib import Path

import shortfin as sf
import shortfin.array as sfnp

from .cache import AttnPageCache
from .config_struct import ModelParams
from .manager import SystemManager
from .messages import InferenceExecRequest, InferencePhase, StrobeMessage
from .tokenizer import Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class GenerateService:
    """Top level service interface for generating text against a model."""

    inference_program: sf.Program
    prefill_functions: dict[int, sf.ProgramFunction]
    decode_functions: dict[int, sf.ProgramFunction]

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
    ):
        self.name = name

        # Application objects.
        self.sysman = sysman
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.inference_parameters: list[sf.BaseProgramParameters] = []
        self.inference_modules: list[sf.ProgramModule] = []

        self.main_worker = sysman.ls.create_worker(f"{name}-inference")
        self.main_fiber = sysman.ls.create_fiber(self.main_worker)

        # Scope dependent objects.
        self.batcher = BatcherProcess(self)
        self.page_cache = AttnPageCache(
            devices=self.main_fiber.devices_dict.values(), model_params=model_params
        )

    def load_inference_module(self, vmfb_path: Path):
        self.inference_modules.append(sf.ProgramModule.load(self.sysman.ls, vmfb_path))

    def load_inference_parameters(
        self, *paths: Path, parameter_scope: str, format: str = ""
    ):
        p = sf.StaticProgramParameters(self.sysman.ls, parameter_scope=parameter_scope)
        for path in paths:
            logging.info("Loading parameter fiber '%s' from: %s", parameter_scope, path)
            p.load(path, format=format)
        self.inference_parameters.append(p)

    def start(self):
        self.inference_program = sf.Program(
            modules=[
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters
                )
            ]
            + self.inference_modules,
            fiber=self.main_fiber,
            trace_execution=False,
        )
        # Resolve prefill entrypoints.
        self.prefill_functions = {}
        for bs in self.model_params.prefill_batch_sizes:
            self.prefill_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.prefill_bs{bs}"
            ]
        # Resolve decode entrypoints.
        self.decode_functions = {}
        for bs in self.model_params.decode_batch_sizes:
            self.decode_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.decode_bs{bs}"
            ]

        # Start persistent processes.
        self.batcher.launch()

    def shutdown(self):
        self.batcher.shutdown()

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  page_cache={self.page_cache}\n"
            f")"
        )


########################################################################################
# Batcher
########################################################################################

import math


class BatcherProcess(sf.Process):
    """The batcher is a persistent process responsible for flighting incoming work
    into batches and handling the requisite cache allocations (since every batch needs
    committed cache state).
    """

    STROBE_SHORT_DELAY = 0.1
    STROBE_LONG_DELAY = 0.25

    def __init__(self, service: GenerateService):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.batcher_infeed = self.system.create_queue()
        self.pending_prefills: set[InferenceExecRequest] = set()
        self.pending_decodes: set[InferenceExecRequest] = set()
        self.strobe_enabled = True
        self.strobes: int = 0
        # TODO: There is no "ideal" batch size. Use prefill/decode dynamic
        # batching in the scheduling algo.
        self.ideal_batch_size: int = max(service.model_params.prefill_batch_sizes)
        self.page_seq_stride = service.model_params.paged_kv_cache.block_seq_stride

    def shutdown(self):
        self.batcher_infeed.close()

    def submit(self, request: StrobeMessage | InferenceExecRequest):
        self.batcher_infeed.write_nodelay(request)

    async def _background_strober(self):
        while not self.batcher_infeed.closed:
            await asyncio.sleep(
                BatcherProcess.STROBE_SHORT_DELAY
                if len(self.pending_prefills) > 0
                else BatcherProcess.STROBE_LONG_DELAY
            )
            if self.strobe_enabled:
                self.submit(StrobeMessage())

    async def run(self):
        logger.info("Starting InferenceExecutorProcess run")
        strober_task = asyncio.create_task(self._background_strober())
        reader = self.batcher_infeed.reader()
        while item := await reader():
            self.strobe_enabled = False
            if isinstance(item, InferenceExecRequest):
                phase = item.phase
                if phase == InferencePhase.PREFILL:
                    self.pending_prefills.add(item)
                elif phase == InferencePhase.DECODE:
                    self.pending_decodes.add(item)
                else:
                    logger.error("Illegal InferenceExecRequest phase: %r", phase)
            elif isinstance(item, StrobeMessage):
                self.strobes += 1
            else:
                logger.error("Illegal message received by batcher: %r", item)
            self.board_flights()
            self.strobe_enabled = True
        await strober_task

    def board_flights(self):
        logger.info(
            f"Boarding flights. Prefills: {len(self.pending_prefills)}, Decodes: {len(self.pending_decodes)}"
        )
        waiting_count = len(self.pending_prefills) + len(self.pending_decodes)
        if waiting_count == 0:
            return
        if waiting_count < self.ideal_batch_size and self.strobes < 2:
            logger.info("Waiting a bit longer to fill flight")
            return
        self.strobes = 0
        cache = self.service.page_cache

        # TODO: This is a very naive cache management algorithm. Burn with fire
        # and implement a real one.
        self.board_prefills(cache)
        self.board_decodes(cache)

        # For now, kill anything that is left.
        for prefill_request in self.pending_prefills:
            prefill_request.done.set_success()
        self.pending_prefills.clear()
        logger.debug("Post boarding cache state: %r", cache)

    def board_prefills(self, cache: AttnPageCache):
        # Fill prefill flights.
        pending_prefills = self.pending_prefills
        if len(pending_prefills) == 0:
            return
        exec_process = InferenceExecutorProcess(
            self.service,
            InferencePhase.PREFILL,
            self.page_seq_stride,
            cache.page_tables,
        )
        for prefill_request in pending_prefills:
            assert prefill_request.phase == InferencePhase.PREFILL
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            needed_pages = math.ceil(
                len(prefill_request.input_token_ids) / self.page_seq_stride
            )
            pages = cache.acquire_free_pages(needed_pages)
            if pages is None:
                logger.debug("Cannot fulfill request for %d pages", needed_pages)
                continue
            else:
                # TODO: log the pages here
                # specifically what pages
                logger.debug(f"Pages: {pages}")
                logger.debug("Allocated %d cache pages to request", len(pages))
                prefill_request.lock_initial_cache_pages(cache, pages)

            # Can flight this request.
            exec_process.exec_requests.append(prefill_request)

        # We've filled our flight. Remove from the boarding area.
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending_prefills.remove(flighted_request)
            # And takeoff.
            exec_process.launch()

    def board_decodes(self, cache: AttnPageCache):
        # Fill decode flights.
        pending_decodes = self.pending_decodes
        if len(pending_decodes) == 0:
            return
        exec_process = InferenceExecutorProcess(
            self.service, InferencePhase.DECODE, self.page_seq_stride, cache.page_tables
        )
        for decode_request in pending_decodes:
            assert decode_request.phase == InferencePhase.DECODE
            if len(exec_process.exec_requests) >= self.ideal_batch_size:
                break
            needed_pages = math.ceil(
                len(decode_request.input_token_ids) / self.page_seq_stride
            )
            if needed_pages > len(decode_request.locked_pages):
                pages = cache.acquire_free_pages(needed_pages)
                if pages is None:
                    logger.debug(
                        "Cannot fulfill decode request for %d pages", needed_pages
                    )
                    continue
                else:
                    logger.debug(
                        "Allocated %d cache pages to decode request", len(pages)
                    )
                decode_request.lock_new_cache_pages(cache, pages)

            # Can flight this request.
            exec_process.exec_requests.append(decode_request)

        # We've filled our flight. Remove from the boarding area.
        if exec_process.exec_requests:
            for flighted_request in exec_process.exec_requests:
                self.pending_decodes.remove(flighted_request)
            # And takeoff.
            exec_process.launch()


########################################################################################
# Inference Executor
########################################################################################

# debug_utils.py
from typing import NamedTuple
import logging
import datetime
from pathlib import Path
import numpy as np
from scipy import stats
import array
import shortfin.array as sfnp

logger = logging.getLogger(__name__)


class TensorStats(NamedTuple):
    """Container for tensor statistics to avoid recalculating values."""

    nan_count: int
    total_zeros: int
    leading_zeros: int
    min_val: float
    max_val: float
    mean_val: float
    mode_val: float
    first_elements: np.ndarray
    last_elements: np.ndarray


class TensorDebug:
    """Utilities for debugging tensor operations and contents."""

    def __init__(self, dump_path: Path = Path("/tmp/sharktank/shortfin_llm")):
        self.dump_path = dump_path
        self.dump_path.mkdir(parents=True, exist_ok=True)

    async def get_tensor(self, tensor, name: str = "Tensor") -> np.ndarray:
        """Convert various tensor types to numpy array for analysis."""
        if isinstance(tensor, sfnp.device_array):
            logger.info(f"tensor {name} is a device array; converting to host array")
            host_tensor = tensor.for_transfer()
            host_tensor.copy_from(tensor)
            await tensor.device
            tensor = host_tensor
        else:
            logger.info(
                f"tensor {name} is not a device array; assuming it is a host array"
            )

        if isinstance(tensor, sfnp.base_array):
            logger.info(f"tensor {name} is a base_array; converting to array.array")
            fp16_hack = tensor.dtype == sfnp.float16
            tensor = tensor.items
        else:
            logger.info(
                f"tensor {name} is not a base array; assuming it is an array.array"
            )
            fp16_hack = False

        if isinstance(tensor, array.array):
            logger.info(f"tensor {name} is an array.array")
            dtype = np.float16 if fp16_hack else tensor.typecode
            tensor = np.frombuffer(tensor, dtype=dtype)
        else:
            logger.info(f"tensor {name} is not an array.array")

        assert isinstance(tensor, np.ndarray)
        return tensor

    def _compute_tensor_stats(self, tensor: np.ndarray) -> TensorStats:
        """Compute all tensor statistics in a single pass."""
        # Flatten the tensor once
        flat_tensor = tensor.ravel()

        # Create boolean masks once
        nan_mask = np.isnan(flat_tensor)
        zero_mask = flat_tensor == 0

        # Compute counts
        nan_count = np.sum(nan_mask)
        total_zeros = np.sum(zero_mask)

        # Compute leading zeros efficiently
        leading_zeros = np.searchsorted(zero_mask == False, True, side="right")

        # Get valid values (non-NaN) for statistics
        valid_tensor = flat_tensor[~nan_mask]

        if len(valid_tensor) > 0:
            # Compute basic statistics
            min_val = np.min(valid_tensor)
            max_val = np.max(valid_tensor)
            mean_val = np.mean(valid_tensor)
            mode_val = stats.mode(valid_tensor)[0]

            # Get first and last elements
            first_elements = valid_tensor[:10]
            last_elements = valid_tensor[-10:]
        else:
            min_val = max_val = mean_val = mode_val = np.nan
            first_elements = last_elements = np.array([])

        return TensorStats(
            nan_count=nan_count,
            total_zeros=total_zeros,
            leading_zeros=leading_zeros,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            mode_val=mode_val,
            first_elements=first_elements,
            last_elements=last_elements,
        )

    async def log_tensor_stats(self, tensor, name: str = "Tensor"):
        """Log comprehensive tensor statistics and dump to file."""
        tensor = await self.get_tensor(tensor, name)
        name = name.replace(" ", "_")

        # Save tensor to file (async file I/O could be implemented here if needed)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dump_file = self.dump_path / f"{name}_{timestamp}.npy"
        logger.info(f"Dumping tensor {name} to {dump_file}")
        np.save(dump_file, tensor)

        # Compute all stats in one pass
        stats = self._compute_tensor_stats(tensor)

        # Log information
        logger.info(f"{name} stats:")
        logger.info(f"  type: {type(tensor)}")
        logger.info(f"  dtype: {tensor.dtype}")
        logger.info(f"  shape: {tensor.shape}")
        logger.info(f"  NaN count: {stats.nan_count} / {tensor.size}")

        if len(stats.first_elements) > 0:
            logger.info(f"  Min (excluding NaN): {stats.min_val}")
            logger.info(f"  Max (excluding NaN): {stats.max_val}")
            logger.info(f"  Mean (excluding NaN): {stats.mean_val}")
            logger.info(f"  Mode (excluding NaN): {stats.mode_val}")
            logger.info(f"  First 10 elements: {stats.first_elements}")
            logger.info(f"  Last 10 elements: {stats.last_elements}")
        else:
            logger.warning(f"  All values are NaN in {name}")

        logger.info(f"  Leading zeros: {stats.leading_zeros}")
        logger.info(f"  Total zeros: {stats.total_zeros}")

    async def log_tensor_values(self, tensor, name: str = "Tensor"):
        """Log raw tensor values."""
        tensor = await self.get_tensor(tensor, name)
        name = name.replace(" ", "_")
        logger.info(f"{name} values:")
        logger.info(f"  {tensor}")


class CacheDebug:
    """Utilities for debugging cache operations."""

    def __init__(self, tensor_debug: TensorDebug):
        self.tensor_debug = tensor_debug

    async def dump_cache_contents(self, page_tables):
        """Dump and analyze contents of all page tables."""
        logger.info("Skipping cache dump for now")
        return
        logger.info("Dumping KV cache contents:")
        for i, page_table in enumerate(page_tables):
            await self.tensor_debug.log_tensor_stats(
                page_table, f"Page table {i} contents"
            )


class InferenceDebug:
    """Utilities for debugging inference operations."""

    def __init__(self, tensor_debug: TensorDebug):
        self.tensor_debug = tensor_debug

    async def log_inference_inputs(
        self, *, tokens, seq_lens, start_positions=None, seq_block_ids=None
    ):
        """Log all inference input tensors."""
        logger.info("Logging inference inputs:")
        await self.tensor_debug.log_tensor_values(tokens, "tokens")
        await self.tensor_debug.log_tensor_values(seq_lens, "seq_lens")
        if start_positions is not None:
            await self.tensor_debug.log_tensor_values(
                start_positions, "start_positions"
            )
        if seq_block_ids is not None:
            await self.tensor_debug.log_tensor_values(seq_block_ids, "seq_block_ids")


class InferenceExecutorProcess(sf.Process):
    """Executes a prefill or decode batch."""

    def __init__(
        self,
        service: GenerateService,
        phase: InferencePhase,
        seq_stride: int,
        page_tables,
    ):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.phase = phase
        self.seq_stride = seq_stride
        self.exec_requests: list[InferenceExecRequest] = []
        self.page_tables = page_tables

    async def run(self):
        logger.info("Starting InferenceExecutorProcess run")
        logger.info(f"main fiber: {self.service.main_fiber}")
        logger.info(f"main fiber: {self.service.main_fiber}")
        try:
            is_decode = self.phase == InferencePhase.DECODE
            req_bs = len(self.exec_requests)
            seq_stride = self.seq_stride
            # Select an entrypoint for the batch.
            if is_decode:
                entrypoints = self.service.decode_functions
            else:
                entrypoints = self.service.prefill_functions
            for bs, fn in entrypoints.items():
                if bs >= req_bs:
                    break
            else:
                raise RuntimeError(f"No available entry point for bs {req_bs}")

            # Compute block sequence length as maximum sequence length, rounded
            # up to the seq_stride.
            bsl = max(len(r.input_token_ids) for r in self.exec_requests)
            bsl = int(math.ceil(bsl / seq_stride) * seq_stride)
            block_count = bsl // seq_stride
            req_count = len(self.exec_requests)
            logger.info("Prefill bs=%d, bsl=%d", bs, bsl)

            # Prepare inputs.
            # TODO: Better support in shortfin for h2d. The best way to do it is
            # device dependent.
            device0 = self.fiber.device(0)
            int_dtype = sfnp.int64
            if is_decode:
                tokens = sfnp.device_array.for_device(device0, [bs, 1], int_dtype)
                start_positions = sfnp.device_array.for_device(device0, [bs], int_dtype)
            else:
                tokens = sfnp.device_array.for_device(device0, [bs, bsl], int_dtype)
            seq_lens = sfnp.device_array.for_device(device0, [bs], int_dtype)
            seq_block_ids = sfnp.device_array.for_device(
                device0, [bs, block_count], int_dtype
            )

            # Populate tokens.
            tokens_host = tokens.for_transfer()
            for i in range(bs):
                with tokens_host.view(i).map(discard=True) as m:
                    m.fill(0)
                    if i < req_count:
                        m.items = self.exec_requests[i].input_token_ids
            tokens_host.copy_to(tokens)

            # Populate seq_lens.
            seq_lens_host = seq_lens.for_transfer()
            with seq_lens_host.map(discard=True) as m:
                m.fill(0)
                m.items = [len(req.input_token_ids) for req in self.exec_requests]
            seq_lens_host.copy_to(seq_lens)

            # For decode, populate start_positions.
            if self.phase == InferencePhase.DECODE:
                start_positions_host = start_positions.for_transfer()
                with start_positions_host.map(discard=True) as m:
                    m.fill(0)
                    m.items = [req.start_position for req in self.exec_requests]
                start_positions_host.copy_to(start_positions)

            # Populate cache pages.
            seq_block_ids_host = seq_block_ids.for_transfer()
            for i in range(bs):
                with seq_block_ids_host.view(i).map(discard=True) as m:
                    m.fill(0)
                    if i < req_count:
                        m.items = self.exec_requests[i].cache_page_indices(block_count)
            seq_block_ids_host.copy_to(seq_block_ids)

            # V1 args:
            #  prefill:
            #    tokens: [bs, bsl]
            #    seq_lens: [bs]
            #    seq_block_ids: [bs, blocks]
            #    cache_slabs: ...
            #  decode:
            #    tokens: [bs, 1]
            #    seq_lens: [bs]
            #    start_positions: [bs]
            #    seq_block_ids: [bs, blocks]
            #    cache_slabs: ...
            if is_decode:
                args = [tokens, seq_lens, start_positions, seq_block_ids]
            else:
                args = [tokens, seq_lens, seq_block_ids]
            args.extend(self.page_tables)
            logger.info(
                "INVOKE %r: %s",
                fn,
                "".join([f"\n  {i}: {ary.shape}" for i, ary in enumerate(args)]),
            )

            self.tensor_debug = TensorDebug()
            self.cache_debug = CacheDebug(self.tensor_debug)
            self.inference_debug = InferenceDebug(self.tensor_debug)
            await device0
            logger.info("Pre-invoke debug information:")

            logger.info("Tokens information:")
            await self.tensor_debug.log_tensor_stats(tokens_host, "tokens_host")
            await self.tensor_debug.log_tensor_stats(tokens, "tokens")

            logger.info("Cache information:")
            await self.cache_debug.dump_cache_contents(self.page_tables)

            logger.info("Inference inputs:")
            await self.inference_debug.log_inference_inputs(
                tokens=tokens,
                seq_lens=seq_lens,
                start_positions=start_positions if is_decode else None,
                seq_block_ids=seq_block_ids,
            )

            # Invoke function
            (logits,) = await fn(*args)

            logger.info("Post-invoke debug information:")
            await self.cache_debug.dump_cache_contents(self.page_tables)
            await self.tensor_debug.log_tensor_stats(logits, "logits")
            # Return results.
            for i in range(req_count):
                req = self.exec_requests[i]
                sl = len(req.input_token_ids)
                logger.info("Picking logit slice for request %d", i)
                logger.info("  Request token count: %d", sl)
                if req.return_all_logits:
                    logits_item = logits.view(i, slice(0, sl))
                else:
                    logits_item = logits.view(i, sl - 1)
                if req.return_host_array:
                    req.result_logits = logits_item.for_transfer()
                    req.result_logits.copy_from(logits_item)
                    await device0
                else:
                    req.result_logits = logits_item
                req.done.set_success()

        except Exception:
            logger.exception("Fatal error in prefetch invocation")
            # TODO: Cancel and set error correctly
            for req in self.exec_requests:
                req.result_logits = None
                req.free_cache_pages()
                req.done.set_success()
