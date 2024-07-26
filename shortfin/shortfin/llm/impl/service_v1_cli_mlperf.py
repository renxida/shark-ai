# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import argparse
import numpy
import sys
import ast
import pandas as pd
from transformers import LlamaTokenizer  # type: ignore

from iree.runtime import (  # type: ignore
    HalElementType,
    DeviceArray
)

from shortfin.framework.session import DeviceSession

from shortfin.llm.attn_block_cache import (
    create_attn_block_cache_module,
    AttnBlockCache,
)

from shortfin.llm.config import (
    CacheParams,
    ModelParams,
    ServiceParams,
)

from shortfin.llm.impl.service_v1 import GenerateServiceV1
from shortfin.llm.service import GenerateRequest


def setup(vmfb_path, config_path, gguf_path):
    from iree.runtime._binding import disable_leak_checker  # type: ignore

    model_params = ModelParams.load_json(config_path)

    device_block_count = model_params.max_seq_len // model_params.block_seq_stride
    cache_params = CacheParams(
        model=model_params,
        device_block_count=device_block_count,
        block_pos_stride=model_params.block_seq_stride,
    )

    disable_leak_checker()
    session = DeviceSession(uri="hip://3", queue_count=2)
    attn_block_cache = AttnBlockCache(session, cache_params)

    device = session.device
    lms = session.create_module_set(model_params.module_name, context_count=1)
    lms.load_io_module(gguf_path)
    lms.load_vmfb(vmfb_path)
    lms.add(create_attn_block_cache_module(attn_block_cache))
    lms.initialize()

    params = ServiceParams(cache=cache_params, model=model_params)
    service = GenerateServiceV1(session=session, params=params, cache=attn_block_cache)
    return service, device


def map_buffer(value, device):
    return DeviceArray(device, value, override_dtype=HalElementType.map_to_dtype(value.element_type))


async def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="name of hugginface tokenizer to use")
    parser.add_argument("--config", help="json config file with hyperparameters")
    parser.add_argument("--vmfb", help="vmfb with compiler LLM kernels")
    parser.add_argument("--gguf", help="gguf file containing modle coefficients")
    parsed = parser.parse_args(argv)

    hf_path = parsed.tokenizer
    config_path = parsed.config
    vmfb_path = parsed.vmfb
    gguf_path = parsed.gguf

    service, device = setup(vmfb_path, config_path, gguf_path)
    tokenizer = LlamaTokenizer.from_pretrained(hf_path)
    state = service.start()

    df = pd.read_csv('golden_outputs_llama2_70b_fp8_latest.csv')  
    prompt_tokens = list(df['groundtruth_tok_input'])
    promt_list = list(df['groundtruth_input'])
    prompt_test_bool = []
    i = 0
    for idx, line in enumerate(prompt_tokens):
        input_ids = ast.literal_eval(line)
        if not prompt:
            break
        prompt = tokenizer.batch_decode(
                input_ids, skip_special_tokens=True)
        if i ==0:
            print(prompt[:20])
        prompt_test_bool.append(promt_list[idx] == prompt)
        request = GenerateRequest("request_id", prompt, input_ids)
        await state.set_sequences([request])
        logits = await state.prefill()

        seq_len = len(input_ids)
        mapped_logits = map_buffer(logits.value, device).to_host()
        predicted_tokens = numpy.argmax(mapped_logits[0, :seq_len], axis=-1)
        predicted_token = predicted_tokens[-1]
        decoded_token = tokenizer.decode(predicted_token)
        print(f"Prefill predicted token: {predicted_token}, decoded: '{decoded_token}'")

        # TODO(scotttodd): sanity check tokenizer use, document inputs/outputs
        #   'prefill' is for initialization with multiple steps at once
        #   'decode' is for hypothesis exploration, one step at a time
        await state.set_decode_step([predicted_token])
        logits = await state.decode()
        mapped_logits = map_buffer(logits.value, device).to_host()
        predicted_tokens = numpy.argmax(mapped_logits, axis=-1)
        predicted_token = predicted_tokens[0]
        decoded_token = tokenizer.decode(predicted_token)
        print(f"Decode predicted token: {predicted_token}, decoded: '{decoded_token}'")
        i =+ 1
        await state.recycle()
        
    print('sum(prompt_test_bool), len(prompt_test_bool)', sum(prompt_test_bool), len(prompt_test_bool))
    service.shutdown()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
