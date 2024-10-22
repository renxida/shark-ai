import logging
from pathlib import Path

import shortfin as sf
from shortfin_apps.llm.components.service import GenerateService
from shortfin_apps.llm.components.manager import SystemManager
from shortfin_apps.llm.components.tokenizer import Tokenizer
from shortfin_apps.llm.components.config_struct import ModelParams
from shortfin_apps.llm.components.messages import InferenceExecRequest, InferencePhase

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def configure():
    sysman = SystemManager(device="hip")
    tokenizer = Tokenizer.from_tokenizer_json_file(
        "/tmp/sharktank/llama/tokenizer.json"
    )
    model_params = ModelParams.load_json("/tmp/sharktank/llama/edited_config.json")

    service = GenerateService(
        name="test", sysman=sysman, tokenizer=tokenizer, model_params=model_params
    )
    service.load_inference_module("/tmp/sharktank/llama/model.vmfb")
    service.load_inference_parameters(
        "/tmp/sharktank/llama/open-llama-3b-v2-f16.gguf", parameter_scope="model"
    )

    return sysman, service, tokenizer


async def main():
    sysman, service, tokenizer = configure()

    try:
        service.start()

        # Test input
        input_text = "1 2 3 4 5"
        input_tokens = tokenizer.encode([input_text])

        # Create inference request
        request = InferenceExecRequest(InferencePhase.PREFILL, input_tokens)

        # Submit request
        await service.batcher.submit(request)

        # Wait for completion
        await request.done.wait()

        # Decode output
        output_tokens = request.result_logits.argmax(axis=-1).tolist()
        output_text = tokenizer.decode(output_tokens)

        logger.info(f"Input: {input_text}")
        logger.info(f"Output: {output_text}")

        # Dump cache contents
        service.page_cache.dump_cache_contents()

    finally:
        service.shutdown()
        sysman.shutdown()


import asyncio

asyncio.run(main())
