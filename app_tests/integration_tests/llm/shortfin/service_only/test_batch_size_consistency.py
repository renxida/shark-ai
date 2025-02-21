from app_tests.integration_tests.llm.server_management import (
    ServerInstance,
    ServerConfig,
)
from app_tests.integration_tests.llm.model_management import TEST_MODELS, ModelProcessor
from app_tests.integration_tests.llm.device_settings import CPU

import numpy as np
import shortfin as sf
from shortfin_apps.llm.components.messages import InferencePhase, InferenceExecRequest
from shortfin_apps.llm.components.service import InferenceExecutorProcess


processor = ModelProcessor(base_dir="/tmp/model_management")
model_config = TEST_MODELS["tinystories_llama2_25m"]
model_config.device_settings = CPU
artifacts = processor.process_model(TEST_MODELS["tinystories_llama2_25m"])
sconf = ServerConfig(
    artifacts=artifacts,
    device_settings=CPU,
    prefix_sharing_algorithm="none",
)

sinst = ServerInstance(sconf)
sinst.port = 0


class TestProcess(sf.Process):
    """Process for testing batch consistency across different batch sizes."""

    def __init__(self, service, input_tokens, batch_sizes, max_response_length=0):
        super().__init__(fiber=service.main_fiber)
        self.service = service
        self.input_tokens = input_tokens
        self.batch_sizes = batch_sizes
        self.max_response_length = max_response_length

    async def run_requests(self, phase, requests):
        """Run a batch of requests through InferenceExecutorProcess."""
        exec_process = InferenceExecutorProcess(
            self.service,
            phase,
            self.service.model_params.paged_kv_cache.block_seq_stride,
            self.service.page_cache.page_pool.page_tables,
        )
        exec_process.exec_requests = requests
        await exec_process.run()
        return [req.result_logits for req in requests]

    async def run(self):
        try:
            # Store results for each batch size
            prefill_results = {}
            decode_results = {}

            # Run prefill for each batch size
            for batch_size in self.batch_sizes:
                # Create batch of identical requests
                requests = []
                for i in range(batch_size):
                    request = InferenceExecRequest(
                        phase=InferencePhase.PREFILL,
                        input_token_ids=self.input_tokens,
                        rid=f"prefill_bs{batch_size}_{i}",
                    )
                    request.return_all_logits = True
                    requests.append(request)

                # Run the batch
                results = await self.run_requests(InferencePhase.PREFILL, requests)
                prefill_results[batch_size] = results

            # Verify all prefill results are identical across batch sizes
            first_bs = self.batch_sizes[0]
            first_result = prefill_results[first_bs][0]
            for bs in self.batch_sizes[1:]:
                for i in range(bs):
                    assert np.array_equal(
                        first_result.items, prefill_results[bs][i].items
                    ), f"Prefill results differ between batch sizes {first_bs} and {bs}"

            # Run decode phase if max_response_length > 0
            if self.max_response_length > 0:
                # For each decode step
                for step in range(self.max_response_length):
                    # For each batch size
                    for batch_size in self.batch_sizes:
                        # Create batch of identical requests
                        requests = []
                        for i in range(batch_size):
                            request = InferenceExecRequest(
                                phase=InferencePhase.DECODE,
                                input_token_ids=self.input_tokens,
                                rid=f"decode_bs{batch_size}_{i}_step{step}",
                            )
                            request.return_all_logits = True
                            request.start_position = len(self.input_tokens) + step
                            requests.append(request)

                        # Run the batch
                        results = await self.run_requests(
                            InferencePhase.DECODE, requests
                        )
                        decode_results.setdefault(step, {})[batch_size] = results

                    # Verify decode results are identical across batch sizes for this step
                    first_result = decode_results[step][first_bs][0]
                    for bs in self.batch_sizes[1:]:
                        for i in range(bs):
                            assert np.array_equal(
                                first_result.items, decode_results[step][bs][i].items
                            ), f"Decode results differ between batch sizes {first_bs} and {bs} at step {step}"

        except Exception as e:
            print(f"Error in TestProcess: {e}")
            raise


def batch_and_nobatch_consistency_test(generate_service):
    """Test that requests produce identical results regardless of batch size."""
    # Test parameters
    input_tokens = [1, 2, 3, 4]  # Initial sequence
    batch_sizes = [1, 2, 4]  # Different batch sizes to test
    max_response_length = 3  # Number of decode steps

    # Run the test process
    test_process = TestProcess(
        generate_service, input_tokens, batch_sizes, max_response_length
    )
    test_process.launch()


with sinst.start_service_only() as generate_service:
    print("Service started")
    batch_and_nobatch_consistency_test(generate_service)
    print("Test completed successfully")


print("Service stopped")
