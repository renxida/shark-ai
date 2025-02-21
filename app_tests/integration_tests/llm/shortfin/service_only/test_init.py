from app_tests.integration_tests.llm.server_management import (
    ServerInstance,
    ServerConfig,
)
from app_tests.integration_tests.llm.model_management import TEST_MODELS, ModelProcessor
from app_tests.integration_tests.llm.device_settings import CPU


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


from shortfin_apps.llm.components.service import InferenceExecutorProcess
from shortfin_apps.llm.components.messages import InferencePhase, InferenceExecRequest


def batch_and_nobatch_consistency_test(generate_service):
    service = generate_service
    cache = service.page_cache

    exec_process = InferenceExecutorProcess(
        service,
        InferencePhase.PREFILL,
        service.model_params.paged_kv_cache.block_seq_stride,
        cache.page_pool.page_tables,
    )


with sinst.start_service_only() as generate_service:
    print("Service started")
    print(generate_service)
    batch_and_nobatch_consistency_test(generate_service)
    print(type(generate_service))


print("Service stopped")
