from typing import Tuple
from dataclasses import dataclass


@dataclass
class DeviceSettings:
    compile_flags: Tuple[str]
    server_flags: Tuple[str]


CPU = DeviceSettings(
    compile_flags=(
        "-iree-hal-target-backends=llvm-cpu",
        "--iree-llvmcpu-target-cpu=host",
    ),
    server_flags=("--device=local-task",),
)

GFX942 = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-backends=rocm",
        "--iree-hip-target=gfx942",
    ),
    server_flags=("--device=hip",),
)

def get_device_based_on_env_variable():
    import os

    device_name = os.environ.get("SHORTFIN_INTEGRATION_TEST_DEVICE", "cpu").lower()

    table = {
        "gpu": GFX942,
        "amdgpu": GFX942,
        "gfx942": GFX942,
        "host": CPU,
        "hostcpu": CPU,
        "local-task": CPU,
        "cpu": CPU,
    }
    if device_name in table:
        return table[device_name]

    raise ValueError(
        f"os.environ['SHORTFIN_INTEGRATION_TEST_DEVICE']=={device_name} but is not recognized. Supported device names: {list(table.keys())}"
    )