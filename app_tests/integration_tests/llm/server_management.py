"""Handles server lifecycle and configuration."""
import socket
from contextlib import closing
from dataclasses import dataclass
import subprocess
import time
import requests
import sys
from typing import Optional

from .device_settings import DeviceSettings
from .model_management import ModelArtifacts


@dataclass
class ServerConfig:
    """Configuration for server instance."""

    artifacts: ModelArtifacts
    device_settings: DeviceSettings
    prefix_sharing_algorithm: str = "none"


class ServerInstance:
    """An instance of the shortfin llm inference server.

    Example usage:

    ```
        from shortfin_apps.llm.server_management import ServerInstance, ServerConfig
        # Create and start server
        server = Server(config=ServerConfig(
            artifacts=model_artifacts,
            device_settings=device_settings,
            prefix_sharing_algorithm="none"
        ))

        server.start()  # This starts the server and waits for it to be ready

        # Use the server
        print(f"Server running on port {server.port}")

        # Cleanup when done
        server.stop()
    ```
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None

    @staticmethod
    def find_available_port() -> int:
        """Finds an available port for the server."""
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def start(self) -> None:
        """Starts the server process."""
        if self.process is not None:
            raise RuntimeError("Server is already running")

        self.port = self.find_available_port()

        cmd = [
            sys.executable,
            "-m",
            "shortfin_apps.llm.server",
            f"--tokenizer_json={self.config.artifacts.tokenizer_path}",
            f"--model_config={self.config.artifacts.config_path}",
            f"--vmfb={self.config.artifacts.vmfb_path}",
            f"--parameters={self.config.artifacts.weights_path}",
            f"--port={self.port}",
            f"--prefix_sharing_algorithm={self.config.prefix_sharing_algorithm}",
        ]
        cmd.extend(self.config.device_settings.server_flags)

        self.process = subprocess.Popen(cmd)
        self.wait_for_ready()

    def wait_for_ready(self, timeout: int = 30) -> None:
        """Waits for server to be ready and responding to health checks."""
        if self.port is None:
            raise RuntimeError("Server hasn't been started")

        start = time.time()
        while time.time() - start < timeout:
            try:
                requests.get(f"http://localhost:{self.port}/health")
                return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise TimeoutError(f"Server failed to start within {timeout} seconds")

    def stop(self) -> None:
        """Stops the server process."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.port = None
