# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Server configuration management."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dataclasses_json import dataclass_json, Undefined


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ServerParams:
    """Server configuration parameters."""

    # KV cache configuration
    prefix_sharing_algorithm: str = "none"  # none or trie

    # Server runtime configuration
    host: Optional[str] = None
    port: int = 8000
    root_path: Optional[str] = None
    timeout_keep_alive: int = 5

    # Program isolation configuration
    program_isolation: str = "per_call"

    # Device configuration
    device_ids: list[str] = field(default_factory=list)
    amdgpu_async_allocations: bool = False
    amdgpu_allocators: Optional[str] = None

    @staticmethod
    def load(config_path: Optional[Path] = None) -> "ServerParams":
        """Load server configuration from a file or use defaults.

        Args:
            config_path: Path to config file. If None, will check standard locations.

        Returns:
            ServerParams instance with defaults or loaded values
        """
        if config_path is None:
            # Check standard locations
            config_paths = [
                Path.home() / ".shortfin" / "server_config.json",
                Path.home() / ".config" / "shortfin" / "server_config.json",
                Path("/etc/shortfin/server_config.json"),
            ]

            for path in config_paths:
                if path.exists():
                    config_path = path
                    break

        # Start with defaults
        params = ServerParams()

        # Override with config file if it exists
        if config_path and config_path.exists():
            with open(config_path) as f:
                file_params = ServerParams.from_json(f.read())
                # Update only non-None values from file
                for field in params.__dataclass_fields__:
                    file_value = getattr(file_params, field)
                    if file_value is not None:
                        setattr(params, field, file_value)

        return params

    def update_from_args(self, args) -> None:
        """Update configuration from command line arguments.

        Args:
            args: Parsed command line arguments

        Command line arguments take highest priority.
        """
        # Only update fields that are present in args
        for field in self.__dataclass_fields__:
            if hasattr(args, field):
                arg_value = getattr(args, field)
                if arg_value is not None:  # Only override if arg was provided
                    setattr(self, field, arg_value)

    def save(self, config_path: Optional[Path] = None):
        """Save configuration to a file.

        Args:
            config_path: Path to save to. If None, saves to ~/.shortfin/server_config.json
        """
        if config_path is None:
            config_path = Path.home() / ".shortfin" / "server_config.json"

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
