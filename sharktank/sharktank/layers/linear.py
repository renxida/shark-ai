# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import torch

from .. import ops
from .base import Theta, ThetaLayer
from ..types.layout_utils import saturate_cast
from ..types import (
    DynamicScaledQuantizer,
    QuantizedTensor,
    QuantizerTensor,
    StaticScaledQuantizer,
    TensorScaledLayout,
)

__all__ = [
    "LinearLayer",
]


class LinearLayer(ThetaLayer):
    """Linear layer which computes:

    ```
    if premul_input is not None:
      x = x * premul_input
    matmul(x, weight.T) + bias
    ```
    """

    def __init__(
        self,
        theta: Theta,
        *,
        weight_name: str = "weight",
        bias_name: str = "bias",
    ):
        super().__init__(theta)
        self._simulate_native_quant = True
        self.weight = self.theta_tensor(weight_name)
        self.bias = None
        if bias_name in self.theta.keys:
            self.bias = self.theta_tensor(bias_name)

        # Input premultiplier.
        self.premul_input = theta.optional_tensor("premul_input")
        self.q_input: Optional[QuantizerTensor] = theta.optional_tensor("q_input")
        self.qdq_input: Optional[QuantizedTensor] = theta.optional_tensor("qdq_input")
        if self.q_input is not None and self.qdq_input is not None:
            raise AssertionError(f"LinearLayer cannot have both q_input and qdq_input")

    def forward(self, x):
        weight = self.weight
        bias = self.bias
        q_input = self.q_input
        qdq_input = self.qdq_input

        if self.premul_input is not None:
            x = ops.elementwise(torch.mul, x, self.premul_input)

        if q_input is not None:
            x = q_input.quantize(x)
        elif qdq_input is not None:
            x = qdq_input.quantize(x).unpack().dequant()
        if "attn_q" in weight.name:
            weight._data = (
                weight.as_torch()
                .reshape(64, weight.shape[0] // 64 // 2, 2, *weight.shape[1:])
                .swapaxes(1, 2)
                .reshape(weight.shape)
            )
        if "attn_k" in weight.name:
            weight._data = (
                weight.as_torch()
                .reshape(8, weight.shape[0] // 8 // 2, 2, *weight.shape[1:])
                .swapaxes(1, 2)
                .reshape(weight.shape)
            )
        y = ops.linear(x, weight, bias)

        # Unconditionally dequantize.
        # TODO: Support a q_output specifier that signals the layer to let
        # the QuantizedTensor escape.
        if isinstance(y, QuantizedTensor):
            y = y.unpack().dequant()
        return y
