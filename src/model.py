"""SIREN: Sinusoidal Representation Network (Sitzmann et al., NeurIPS 2020).

The network uses sine activations so its output can keep oscillating outside
the training window. Frequencies are learned from data. This module does not
import the ground-truth function and does not set weights from a known period.
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class SineLayer(nn.Module):
    """One SIREN layer: linear map followed by sin(omega_0 * x)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self) -> None:
        # Sitzmann et al.: first layer covers the input; later layers keep
        # the sine input well scaled so activations do not explode.
        with torch.no_grad():
            fan_in = self.linear.weight.shape[1]
            if self.is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / self.omega_0
            nn.init.uniform_(self.linear.weight, -bound, bound)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Stack of sine layers plus a linear output head."""

    def __init__(
        self,
        in_features: int = 1,
        hidden_dims: Optional[Sequence[int]] = None,
        out_features: int = 1,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]
        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one layer")

        dims: List[int] = list(hidden_dims)
        layers: List[nn.Module] = [
            SineLayer(in_features, dims[0], omega_0=omega_0, is_first=True)
        ]
        for i in range(1, len(dims)):
            layers.append(SineLayer(dims[i - 1], dims[i], omega_0=omega_0))

        # No sine on the output. The target range is unknown and bounded,
        # but the head should be free to match whatever scale f uses.
        out_layer = nn.Linear(dims[-1], out_features)
        with torch.no_grad():
            fan_in = out_layer.weight.shape[1]
            bound = math.sqrt(6.0 / fan_in) / omega_0
            nn.init.uniform_(out_layer.weight, -bound, bound)
            nn.init.zeros_(out_layer.bias)
        layers.append(out_layer)

        self.omega_0 = omega_0
        self.hidden_dims = dims
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Inference. Runs the network only and does not call the ground truth.

    Accepts shape (N, 1) or (N,). Returns shape (N, out_features).
    """
    if x.ndim == 1:
        x = x.unsqueeze(1)
    model.eval()
    return model(x)


class ReLUNet(nn.Module):
    """MLP with the same linear sizes as SIREN, using ReLU.

    ReLU has no parameters of its own, so the parameter count matches a
    SIREN of the same depth and width. Weights use PyTorch's default
    initialization, which is the standard init for this activation.
    Outside the training inputs a ReLU network extrapolates as a straight
    line. It does not keep oscillating.
    """

    def __init__(
        self,
        in_features: int = 1,
        hidden_dims: Optional[Sequence[int]] = None,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]
        layers: List[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TanhNet(nn.Module):
    """MLP with the same linear sizes as SIREN, using Tanh.

    Parameter count matches a SIREN of the same depth and width. Weights
    use PyTorch's default initialization. Tanh saturates, so far from the
    training inputs the network stops oscillating and flattens.
    """

    def __init__(
        self,
        in_features: int = 1,
        hidden_dims: Optional[Sequence[int]] = None,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 256]
        layers: List[nn.Module] = []
        prev = in_features
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.Tanh())
            prev = dim
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
