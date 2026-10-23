"""SIREN architecture, initialization, and inference isolation."""

import inspect
import math

import torch

from src.model import SIREN, SineLayer, predict


def test_forward_shape():
    model = SIREN(hidden_dims=[16, 16], omega_0=30.0)
    y = model(torch.zeros(7, 1))
    assert y.shape == (7, 1)
    y_vec = predict(model, torch.linspace(-1, 1, 5))
    assert y_vec.shape == (5, 1)


def test_default_parameter_count():
    model = SIREN()
    # 4 x 256 sine layers + linear head, as counted in the plan.
    assert model.count_parameters() == 198_145


def test_first_layer_init_bounds():
    layer = SineLayer(1, 32, omega_0=30.0, is_first=True)
    weight = layer.linear.weight
    bound = 1.0 / weight.shape[1]
    assert torch.all(weight >= -bound - 1e-6)
    assert torch.all(weight <= bound + 1e-6)
    assert torch.all(layer.linear.bias == 0)
    assert torch.any(weight != 0)


def test_hidden_and_output_init_bounds():
    omega_0 = 30.0
    model = SIREN(hidden_dims=[16, 8], omega_0=omega_0)

    hidden = model.net[1]
    fan_in = hidden.linear.weight.shape[1]
    bound = math.sqrt(6.0 / fan_in) / omega_0
    assert torch.all(hidden.linear.weight >= -bound - 1e-6)
    assert torch.all(hidden.linear.weight <= bound + 1e-6)
    assert torch.all(hidden.linear.bias == 0)

    out = model.net[-1]
    out_bound = math.sqrt(6.0 / out.weight.shape[1]) / omega_0
    assert torch.all(out.weight >= -out_bound - 1e-6)
    assert torch.all(out.weight <= out_bound + 1e-6)
    assert torch.all(out.bias == 0)


def test_model_source_does_not_reference_black_box_or_period():
    source = inspect.getsource(SIREN) + inspect.getsource(SineLayer) + inspect.getsource(predict)
    lowered = source.lower()
    assert "black_box" not in lowered
    assert "import src" not in lowered
    # The placeholder formula is not baked into the architecture.
    assert "sin(2" not in lowered
    assert "2.0 * x" not in lowered
