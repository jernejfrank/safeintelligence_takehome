"""Tests for the PGD module.

To ensure consistency in testing we fix the model structure and weights. We are using
the model in provided.network.SimpleNN that has the following architecture:

self.fc1 = nn.Linear(3, 10)  # Input layer (3 features) -> Hidden layer (10 neurons)
self.fc2 = nn.Linear(10, 1)  # Hidden layer (10 neurons) -> Output layer (1 output)
self.relu = nn.ReLU() # Non-linear activation for hidden layer

The weights are stored in tests.model_cache.cached_model_weights.pth.
"""

from pathlib import Path

import pytest
import torch

from provided.network import SimpleNN
from provided.pgd import PGDAttack


@pytest.fixture(scope="module")
def cached_model() -> SimpleNN:
    model = SimpleNN()
    cached_weights = Path(__file__).parent / "model_cache" / "cached_model_weights.pth"
    model.load_state_dict(torch.load(cached_weights))
    model.to("cpu")
    model.eval()
    yield model


@pytest.fixture(scope="module")
def pgd(cached_model) -> PGDAttack:
    return PGDAttack(cached_model, epsilon=0.5, steps=1)


def test_is_a_counterexample_true(pgd: PGDAttack):
    dummy_input = torch.tensor([[1, 1, 1]], dtype=torch.float32).to("cpu")
    assert pgd.is_a_counterexample(x_adv=dummy_input, label=1, step=0)


def test_is_a_counterexample_false(pgd: PGDAttack):
    dummy_input = torch.tensor([[1, 1, 1]], dtype=torch.float32).to("cpu")
    assert not pgd.is_a_counterexample(x_adv=dummy_input, label=0, step=0)
