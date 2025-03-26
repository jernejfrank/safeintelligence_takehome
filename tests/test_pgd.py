"""Tests for the PGD module."""

import torch

from provided.pgd import PGDAttack


def test_is_a_counterexample():
    dummy_input = torch.tensor([[1, 1, 1]], dtype=torch.float32)
    correct_output = torch.tensor([1])
    correct_label = 1
    incorrect_label = 0
    assert not PGDAttack.is_a_counterexample(
        x_adv=dummy_input,
        perturbed_prediction=correct_output,
        label=correct_label,
        current_epsilon=0,
    )
    assert PGDAttack.is_a_counterexample(
        x_adv=dummy_input,
        perturbed_prediction=correct_output,
        label=incorrect_label,
        current_epsilon=0,
    )
