"""Focused regression tests for graph construction and imbalance sampling."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from epilepsy.data import EpochBalancedSampler
from epilepsy.models.gat import DEFAULT_CHANNEL_NAMES, PriorMatrixBuilder, compute_plv_batch


class ModelRegressionTests(unittest.TestCase):
    def test_plv_is_symmetric_with_unit_diagonal(self) -> None:
        x = torch.randn(3, len(DEFAULT_CHANNEL_NAMES), 128)
        plv = compute_plv_batch(x)
        self.assertEqual(plv.shape, (3, 22, 22))
        self.assertTrue(torch.allclose(plv, plv.transpose(1, 2), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                plv.diagonal(dim1=1, dim2=2), torch.ones(3, 22), atol=1e-6
            )
        )

    def test_graph_for_sample_does_not_depend_on_batch_neighbors(self) -> None:
        builder = PriorMatrixBuilder(DEFAULT_CHANNEL_NAMES)
        target = torch.eye(22)
        first = builder(torch.stack((target, torch.zeros_like(target))))[0]
        second = builder(torch.stack((target, torch.ones_like(target))))[0]
        self.assertTrue(torch.allclose(first, second, atol=1e-7))

    def test_negative_sampling_ratio(self) -> None:
        labels = np.asarray([1] * 10 + [0] * 100)
        sampler = EpochBalancedSampler(labels, seed=7, negative_ratio=3.0)
        selected = np.asarray(list(iter(sampler)))
        self.assertEqual(len(selected), 40)
        self.assertEqual(int(np.sum(labels[selected] == 1)), 10)
        self.assertEqual(int(np.sum(labels[selected] == 0)), 30)


if __name__ == "__main__":
    unittest.main()
