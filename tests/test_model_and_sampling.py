"""Focused regression tests for graph construction and imbalance sampling."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from epilepsy.data import EpochBalancedSampler
from epilepsy.chbmit import TARGET_CHANNELS, resolve_montage
from epilepsy.models.gat import DEFAULT_CHANNEL_NAMES, PriorMatrixBuilder, compute_plv_batch
from epilepsy.train import calculate_binary_metrics, select_decision_threshold


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

    def test_common_chbmit_montage_has_18_channels(self) -> None:
        montage = resolve_montage(TARGET_CHANNELS)
        self.assertEqual(len(montage), 18)
        self.assertTrue(all(len(sources) == 1 for sources in montage))

    def test_referential_channels_are_reconstructed_as_bipolar(self) -> None:
        electrodes = []
        for channel in TARGET_CHANNELS:
            for electrode in channel.split("-"):
                if electrode not in electrodes:
                    electrodes.append(electrode)
        names = [f"{electrode}-CS2" for electrode in electrodes]
        montage = resolve_montage(names)
        self.assertEqual(len(montage), 18)
        self.assertTrue(all(len(sources) == 2 for sources in montage))
        self.assertTrue(all(sum(weight for _, weight in sources) == 0 for sources in montage))

    def test_false_positives_per_hour_uses_negative_duration(self) -> None:
        metrics, _ = calculate_binary_metrics(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.7, 0.1, 0.8, 0.2]),
            loss=0.0,
            threshold=0.5,
            segment_duration=1800.0,
        )
        self.assertEqual(metrics.fpr_per_hour, 1.0)

    def test_fixed_threshold_protocol_does_not_tune_on_validation(self) -> None:
        threshold, _ = select_decision_threshold(
            np.asarray([0, 0, 1, 1]),
            np.asarray([0.8, 0.7, 0.6, 0.5]),
            metric_name="fixed",
            loss=0.0,
        )
        self.assertEqual(threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
