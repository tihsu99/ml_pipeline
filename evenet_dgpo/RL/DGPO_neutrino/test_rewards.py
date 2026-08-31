"""Unit tests for RL/DGPO_neutrino/rewards.py."""

from __future__ import annotations

import os
import sys
import unittest

import torch

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _REPO_ROOT)

from RL.DGPO_neutrino.rewards import (
    CalibrationMagnitudeReward,
    ComponentNormalizedTruthDistanceReward,
    RewardAggregator,
    cartesian_to_log_pt_eta_phi,
    get_event_valid_mask,
    log_pt_eta_phi_to_cartesian,
)


def _random_invisible(B: int, N: int, F: int) -> torch.Tensor:
    x = torch.zeros(B, N, F)
    x[..., 0] = torch.log1p(torch.rand(B, N) * 80.0 + 1e-3)
    x[..., 1] = torch.randn(B, N) * 0.4
    x[..., 2] = (torch.rand(B, N) - 0.5) * 6.2
    return x


class TestLogPtEtaPhiToCartesian(unittest.TestCase):
    def test_matches_preprocessing_convention(self):
        log_pt = torch.log1p(torch.tensor(30.0))
        eta = torch.tensor(0.0)
        phi = torch.tensor(0.0)
        c = log_pt_eta_phi_to_cartesian(log_pt, eta, phi)
        torch.testing.assert_close(c, torch.tensor([30.0, 0.0, 0.0]))

    def test_cartesian_roundtrip(self):
        log_pt = torch.log1p(torch.tensor([10.0, 2.0]))
        eta = torch.tensor([0.3, -0.1])
        phi = torch.tensor([0.5, 1.2])
        px, py, pz = log_pt_eta_phi_to_cartesian(log_pt, eta, phi).unbind(-1)
        lp2, e2, p2 = cartesian_to_log_pt_eta_phi(px, py, pz)
        torch.testing.assert_close(lp2, log_pt, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(e2, eta, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(p2, phi, rtol=1e-5, atol=1e-5)


class TestComponentNormalizedTruthDistanceReward(unittest.TestCase):
    def test_reward_uses_root_sum_squared_distance(self):
        truth = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        batch = {
            "x_invisible": truth,
            "x_invisible_mask": torch.ones(1, 2),
        }
        candidates = truth.unsqueeze(0).clone()
        candidates[0, 0, 0, 0] = 3.0
        candidates[0, 0, 1, 1] = 4.0

        reward = ComponentNormalizedTruthDistanceReward(
            {
                "nu1_theta": 1.0,
                "nu1_phi": 1.0,
                "nu2_theta": 1.0,
                "nu2_phi": 1.0,
            },
            feature_names=("theta", "phi"),
        )
        value = reward.compute(candidates, batch)

        self.assertAlmostEqual(float(value[0, 0]), -5.0, places=5)

    def test_exact_match_highest_reward(self):
        B, K, F = 2, 4, 7
        truth = _random_invisible(B, 2, F)
        batch = {
            "x_invisible": truth,
            "x_invisible_mask": torch.ones(B, 2),
        }
        scales = [1.0] * 6
        c0 = torch.zeros(K, B, 2, F)
        c0[0, 0] = truth[0]
        c0[1, 0] = truth[0].clone()
        c0[1, 0, :, 0] += 2.0
        for k in range(K):
            c0[k, 1] = truth[1].clone()
            c0[k, 1, :, 0] += float(k) * 0.3

        r = ComponentNormalizedTruthDistanceReward(scales).compute(c0, batch)
        self.assertEqual(r.shape, (K, B))
        self.assertGreater(r[0, 0].item(), r[1, 0].item())

    def test_theta_phi_topology_metrics_are_exposed(self):
        truth = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        batch = {
            "x_invisible": truth,
            "x_invisible_mask": torch.ones(1, 2),
            "lead_a_visible_px": torch.tensor([1.0]),
            "lead_a_visible_py": torch.tensor([0.0]),
            "lead_a_visible_pz": torch.tensor([0.0]),
            "lead_b_visible_px": torch.tensor([-1.0]),
            "lead_b_visible_py": torch.tensor([0.0]),
            "lead_b_visible_pz": torch.tensor([0.0]),
        }
        candidates = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.4], [0.0, -0.6]]],
            ],
            dtype=torch.float32,
        )
        reward = ComponentNormalizedTruthDistanceReward(
            {
                "nu1_theta": 1.0,
                "nu1_phi": 1.0,
                "nu2_theta": 1.0,
                "nu2_phi": 1.0,
            },
            feature_names=("theta", "phi"),
        )

        reward.compute(candidates, batch)
        topology = reward.last_topology_metrics()

        self.assertIsNotNone(topology)
        assert topology is not None
        self.assertIn("back_to_back_loss", topology)
        self.assertAlmostEqual(float(topology["cos_opening"][0, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(topology["delta_phi_to_pi"][0, 0]), 0.0, places=5)
        self.assertLess(
            float(topology["back_to_back_loss"][0, 0]),
            float(topology["back_to_back_loss"][1, 0]),
        )


class TestGetEventValidMask(unittest.TestCase):
    def test_event_weight_zeros_event(self):
        batch = {"event_weight": torch.tensor([1.0, 0.0])}
        m = get_event_valid_mask(batch, 2, torch.device("cpu"), torch.float32)
        self.assertEqual(float(m[0]), 1.0)
        self.assertEqual(float(m[1]), 0.0)


class TestCalibrationMagnitudeReward(unittest.TestCase):
    def test_back_to_back_candidate_has_higher_reward(self):
        truth = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        batch = {
            "x_invisible": truth,
            "x_invisible_mask": torch.ones(1, 2),
            "lead_a_visible_px": torch.tensor([1.0]),
            "lead_a_visible_py": torch.tensor([0.0]),
            "lead_a_visible_pz": torch.tensor([0.0]),
            "lead_b_visible_px": torch.tensor([-1.0]),
            "lead_b_visible_py": torch.tensor([0.0]),
            "lead_b_visible_pz": torch.tensor([0.0]),
        }
        candidates = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.4], [0.0, -0.6]]],
            ],
            dtype=torch.float32,
        )
        reward = CalibrationMagnitudeReward(feature_names=("theta", "phi"))
        values = reward.compute(candidates, batch)
        self.assertGreater(float(values[0, 0]), float(values[1, 0]))
        topology = reward.last_topology_metrics()
        self.assertIsNotNone(topology)
        assert topology is not None
        self.assertAlmostEqual(float(topology["calibration_deltaR_sum"][0, 0]), 0.0, places=5)


class TestRewardAggregator(unittest.TestCase):
    def test_single_source(self):
        B, K, F = 1, 2, 7
        truth = _random_invisible(B, 2, F)
        batch = {"x_invisible": truth, "x_invisible_mask": torch.ones(B, 2)}
        candidates = truth.unsqueeze(0).expand(K, -1, -1, -1).clone()
        agg = RewardAggregator()
        agg.add(ComponentNormalizedTruthDistanceReward([1.0] * 6), 1.0)
        total, breakdown = agg.compute(candidates, batch)
        self.assertEqual(total.shape, (K, B))
        self.assertIn("component_normalized_truth_distance", breakdown)

    def test_multiple_sources_are_added(self):
        B, K = 1, 2
        truth = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        batch = {
            "x_invisible": truth,
            "x_invisible_mask": torch.ones(1, 2),
            "lead_a_visible_px": torch.tensor([1.0]),
            "lead_a_visible_py": torch.tensor([0.0]),
            "lead_a_visible_pz": torch.tensor([0.0]),
            "lead_b_visible_px": torch.tensor([-1.0]),
            "lead_b_visible_py": torch.tensor([0.0]),
            "lead_b_visible_pz": torch.tensor([0.0]),
        }
        candidates = torch.tensor(
            [
                [[[0.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 0.4], [0.0, -0.6]]],
            ],
            dtype=torch.float32,
        )
        mse_reward = ComponentNormalizedTruthDistanceReward(
            {
                "nu1_theta": 1.0,
                "nu1_phi": 1.0,
                "nu2_theta": 1.0,
                "nu2_phi": 1.0,
            },
            feature_names=("theta", "phi"),
        )
        calib_reward = CalibrationMagnitudeReward(feature_names=("theta", "phi"))
        agg = RewardAggregator()
        agg.add(mse_reward, 0.5)
        agg.add(calib_reward, 2.0)

        total, breakdown = agg.compute(candidates, batch)

        expected = 0.5 * breakdown["component_normalized_truth_distance"] + 2.0 * breakdown["calibration_magnitude"]
        torch.testing.assert_close(total, expected)


if __name__ == "__main__":
    unittest.main()
