from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import extract_keyframes


class ExtractKeyframesTests(unittest.TestCase):
    def tearDown(self) -> None:
        extract_keyframes._HPSv3Ctx.model = None
        extract_keyframes._HPSv3Ctx.device = None

    def test_quality_model_is_frozen_when_cached(self):
        class FakeInferencer:
            def __init__(self, device):
                self.device = device
                self.model = torch.nn.Linear(2, 1)

        with patch("extract_keyframes.HPSv3RewardInferencer", FakeInferencer):
            model = extract_keyframes._get_quality_model(device="cpu")

        self.assertFalse(model.model.training)
        self.assertTrue(all(not param.requires_grad for param in model.model.parameters()))

    def test_quality_reward_runs_under_inference_mode(self):
        class FakeQualityModel:
            def __init__(self):
                self.grad_enabled = None
                self.inference_enabled = None

            def reward(self, image_paths, prompts):
                self.grad_enabled = torch.is_grad_enabled()
                self.inference_enabled = torch.is_inference_mode_enabled()
                return torch.tensor([[3.5]])

        quality_model = FakeQualityModel()
        frame = torch.zeros((3, 8, 8), dtype=torch.uint8)

        self.assertFalse(extract_keyframes.is_low_quality(frame, quality_model, threshold=3.0))
        self.assertFalse(quality_model.grad_enabled)
        self.assertTrue(quality_model.inference_enabled)


if __name__ == "__main__":
    unittest.main()
