import unittest

import torch

from evaluate import _head_from_checkpoint
from train import CONDITION_GROUPS, LABEL_FUSION_HEADS, QualityMLP


class LabelFusionExperimentTests(unittest.TestCase):
    def test_all_fusions_forward_backward(self):
        groups = torch.tensor([0, 1, 2, 3, 4])
        for name, head_class in LABEL_FUSION_HEADS.items():
            with self.subTest(fusion=name):
                head = head_class(
                    input_dim=4, num_groups=len(CONDITION_GROUPS), hidden_dim=8,
                    label_dim=3, dropout=0.0, condition_dropout=0.0,
                )
                output = head(torch.randn(5, 4), groups)
                self.assertEqual(output.shape, (5,))
                output.sum().backward()
                self.assertTrue(any(p.grad is not None for p in head.parameters()))

    def test_additive_and_film_start_as_identity_conditioning(self):
        features = torch.randn(5, 4)
        first = torch.zeros(5, dtype=torch.long)
        second = torch.ones(5, dtype=torch.long)
        for name in ("additive", "film_input", "film_hidden"):
            with self.subTest(fusion=name):
                head = LABEL_FUSION_HEADS[name](
                    4, len(CONDITION_GROUPS), hidden_dim=8, label_dim=3,
                    dropout=0.0, condition_dropout=0.0,
                ).eval()
                torch.testing.assert_close(head(features, first), head(features, second))
                torch.testing.assert_close(
                    head(features, first), head(features, first, zero_condition=True)
                )

    def test_residual_gate_starts_near_zero_and_zero_condition_is_baseline(self):
        head = LABEL_FUSION_HEADS["residual_gate"](
            4, len(CONDITION_GROUPS), hidden_dim=8, label_dim=3,
            dropout=0.0, condition_dropout=0.0,
        ).eval()
        condition = head.group_embedding(torch.tensor([0, 1]))
        gate = torch.sigmoid(head.gate(condition))
        self.assertTrue(torch.all(gate < 0.02))
        features = torch.randn(2, 4)
        normalized = head.feature_norm(features)
        torch.testing.assert_close(
            head(features, torch.tensor([0, 1]), zero_condition=True),
            head.base_head(normalized).squeeze(-1),
        )

    def test_patch_attention_zero_condition_uses_class_token_fallback(self):
        head = LABEL_FUSION_HEADS["patch_attention"](
            8, len(CONDITION_GROUPS), hidden_dim=16, label_dim=3,
            dropout=0.0, condition_dropout=0.0,
        ).eval()
        tokens = torch.randn(2, 5, 8)
        groups = torch.tensor([0, 1])
        expected = head.net(head.feature_norm(tokens)[:, 0]).squeeze(-1)
        torch.testing.assert_close(
            head(tokens, groups, zero_condition=True),
            expected,
        )

    def test_low_rank_hypernetwork_starts_as_unconditional_path(self):
        head = LABEL_FUSION_HEADS["low_rank_hypernetwork"](
            8, len(CONDITION_GROUPS), hidden_dim=16, label_dim=3,
            dropout=0.0, condition_dropout=0.0, rank=2,
        ).eval()
        features = torch.randn(2, 8)
        first = torch.tensor([0, 0])
        second = torch.tensor([1, 1])
        torch.testing.assert_close(head(features, first), head(features, second))
        torch.testing.assert_close(
            head(features, first),
            head(features, first, zero_condition=True),
        )

    def test_new_heads_match_baseline_parameter_budget(self):
        baseline = QualityMLP(768, hidden_dim=256)
        baseline_parameters = sum(p.numel() for p in baseline.parameters())
        for name in ("patch_attention", "low_rank_hypernetwork"):
            with self.subTest(fusion=name):
                head = LABEL_FUSION_HEADS[name](
                    768, len(CONDITION_GROUPS), hidden_dim=256, label_dim=32,
                )
                parameters = sum(p.numel() for p in head.parameters())
                self.assertLessEqual(parameters, baseline_parameters)
                self.assertLess(
                    baseline_parameters - parameters,
                    baseline_parameters * 0.01,
                )

    def test_checkpoint_reconstructs_each_new_fusion(self):
        for name in (
            "additive", "film_input", "film_hidden", "residual_gate",
            "patch_attention", "low_rank_hypernetwork",
        ):
            with self.subTest(fusion=name):
                original = LABEL_FUSION_HEADS[name](
                    4, len(CONDITION_GROUPS), hidden_dim=7, label_dim=3,
                )
                checkpoint = {
                    "head": original.state_dict(), "backbone": "clip-base",
                    "feature_dim": 4, "hidden_dim": 7, "conditioning": "label",
                    "label_fusion": name, "label_dim": 3,
                    "low_rank_dim": 4,
                    "groups": list(CONDITION_GROUPS),
                }
                restored, conditioning, _ = _head_from_checkpoint(
                    checkpoint, torch.device("cpu")
                )
                self.assertIsInstance(restored, LABEL_FUSION_HEADS[name])
                self.assertEqual(conditioning, "label")


if __name__ == "__main__":
    unittest.main()
