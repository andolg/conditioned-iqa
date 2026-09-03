import unittest

import torch

from types import SimpleNamespace

from evaluate import _head_from_checkpoint, _model_complexity
from train import CONDITION_GROUPS, LabelConditionedQualityMLP, QualityMLP


class CheckpointLoadingTests(unittest.TestCase):
    def test_loads_legacy_baseline_checkpoint_without_hidden_dim_metadata(self):
        original = QualityMLP(4, hidden_dim=7)
        checkpoint = {
            "head": original.state_dict(),
            "backbone": "clip-base",
            "feature_dim": 4,
        }

        head, conditioning, feature_dim = _head_from_checkpoint(
            checkpoint, torch.device("cpu"),
        )

        self.assertIsInstance(head, QualityMLP)
        self.assertEqual(conditioning, "none")
        self.assertEqual(feature_dim, 4)
        self.assertEqual(head.net[1].out_features, 7)

    def test_loads_label_conditioned_checkpoint(self):
        original = LabelConditionedQualityMLP(
            4, len(CONDITION_GROUPS), hidden_dim=7, label_dim=3,
        )
        checkpoint = {
            "head": original.state_dict(),
            "backbone": "clip-base",
            "feature_dim": 4,
            "hidden_dim": 7,
            "conditioning": "label",
            "label_dim": 3,
            "groups": list(CONDITION_GROUPS),
        }

        head, conditioning, feature_dim = _head_from_checkpoint(
            checkpoint, torch.device("cpu"),
        )

        self.assertIsInstance(head, LabelConditionedQualityMLP)
        self.assertEqual(conditioning, "label")
        self.assertEqual(feature_dim, 4)
        self.assertEqual(head.group_embedding.embedding_dim, 3)

    def test_model_complexity_counts_parameters_and_flops(self):
        class TinyBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(12, 4)

            def forward(self, pixel_values):
                return SimpleNamespace(
                    pooler_output=self.projection(pixel_values.flatten(1)),
                )

        backbone = TinyBackbone().eval().requires_grad_(False)
        head = QualityMLP(4, hidden_dim=3).eval()
        complexity = _model_complexity(
            backbone, head, image_size=2, device=torch.device("cpu"),
            conditioned=False,
        )

        expected_total = sum(
            parameter.numel()
            for module in (backbone, head)
            for parameter in module.parameters()
        )
        self.assertEqual(complexity["total_parameters"], expected_total)
        self.assertEqual(
            complexity["trainable_parameters"],
            sum(parameter.numel() for parameter in head.parameters()),
        )
        self.assertGreater(complexity["gflops_per_image"], 0)


if __name__ == "__main__":
    unittest.main()
