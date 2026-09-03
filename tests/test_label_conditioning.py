from types import SimpleNamespace
import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from train import (
    CONDITION_GROUPS,
    LABEL_FUSION_HEADS,
    LabelConditionedQualityMLP,
    _checkpoint_paths,
    encode_groups,
    evaluate,
)


class _FeatureDataset(Dataset):
    def __init__(self):
        groups = ["blur", "noise", "compression", "tone"] * 2
        self.rows = pd.DataFrame({"group": groups})
        values = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        self.features = torch.sin(values) + torch.cos(values / 3)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {
            "image": self.features[index],
            "target": torch.tensor(index / 7, dtype=torch.float32),
            "reference": "reference-1",
            "dataset": "tiny",
            "group": self.rows.iloc[index]["group"],
        }


class _IdentityBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        return SimpleNamespace(pooler_output=pixel_values)


class _TokenBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.vision_model = SimpleNamespace(post_layernorm=torch.nn.Identity())

    def forward(self, pixel_values):
        self.calls += 1
        tokens = torch.stack(
            (pixel_values, pixel_values + 0.25, pixel_values - 0.25), dim=1,
        )
        return SimpleNamespace(
            pooler_output=pixel_values,
            last_hidden_state=tokens,
        )


class LabelConditioningTests(unittest.TestCase):
    def test_checkpoint_paths_use_experiment_name(self):
        best, last = _checkpoint_paths("weights", "label_conditioned")
        self.assertEqual(best.name, "label_conditioned_best.pth")
        self.assertEqual(last.name, "label_conditioned_last.pth")

    def test_group_encoding_has_stable_unknown_and_colour_alias(self):
        ids = encode_groups(["blur", "colour", "", "new-distortion"]).tolist()
        self.assertEqual(ids[:2], [2, 4])
        self.assertEqual(ids[2:], [len(CONDITION_GROUPS) - 1] * 2)

    def test_conditioned_head_forward_and_backward(self):
        head = LabelConditionedQualityMLP(4, len(CONDITION_GROUPS), label_dim=3)
        output = head(torch.randn(5, 4), encode_groups(["blur"] * 5))
        self.assertEqual(output.shape, (5,))
        output.sum().backward()
        self.assertIsNotNone(head.group_embedding.weight.grad)

    def test_evaluation_reuses_features_for_all_condition_ablations(self):
        loader = DataLoader(_FeatureDataset(), batch_size=4)
        backbone = _IdentityBackbone()
        head = LabelConditionedQualityMLP(
            4, len(CONDITION_GROUPS), hidden_dim=8, label_dim=3,
        )

        scores = evaluate(
            backbone, head, loader, torch.device("cpu"), conditioned=True,
        )

        self.assertEqual(backbone.calls, len(loader))
        self.assertEqual(
            set(scores["condition_ablations"]), {"shuffled", "wrong", "zeroed"},
        )
        self.assertIn("tiny", scores["per_dataset"])
        self.assertGreater(scores["images_per_second"], 0)
        self.assertGreaterEqual(scores["latency_p95_ms"], scores["latency_p50_ms"])
        self.assertIsNone(scores["peak_memory_mb"])

    def test_patch_attention_evaluation_requests_backbone_tokens(self):
        loader = DataLoader(_FeatureDataset(), batch_size=4)
        backbone = _TokenBackbone()
        head = LABEL_FUSION_HEADS["patch_attention"](
            4, len(CONDITION_GROUPS), hidden_dim=16, label_dim=3,
        )

        scores = evaluate(
            backbone, head, loader, torch.device("cpu"), conditioned=True,
        )

        self.assertEqual(backbone.calls, len(loader))
        self.assertIn("tiny", scores["per_dataset"])
        self.assertEqual(
            set(scores["condition_ablations"]), {"shuffled", "wrong", "zeroed"},
        )


if __name__ == "__main__":
    unittest.main()
