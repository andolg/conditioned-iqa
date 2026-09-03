from types import SimpleNamespace
import unittest

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from arniqa import embed_arniqa
from evaluate import _head_from_checkpoint
from train import ARNIQAConditionedQualityMLP, evaluate


class _MeanEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, means.square()), dim=1)


class _IdentityBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, pixel_values):
        self.calls += 1
        return SimpleNamespace(pooler_output=pixel_values)


class _ARNIQADataset(Dataset):
    def __init__(self):
        self.rows = pd.DataFrame({"group": ["blur"] * 8})
        values = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        self.features = torch.sin(values)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        condition = torch.full((5, 1, 1, 1), float(index + 1))
        return {
            "image": self.features[index],
            "arniqa_image": condition,
            "arniqa_image_ds": condition / 2,
            "target": torch.tensor(index / 7, dtype=torch.float32),
            "reference": "reference-1",
            "dataset": "tiny",
            "group": "blur",
        }


class ARNIQAConditioningTests(unittest.TestCase):
    def test_embedding_concatenates_scales_and_averages_crops(self):
        encoder = _MeanEncoder()
        full = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2, 1, 1)
        half = full + 10

        embedding = embed_arniqa(encoder, full, half, chunk_size=3)

        expected_full = encoder(full.flatten(0, 1)).mean(dim=0)
        expected_half = encoder(half.flatten(0, 1)).mean(dim=0)
        self.assertTrue(
            torch.allclose(embedding[0], torch.cat((expected_full, expected_half)))
        )

    def test_conditioned_head_forward_and_backward(self):
        head = ARNIQAConditionedQualityMLP(
            input_dim=4, arniqa_dim=6, hidden_dim=8, condition_dim=3
        )
        output = head(torch.randn(5, 4), torch.randn(5, 6))
        self.assertEqual(output.shape, (5,))
        output.sum().backward()
        self.assertIsNotNone(head.condition_projection[0].weight.grad)

    def test_checkpoint_reconstructs_arniqa_head(self):
        original = ARNIQAConditionedQualityMLP(
            input_dim=4, arniqa_dim=6, hidden_dim=7, condition_dim=3
        )
        checkpoint = {
            "head": original.state_dict(),
            "backbone": "clip-base",
            "feature_dim": 4,
            "hidden_dim": 7,
            "conditioning": "arniqa",
            "condition_dim": 3,
            "arniqa_feature_dim": 6,
        }

        head, conditioning, feature_dim = _head_from_checkpoint(
            checkpoint, torch.device("cpu")
        )

        self.assertIsInstance(head, ARNIQAConditionedQualityMLP)
        self.assertEqual(conditioning, "arniqa")
        self.assertEqual(feature_dim, 4)
        self.assertEqual(head.condition_projection[0].out_features, 3)

    def test_evaluation_reuses_both_encoders_for_ablations(self):
        loader = DataLoader(_ARNIQADataset(), batch_size=4)
        backbone = _IdentityBackbone()
        arniqa_encoder = _MeanEncoder()
        head = ARNIQAConditionedQualityMLP(
            input_dim=4, arniqa_dim=4, hidden_dim=8, condition_dim=3
        )

        scores = evaluate(
            backbone,
            head,
            loader,
            torch.device("cpu"),
            conditioned=True,
            conditioning="arniqa",
            arniqa_encoder=arniqa_encoder,
            arniqa_batch_size=8,
        )

        self.assertEqual(backbone.calls, len(loader))
        # Each loader batch has 40 crop tensors and is encoded in five chunks.
        self.assertEqual(arniqa_encoder.calls, len(loader) * 5)
        self.assertEqual(set(scores["condition_ablations"]), {"shuffled", "zeroed"})
        self.assertIn("tiny", scores["per_dataset"])


if __name__ == "__main__":
    unittest.main()
