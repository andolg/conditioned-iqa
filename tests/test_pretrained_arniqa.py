import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from arniqa import scale_arniqa_score
from evaluate_arniqa import evaluate_pretrained_arniqa


class _MeanEncoder(torch.nn.Module):
    def forward(self, images):
        means = images.mean(dim=(1, 2, 3))
        return torch.stack((means, means.square()), dim=1)


class _MetricDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        condition = torch.full((5, 1, 1, 1), 1.0 + index / 7)
        return {
            "arniqa_image": condition,
            "arniqa_image_ds": condition / 2,
            "target": torch.tensor(index / 7, dtype=torch.float32),
            "reference": "reference-1",
            "dataset": "tiny",
        }


class PretrainedARNIQATests(unittest.TestCase):
    def test_official_score_scaling_and_dmos_direction(self):
        kadid = scale_arniqa_score(torch.tensor([1.0, 5.0]), "kadid10k")
        live = scale_arniqa_score(torch.tensor([1.0, 100.0]), "live")
        self.assertTrue(torch.allclose(kadid, torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.allclose(live, torch.tensor([1.0, 0.0])))

    def test_standalone_evaluation_reports_iqa_and_runtime_metrics(self):
        regressor = torch.nn.Linear(4, 1)
        with torch.no_grad():
            regressor.weight.fill_(0.25)
            regressor.bias.fill_(1.0)
        scores = evaluate_pretrained_arniqa(
            _MeanEncoder(),
            regressor,
            DataLoader(_MetricDataset(), batch_size=4),
            torch.device("cpu"),
            arniqa_batch_size=8,
        )

        self.assertIn("tiny", scores["per_dataset"])
        self.assertGreater(scores["images_per_second"], 0)
        self.assertGreaterEqual(scores["latency_p95_ms"], scores["latency_p50_ms"])
        self.assertIsNone(scores["peak_memory_mb"])


if __name__ == "__main__":
    unittest.main()
