"""Lightweight end-to-end test for training, evaluation, and MLflow logging."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlflow
import pandas as pd
import torch
import yaml

import train


class TinyDataset(torch.utils.data.Dataset):
    """Synthetic IQA data that avoids model and dataset downloads."""

    def __init__(self, source, *args, **kwargs):
        if isinstance(source, pd.DataFrame):
            self.rows = source.reset_index(drop=True)
        else:
            self.rows = pd.DataFrame({
                "target": torch.linspace(0.0, 1.0, 20).tolist(),
                "reference": [f"ref-{index:02d}" for index in range(20)],
                "dataset": ["synthetic"] * 20,
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        target = float(row["target"])
        return {
            "image": torch.full((3, 2, 2), target),
            "target": torch.tensor(target, dtype=torch.float32),
            "reference": row["reference"],
            "dataset": row["dataset"],
            "distortion": "blur",
            "level": index % 5,
        }

    def subset(self, rows):
        return TinyDataset(rows)


class TinyBackbone(torch.nn.Module):
    def forward(self, pixel_values):
        value = pixel_values[:, 0].mean(dim=(1, 2))
        pooled = torch.stack((value, value.square(), torch.sin(value)), dim=1)
        return SimpleNamespace(pooler_output=pooled)


class MLflowIntegrationTest(unittest.TestCase):
    def test_training_evaluation_and_artifact_are_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            tracking_uri = f"sqlite:///{Path(directory, 'mlflow.db')}"
            output = Path(directory, "head.pt")
            config_dir = Path(directory, "configs")
            input_config = Path(directory, "input.yaml")
            client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
            client.create_experiment(
                "integration-test",
                artifact_location=Path(directory, "artifacts").as_uri(),
            )
            input_config.write_text(yaml.safe_dump({
                "data": "synthetic.csv",
                "device": "cpu",
                "epochs": 3,
                "batch_size": 4,
                "workers": 0,
                "hidden_dim": 8,
                "out": str(output),
                "mlflow": True,
                "mlflow_tracking_uri": tracking_uri,
                "mlflow_experiment": "integration-test",
                "mlflow_run_name": "tiny-run",
                "config_dir": str(config_dir),
            }))
            argv = ["train.py", "--config", str(input_config), "--epochs", "2"]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(train, "IQADataset", TinyDataset), \
                 mock.patch.object(
                     train, "load_backbone",
                     return_value=(TinyBackbone(), 2, 3),
                 ):
                train.main()

            experiment = client.get_experiment_by_name("integration-test")
            self.assertIsNotNone(experiment)
            runs = client.search_runs([experiment.experiment_id])
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.info.status, "FINISHED")
            self.assertEqual(run.data.params["backbone"], "clip-base")
            self.assertEqual(run.data.params["config"], str(input_config))
            self.assertEqual(run.data.params["train_size"], "16")
            self.assertIn("train/loss", run.data.metrics)
            self.assertIn("validation/synthetic/srcc", run.data.metrics)
            self.assertIn("validation/synthetic/plcc", run.data.metrics)
            self.assertEqual(
                len(client.get_metric_history(run.info.run_id, "train/loss")), 8
            )
            self.assertEqual(
                [metric.step for metric in client.get_metric_history(
                    run.info.run_id, "train/loss"
                )],
                list(range(8)),
            )
            self.assertEqual(
                len(client.get_metric_history(run.info.run_id, "train/epoch_loss")), 2
            )
            epoch_history = client.get_metric_history(run.info.run_id, "train/epoch")
            self.assertEqual(len(epoch_history), 2)
            self.assertEqual([metric.value for metric in epoch_history], [1.0, 2.0])
            artifacts = client.list_artifacts(run.info.run_id, "checkpoints")
            self.assertEqual([item.path for item in artifacts], ["checkpoints/quality_head.pt"])
            config_artifacts = client.list_artifacts(run.info.run_id, "configs")
            self.assertEqual(
                [item.path for item in config_artifacts],
                [f"configs/{run.info.run_id}.yaml"],
            )
            config_path = config_dir / f"{run.info.run_id}.yaml"
            with config_path.open(encoding="utf-8") as stream:
                config = yaml.safe_load(stream)
            self.assertEqual(config["epochs"], 2)
            self.assertEqual(config["batch_size"], 4)
            self.assertEqual(config["train_size"], 16)
            self.assertEqual(config["device_used"], "cpu")
            self.assertEqual(config["mlflow_run_id"], run.info.run_id)
            self.assertIn("--config", config["command"])
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
