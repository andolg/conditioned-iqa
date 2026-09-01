"""Unit tests for the isolated text-conditioning heads."""

import torch

from text_conditioning.models import DatasetScaleHead, ResidualTextHead
from train import QualityMLP


def test_residual_head_zero_text_is_exact_unconditioned_path():
    """Zero must mean no correction, not merely a new learned prompt."""
    torch.manual_seed(0)
    head = ResidualTextHead(vision_dim=6, text_dim=4, fusion_dim=5, hidden_dim=7)
    head.eval()
    vision = torch.randn(3, 6)
    zero_text = torch.zeros(3, 4)

    with torch.no_grad():
        expected = head.base(vision).squeeze(-1)
        actual = head(vision, zero_text)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dataset_scale_head_calibrates_known_and_falls_back_for_unknown():
    torch.manual_seed(0)
    head = DatasetScaleHead(QualityMLP(6, 7), ["a", "b"])
    head.eval()
    vision = torch.randn(3, 6)
    with torch.no_grad():
        latent = head(vision)
        calibrated = head(vision, datasets=["a", "unknown", "b"])
    # The unseen-dataset path must expose the shared latent score, while known
    # datasets are passed through a monotonic bounded calibration.
    torch.testing.assert_close(calibrated[1], latent[1], rtol=0, atol=0)
    assert torch.all((calibrated[[0, 2]] >= 0) & (calibrated[[0, 2]] <= 1))
