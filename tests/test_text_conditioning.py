"""Unit tests for the isolated text-conditioning heads."""

import pandas as pd
import torch

from text_conditioning.data import ConditionedIQADataset, dataset_score_ranges, target_from_row
from text_conditioning.models import DatasetScaleHead, MDTVSFAHead, PatchWeightedHead, ResidualTextHead, TextPatchWeightedHead
from train import QualityMLP
from train_text_conditioned import (
    error_induced_loss,
    linearity_induced_loss,
    monotonicity_induced_loss,
    split_from_manifest,
)


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


def test_faithful_mdtvsfa_head_exposes_three_distinct_stages():
    torch.manual_seed(0)
    head = MDTVSFAHead(
        QualityMLP(6, 7), ["a", "b"], {"a": (1.0, 5.0), "b": (10.0, 20.0)}
    )
    head.eval()
    vision = torch.randn(3, 6)
    with torch.no_grad():
        relative, perceptual, aligned = head.stages(vision, datasets=["a", "unknown", "b"])
        unknown = head(vision, datasets=["unknown", "unknown", "unknown"])
    assert torch.all((relative >= 0) & (relative <= 1))
    assert not torch.equal(relative, perceptual)
    torch.testing.assert_close(aligned[1], perceptual[1], rtol=0, atol=0)
    torch.testing.assert_close(unknown, perceptual, rtol=0, atol=0)
    assert sum(parameter.numel() for parameter in head.nonlinear.parameters()) == 4
    assert not head.nonlinear[2].weight.requires_grad
    assert not head.nonlinear[2].bias.requires_grad


def test_faithful_mdtvsfa_losses_follow_order_linearity_and_scale():
    target = torch.tensor([1.0, 2.0, 3.0])
    increasing = torch.tensor([0.1, 0.2, 0.3])
    decreasing = torch.tensor([0.3, 0.2, 0.1])
    assert monotonicity_induced_loss(increasing, target).item() == 0.0
    assert monotonicity_induced_loss(decreasing, target).item() > 0.0
    assert linearity_induced_loss(increasing, target).item() < 1e-6
    assert linearity_induced_loss(decreasing, target).item() > 0.99
    assert error_induced_loss(target, target, (1.0, 5.0)).item() == 0.0


def test_oriented_raw_scores_flip_dmos_and_compute_training_ranges():
    rows = pd.DataFrame({
        "dataset": ["csiq", "csiq", "spaq", "spaq"],
        "original_subjective_score": [1.0, 3.0, 20.0, 80.0],
    })
    assert target_from_row(rows.iloc[0], "oriented_subjective_score") == -1.0
    assert target_from_row(rows.iloc[2], "oriented_subjective_score") == 20.0
    assert dataset_score_ranges(rows, "oriented_subjective_score") == {
        "csiq": (-3.0, -1.0), "spaq": (20.0, 80.0)
    }


def test_split_manifest_keeps_official_test_rows_out_and_preserves_references(tmp_path):
    rows = pd.DataFrame({
        "path": ["a.png", "b.png", "c.png"],
        "dataset": ["kadid10k", "kadid10k", "uhdiqa"],
        "reference": ["reference-a", "reference-a", "reference-c"],
        "scaled_subjective_score": [0.2, 0.8, 0.5],
        "group": ["blur", "blur", "authentic"],
    })
    manifest = pd.DataFrame({
        "path": ["a.png", "b.png", "c.png"],
        "partition": ["train", "train", "validation"],
    })
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)

    train, validation = split_from_manifest(ConditionedIQADataset(rows), path)

    assert train.rows["path"].tolist() == ["a.png", "b.png"]
    assert validation.rows["path"].tolist() == ["c.png"]


def test_split_manifest_rejects_test_partition(tmp_path):
    rows = pd.DataFrame({
        "path": ["a.png"],
        "dataset": ["uhdiqa"],
        "reference": ["a"],
        "scaled_subjective_score": [0.5],
        "group": ["authentic"],
    })
    path = tmp_path / "manifest.csv"
    pd.DataFrame({"path": ["a.png"], "partition": ["test"]}).to_csv(path, index=False)

    try:
        split_from_manifest(ConditionedIQADataset(rows), path)
    except ValueError as error:
        assert "forbidden partitions" in str(error)
    else:
        raise AssertionError("manifest with a test row must be rejected")


def test_patch_weighted_head_is_permutation_invariant_and_normalized():
    torch.manual_seed(0)
    head = PatchWeightedHead(6, hidden_dim=4, dropout=0.0).eval()
    patches = torch.randn(2, 9, 6)
    with torch.no_grad():
        score = head(patches)
        permutation = torch.randperm(patches.shape[1])
        permuted = head(patches[:, permutation])
    torch.testing.assert_close(score, permuted, rtol=0, atol=1e-6)


def test_text_patch_weighted_head_responds_to_condition_and_preserves_shape():
    torch.manual_seed(0)
    head = TextPatchWeightedHead(6, 5, fusion_dim=4, hidden_dim=3, dropout=0.0).eval()
    patches, text = torch.randn(2, 9, 6), torch.randn(2, 5)
    with torch.no_grad():
        score = head(patches, text)
        changed = head(patches, torch.zeros_like(text))
    assert score.shape == (2,)
    assert not torch.equal(score, changed)
