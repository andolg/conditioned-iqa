"""Unit tests for the isolated text-conditioning heads."""

import torch

from text_conditioning.models import ResidualTextHead


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
