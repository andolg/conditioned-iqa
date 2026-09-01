"""Offline/mirror-aware frozen text encoder for native CLIP or SigLIP text."""

from __future__ import annotations

from pathlib import Path

import torch

from hf_mirror_utils import download_model_snapshot

TEXT_PATTERNS = (
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "sentencepiece.model",
    "spiece.model",
)


def load_frozen_text_encoder(
    model_id: str, weights: str | None, device: torch.device
) -> tuple[object, object, int]:
    """Load a native text tower locally or via the project mirror helper."""
    from transformers import AutoTokenizer, CLIPTextModel, SiglipTextModel

    snapshot = Path(weights).expanduser() if weights else download_model_snapshot(
        model_id, allow_patterns=TEXT_PATTERNS
    )
    model_class = SiglipTextModel if "siglip" in model_id else CLIPTextModel
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = model_class.from_pretrained(snapshot, local_files_only=True)
    model = model.eval().requires_grad_(False).to(device)
    return tokenizer, model, model.config.hidden_size


@torch.no_grad()
def encode_prompts(tokenizer, model, prompts: list[str], device: torch.device) -> torch.Tensor:
    """Encode prompts once and return CPU float vectors for cheap batch lookup."""
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    outputs = model(**inputs)
    pooled = outputs.pooler_output
    return pooled.float().cpu()
