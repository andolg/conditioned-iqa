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
    model_id: str, weights: str | None, device: torch.device, *, native: bool = True
) -> tuple[object, object, int]:
    """Load a frozen text encoder locally or through the project mirror helper."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPTextModel, SiglipTextModel, T5EncoderModel

    snapshot = Path(weights).expanduser() if weights else download_model_snapshot(
        model_id, allow_patterns=TEXT_PATTERNS
    )
    if native:
        model_class = SiglipTextModel if "siglip" in model_id else CLIPTextModel
    else:
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        model_class = T5EncoderModel if config.model_type == "t5" else AutoModel
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = model_class.from_pretrained(snapshot, local_files_only=True)
    model = model.eval().requires_grad_(False).to(device)
    text_dim = getattr(model.config, "hidden_size", None)
    if text_dim is None:
        text_dim = model.config.d_model
    return tokenizer, model, text_dim


@torch.no_grad()
def encode_prompts(tokenizer, model, prompts: list[str], device: torch.device) -> torch.Tensor:
    """Encode prompts once and return CPU float vectors for cheap batch lookup."""
    inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}
    outputs = model(**inputs)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        tokens = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return pooled.float().cpu()
