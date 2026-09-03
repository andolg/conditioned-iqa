"""Small Hugging Face model-download helpers for networks in China."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


CHINA_HF_ENDPOINTS = (
    "https://alpha.hf-mirror.com",
    "https://hf-mirror.com",
    "https://huggingface.co",
)

# Enough for built-in Transformers model classes without also fetching Flax,
# TensorFlow, ONNX, tokenizer, or processor artifacts.
TRANSFORMERS_MODEL_PATTERNS = (
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "pytorch_model*.bin",
)


def hf_endpoints(endpoints: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return custom/env endpoints followed by the China-friendly defaults."""
    requested = list(endpoints or ())
    if configured := os.environ.get("HF_ENDPOINT"):
        requested.append(configured)
    requested.extend(CHINA_HF_ENDPOINTS)
    return tuple(dict.fromkeys(endpoint.rstrip("/") for endpoint in requested))


def download_model_snapshot(
    repo_id: str,
    *,
    endpoints: Iterable[str] | None = None,
    allow_patterns: Iterable[str] | None = TRANSFORMERS_MODEL_PATTERNS,
    **download_kwargs,
) -> Path:
    """Download one model snapshot, retrying compatible Hub endpoints."""
    from huggingface_hub import snapshot_download

    failures = []
    sources = hf_endpoints(endpoints)
    for number, endpoint in enumerate(sources, start=1):
        print(f"Hugging Face source {number}/{len(sources)}: {endpoint}")
        try:
            snapshot = snapshot_download(
                repo_id,
                endpoint=endpoint,
                allow_patterns=list(allow_patterns) if allow_patterns else None,
                **download_kwargs,
            )
            return Path(snapshot)
        except Exception as error:
            summary = str(error).splitlines()[0] if str(error) else type(error).__name__
            failures.append(f"{endpoint}: {summary}")
            if number < len(sources):
                print(f"  failed: {summary}\n  trying the next source...")

    raise RuntimeError(
        f"could not download {repo_id} from any Hugging Face source:\n  "
        + "\n  ".join(failures)
    )


def load_transformers_model_from_mirrors(
    model_class,
    repo_id: str,
    *,
    endpoints: Iterable[str] | None = None,
    download_kwargs: dict | None = None,
    **model_kwargs,
):
    """Download through mirrors, then load without making another HTTP call."""
    snapshot = download_model_snapshot(
        repo_id, endpoints=endpoints, **(download_kwargs or {})
    )
    model_kwargs["local_files_only"] = True
    return model_class.from_pretrained(snapshot, **model_kwargs)
