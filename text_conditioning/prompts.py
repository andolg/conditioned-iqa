"""Canonical natural-language conditions for the shared distortion taxonomy."""

from __future__ import annotations

GROUP_PROMPTS = {
    "blur": "Assess loss of sharpness, defocus, motion blur, and missing fine detail.",
    "noise": "Assess random grain, sensor noise, speckle, and unwanted pixel variation.",
    "compression": "Assess blocking, ringing, mosquito noise, and loss caused by compression.",
    "colour": "Assess unnatural colour, chromatic errors, saturation shifts, and colour quantization.",
    "tone": "Assess exposure, contrast, brightness, and tone reproduction.",
    "spatial": "Assess geometric deformation, resampling, aliasing, and spatial discontinuities.",
    "generative": "Assess artificial textures, hallucinated detail, restoration artifacts, and structural inconsistencies.",
    "authentic": "Assess the overall perceptual quality of this real photograph and naturally occurring defects.",
}

# Never used for training in the initial canonical-prompt runs.  These express
# the same condition with materially different wording, making them a cheap
# semantic generalisation check rather than a second learned prompt ID.
HELD_OUT_GROUP_PROMPTS = {
    "blur": "Judge how much the picture has been softened or lost crisp detail through defocus or motion.",
    "noise": "Judge the severity of unwanted random pixel fluctuations, grain, or speckle.",
    "compression": "Judge visible coding artifacts such as blocks, ringing, and mosquito patterns.",
    "colour": "Judge colour fidelity, including casts, oversaturation, and chromatic distortions.",
    "tone": "Judge whether brightness, contrast, and exposure are reproduced naturally.",
    "spatial": "Judge warping, jagged resampling artifacts, and discontinuities in spatial structure.",
    "generative": "Judge implausible synthetic texture, invented detail, and inconsistent restored structure.",
    "authentic": "Rate this photograph's overall visual fidelity, including any real-world quality defects.",
}

GENERIC_PROMPT = "Assess the overall perceptual quality of this image."
GROUPS = tuple(GROUP_PROMPTS)


def prompt_for_group(group: str) -> str:
    """Return the canonical prompt, with a safe authentic fallback."""
    return GROUP_PROMPTS.get(group, GROUP_PROMPTS["authentic"])


def wrong_group(group: str) -> str:
    """A deterministic wrong family used for the intervention evaluation."""
    try:
        return GROUPS[(GROUPS.index(group) + 1) % len(GROUPS)]
    except ValueError:
        return "blur"
