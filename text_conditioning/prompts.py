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

# Canonical wording remains the first item, preserving E1/E2 evaluation.  E4
# samples only these training paraphrases; held-out wording is never sampled.
TRAINING_GROUP_PROMPTS = {
    "blur": (GROUP_PROMPTS["blur"], "Evaluate blur from defocus or motion and the resulting loss of crisp fine detail.", "Assess whether the image has become soft, smeared, or lacking in sharp detail.", "Judge degradation caused by motion blur, defocus, and reduced edge clarity.", "Rate the severity of lost sharpness and missing high-frequency image detail."),
    "noise": (GROUP_PROMPTS["noise"], "Evaluate visible grain, speckle, and random sensor-like pixel noise.", "Assess how strongly random fluctuations and unwanted grain affect the picture.", "Judge noise artifacts such as speckle and irregular pixel-level variation.", "Rate the severity of distracting random grain and sensor noise."),
    "compression": (GROUP_PROMPTS["compression"], "Evaluate coding damage including blocks, ringing, and mosquito artifacts.", "Assess visible lossy-compression artifacts and the detail they remove.", "Judge block boundaries, ringing, and compression-induced texture damage.", "Rate degradation from image compression, including blocking and lost detail."),
    "colour": (GROUP_PROMPTS["colour"], "Evaluate colour casts, incorrect saturation, and chromatic rendering errors.", "Assess whether colours look unnatural, shifted, quantized, or poorly reproduced.", "Judge chromatic fidelity, including hue errors and saturation distortions.", "Rate visible colour artifacts and departures from natural colour appearance."),
    "tone": (GROUP_PROMPTS["tone"], "Evaluate exposure, brightness, contrast, and faithful tonal reproduction.", "Assess tonal problems such as poor exposure or unnatural contrast.", "Judge brightness and contrast rendering across the image.", "Rate degradation in exposure and global or local tone reproduction."),
    "spatial": (GROUP_PROMPTS["spatial"], "Evaluate warping, aliasing, resampling defects, and broken spatial structure.", "Assess geometric distortion, jagged resampling, and spatial discontinuities.", "Judge deformation and spatial artifacts such as aliasing or discontinuities.", "Rate defects in image geometry and spatial sampling."),
    "generative": (GROUP_PROMPTS["generative"], "Evaluate hallucinated detail, artificial texture, and inconsistent generated structure.", "Assess synthetic-looking textures and implausible restored or invented detail.", "Judge artifacts caused by image generation or restoration, including structural errors.", "Rate the severity of artificial textures and generative-image inconsistencies."),
    "authentic": (GROUP_PROMPTS["authentic"], "Evaluate the overall perceptual quality of this real-world photograph and its natural defects.", "Assess how naturally occurring capture or processing defects affect this photograph.", "Judge the general visual fidelity of this authentic image.", "Rate overall quality and any real photographic imperfections."),
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

HELD_OUT_PARAPHRASES = {
    "blur": (HELD_OUT_GROUP_PROMPTS["blur"], "How severely do soft focus and motion smear reduce visible detail in this image?", "Rate the impact of lost edge definition and fine-detail clarity."),
    "noise": (HELD_OUT_GROUP_PROMPTS["noise"], "How objectionable are the random grainy fluctuations visible in this image?", "Rate the impact of visible pixel grain and speckling."),
    "compression": (HELD_OUT_GROUP_PROMPTS["compression"], "How much do encoded blocks and ringing damage the visual quality of this image?", "Rate the impact of lossy coding artifacts on the picture."),
    "colour": (HELD_OUT_GROUP_PROMPTS["colour"], "How much do inaccurate hues and saturation make the colours look wrong?", "Rate the impact of chromatic errors on visual fidelity."),
    "tone": (HELD_OUT_GROUP_PROMPTS["tone"], "How much do exposure and contrast problems harm the tonal appearance?", "Rate the impact of tonal and illumination errors on the picture."),
    "spatial": (HELD_OUT_GROUP_PROMPTS["spatial"], "How much do warping and sampling artifacts disrupt the image structure?", "Rate the impact of geometric and resampling defects on the picture."),
    "generative": (HELD_OUT_GROUP_PROMPTS["generative"], "How much do invented texture and implausible structure reduce image quality?", "Rate the impact of artificial generated-image artifacts on the picture."),
    "authentic": (HELD_OUT_GROUP_PROMPTS["authentic"], "How good does this real photograph look overall, considering its natural flaws?", "Rate the visual fidelity of this naturally captured photograph."),
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
