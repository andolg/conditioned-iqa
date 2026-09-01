"""Prompt taxonomy checks for the E4 paraphrase experiment."""

from text_conditioning.prompts import GROUPS, HELD_OUT_PARAPHRASES, TRAINING_GROUP_PROMPTS


def test_training_and_held_out_prompt_sets_are_disjoint_and_sufficiently_diverse():
    for group in GROUPS:
        training = set(TRAINING_GROUP_PROMPTS[group])
        held_out = set(HELD_OUT_PARAPHRASES[group])
        assert len(training) >= 5
        assert len(held_out) >= 3
        assert training.isdisjoint(held_out)
