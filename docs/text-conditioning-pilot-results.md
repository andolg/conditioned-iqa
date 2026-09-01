# Text-conditioning pilot results

All figures below are final-epoch macro SRCC on the 2,000-image KADID
reference-split validation set. Each CLIP-Base result uses 8,125 training
images, five epochs, the same frozen vision snapshot, and seeds 0--2. Runs and
resolved configurations are in MLflow experiment `conditioned-iqa-text`.

| method | seed 0 | seed 1 | seed 2 | mean | sample SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| image-only baseline | 0.7409 | 0.7630 | 0.7757 | 0.7599 | 0.0176 |
| text concat | 0.7356 | 0.7671 | 0.7916 | 0.7648 | 0.0281 |
| text interaction `[v,t,v*t]` | 0.7804 | 0.7873 | 0.7866 | 0.7847 | 0.0038 |
| residual text correction | 0.7460 | 0.7786 | 0.8013 | 0.7753 | 0.0278 |

Interaction improves over the paired baseline on all three seeds (mean paired
gain +0.0249). Plain concatenation is inconsistent (+0.0049 mean paired
gain). Residual conditioning gives a valid image-only fallback, but its final
epoch gain is smaller (+0.0154); its zero-condition scores are meaningful
because training used 15% condition dropout.

## Semantic intervention

The interaction model was trained only with canonical group descriptions. It
was then evaluated with alternate held-out wording that was never passed to
the scoring head during training:

| seed | canonical | held-out paraphrase | generic | wrong | shuffled |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7797 | 0.7689 | 0.7401 | 0.7047 | 0.7157 |
| 1 | 0.7920 | 0.7863 | 0.7484 | 0.7223 | 0.7209 |
| 2 | 0.7865 | 0.7705 | 0.6910 | 0.7537 | 0.7203 |
| mean | 0.7861 | 0.7752 | 0.7265 | 0.7269 | 0.7190 |

Held-out wording retains most of the canonical score (mean drop 0.0109) and
is substantially better than generic, wrong, or shuffled conditions. This is
evidence for semantic use of the text, but not proof of broad language
understanding: all descriptions still name the same known distortion groups.

## Zero-shot CLIP-IQA reference

The direct CLIP similarity difference between `a good quality image` and `a bad
quality image` scores 0.5261 SRCC / 0.5409 PLCC. It is therefore a useful
zero-shot reference but not competitive with the trained image-only scorer or
the learned text-conditioned head on KADID.

## Interpretation

The strongest initial candidate is the interaction head. It should be
validated next on an unseen distortion group and with training-time paraphrase
sampling. Model selection should use paired multi-seed results and semantic
interventions, not the highest single epoch or single seed.

## CLIP-Large confirmation

One five-epoch CLIP-Large (ViT-L/14@336px) seed-0 pair confirms the scale
trend: the image-only baseline finishes at **0.8469**, while interaction
conditioning finishes at **0.8896** (paired gain +0.0427). Its intervention
scores are zero 0.8239, generic 0.8522, wrong 0.8291, and shuffled 0.8045,
again placing the correct condition first.
