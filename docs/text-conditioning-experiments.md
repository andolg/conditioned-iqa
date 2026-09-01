   # Text conditioning: implementation and experiment design

## Objective

Determine whether telling an NR-IQA model which distortion family to assess
improves agreement with subjective quality scores, and whether the model uses
the *meaning* of the text rather than treating eight prompt vectors as arbitrary
IDs.

The current reference model is deliberately small:

```text
image -> frozen CLIP/SigLIP vision encoder -> pooled feature -> QualityMLP -> score
```

The KADID-10k CLIP-L baseline in `baseline.txt` reports 0.8469 SRCC and 0.8190
PLCC after five epochs. Every conditioned result must be compared with a newly
run baseline using the same backbone checkpoint, split, seed, train subset,
batch size, sampler, loss, optimizer, and number of epochs. Historical numbers
are a smoke reference, not a valid paired comparison.

## Collaboration boundary and proposed files

Do not implement text conditioning inside `train.py`. It is the shared baseline
and other colleagues are developing unrelated features in parallel. Keep this
work isolated so it can be reviewed, benchmarked, and merged independently:

```text
train.py                              unchanged baseline
train_text_conditioned.py             separate entry point
text_conditioning/
    data.py                            group-aware dataset wrapper
    prompts.py                         taxonomy and prompt sampling
    text_encoder.py                    mirror-aware frozen text encoder
    models.py                          pooled text-fusion heads
    evaluate.py                        condition interventions and IQA metrics
configs/text_conditioning/
    00_baseline_parity.yaml
    01_text_concat.yaml
    02_text_interaction.yaml
    03_text_residual.yaml
    04_text_encoder_comparison.yaml
    05_prompt_paraphrases.yaml
tests/text_conditioning/
```

It is acceptable to import stable baseline definitions such as `BACKBONES` or
`QualityMLP`, but the conditioned runner must not require edits to `train.py`.
If shared logic eventually needs extraction, do that in a small dedicated PR
after the first experiments, not as part of a conditioning implementation.

`IQADataset.__getitem__` currently omits the CSV's `group` column. Prefer a
`ConditionedIQADataset` subclass/wrapper in `text_conditioning/data.py` and
override `subset()` so reference splitting preserves the conditioned type.
Avoid changing `dataset.py` merely to start these experiments, because it is
another likely integration hotspot.

## Condition taxonomy and prompts

Use the existing broad `group` values, never the dataset-specific distortion
code. Fine-grained codes leak corpus vocabulary and do not transfer between
KADID, TID2013, and CSIQ.

| group | canonical description |
| --- | --- |
| `blur` | Assess loss of sharpness, defocus, motion blur, and missing fine detail. |
| `noise` | Assess random grain, sensor noise, speckle, and unwanted pixel variation. |
| `compression` | Assess blocking, ringing, mosquito noise, and loss caused by compression. |
| `colour` | Assess unnatural colour, chromatic errors, saturation shifts, and colour quantization. |
| `tone` | Assess exposure, contrast, brightness, and local or global tone reproduction. |
| `spatial` | Assess geometric deformation, resampling, aliasing, and spatial discontinuities. |
| `generative` | Assess artificial textures, hallucinated detail, restoration artifacts, and structural inconsistencies. |
| `authentic` | Assess the overall perceptual quality of this real photograph and any naturally occurring defects. |

Store prompts as data in `text_conditioning/prompts.py` or YAML, not inline in
model code. Each group needs:

- one canonical description for initial experiments;
- 4-8 training paraphrases sampled per example or epoch;
- at least three held-out paraphrases never used during training;
- a deliberately wrong prompt and a generic `assess overall image quality`
  control.

The first experiments may use only canonical prompts. Do not make a semantic
generalization claim until held-out paraphrases and unseen-group evaluation are
run.

## Hugging Face access on this server

Direct traffic to Hugging Face is unreliable/blocked. A remote model ID must
never be passed directly to `from_pretrained`. Follow the existing backbone
strategy:

1. Accept `text_weights` in YAML. When provided, expand `~` and load only from
   that local snapshot with `local_files_only=True`.
2. Otherwise download through `hf_mirror_utils.download_model_snapshot`, whose
   order is `HF_ENDPOINT`, `alpha.hf-mirror.com`, `hf-mirror.com`, then the
   official Hub, all sharing the Hub cache.
3. Extend the text snapshot allow-list to include tokenizer assets in addition
   to model weights: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
   `merges.txt`, `special_tokens_map.json`, `sentencepiece.model`, and
   `spiece.model`. Do this in a text-specific helper or a backward-compatible
   addition to `hf_mirror_utils.py`.
4. Load the tokenizer and model from the returned local path, never from the
   remote ID. Add an offline test that patches Hub access to fail and proves a
   supplied local snapshot still loads.
5. Freeze the text encoder, put it in evaluation mode, and precompute embeddings
   for the small prompt vocabulary. Save the prompt-to-vector cache as an
   MLflow artifact with the encoder ID/revision and prompt-set hash.

Start with the matching CLIP or SigLIP text tower. It is already trained with
visual language and its full checkpoint may be present in the vision snapshot,
although tokenizer files may need one mirror download. Do not begin with Qwen,
INSTRUCTOR, or another large sentence encoder: they add download, memory, and
representation confounds before basic conditioning is proven.

For learned fusion, image and text features do not have to share a CLIP
projection space; learned projections can map both into a common fusion width.
Only cosine-similarity experiments require the official visual/text projection
layers.

## Common training and logging protocol

All runs use the existing YAML/MLflow workflow. Put reusable inputs under
`configs/text_conditioning/`; resolved manifests remain under `runs/configs/`
and are registered as artifacts.

Log at minimum:

- `conditioning/method`, text encoder ID/revision, prompt-set hash, fusion
  width, trainable parameter count, and whether embeddings are cached;
- step-level `train/loss` and epoch-level loss/SRCC/PLCC;
- per-dataset, macro, worst-dataset, and PIPAL within-reference metrics;
- seed, split manifest, predictions, checkpoint, reusable input YAML, resolved
  YAML, and prompt definitions;
- intervention metrics with explicit suffixes: `correct`, `zero`, `shuffled`,
  `wrong`, `generic`, and later `paraphrase_worst`.

Use reference splits for synthetic datasets. Use the exact same split manifest
across baseline and conditioned variants. Initially run seeds 0, 1, and 2; use
five seeds for a finalist. Compare paired per-seed differences and report mean,
standard deviation, and a bootstrap confidence interval over validation
references. Do not pool correlations across datasets.

Parameter count is a confound. For each conditioned head, record both:

- the ordinary baseline; and
- a widened unconditioned MLP with approximately the same trainable parameter
  count, or a constant-vector version of the same fusion head.

## Experiment order

This workstream follows the brief's **Text conditioning** direction: represent
the condition as a sentence through a frozen text embedder, test whether the
model uses meaning rather than prompt identity, compare text embedders, and test
whether descriptions transfer to an unseen distortion group.

The brief lists **Learnable queries** as a separate research direction. It can
eventually consume text, but implementing a new patch-query scorer here would
confound text conditioning with a simultaneous image-head change and may overlap
another colleague's work. Keep the frozen pooled vision representation fixed in
the core experiments below. Treat learned queries, extra tokens, FiLM, and
generated heads as optional future integration work, not prerequisites.

Learned-label conditioning is also out of scope because another team owns it.
Use colleagues' final metrics as external comparisons when available, without
copying their implementations into this branch.

### E0 - Baseline parity in the separate runner

Implement an unconditioned mode in `train_text_conditioned.py` that reproduces
`train.py` without text components.

**Reason:** otherwise a gain may come from a different split, loader, seed,
evaluation path, or optimizer rather than conditioning.

**Expected result:** per-seed SRCC/PLCC should match `train.py` within normal
floating-point/run variance. Treat an absolute SRCC difference above 0.005 on
the identical deterministic setup as a bug to investigate.

**Gate:** no conditioned experiment is valid until parity passes.

### E1 - Frozen native text embedding plus concatenation

```text
t = frozen_text(canonical_description)       # cached
v' = project_vision(v)
t' = project_text(t)
score = shared_MLP([v', t', v' * t'])
```

First test plain `[v', t']`; add the elementwise interaction only as a recorded
ablation. Compare against both the ordinary baseline and a widened
unconditioned head with approximately the same trainable parameter count.

**Reason:** this is the smallest direct text-conditioning experiment and changes
only one thing relative to the baseline: a sentence embedding is supplied to
the quality head. It also proves prompts, cached embeddings, intervention
evaluation, and MLflow artifacts work end to end.

**Expected result:** the main early gain should occur on KADID, where multiple
groups occur within the same dataset and content is controlled. Little or no
gain is expected on SPAQ/GFIQA because `authentic` is constant. A large gain
over the matched-capacity baseline should trigger a capacity/data-leak audit
before being celebrated.

**Gate:** correct text must beat zero/generic text; shuffled or wrong text must
erase the gain. Otherwise the text branch is decorative.

### E2 - Pooled text-fusion ablations

Keep the frozen pooled image vector identical and compare small ways to combine
it with the cached text vector:

1. plain concatenation `[v', t']`;
2. concatenation with elementwise interaction `[v', t', v' * t']`; and
3. a residual correction
   `score = baseline_score(v) + correction(v,t)`.

For the residual variant, use condition dropout during training and define a
literal all-zero text vector to return exactly `baseline_score(v)`. This makes
the zero-condition evaluation a valid unconditioned fallback measurement,
rather than an out-of-distribution vector passed through a learned text branch.

Each variant needs a constant-vector or widened unconditioned control with
approximately matched parameter count.

**Reason:** this isolates whether the result depends on how text is presented
to the existing scoring head without changing the vision representation. It is
closer to the text-conditioning question than replacing the scorer with a patch
query architecture.

**Expected result:** interaction or a gated correction may use the condition
more strongly than plain concatenation, but also has more opportunity to
memorize the eight prompt vectors. Select by correct-versus-wrong intervention
behavior and paired multi-seed SRCC, not by the best single run.

### E3 - Which frozen text embedder preserves the condition

Hold the best E1/E2 fusion, prompts, vision features, split, and training setup
fixed. Compare:

1. the native CLIP or SigLIP text tower matching the vision backbone;
2. INSTRUCTOR as the first external semantic encoder, if it can be obtained
   reliably through mirrors; and
3. Qwen3-Embedding only later, because its size and download cost add a major
   resource confound.

**Reason:** “which embedder preserves the meaning” is one of the explicit open
questions in the Text conditioning section of the brief.

**Expected result:** the native tower may perform best in-domain because it is
vision-language aligned. A general instruction embedder may not improve
canonical prompts, but could be better on held-out paraphrases. Do not call an
embedder better unless it improves semantic controls, not just fixed-prompt
SRCC.

### E4 - Prompt diversity and semantic consistency

Train the best pooled text-conditioned model with random training paraphrases.
Optionally add:

```text
L_total = L_MOS + lambda * abs(score(image, paraphrase_a)
                             - score(image, paraphrase_b))
```

Start with `lambda` 0, 0.01, and 0.05. Evaluate the worst held-out paraphrase,
not only the average.

**Reason:** fixed canonical prompts allow the model to memorize eight vectors.
Paraphrase sampling is the cheapest way to demand local semantic invariance.

**Expected result:** average canonical accuracy may stay flat or fall slightly,
while worst-paraphrase accuracy and variance should improve. If all conditions
become interchangeable, the consistency weight is too large; verify with wrong
prompts.

### E5 - Leave-one-group-out semantic transfer

For blur, noise, and compression in turn, exclude the entire group from
training and score it from text at evaluation. Compare with:

- E0 unconditioned;
- generic text;
- nearest seen-group text;
- an oracle model trained with the held-out group.

If the colleagues' label-conditioned checkpoint is available, include its fresh
untrained held-out label as a reported external comparator; do not implement it
in this branch.

**Reason:** this is the decisive advantage text could have over labels.

**Expected result:** a semantic text embedder and pooled fusion may recover part
of the oracle gap, but a large gain is not expected from only eight groups. A
learned label has no meaningful zero-shot representation, which is why
colleagues' label result is useful context if available. Blur/noise/compression
are the best first cases because their semantics are concrete and they appear
across KADID, TID2013, and CSIQ.

### E6 - Optional later integration with learned-query work

If a colleague produces a stable learned-query patch scorer, integrate the best
text embedding into it in a separate follow-up comparison: query shift, extra
text token, post-pooling FiLM, or generated head weights. Compare against that
scorer's own unconditioned checkpoint.

**Reason:** this is relevant to the broader project brief, but it is not needed
to establish text conditioning and should not block E1-E5.

## Dataset schedule

1. **KADID-10k development:** fastest causal laboratory; multiple groups,
   reference-controlled content, and severity levels. Use small `limit` runs
   only for plumbing, then full training for comparisons.
2. **TID2013 and CSIQ transfer:** never select hyperparameters on these held-out
   sets. They test whether the broad group taxonomy transfers across dataset
   vocabularies.
3. **Combined designated training sets:** use `by_dataset` sampling. Read PIPAL
   by within-reference SRCC. Treat SPAQ/GFIQA's constant `authentic` prompt as a
   dataset-control condition, not evidence of semantic use.
4. **AIGCIQA prompt caution:** its generation prompt describes intended image
   content, while this project's condition describes the defect to assess.
   Keep them as separate fields and ablate them separately.

For synthetic data, also report severity monotonicity: within a reference and
distortion type, predicted quality should decrease as severity increases.

## Required intervention matrix

Every serious conditioned checkpoint is evaluated without retraining under:

| evaluation condition | interpretation |
| --- | --- |
| correct | headline conditioned score |
| zero/unknown | should return near the unconditioned model |
| shuffled within dataset | removes image-condition pairing but preserves frequencies |
| deliberately wrong family | should be worse than unconditioned if the instruction drives scoring |
| generic quality prompt | tests whether specificity matters |
| held-out paraphrases | tests local semantic invariance |
| keyword removed | tests description meaning beyond the family name |

Also train one model with conditions permanently permuted. If it matches the
correctly paired model, any apparent gain is extra capacity or dataset identity,
not useful conditioning.

## Decision rules

- Advance plain text concatenation to interaction or gated residual fusion only
  if correct conditioning passes shuffled/wrong interventions.
- Compare every fusion against the same pooled E0 and a matched-capacity
  unconditioned control; keep the image representation fixed.
- Compare text embedders on canonical prompts, held-out paraphrases, and unseen
  groups, not on one fixed-prompt SRCC number alone.
- Claim semantic text use only after held-out paraphrases. Compare against the
  colleagues' label-conditioning results only after protocols and splits are
  confirmed equivalent; claim a text-over-label advantage only after
  leave-one-group-out transfer.
- Reject any model whose gain disappears against matched capacity, whose zero
  condition does not fall back near baseline, or whose worst-dataset score
  collapses despite a macro gain.

## Relevant starting references

- [LIQE: vision-language correspondence for IQA](https://arxiv.org/abs/2303.14968)
- [FiLM conditioning](https://arxiv.org/abs/1709.07871)
- [PromptIQA](https://arxiv.org/abs/2403.04993)
- [HyperIQA](https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.html)
- [INSTRUCTOR text embeddings](https://aclanthology.org/2023.findings-acl.71/)
- [FollowIR instruction-use evaluation](https://aclanthology.org/2025.naacl-long.597/)
