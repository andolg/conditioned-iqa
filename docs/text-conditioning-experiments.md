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

## Performance-first follow-up experiments

The original E0--E5 sequence establishes that a text branch can be trained and
whether it reacts to prompt interventions. It does not, by itself, solve the
larger problem exposed by the held-out evaluations: a model that is strong on
KADID can still be weak on authentic and high-resolution test sets. The next
experiments therefore optimize the training objective and dataset mixture
first. All of them use CLIP-B/16, the same reference split, stretch
preprocessing, five epochs, best-validation-epoch restoration, and the full
held-out evaluation suite unless a row explicitly says otherwise.

### P1 - Native CLIP interaction on the clean mixture

Train the interaction head on KADID-10k + SPAQ + AIGCIQA2023, excluding PIPAL.
Run seeds 0, 1, and 2, and pair these with clean-mixture image-only baselines
for seeds 1 and 2 (seed 0 already exists). Evaluate TID2013, CSIQ, CID2013,
KonIQ-10k, CLIVE, AGIQA-3K, GFIQA-20K, PIPAL, and the official UHD-IQA test
split. Keep the current image-only clean mixture as the primary comparator.

**Question:** does native CLIP text interaction improve the strongest current
cross-dataset model, rather than only improving KADID validation?

**Gate:** advance text conditioning only if the paired mean external SRCC
improves without a collapse in the worst dataset or in zero/wrong-prompt
interventions.

### M1 - MDTVSFA-style dataset-scale alignment

The current joint regression uses per-dataset min-max scores as if they had a
shared perceptual origin. MDTVSFA identifies this as a source of cross-dataset
conflict and instead learns a shared relative-quality representation followed
by dataset-specific nonlinear scale alignment and dataset-aware losses
([paper](https://arxiv.org/abs/2011.04263), [reference implementation](https://github.com/lidq92/MDTVSFA)).

Adapt the idea to image IQA as follows:

1. predict one shared latent quality value `z` from the frozen CLIP feature
   (and, in the conditioned variant, the text feature);
2. learn one monotonic calibration function per training dataset,
   `y_hat_d = sigmoid(alpha_d + softplus(beta_d) * z)`, rather than applying
   one global min-max scale;
3. optimize calibrated SmoothL1 regression within each dataset;
4. add a within-dataset pairwise ranking loss, including within-reference
   ranking for synthetic distortions; and
5. compare fixed equal dataset weighting with softmax weighting of the current
   per-dataset losses, as used by MDTVSFA.

The first M1 comparison excludes PIPAL. PIPAL is added only in M3 because its
Elo-like labels are reliable for within-reference ordering but are not a
shared absolute MOS scale. On unseen test datasets, report the shared latent
score without an unavailable test-specific calibrator; optionally report a
second number after fitting a calibrator on a clearly separated support split.

**Question:** does learned dataset-scale alignment improve weak datasets without
giving up the KADID/AGIQA gains?

**Gate:** retain M1 only if external macro SRCC and the worst-dataset SRCC both
improve over the clean-mixture baseline, or if it gives a clear reduction in
between-dataset calibration error without hurting rank correlation.

### M2 - Dataset-mixture and loss-weighting ablation

Using the same shared head and seed protocol, compare:

- KADID only;
- KADID + authentic photographs (SPAQ);
- KADID + generative images (AIGCIQA2023);
- the full clean mixture;
- proportional sampling;
- equal dataset sampling (`by_dataset`); and
- M1 adaptive softmax dataset-loss weighting.

Do not select the mixture using test labels. Select using the validation macro
SRCC, then report every held-out dataset and the mean/worst external metrics.

**Question:** is the current `by_dataset` sampler the problem, or is the
global target scale the problem?

### D1 - GFIQA-20k expansion after KonIQ overlap screening

Add GFIQA-20k as a fourth training source, but screen it before making the
train/validation split. GFIQA filenames expose a Flickr source ID followed by
a crop/variant suffix, while KonIQ filenames use the source ID directly. In
the downloaded data, 213 source IDs in the labeled KonIQ table correspond to
235 GFIQA rows; these rows are known same-photo crop/format near-duplicates and
must not enter training. A second visual near-duplicate pass should be run to
catch matches whose filenames do not preserve the source ID. Keep the screening
manifest and its thresholds as an artifact of the run.

Build a new table from KADID-10k + SPAQ + the screened GFIQA-20k rows +
AIGCIQA2023. Use the same CLIP-B/16 representation, reference split, native
interaction head and matched image-only control as the current finalist, with
equal per-dataset sampling, best-validation-epoch restoration, and seeds 0--2.
Do not use KonIQ labels during screening, splitting, model selection, or
calibration.

Evaluate TID2013, CSIQ, CID2013, KonIQ-10k, CLIVE, AGIQA-3K, PIPAL, and
UHD-IQA as untouched external datasets. Report GFIQA validation metrics
separately because GFIQA is now in training, and exclude it from the
zero-shot external macro. Compare the new models with the existing
three-dataset clean-mixture runs only on the common external datasets.

**Question:** does adding a leakage-screened authentic face source improve
authentic/high-resolution transfer, or does it mainly introduce a face-domain
bias without helping the held-out datasets?

**Gate:** keep the expansion only if the paired external macro and the worst
held-out dataset improve without a material regression on KADID/SPAQ
validation or a collapse in the prompt-intervention checks.

**Execution record (2026-09-02):** The deterministic source-ID screen retained
19,763 of 19,998 GFIQA rows and removed 235 rows sharing 213 labeled KonIQ
source IDs. The shared training table therefore contains 43,413 rows, with no
duplicate paths or missing files. The interaction head selected epochs 1, 4,
and 3 for seeds 0--2 (macro validation SRCC 0.8571, 0.8517, and 0.8547); the
matched image-only control selected epochs 4, 3, and 3 (0.8377, 0.8394, and
0.8398). Across seeds, the interaction model reached 0.8545 validation SRCC
and 0.6166 external SRCC, versus 0.8390 and 0.5971 for the matched control.
The external aggregate covers TID2013, CSIQ, CID2013, KonIQ-10k, CLIVE,
AGIQA-3K, PIPAL, and UHD-IQA; GFIQA is reported only as a training/validation
dataset. Both rows use the matched 86.39M total model-parameter footprint.

### M3 - PIPAL within-reference ranking extension

Add PIPAL to the best M1/M2 configuration, but never train it as ordinary
absolute regression. For pairs of images sharing a pristine reference, enforce
the observed quality ordering with a logistic or margin ranking loss. Keep the
absolute calibrated loss for KADID, SPAQ, and AIGCIQA2023. Report PIPAL
within-reference SRCC separately and report the effect on every other external
dataset.

**Question:** can PIPAL contribute useful restoration-artifact ordering without
miscalibrating the other datasets?

### H1 - Quality-head capacity and conditioning architecture

On the winning data objective, compare the current pooled MLP with a matched-
parameter widened image-only MLP, interaction fusion, and a gated residual
correction. Keep the CLIP-B representation fixed. Use the correct/zero/wrong/
shuffled intervention matrix for conditioned heads.

**Question:** are gains due to text semantics or simply additional head
capacity?

The seed-0 matched control is now complete on the exact row-14 training
protocol: KADID-10k, SPAQ, and AIGCIQA2023; reference split; equal
per-dataset M1 loss; and three calibrators. Its 529,763-parameter image-only
head selected validation macro SRCC/PLCC `0.8278/0.8295`. The 529,415-parameter
row-14 interaction head obtained `0.8407/0.8558` on the same split, while the
ordinary 198,663-parameter image-only baseline obtained `0.8269/0.8326`.
On the nine-dataset external suite, the matched-capacity control averaged
`0.5747/0.5982` SRCC/PLCC, below row 14's `0.6238/0.6561`; it was lower on
seven datasets and higher only on AGIQA-3K and PIPAL. Thus, in this one seed,
capacity alone does not explain the interaction gain. Seeds 1--2 are still
required before a final claim.

### E7 - External text encoders after objective stabilization

Only after P1--M3, compare native CLIP text, INSTRUCTOR, and (resources
permitting) Qwen embeddings on the same winning objective, prompts, seeds, and
external test suite. Select by paired external performance plus semantic
intervention behavior, not by KADID validation alone.

### E8 - Semantic transfer and robustness finalist

For the finalist, rerun held-out paraphrases, wrong and shuffled prompts, and
leave-one-group-out blur/noise/compression transfer. Use five seeds and report
paired confidence intervals. This is the final semantic claim, not a tuning
criterion for P1--M3.

## Execution record

The performance-first runs were completed on 2026-09-01. Every external row
below is a zero-retraining evaluation on all nine test datasets; UHD-IQA uses
its official 900-image test split. The values are macro means of the nine
per-dataset correlations, not pooled correlations.

| run | MLflow source run | external SRCC | external PLCC |
| --- | --- | ---: | ---: |
| P1 clean interaction, seed 0 | `942bb6f9` | 0.6258 | 0.6645 |
| P1 clean interaction, seed 1 | `68898ac1` | 0.6115 | 0.6561 |
| P1 clean interaction, seed 2 | `cc3df954` | 0.6200 | 0.6503 |
| P1 clean image-only, seed 1 | `b29f01c3` | 0.5780 | 0.6095 |
| P1 clean image-only, seed 2 | `88c358b3` | 0.5980 | 0.6209 |
| M1 calibrated image-only, equal loss | `18d3fb7d` | 0.5965 | 0.6194 |
| M1 calibrated interaction | `7e322994` | 0.6238 | 0.6561 |
| M2 calibrated image-only, softmax loss | `b8b3177d` | 0.5973 | 0.6174 |
| M2 global regression, proportional sampling | `fe077cf6` | 0.5867 | 0.6064 |
| M2 KADID + SPAQ | `d1d1bf3c` | 0.5741 | 0.5931 |
| M2 KADID + AIGCIQA2023 | `580585be` | 0.5434 | 0.5562 |
| M3 PIPAL within-reference ranking | `1d0ab6d5` | 0.5789 | 0.5999 |
| H1 residual interaction head | `334e189f` | 0.5882 | 0.6139 |
| E7 INSTRUCTOR interaction | `20c1b624` | 0.6112 | 0.6464 |

P1 interaction averaged 0.6191 SRCC (standard deviation 0.0072) versus
0.5880 (0.0142) for the two newly paired image-only seeds. The interaction
gain is consistent, although it is not evidence that the condition is useful
on every individual dataset.

The E8 leave-one-group-out probes were run with the native interaction head.
On blur, SRCC was 0.7471/0.7842 on TID2013/CSIQ versus 0.7988/0.7983 for the
clean image-only comparator. On noise it was 0.5743/0.7025 versus
0.6850/0.7493. On compression it was 0.7993/0.8693 versus 0.7977/0.8370.
These are transfer diagnostics, not model-selection numbers; they show that
semantic transfer is condition- and dataset-dependent.

**Current conclusion:** P1 native CLIP interaction is the best tested
performance configuration. M1 calibration is promising for the conditioned
head but did not beat P1 on this first seed. PIPAL ranking-only training and
the residual head did not improve the external macro. Keep the clean mixture
and native interaction as the finalist, and treat five-seed confirmation and
support-set calibration for unseen datasets as publication-grade follow-up
rather than silently mixing calibrated and latent scores.

## Detailed results and interpretation

The macro table above is useful for model selection, but it hides which test
sets move. The following breakdown uses the same zero-retraining external
evaluations. SRCC is the primary IQA ranking metric; the complete SRCC/PLCC
rows, sample counts, latency, and MLflow IDs remain in `runs/results.csv`.

### P1: clean-mixture native CLIP interaction

The P1 interaction head was trained on 18,945 images and selected on 4,705
reference-held-out images. It has 529,409 trainable parameters. The selected
epochs and validation scores were:

| seed | selected epoch | validation SRCC | validation PLCC | worst validation SRCC |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 0.8335 | 0.8519 | 0.7704 |
| 1 | 2 | 0.8458 | 0.8503 | 0.7905 |
| 2 | 1 | 0.8317 | 0.8343 | 0.7631 |

The prompt interventions on the selected checkpoints show that the head did
not simply ignore text. In seed 0, for example, macro validation SRCC was
0.8335 with the correct prompt, 0.8294 with a held-out paraphrase, 0.7896
with a generic prompt, 0.7667 with the wrong family, 0.7571 after shuffling
conditions, and 0.5169 with zero text. The corresponding values for seeds 1
and 2 were correct/held-out/wrong/shuffled/zero =
`0.8458/0.8405/0.7756/0.7578/0.3303` and
`0.8317/0.8252/0.7697/0.7466/0.7009`.

The three-seed external SRCC means make the transfer pattern clearer:

| held-out dataset | interaction | image-only baseline | difference |
| --- | ---: | ---: | ---: |
| AGIQA-3K | 0.7187 | 0.7027 | +0.0160 |
| CID2013 | 0.6283 | 0.6230 | +0.0053 |
| CLIVE | 0.7411 | 0.7007 | +0.0404 |
| CSIQ | 0.8232 | 0.7867 | +0.0365 |
| GFIQA-20K | 0.6686 | 0.6552 | +0.0134 |
| KonIQ-10k | 0.6308 | 0.5785 | +0.0523 |
| PIPAL | 0.4334 | 0.4422 | -0.0088 |
| TID2013 | 0.6677 | 0.7213 | -0.0536 |
| UHD-IQA | 0.2601 | 0.1723 | +0.0878 |
| **macro mean** | **0.6191** | **0.5954** | **+0.0237** |

The interaction gain is therefore real in the aggregate, but it is not a
uniform distortion-family transfer result: TID2013 and PIPAL decline, while
the largest numerical gain is on the authentic UHD-IQA set, where both models
are still weak. Also, the interaction head is not capacity matched to the
baseline (529,409 versus 198,657 parameters), so this table cannot isolate a
semantic-text gain.

### M1: MDTVSFA-style scale alignment

M1 adds a shared latent score and a monotonic, learned calibration per training
dataset, with equal per-dataset regression losses and a within-dataset ranking
term. The image-only run selected epoch 4 (`0.8269` validation SRCC,
`0.8326` PLCC) and reached external `0.5965/0.6194` SRCC/PLCC. Its strongest
per-dataset SRCCs were AGIQA-3K `0.7146`, CID2013 `0.6317`, CSIQ `0.7615`, and
PIPAL `0.4508`; it reduced CSIQ and authentic-dataset transfer relative to P1.

The conditioned M1 run selected epoch 1 (`0.8407/0.8558` validation
SRCC/PLCC) and reached external `0.6238/0.6561`. Its per-dataset SRCCs were:

```text
AGIQA 0.6993  CID2013 0.6999  CLIVE 0.7311  CSIQ 0.8307
GFIQA 0.6691  KonIQ 0.6112  PIPAL 0.4071  TID2013 0.6755  UHD 0.2901
```

This is close to P1, but it is only one seed. More importantly, the external
datasets do not have learned calibrators: evaluation falls back to the shared
latent score. Because the calibration is monotonic, it cannot change an
unseen-dataset SRCC directly. Thus this run tests whether calibration changes
the learned representation indirectly, not whether calibrated external MOS
values are solved.

### M2: dataset mixtures and loss weighting

The ablations were image-only controls, so they ask whether the data objective
helps independently of text:

| variant | train/validation images | selected epoch | validation SRCC | external SRCC/PLCC | result |
| --- | ---: | ---: | ---: | ---: | --- |
| full clean mixture, global proportional | 18,945/4,705 | 2 | 0.8194 | 0.5867/0.6064 | worse than equal dataset sampling |
| KADID + SPAQ | 17,025/4,225 | 1 | 0.8231 | 0.5741/0.5931 | authentic data alone did not transfer |
| KADID + AIGCIQA | 10,045/2,480 | 4 | 0.7841 | 0.5434/0.5562 | weakest mixture |
| calibrated full mixture, softmax loss | 18,945/4,705 | 4 | 0.8300 | 0.5973/0.6174 | small calibration gain, still below P1 |

The full proportional run had SRCC `0.6918/0.5996/0.7061/0.7730/0.6501/
0.5539/0.4384/0.6930/0.1743` on AGIQA-3K/CID2013/CLIVE/CSIQ/GFIQA/KonIQ/
PIPAL/TID2013/UHD. The KADID+SPAQ run was `0.6888/0.5337/0.7056/0.8001/
0.5874/0.5565/0.4288/0.6719/0.1941`; KADID+AIGCIQA was
`0.7144/0.5056/0.5681/0.7532/0.5837/0.4896/0.4046/0.6647/0.2068`.
The mixture choice clearly matters, but none of these one-seed controls
establishes a generally best training distribution.

### M3: PIPAL within-reference ranking

M3 used 33,185 training and 8,265 validation images, excluded PIPAL from
absolute regression, and added within-reference ranking. It selected epoch 3
with validation SRCC/PLCC `0.7408/0.7435`, far below the clean-mixture
controls. External SRCC/PLCC was `0.5789/0.5999`.

PIPAL SRCC rose slightly to `0.4538`, compared with about `0.4334` for the P1
interaction mean, but the other datasets fell: CID2013 `0.5359`, CLIVE
`0.6748`, GFIQA `0.6152`, TID2013 `0.6482`, and UHD-IQA `0.2243`. The ranking
loss therefore improved the one metric it was designed to target only
slightly, while damaging cross-dataset transfer. It also operates on pairs
that happen to share a reference within a mini-batch, so its effective signal
is sparse under random batch composition.

### H1: residual conditioning head

H1 used condition dropout (`0.15`) and a residual correction so zero text has
an explicit image-only path. It selected epoch 4 with validation
SRCC/PLCC `0.8298/0.8293` and had 728,066 trainable parameters. External
SRCC/PLCC was `0.5882/0.6139`, below P1 and nearly at the image-only level.

Its intervention scores were correct `0.8298`, held-out paraphrase `0.8261`,
generic `0.8128`, wrong `0.8058`, shuffled `0.8015`, and zero `0.8179`.
The narrow spread means the residual model mostly learned a strong visual
score and made only a small condition-dependent correction. This is useful as
a fallback design, but not evidence that residual text improves transfer.

### E7: INSTRUCTOR text encoder

E7 replaced the native CLIP text tower with frozen INSTRUCTOR while keeping
the CLIP-B image encoder, clean mixture, interaction fusion, and training
budget fixed. It selected epoch 2 with validation SRCC/PLCC `0.8288/0.8441`,
used 595,457 trainable head parameters, and reached external `0.6112/0.6464`.
Per-dataset SRCC was AGIQA-3K `0.6897`, CID2013 `0.6567`, CLIVE `0.7372`, CSIQ
`0.8165`, GFIQA `0.6760`, KonIQ `0.6321`, PIPAL `0.4135`, TID2013 `0.6175`,
and UHD-IQA `0.2616`.

The interventions were correct `0.8288`, held-out `0.8304`, generic `0.8100`,
wrong `0.7523`, shuffled `0.7582`, and zero `0.3531`. INSTRUCTOR therefore
retained prompt sensitivity but did not beat the native CLIP interaction mean
(`0.6191/0.6570`). This suggests that better general-purpose sentence
similarity is not enough when the frozen visual representation is the limiting
factor.

### E8: leave-one-group-out semantic transfer

E8 excluded one KADID group at a time, then evaluated the corresponding group
in TID2013 and CSIQ. These were one-seed diagnostic probes, not five-seed
finalist estimates:

| group excluded from training | train/validation images | selected epoch | validation SRCC | TID2013 SRCC | CSIQ SRCC | clean image-only comparator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blur | 17,970/4,465 | 2 | 0.8273 | 0.7471 | 0.7842 | 0.7988/0.7983 |
| noise | 17,320/4,305 | 2 | 0.8406 | 0.5743 | 0.7025 | 0.6850/0.7493 |
| compression | 18,295/4,545 | 4 | 0.8271 | 0.7993 | 0.8693 | 0.7977/0.8370 |

Leaving out blur hurt TID2013 and leaving out noise hurt both test sets, while
leaving out compression improved both slightly. The result is not a stable
zero-shot semantic advantage. The held-out prompt is a new vector to a head
trained on the other seven vectors, and the pooled CLIP feature still has to
detect the relevant low-level artifact; the text alone cannot perform that
extrapolation.

### UHD-IQA official-split domain-adaptation probe

UHD-IQA is a deliberately separate domain-adaptation experiment, not a
zero-shot result. Its published partitions are retained exactly: 4,269 UHD
images are eligible for optimization, 904 for epoch selection, and the 900
official test images are written to a separate CSV that the training runner
refuses to load. The old KADID/SPAQ/AIGCIQA rows receive a deterministic,
reference-disjoint seed-0 split in the same manifest. The manifest is shared
by every comparison and logged as an MLflow artifact.

The first one-seed batch keeps the M1 objective fixed and compares the ordinary
image-only calibrated baseline and calibrated native-CLIP interaction head.
Both use equal per-dataset sampling and four learned training-dataset
calibrators (KADID-10k, SPAQ, AIGCIQA2023, UHD-IQA). Only after epoch selection
are they evaluated once on `uhd_official_test.csv`. The protocol builder is
`build_uhd_official_training_data.py`; the valid UHD configs are 39 and 41.

The first seed (`seed=0`) produced the following official 900-image UHD test
scores:

| head | validation macro SRCC | UHD test SRCC | UHD test PLCC |
| --- | ---: | ---: | ---: |
| ordinary image-only (198,663 parameters) | 0.7153 | 0.3869 | 0.3898 |
| native-CLIP conditioned interaction (529,417) | 0.7556 | 0.5974 | 0.5338 |

The earlier widened UHD image-only run is retained as a historical auxiliary
result, but it cannot answer the head-capacity question: it belongs to a
different four-dataset training setup than the row-14 best model. The valid
capacity control is config 42, paired only with configs 25 and 27 on the
original KADID-10k/SPAQ/AIGCIQA2023 mixture. This is a one-seed UHD result, not
a final conclusion; repeat seeds 1--2 before reporting a mean or choosing a
default.

### Faithful MDTVSFA implementation

The earlier M1 rows are explicitly an approximation: one learned sigmoid
calibrator was trained with regression/ranking losses. They should not be
described as a reproduction of MDTVSFA. The new
`38_clean_multi_mdtvsfa_faithful_interaction.yaml` implementation follows the
paper/reference design adapted to frozen-image/text features:

1. a shared relative score `sigmoid(z)` trained with the vectorized
   monotonicity-induced loss;
2. a shared four-parameter logistic nonlinear mapping trained with the
   centered-cosine (PLCC) linearity-induced loss; and
3. a separate affine alignment layer per training dataset, initialized from
   that dataset's raw-score range and trained with normalized L1 error.

Each dataset contributes the sum of those three losses. Dataset batches are
kept separate and shorter loaders are cycled, matching the reference mixed
dataset procedure; the final aggregate is the reference softmax-weighted
loss. Raw scores are used in their original units, with CSIQ/LIVE DMOS
datasets oriented so higher still means better. Unknown test datasets use the
shared perceptual stage because their alignment layer is unavailable.

The three-seed pilot selected epochs 1, 1, and 0, respectively. Its held-out
macro mean was `0.6169/0.6441` SRCC/PLCC, compared with
`0.6191/0.6570` for the matched clean-mixture interaction control. Per-dataset
SRCC means were:

| AGIQA-3K | CID2013 | CLIVE | CSIQ | GFIQA-20K | KonIQ-10k | PIPAL | TID2013 | UHD-IQA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7226 | 0.5983 | 0.7543 | 0.8239 | 0.6810 | 0.6748 | 0.4236 | 0.6607 | 0.2133 |

The faithful objective improved KonIQ, CLIVE, GFIQA, and AGIQA slightly, but
lost on CID2013, TID2013, PIPAL, and especially UHD-IQA. It therefore does
not replace the current clean-interaction configuration as the default. The
implementation is retained for controlled ablations and future objective
work; the earlier M1 numbers remain the legacy approximation and must not be
combined with this result.

### H1: pooled MLP depth

The depth follow-up tests whether the one-hidden-layer score MLP is limiting
the conditioned interaction head. The implementation adds hidden-to-hidden
GELU/dropout blocks while keeping `mlp_layers=1` identical to the historical
head. All four runs use CLIP-B/16, the KADID-10k + SPAQ + AIGCIQA2023 clean
mixture, the reference split, equal per-dataset legacy MDTVSFA weighting,
three calibrators, five epochs, and seed 0. The image-only controls match the
conditioned heads' total parameter counts: 595,115 vs. 595,207 for depth 2 and
661,517 vs. 660,999 for depth 3.

The external suite uses the eight unseen benchmark CSVs plus the complete
6,073-image UHD-IQA CSV. UHD is zero-shot for these models because no UHD
images enter their training table; the official 900-image split is reserved
for the separate UHD-training experiments and is not mixed into this depth
comparison.

| head | selected epoch | validation SRCC/PLCC | full external SRCC/PLCC |
| --- | ---: | ---: | ---: |
| conditioned interaction, one hidden layer (reference) | 1 | 0.8407/0.8558 | 0.6238/0.6561* |
| conditioned interaction, two hidden layers | 1 | 0.8452/0.8574 | 0.6246/0.6525 |
| conditioned interaction, three hidden layers | 1 | 0.8406/0.8514 | 0.6171/0.6392 |
| matched image-only, two hidden layers | 3 | 0.8282/0.8379 | 0.5879/0.6192 |
| matched image-only, three hidden layers | 3 | 0.8353/0.8460 | 0.5868/0.6165 |

\*The one-layer reference value is the existing row-14 result under the
recent nine-dataset protocol; its UHD component was evaluated on the official
900-image split, so it is shown only as a contextual reference and is not a
strictly identical full-UHD comparison.

Depth 2 is the only candidate with a positive validation and external
movement over the one-layer interaction on this seed. On the eight datasets
that are identical between the runs (excluding UHD), it reaches
`0.6707/0.6979` versus `0.6655/0.6909` for the one-layer seed-0 reference
(`+0.0052/+0.0070`), while its nine-dataset full-UHD mean is only `+0.0008`
SRCC above the reference number because the UHD protocols differ. This still
needs seeds 1--2. Depth 3 is worse externally,
while both matched image-only controls remain substantially below their
conditioned counterparts. The current evidence therefore does not support
“the MLP is simply too small” as the sole explanation: extra depth may help
slightly at depth 2, but it does not reproduce a broad generalization gain.

### Backbone and representation follow-up (single seed)

The current-best protocol was held fixed for the backbone comparison: frozen
native image/text encoders, the clean KADID-10k + SPAQ + AIGCIQA2023 mixture,
stretch preprocessing, equal per-dataset sampling, five epochs, and the
best-validation checkpoint. Thus the CLIP-L and SigLIP rows are not selected
from the later architecture search; they are direct backbone substitutions on
the pre-existing best configuration. Every external value below uses all nine
held-out CSVs, including the complete zero-shot UHD-IQA set.

| configuration | selected epoch | validation mean SRCC/PLCC | external mean SRCC/PLCC | val+external mean SRCC/PLCC |
| --- | ---: | ---: | ---: | ---: |
| frozen SigLIP-L/16@256, native interaction | 2 | 0.8435/0.8511 | 0.6258/0.6464 | 0.7347/0.7487 |
| frozen CLIP-L/14@336, native interaction | 1 | 0.8652/0.8783 | 0.6292/0.6570 | 0.7472/0.7677 |

The same queue also tested representation/head hypotheses with CLIP-B/16. A
five-view head (one global view plus four local tiles) improved both validation
and external means over its matched image-only control, so its gain cannot be
attributed to extra views alone. The pooled visual adapter and softmax source
loss did not improve the native interaction control. These are deliberately
single-seed exploratory results and are recorded in the sheet immediately
after the backbone rows.

| configuration | selected epoch | validation mean SRCC/PLCC | external mean SRCC/PLCC |
| --- | ---: | ---: | ---: |
| five-view native text interaction | 1 | 0.8721/0.8848 | 0.6580/0.6884 |
| five-view image-only control | 4 | 0.8553/0.8680 | 0.6310/0.6578 |
| pooled visual adapter + native interaction | 1 | 0.8318/0.8515 | 0.6160/0.6553 |
| softmax source-loss weighting + native interaction | 1 | 0.8393/0.8542 | 0.6165/0.6415 |

The full per-dataset values, parameter footprints, and run metadata are in
the `28s_mur` sheet; the rows were appended after the last existing populated
row and all metric cells remain numeric with the existing four-decimal display
format.

### Original E0--E4 pilot

The initial KADID pilot is reported in detail in
[text-conditioning-pilot-results.md](text-conditioning-pilot-results.md). Its
results establish the narrower in-domain claim:

| pilot | mean KADID validation SRCC | interpretation |
| --- | ---: | --- |
| E0 image-only baseline | 0.7599 | reference control |
| E1 plain concatenation | 0.7648 | inconsistent, small gain |
| E2 interaction `[v,t,v*t]` | 0.7847 | strongest pooled head in-domain |
| E2 residual correction | 0.7753 | useful fallback, smaller gain |
| E4 held-out paraphrase | 0.7752 | retains most canonical performance |

The canonical interaction score was `0.7861`; generic, wrong, and shuffled
conditions were `0.7265`, `0.7269`, and `0.7190`. E0--E4 therefore show that
the model can react to prompt meaning on KADID, but they do not establish
cross-dataset quality transfer. E6 (integration with a separate learned-query
scorer) was intentionally not run because that architecture belongs to a
separate workstream.

### Overall reading

Taken together, the experiments answer two different questions. The first is
positive: a frozen text vector changes the KADID score in a prompt-dependent
way, and native CLIP interaction gives a modest aggregate external gain. The
second is negative: changing the fusion head, text encoder, dataset loss, or
PIPAL objective does not solve the weak UHD-IQA/PIPAL/authentic transfer.
The next comparison must therefore target the image representation and use a
matched-capacity control before making a stronger semantic-conditioning claim.
