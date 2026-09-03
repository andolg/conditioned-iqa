# Conditioned IQA — quick start

A quality metric usually gets an image and nothing else. This project asks
whether telling it what kind of distortion it is looking at makes its scores
agree better with human ones.

The full statement — research directions, how it is evaluated, what to hand
in — is at
[dreminm.github.io/iqa-summer-school/project-1.html](https://dreminm.github.io/iqa-summer-school/project-1.html).
This repository is where you start from.

Four scripts and a note to get you to a number today:

```
download_data.py   fetch a dataset and unpack it
prepare_data.py    its labels -> one CSV, same columns for every dataset
dataset.py         a torch Dataset over that CSV, with splitting and sampling
train.py           frozen CLIP + an MLP, trained to predict quality
datasets.md        what trains, what is held out, and why
```

## Run it

```
uv venv --python 3.12
source .venv/bin/activate        # .venv\Scripts\activate on Windows
uv pip install -e .

python download_data.py --list
python download_data.py kadid10k --data-root ~/iqa-data     # 2.9 GB, start here
python prepare_data.py ~/iqa-data/kadid10k
python train.py --data ~/iqa-data/kadid10k/labels.csv --epochs 5
```

To run the learned-label conditioning experiment on the same frozen backbone:

```bash
python train.py --data ~/iqa-data/kadid10k/labels.csv --epochs 5 \
  --conditioning label --label-dim 32 \
  --save-dir ./weights --name label_conditioned_clip_base
```

The parameter-matched patch-attention and low-rank hypernetwork experiments
each have matching CLIP-Base and CLIP-Large launchers:

```bash
bash scripts/train/train_label_patch_attention_base.sh
bash scripts/train/train_label_patch_attention_large.sh
bash scripts/train/train_label_low_rank_hypernetwork_base.sh
bash scripts/train/train_label_low_rank_hypernetwork_large.sh
```

Both use the parameter budget of a 256-wide baseline head. Patch attention
spends part of that budget on a label-guided query over frozen spatial tokens;
the low-rank hypernetwork spends it on a rank-4 label-generated update. Override
the latter with `LOW_RANK_DIM=8`, and override either run's five-epoch default
with `EPOCHS=N`.

`run_label_conditioned.sh` mirrors `run_baseline_large.sh` exactly (CLIP-Large,
data, device, and local weights) and changes only the head conditioning, so
those two runs are the direct comparison for the current setup. Use
`run_baseline_base.sh` for the CLIP-Base baseline. The required permuted-training
control can use that same script while overriding its output:

```bash
bash run_label_conditioned.sh --permute-training-labels \
  --save-dir ~/conditioned-iqa/28d_evs/conditioned-iqa/weights \
  --name label_permuted_clip_large
```

This adds only a small group embedding and one narrow slice to the existing
MLP input; it does not enlarge or fine-tune CLIP. Validation reports the
oracle (correct) group normally, then reports shuffled, deliberately wrong,
and zeroed conditions from the same frozen image features. Those checks show
whether a gain came from using the label rather than merely adding parameters.
`--condition-dropout 0.1` (the default) zeroes a small fraction of training
conditions so the zeroed-label result is a learned fallback.

Each validation pass also prints inference latency p50/p95 in milliseconds
per image, peak allocated CUDA memory in MB, and image throughput. Timing
covers device transfer, the frozen vision backbone, and the normal prediction
head at the configured batch size; metric computation and condition-ablation
forwards are excluded. Peak memory is reported as `N/A` outside CUDA.
After training, the macro validation SRCC/PLCC from the best epoch is printed.
The best epoch is selected by macro validation SRCC. Training always writes
`NAME_best.pth` and `NAME_last.pth` below `--save-dir`.

For per-image distortion-manifold conditioning, run:

    bash scripts/train/train_arniqa_conditioned_base.sh
    bash scripts/train/train_arniqa_conditioned_large.sh
    bash scripts/test/evaluate_arniqa_base.sh
    bash scripts/test/evaluate_arniqa_large.sh

For the comparison row using pretrained ARNIQA itself as the quality metric:

    bash scripts/test/evaluate_arniqa.sh

This defaults to ARNIQA's official KADID-10k regressor and evaluates the same
prepared datasets without CLIP or a trained conditioned head. Override the
official regressor with `REGRESSOR_DATASET`, or provide local files through
`ARNIQA_WEIGHTS` and `ARNIQA_REGRESSOR_WEIGHTS`.

This path uses only ARNIQA's frozen self-supervised encoder, not any of its
dataset-specific quality regressors. It averages the official center and
corner crops at full and half scale, projects the resulting 4096-dimensional
condition through a learned 32-dimensional bottleneck, and concatenates that
with the normalized frozen CLIP feature. Set
`ARNIQA_WEIGHTS=/path/to/ARNIQA.pth` to use a local checkpoint; otherwise
the official encoder checkpoint is downloaded to PyTorch's cache.

The paired-training control keeps images and targets fixed while globally
permuting their ARNIQA condition donors:

    bash run_arniqa_conditioned.sh --permute-training-conditions --name arniqa_permuted_clip_large

ARNIQA validation reports correct, whole-dataset shuffled, and zeroed
conditions. Normal-path latency and memory include both frozen encoders and
all ten ARNIQA crops.

Evaluate one trained checkpoint on every prepared dataset under
`~/conditioned-iqa/data` without retraining or reloading the backbone between
datasets:

```bash
bash scripts/evaluate_all.sh
```

The default is `./weights/label_conditioned_clip_large_best.pth` on `cuda:7`.
Override those settings with environment variables and pass remaining
evaluation options after the script name:

```bash
CHECKPOINT=./weights/baseline_clip_large_best.pth DEVICE=cuda:0 \
  bash scripts/evaluate_all.sh --batch-size 64 --workers 8
```

The evaluator prints total/trainable/head parameter counts and estimated
GFLOPs for one normal-path image in addition to IQA and runtime metrics.

Use `--limit 2000` while you are still wiring things up — it samples that many
training images at random, and leaves the held-out split whole.

Downloads run in parallel byte ranges, because the mirror throttles a single
sustained connection to a crawl. Pass `--connections 1` if a proxy dislikes
range requests.

## What prepare_data does

Every release ships its labels differently, so this reads whichever format
it finds and writes one table:

| column | |
| --- | --- |
| `path` | the image |
| `original_subjective_score` | the score as the release published it |
| `scaled_subjective_score` | the same, min-maxed to [0, 1], higher = better |
| `dataset` · `reference` | which set it came from, and of which pristine image |
| `distortion` · `level` | the type and severity the release recorded |
| `group` | that type folded into one of eight distortion groups |

The `group` column is the condition this project studies. It names a family
— blur, noise, compression, colour, tone, spatial, generative, or
`authentic` for photographs nobody degraded on purpose — rather than an
individual type, because no two releases share a type vocabulary and a
per-type label teaches the corpus instead of the distortion.

Point it at several directories with `--out all.csv` to get one table for
all of them — `python prepare_data.py ~/iqa-data/*/ --out ~/iqa-data/all.csv`
prepares everything you have downloaded. On a table like that, train with
`--sampler by_dataset` so the largest set does not decide the batch, and read
the per-dataset rows rather than one pooled number.

## Splitting

```python
from dataset import IQADataset, split_by, make_sampler

data = IQADataset("~/iqa-data/kadid10k/labels.csv", image_size=224)
train, val = split_by(data, "reference")        # or "random"
sampler = make_sampler(train, "balanced")       # or "random", "by_level", "by_dataset"
```

`split_by` keeps a pristine reference whole on one side, and takes its share
from every dataset separately. Both defaults matter. In KADID a hundred and
twenty-five rows are one photograph seen through twenty-five distortions, so
splitting them apart lets the model score the held-out ones by recognising
the picture — on frozen features that is worth up to 0.44 SRCC, more than any
effect you are looking for. And a reference means different things in
different releases, one photograph here and a hundred and twenty-five rows
there, so drawing the held-out share from the pool would let one release
decide the split. Use `"random"` for photographs, where every image is its
own scene.

`make_sampler` weights batches by distortion type, severity or dataset
instead of letting the counts decide.

## Where to go next

`train.py` is short and meant to be edited. `--backbone clip-large`,
`siglip2-base` or `siglip2-large`, `QualityMLP` for a different head,
`embed()` if you want patch tokens instead of the pooled embedding. Every
batch carries `distortion`, `level` and `group`; `--conditioning label` uses
the coarse `group` field and keeps the individual distortion type out of the
model.

Which datasets train, which are held out and what each one is for:
[datasets.md](datasets.md).
