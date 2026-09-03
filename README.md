# Conditioned IQA — quick start

A quality metric usually gets an image and nothing else. This project asks
whether telling it what kind of distortion it is looking at makes its scores
agree better with human ones.

The full statement — research directions, how it is evaluated, what to hand
in — is at
[dreminm.github.io/iqa-summer-school/project-1.html](https://dreminm.github.io/iqa-summer-school/project-1.html).
This repository is where you start from.

## Submission reproduction

| model | test SRCC | test PLCC |
| --- | ---: | ---: |
| Image-only baseline | 0.4783 | 0.4572 |
| Text-conditioned model | 0.6612 | 0.6852 |
| Label-conditioned model | 0.6380 | 0.6534 |
| Model conditioned with a trained label classifier | 0.5200 | 0.5184 |

All models use CLIP-B/16. The reported averages use the held-out datasets
listed for each run; PIPAL and DFIQA are not included in these aggregates.

### Setup

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e .
export DATA_ROOT="${DATA_ROOT:-$HOME/conditioned-iqa/data}"
for d in kadid10k spaq aigciqa2023 tid2013 csiq cid2013 koniq10k clive agiqa3k gfiqa20k pipal uhdiqa; do
  uv run python download_data.py "$d" --data-root "$DATA_ROOT"
done
mkdir -p "$DATA_ROOT/multi_train_clean"
uv run python prepare_data.py "$DATA_ROOT/kadid10k" "$DATA_ROOT/spaq" "$DATA_ROOT/aigciqa2023" --out "$DATA_ROOT/multi_train_clean/labels.csv"
for d in tid2013 csiq cid2013 koniq10k clive agiqa3k gfiqa20k pipal uhdiqa; do
  uv run python prepare_data.py "$DATA_ROOT/$d"
done
```

### Evaluate the released weights (no training)

Download the three release assets into `weights/` (the filenames are listed
below), then prepare the datasets as in Setup. These commands only load the
checkpoints and report per-dataset and macro SRCC/PLCC.

```bash
uv run python evaluate_text_conditioned.py --checkpoint weights/clipb-five-view-calibrated-interaction-best.pt --config configs/text_conditioning/65_clean_multi_multiview_mdtvsfa_interaction.yaml --data "$DATA_ROOT"/{tid2013,csiq,cid2013,koniq10k,clive,agiqa3k,gfiqa20k,pipal,uhdiqa}/labels.csv --weights "${CLIP_BASE_WEIGHTS:-}" --text-weights "${CLIP_BASE_WEIGHTS:-}" --device cuda:0
```

The label and classifier commands are available after merging the
label-conditioning tree (they use the checkpoint's saved architecture and do
not retrain):

```bash
uv run python evaluate.py --checkpoint weights/label_low_rank_hypernetwork_multitrain_clean_best.pth --data "$DATA_ROOT"/{tid2013,csiq,cid2013,koniq10k,clive,agiqa3k,gfiqa20k,pipal,uhdiqa}/labels.csv --device cuda:0
uv run python evaluate_classifier_checkpoint.py --checkpoint weights/clipb-classifier-layer3-feature-deep-concat-best.pt --data "$DATA_ROOT"/{tid2013,csiq,cid2013,koniq10k,clive,agiqa3k,gfiqa20k,pipal,uhdiqa}/labels.csv --device cuda:0
```

Best release assets:
`clipb-five-view-calibrated-interaction-best.pt`,
`label_low_rank_hypernetwork_multitrain_clean_best.pth`,
`clipb-classifier-layer3-feature-deep-concat-best.pt`.

All three evaluators load the released checkpoint directly; evaluation does
not retrain the models.

### Train from scratch

Use the prepared multi-dataset table from Setup:

```bash
# Text-conditioned model
uv run python train_text_conditioned.py --config configs/text_conditioning/65_clean_multi_multiview_mdtvsfa_interaction.yaml --data "$DATA_ROOT/multi_train_clean/labels.csv" --device cuda:0 --weights "${CLIP_BASE_WEIGHTS:-}" --text-weights "${CLIP_BASE_WEIGHTS:-}" --out weights/text-conditioned-best.pt

# Label-conditioned model
uv run python -m label_and_embed_conditioning.train --data "$DATA_ROOT/multi_train_clean/labels.csv" --backbone clip-base --epochs 5 --batch-size 32 --lr 0.001 --hidden-dim 256 --conditioning label --label-fusion low_rank_hypernetwork --label-dim 32 --low-rank-dim 4 --condition-dropout 0.1 --split reference --sampler random --device cuda:0 --seed 0 --save-dir weights --name label-conditioned

# Model conditioned with a trained label classifier
uv run python -m models.label_cond.train --config configs/label_cond/label_cond.yaml configs/label_cond/16_frozen_layer3_emb_deep.yaml
```

The label-conditioning commands require the label-conditioning tree to be
present in the checkout.

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

Reusable experiment arguments live in `configs/`; command-line options can
override YAML defaults:

```
uv run python train.py --config configs/kadid_smoke.yaml
uv run python train.py --config configs/kadid_smoke.yaml --limit 256 --seed 1
```

For the held-out protocol and transfer results, see [datasets.md](datasets.md)
and [docs/external-evaluation-results.md](docs/external-evaluation-results.md).

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
batch already carries `distortion`, `level` and `group`, so conditioning the
model on them is a change to `train.py` alone.

Which datasets train, which are held out and what each one is for:
[datasets.md](datasets.md).
