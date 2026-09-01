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

Reusable experiment arguments live in `configs/`. Values from a YAML file are
used as defaults, so explicit command-line options can override them:

```
uv run python train.py --config configs/kadid_smoke.yaml
uv run python train.py --config configs/kadid_smoke.yaml --limit 256 --seed 1
```

Add `mlflow: true` to a config (or pass `--mlflow`) to record the run
configuration, training loss after every optimizer step, validation
SRCC/PLCC (including per-dataset metrics), and the final quality-head
checkpoint. By default MLflow uses a local SQLite database, `mlflow.db`:

```
python train.py --data ~/iqa-data/kadid10k/labels.csv --epochs 5 \
  --mlflow --mlflow-experiment conditioned-iqa --mlflow-run-name clip-baseline
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Use `--mlflow-tracking-uri` to send runs to an existing tracking server.
Every tracked run also writes a complete YAML configuration to `runs/configs/`
and registers that file under the run's `configs/` artifacts. Change the local
destination with `--config-dir`. These generated manifests contain resolved
runtime information and can themselves be passed back through `--config`;
runtime-only fields are ignored when reconstructing arguments.

If the virtual environment is already activated, the shorter equivalent is
`mlflow ui --backend-store-uri sqlite:///mlflow.db`. Run either command from
the repository root, where `mlflow.db` is created.

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
