"""Turn a downloaded dataset into one CSV with the same columns as every other.

    python prepare_data.py ~/iqa-data/kadid10k
    python prepare_data.py ~/iqa-data/*/ --out ~/iqa-data/all.csv

Every release ships its labels differently — `dmos.csv` here, a MATLAB
struct there, an xlsx, a text file of "score filename" lines, one label file
per reference. This script reads whichever one it finds and writes:

    path                       absolute path to the image
    original_subjective_score  the score exactly as the release gives it
    scaled_subjective_score    the same, min-maxed into [0, 1], higher = better
    oriented_subjective_score  a runtime-only conditioned-runner view of the
                               raw score with DMOS datasets sign-flipped
    dataset                    which dataset the row came from
    reference                  the pristine image this is a version of, or the
                               image itself for photographs with no reference,
                               prefixed with the dataset name
    distortion, level          the type and severity as the release records them
    group                      that type folded into one of eight distortion
                               groups

After this, `dataset.py` just reads a CSV. Everything that knows about the
quirks of a particular release lives here, and you can open the result in a
spreadsheet and see what you are training on.

Two quirks worth knowing, both handled here:

- Scores come on 1-5, 0-9 and 0-100 scales, so `scaled_subjective_score`
  normalizes each dataset separately. `original_subjective_score` keeps the
  number the release published, untouched, so it stays checkable.
- CSIQ's is a DMOS: higher means *worse*. It gets flipped, so the scaled
  column always means quality.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Releases whose score runs backwards (higher = worse quality). Matched
# exactly, never as a substring: "live" is inside "clive", and CLIVE's MOS
# runs the right way up.
FLIPPED = {"csiq", "liveiqa", "livemd"}


# <reference>_<type>_<level>: I01_02_03.png (KADID), i01_02_3.bmp (TID2013)
SYNTHETIC_NAME = re.compile(r"^(?P<ref>[a-zA-Z]?\d+)_(?P<type>\d+)_(?P<level>\d+)\.\w+$")
# <reference>.<TYPE>.<level>: 1600.AWGN.1.png (CSIQ)
CSIQ_NAME = re.compile(r"^(?P<ref>.+?)\.(?P<type>[A-Za-z0-9]+)\.(?P<level>\d+)\.\w+$")


# --- distortion groups ----------------------------------------------------
#
# The condition this project studies names a *group* of distortions, not an
# individual type: a per-type label leaks which dataset a picture came from,
# and no two releases agree on a type vocabulary anyway. Six groups cover
# every synthetic release below; `generative` covers images a model drew or
# repaired; `authentic` covers photographs nobody applied a distortion to.
#
# The maps are the honest part of this file. Every one of them is a judgment
# call about where a published taxonomy fits, and the first thing to check
# when a group-conditioned model behaves strangely is whether two datasets
# disagree about what a group means.

GROUPS = ("compression", "generative", "blur", "noise", "color", "tone", "spatial", "authentic")

# KADID-10k, in the order of its official 7 families.
KADID_GROUPS = {
    1: "blur", 2: "blur", 3: "blur",
    4: "color", 5: "color", 6: "color", 7: "color", 8: "color",
    9: "compression", 10: "compression",
    11: "noise", 12: "noise", 13: "noise", 14: "noise", 15: "noise",
    16: "tone", 17: "tone", 18: "tone",
    19: "spatial", 20: "spatial", 21: "spatial", 22: "spatial", 23: "spatial",
    24: "tone", 25: "tone",
}

# TID2013's own 24 types. Its vocabulary is not KADID's, which is the point:
# after this map the two become comparable, and a model that learned groups
# rather than one dataset's type table can be tested across them.
TID2013_GROUPS = {
    1: "noise", 2: "noise", 3: "noise", 4: "noise", 5: "noise", 6: "noise", 7: "noise",
    8: "blur", 9: "noise",
    10: "compression", 11: "compression", 12: "compression", 13: "compression",
    14: "spatial", 15: "spatial",
    16: "tone", 17: "tone",
    18: "color",
    19: "noise", 20: "noise",
    21: "compression",
    22: "color", 23: "color",
    24: "compression",
}

# CSIQ names its types instead of numbering them, and spells the same thing
# differently across its label files.
CSIQ_GROUPS = {
    "awgn": "noise", "noise": "noise", "fnoise": "noise",
    "blur": "blur",
    "jpeg": "compression", "jpeg2000": "compression", "jpeg 2000": "compression",
    "contrast": "tone",
}

# Whole datasets whose group never varies.
DATASET_GROUPS = {
    "spaq": "authentic", "koniq10k": "authentic", "clive": "authentic",
    "gfiqa20k": "authentic", "cid2013": "authentic", "uhdiqa": "authentic",
    "agiqa3k": "generative", "aigciqa2023": "generative",
}

TYPE_GROUPS = {"kadid10k": KADID_GROUPS, "tid2013": TID2013_GROUPS, "csiq": CSIQ_GROUPS}


# PIPAL numbers its distortion classes and does not name them anywhere in the
# release. The mapping below is recovered from Table 10 of the journal
# version (arXiv:2011.15002) by counting: each sub-type there has a unique
# number of parameter variants, and those counts match the number of `cc`
# values per class in the data exactly — 12, 16, 10, 24, 13, 14, 27, summing
# to the 116 distortion levels the paper reports. The filename is
# `Aaaaa_bb_cc`: image, distortion class, variant within the class (confirmed
# by the author in issue 13 of the dataset repository).
PIPAL_CLASSES = {
    "00": ("traditional SR", "generative"),
    "01": ("PSNR-oriented SR", "generative"),
    "02": ("SR with kernel mismatch", "generative"),
    "03": ("GAN-based SR", "generative"),
    "04": ("denoising", "generative"),
    "05": ("SR and denoising jointly", "generative"),
    # Class 06 is nine classical distortions mixed together — blur, noise,
    # JPEG, colour quantization, spatial warping. Which `cc` is which follows
    # the order of Table 10, and that order holds for most of them (the two
    # codecs in cc 04-08 separate cleanly, noise and comfort noise fall
    # monotonically), but not for all, so no group is assigned rather than a
    # wrong one.
    "06": ("traditional distortions", None),
    # The NTIRE validation split, a separate set of references.
    "10": ("NTIRE validation", None),
}


def pipal_group(distortion) -> str | None:
    """The group of a PIPAL row, or None where the class does not map to one."""
    # zfill: the code is "00".."06" in the release and becomes 0..6 once a
    # CSV round-trip has turned it into a number.
    entry = PIPAL_CLASSES.get(str(distortion).strip().zfill(2))
    return entry[1] if entry else None


def assign_group(dataset: str, distortion) -> str | None:
    """The distortion group of one image, or None when it cannot be known.

    PIPAL's restoration classes map to `generative`; its class of classical
    distortions does not map to a single group and stays unlabelled.
    """
    dataset = dataset.lower()
    if dataset == "pipal":
        return pipal_group(distortion)
    if dataset in DATASET_GROUPS:
        return DATASET_GROUPS[dataset]
    type_map = TYPE_GROUPS.get(dataset)
    if type_map is None or distortion is None or (isinstance(distortion, float) and pd.isna(distortion)):
        return None
    key = str(distortion).strip().lower()
    if key.isdigit():
        key = int(key)
    return type_map.get(key)


def parse_name(name: str, meta: dict) -> tuple[str, str | None, int | None]:
    """Reference, distortion type and level — from metadata, else the filename."""
    reference = meta.get("ref_id") or meta.get("ref_filename")
    distortion = meta.get("distortion_type")
    level = meta.get("distortion_level")
    if reference is None or distortion is None:
        for pattern in (SYNTHETIC_NAME, CSIQ_NAME):
            match = pattern.match(name)
            if match:
                reference = reference or match["ref"]
                distortion = distortion or match["type"]
                level = level if level is not None else int(match["level"])
                break
    # Case-folded: TID2013 ships both I01_.. and i01_.. for one reference,
    # and a case-sensitive key would split one picture across a boundary.
    reference = str(reference or name).split(".")[0].lower()
    return reference, (str(distortion) if distortion is not None else None), (
        int(level) if level is not None else None
    )


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def index_images(root: Path) -> dict[str, Path]:
    """filename -> path, built once.

    Releases bury images at different depths (`images/`, `dst_imgs/awgn/`,
    `1024x768/`), and looking each one up with a glob is quadratic — on
    KADID that alone takes minutes.
    """
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.name not in index:
            index[path.name] = path
    return index


def read_unified(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """`data.csv` — the shape several repackaged releases already use."""
    rows = []
    for record in pd.read_csv(labels).to_dict("records"):
        meta = record.get("metadata")
        meta = json.loads(meta) if isinstance(meta, str) and meta.strip() else {}
        name = str(record["filename"])
        relative = record.get("path")
        path = (
            labels.parent / str(relative)
            if isinstance(relative, str) and relative
            else images.get(name, root / name)
        )
        # KonIQ ships both a 1-5 MOS and a z-scored 0-100 column; the paper
        # reports the former, and they correlate at 0.992 — close enough to
        # look interchangeable, far enough to move the numbers.
        score = meta["mos"] if "mos" in meta and root.name.lower().startswith("koniq") \
            else record["subjective_score"]
        rows.append({"path": path, "original_subjective_score": float(score),
                     **dict(zip(("reference", "distortion", "level"), parse_name(name, meta)))})
    return rows


def read_kadid(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    df = pd.read_csv(labels)
    return [
        {"path": images.get(name, root / name), "original_subjective_score": float(score),
         **dict(zip(("reference", "distortion", "level"), parse_name(name, {})))}
        for name, score in zip(df["dist_img"].astype(str), df["dmos"])
    ]


def read_koniq(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    df = pd.read_csv(labels)
    column = "MOS" if "MOS" in df.columns else "mos"
    return [
        {"path": images.get(name, root / name), "original_subjective_score": float(score),
         "reference": name.lower(), "distortion": None, "level": None}
        for name, score in zip(df["image_name"].astype(str), df[column])
    ]


def read_tid2013(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    rows = []
    for line in labels.read_text().splitlines():
        if not line.strip():
            continue
        score, name = line.split()
        rows.append({"path": images.get(name, root / name), "original_subjective_score": float(score),
                     **dict(zip(("reference", "distortion", "level"), parse_name(name, {})))})
    return rows


def read_gfiqa(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    df = pd.read_csv(labels)
    return [
        {"path": images.get(name, root / name), "original_subjective_score": float(score),
         "reference": name.lower(), "distortion": None, "level": None}
        for name, score in zip(df["img_name"].astype(str), df["mos"])
    ]


def read_cid2013(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """CID2013: an xlsx of six image sets, scored by a realignment study.

    The realigned column puts all six sets on one scale; the alternative is
    correlating within a set, and the two are not interchangeable. Source_ID
    (IS_I_C01_D01) is the filename, and the reference is the image set plus
    camera cluster — those are the groups subjects actually compared within.
    """
    df = pd.read_excel(labels, sheet_name="CID2013 MOS")
    column = next(c for c in ("Realigned MOS", "Image set specific  MOS") if c in df.columns)
    rows = []
    for source_id, score in zip(df["Source_ID"].astype(str), df[column].astype(float)):
        name = f"{source_id}.jpg"
        rows.append({
            "path": images.get(name, root / name),
            "original_subjective_score": float(score),
            "reference": "_".join(source_id.split("_")[:3]).lower(),
            "distortion": None,
            "level": None,
        })
    return rows


def read_uhdiqa(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """UHD-IQA: expert-rated 4K photographs, MOS already in [0, 1]."""
    df = pd.read_csv(labels)
    return [
        {"path": images.get(name, root / name), "original_subjective_score": float(score),
         "reference": name.lower(), "distortion": None, "level": None}
        for name, score in zip(df["image_name"].astype(str), df["quality_mos"])
    ]


def read_aigciqa2023(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """AIGCIQA2023: generated images with three MOS axes.

    `moz1` is quality — the axis this is about. The release repeats each
    image once per question asked about it, with identical scores, so rows
    are deduplicated by filename. The generation prompt is the reference:
    images from one prompt are variations on one idea.
    """
    records = json.loads(Path(labels).read_text())
    seen, rows = set(), []
    for record in records:
        name = str(record["img"])
        if name in seen:
            continue
        seen.add(name)
        rows.append({
            "path": images.get(name, root / name),
            "original_subjective_score": float(record["moz1"]),
            "reference": str(record.get("prompt", name)).lower(),
            "distortion": None,
            "level": None,
        })
    return rows


def read_csiq_label(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """`csiq_label.txt` — "<filename> <dmos>" lines, the shape CSIQ ships in.

    Note the column order is the reverse of TID2013's `mos_with_names.txt`.
    The score is a DMOS and gets flipped downstream.
    """
    rows = []
    for line in labels.read_text().splitlines():
        if not line.strip():
            continue
        name, score = line.split()
        rows.append({"path": images.get(name, root / name),
                     "original_subjective_score": float(score),
                     **dict(zip(("reference", "distortion", "level"), parse_name(name, {})))})
    return rows


def read_clive(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """CLIVE: filenames and MOS in two MATLAB structs beside each other.

    The first seven rows are the calibration images every subject scored
    before the study proper, and the release excludes them from its 1,162.
    """
    from scipy.io import loadmat

    names = [str(entry[0]) for entry in
             loadmat(labels.parent / "AllImages_release.mat")["AllImages_release"].ravel()]
    scores = loadmat(labels)["AllMOS_release"].ravel()
    return [
        {"path": images.get(name, root / name), "original_subjective_score": float(score),
         "reference": name.split(".")[0].lower(), "distortion": None, "level": None}
        for name, score in zip(names[7:], scores[7:])
    ]


def read_agiqa3k(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """AGIQA-3K: `mos_quality` is the axis this is about, `mos_align` is not.

    The archive on the mirror carries images only; `download_data.py` fetches
    this table from the dataset's own repository. The generation prompt is the
    reference — images from one prompt are variations on one idea.
    """
    df = pd.read_csv(labels)
    return [
        {"path": images.get(name, root / name), "original_subjective_score": float(score),
         "reference": str(prompt).lower(), "distortion": None, "level": None}
        for name, score, prompt in zip(df["name"].astype(str), df["mos_quality"], df["prompt"])
    ]


def read_pipal(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """PIPAL: one `<reference>.txt` of "name,elo" per reference, plus `val_label.txt`.

    `Aaaaa_bb_cc` carries the reference, the distortion class and the variant,
    so the filename is the only metadata needed. Class `10` is the NTIRE
    validation split — a separate set of references, which the training recipe
    excludes; it stays in the table under that class so it can be filtered.
    """
    rows = []
    for path in sorted((labels.parent / "Train_Label").glob("*.txt")) + [labels]:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            name, score = (field.strip() for field in line.split(","))
            rows.append({"path": images.get(name, root / name),
                         "original_subjective_score": float(score),
                         **dict(zip(("reference", "distortion", "level"), parse_name(name, {})))})
    return rows


def read_spaq(root: Path, labels: Path, images: dict[str, Path]) -> list[dict]:
    """SPAQ: an xlsx of MOS beside five perceptual attribute scores.

    Only the overall MOS is read. The scene categories and the EXIF tags sit in
    their own workbooks in the same directory, unread here — a scene-grouped
    split needs them, and that is a change to this reader.
    """
    df = pd.read_excel(labels)
    return [
        {"path": images.get(name, root / name), "original_subjective_score": float(score),
         "reference": name.split(".")[0].lower(), "distortion": None, "level": None}
        for name, score in zip(df["Image name"].astype(str), df["MOS"])
    ]


# Label file to look for, and the reader for it. First match wins, so the
# unified data.csv takes precedence wherever a release ships one.
READERS = (
    ("data.csv", read_unified),
    ("dmos.csv", read_kadid),
    ("koniq10k_scores_and_distributions.csv", read_koniq),
    ("mos_with_names.txt", read_tid2013),
    ("csiq_label.txt", read_csiq_label),
    ("mos_val_rating.csv", read_gfiqa),
    ("CID2013 data*.xlsx", read_cid2013),
    ("MOS and Image attribute scores.xlsx", read_spaq),
    ("AllMOS_release.mat", read_clive),
    ("val_label.txt", read_pipal),
    ("agiqa3k_data.csv", read_agiqa3k),
    ("uhd-iqa-metadata.csv", read_uhdiqa),
    ("aigciqa2023_labels.json", read_aigciqa2023),
    ("AIGIQA2023.json", read_aigciqa2023),
)


def prepare(root: Path) -> pd.DataFrame:
    """One dataset directory -> one dataframe with the standard columns."""
    root = root.expanduser().resolve()
    for filename, reader in READERS:
        hits = sorted(root.glob(f"**/{filename}"))
        if hits:
            images = index_images(root)
            rows = reader(root, hits[0], images)
            break
    else:
        raise FileNotFoundError(
            f"no label file under {root}; looked for {[name for name, _ in READERS]}"
        )

    df = pd.DataFrame(rows)
    df["dataset"] = root.name.lower()
    # Prefixed with the dataset: KADID and TID2013 both call a reference "i01",
    # and in a combined CSV `split_by` would treat those as one picture and put
    # rows from two datasets on the same side for the wrong reason.
    df["reference"] = df["dataset"] + "/" + df["reference"].astype(str)
    df["path"] = df["path"].map(str)
    df["group"] = [assign_group(d, t) for d, t in zip(df["dataset"], df["distortion"])]

    scores = df["original_subjective_score"].to_numpy(dtype=float)
    if df["dataset"].iloc[0] in FLIPPED:
        scores = -scores
    low, high = scores.min(), scores.max()
    if high - low < 1e-12:
        raise ValueError(f"{root.name}: every score is identical, cannot normalize")
    # One scale per dataset, applied to every row. Not per reference, even
    # for PIPAL, whose Elo ratings only rank within a reference: the Elo unit
    # is shared — 200 points is 76% preference in any group — and rescaling
    # each reference separately would destroy that while, measured, changing
    # nothing about what a model learns.
    df["scaled_subjective_score"] = (scores - low) / (high - low)

    missing = ~df["path"].map(lambda p: Path(p).exists())
    if missing.any():
        print(f"  {int(missing.sum())} of {len(df)} images not found on disk, dropped")
        df = df[~missing]
    if df.empty:
        raise FileNotFoundError(
            f"{root}: a label file was found but none of its images are on disk"
        )
    return df[
        [
            "path",
            "original_subjective_score",
            "scaled_subjective_score",
            "dataset",
            "reference",
            "distortion",
            "level",
            "group",
        ]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="dataset directories download_data.py unpacked")
    ap.add_argument("--out", default=None, help="one CSV for all of them (default: one each)")
    args = ap.parse_args()

    frames = []
    for root in args.roots:
        path = Path(root)
        print(f"{path.name}:")
        try:
            df = prepare(path)
        except FileNotFoundError as problem:
            # `~/iqa-data/*/` also matches `archives/`, which download_data.py
            # leaves behind. One directory without labels should not take the
            # whole run down with it.
            if len(args.roots) == 1:
                raise
            print(f"  skipped — {problem}")
            continue
        groups = df["group"].dropna().unique()
        print(f"  {len(df)} rows, {df['reference'].nunique()} references, "
              f"groups: {', '.join(sorted(groups)) if len(groups) else 'none (unmapped)'}, "
              f"score {df['original_subjective_score'].min():.2f}"
              f"..{df['original_subjective_score'].max():.2f}")
        if args.out is None:
            target = path.expanduser() / "labels.csv"
            df.to_csv(target, index=False)
            print(f"  -> {target}")
        frames.append(df)

    if args.out:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(args.out, index=False)
        print(f"\n{len(combined)} rows from {len(frames)} datasets -> {args.out}")


if __name__ == "__main__":
    main()
