"""Download an IQA dataset and unpack it.

    python download_data.py --list
    python download_data.py kadid10k --data-root ~/iqa-data

Archives come from the pyiqa mirror on Hugging Face, which republishes the
public IQA datasets in one place. Downloads resume, so an interrupted run
picks up where it stopped.

Licences travel with the data, not with this script. SPAQ is research and
educational use only; the LIVE sets require acknowledging the UT Austin LIVE
lab; several forbid redistribution. Read the terms of whatever you download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main"

# name -> (archive, size in GB, what it is, scale and direction)
# Everything here is on the pyiqa mirror; the three below it are not.
DATASETS = {
    "kadid10k": ("kadid10k.tgz", 2.9, "81 references x 25 distortions x 5 levels",
                 "MOS 1-5, higher = better (the file calls it dmos)"),
    "koniq10k": ("koniq10k.tgz", 5.9, "10,073 in-the-wild photos", "MOS 1-5, higher = better"),
    "spaq": ("spaq.tgz", 32.4, "11,125 smartphone photos", "MOS 0-100, higher = better"),
    "csiq": ("csiq.tgz", 0.9, "30 references x 6 distortions",
             "DMOS 0-1, higher = WORSE — flip it"),
    "tid2013": ("tid2013.tgz", 1.0, "25 references x 24 distortions", "MOS 0-9, higher = better"),
    "clive": ("live_challenge.tgz", 1.5, "1,162 authentic photos", "MOS 0-100, higher = better"),
    "agiqa3k": ("AGIQA-3K.zip", 1.0, "2,982 generated images", "quality MOS 0-5, higher = better"),
    "pipal": ("pipal.tar", 6.8, "restoration and GAN artifacts", "Elo-style, higher = better"),
    "gfiqa20k": ("gfiqa-20k.tgz", 7.7, "20,000 face photos", "MOS 0-1, higher = better"),
}

# Not on the mirror — each comes from its own home, so these carry full URLs.
ELSEWHERE = {
    "cid2013": (
        "https://zenodo.org/api/records/2647033/files/CID2013.7z/content",
        "CID2013.7z", 0.6, "474 consumer-camera photos in six sets",
        "realigned MOS ~0-100, higher = better",
    ),
    "uhdiqa": (
        "https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/UHD-IQA-database.zip",
        "uhd-iqa-database.zip", 10.7, "6,073 expert-rated 4K photos",
        "MOS 0-1, higher = better",
    ),
    "aigciqa2023": (
        "https://huggingface.co/datasets/IntMeGroup/AIGCIQA2023/resolve/main/allimg.zip",
        "aigciqa2023_images.zip", 0.9, "2,400 generated images (labels below)",
        "quality z-score, higher = better",
    ),
}

# Label files that live beside the images rather than inside the archive.
EXTRA_FILES = {
    "uhdiqa": ["https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/uhd-iqa-metadata.zip"],
    "aigciqa2023": [
        "https://huggingface.co/datasets/IntMeGroup/AIGCIQA2023/resolve/main/AIGIQA2023.json"
    ],
    "cid2013": [],
}


def unpack(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".7z":
        # CID2013 ships as 7z; p7zip is the one external tool this needs.
        if subprocess.run(["7z", "x", "-y", f"-o{target}", str(archive)]).returncode != 0:
            raise SystemExit(
                "unpacking the 7z archive failed — install p7zip "
                "(`brew install p7zip` / `apt install p7zip-full`)"
            )
        return
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
    else:
        with tarfile.open(archive) as t:
            t.extractall(target)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", choices=sorted({**DATASETS, **ELSEWHERE}))
    ap.add_argument("--data-root", default="~/iqa-data")
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--keep-archive", action="store_true")
    args = ap.parse_args()

    if args.list or not args.dataset:
        for name, (archive, size, what, scale) in DATASETS.items():
            print(f"{name:12s} {size:5.1f} GB  {what}\n{'':12s} {scale}")
        print("\nnot on the mirror, fetched from their own homes:")
        for name, (url, archive, size, what, scale) in ELSEWHERE.items():
            print(f"{name:12s} {size:5.1f} GB  {what}\n{'':12s} {scale}")
        return 0

    root = Path(args.data_root).expanduser()
    if args.dataset in DATASETS:
        archive_name, size, _, _ = DATASETS[args.dataset]
        url = f"{MIRROR}/{archive_name}"
    else:
        url, archive_name, size, _, _ = ELSEWHERE[args.dataset]
    archive = root / "archives" / archive_name
    archive.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading {archive_name} (~{size:.1f} GB) -> {archive}")
    result = subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "5", "--progress-bar", "-o", str(archive), url]
    )
    if result.returncode != 0:
        print(
            f"download failed (curl {result.returncode}). The partial file is kept, so "
            "re-running resumes. If nothing transfers at all, your network may be blocking "
            "the CDN that serves file content while allowing huggingface.co itself.",
            file=sys.stderr,
        )
        return 1

    for extra in EXTRA_FILES.get(args.dataset, []):
        name = extra.rsplit("/", 1)[-1]
        print(f"fetching {name}")
        subprocess.run(["curl", "-sL", "-o", str(archive.parent / name), extra])
        if name.endswith(".zip"):
            unpack(archive.parent / name, root / args.dataset)
        else:
            (root / args.dataset).mkdir(parents=True, exist_ok=True)
            (root / args.dataset / name).write_bytes((archive.parent / name).read_bytes())

    print(f"unpacking into {root / args.dataset}")
    unpack(archive, root / args.dataset)
    if not args.keep_archive:
        archive.unlink()
    print(f"done: {root / args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
