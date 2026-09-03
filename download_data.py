"""Download an IQA dataset and unpack it.

    python download_data.py --list
    python download_data.py kadid10k --data-root ~/iqa-data

Archives come from the pyiqa mirror on Hugging Face, which republishes the
public IQA datasets in one place. Downloads resume, so an interrupted run
picks up where it stopped.

Transfers run in parallel segments rather than one stream. That is not
premature optimization: the CDN throttles a single sustained connection —
measured at 0.2 MB/s against 15 MB/s on a fresh one — which turns a 3 GB
archive into an afternoon. Use `--connections 1` if a proxy dislikes ranges.

All twelve together are about 67 GB of archive and 69 GB unpacked, and the
archive is deleted once it is unpacked. Start with kadid10k.

Licences travel with the data, not with this script. SPAQ is research and
educational use only; the LIVE sets require acknowledging the UT Austin LIVE
lab; several forbid redistribution. Read the terms of whatever you download.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import threading
import urllib.request
import zipfile
from pathlib import Path

MIRROR = "https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main"

# name -> (archive, size in GB, what it is, scale and direction)
# Sizes are the archive as the server reports it, not the unpacked directory.
# Everything here is on the pyiqa mirror; the three below it are not.
DATASETS = {
    "kadid10k": ("kadid10k.tgz", 2.9, "81 references x 25 distortions x 5 levels",
                 "MOS 1-5, higher = better (the file calls it dmos)"),
    "koniq10k": ("koniq10k.tgz", 5.9, "10,073 in-the-wild photos", "MOS 1-5, higher = better"),
    "spaq": ("spaq.tgz", 32.4, "11,125 smartphone photos", "MOS 0-100, higher = better"),
    "csiq": ("csiq.tgz", 0.4, "30 references x 6 distortions",
             "DMOS 0-1, higher = WORSE — flip it"),
    "tid2013": ("tid2013.tgz", 1.1, "25 references x 24 distortions", "MOS 0-9, higher = better"),
    "clive": ("live_challenge.tgz", 0.3, "1,162 authentic photos", "MOS 0-100, higher = better"),
    "agiqa3k": ("AGIQA-3K.zip", 0.1, "2,982 generated images (labels below)",
                "quality MOS 0-5, higher = better"),
    "pipal": ("pipal.tar", 5.4, "restoration and GAN artifacts", "Elo-style, higher = better"),
    "gfiqa20k": ("gfiqa-20k.tgz", 7.2, "20,000 face photos", "MOS 0-1, higher = better"),
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
        "uhd-iqa-database.zip", 10.0, "6,073 expert-rated 4K photos",
        "MOS 0-1, higher = better",
    ),
    "aigciqa2023": (
        "https://huggingface.co/datasets/IntMeGroup/AIGCIQA2023/resolve/main/allimg.zip",
        "aigciqa2023_images.zip", 0.8, "2,400 generated images (labels below)",
        "quality z-score, higher = better",
    ),
}

# Label files that live beside the images rather than inside the archive.
# AGIQA-3K's archive is images only — without this the dataset cannot be
# prepared at all — and UHD-IQA and AIGCIQA2023 publish theirs separately.
EXTRA_FILES = {
    "uhdiqa": ["https://datasets.vqa.mmsp-kn.de/archives/UHD-IQA/uhd-iqa-metadata.zip"],
    "aigciqa2023": [
        "https://huggingface.co/datasets/IntMeGroup/AIGCIQA2023/resolve/main/AIGIQA2023.json"
    ],
    "agiqa3k": [
        "https://raw.githubusercontent.com/lcysyzxdxc/AGIQA-3k-Database/main/data.csv"
        "#agiqa3k_data.csv"
    ],
    "cid2013": [],
}

CHUNK = 32 * 1024 * 1024


def remote_size(url: str) -> int | None:
    """Content-Length of the target, or None when the server will not say."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            return int(response.headers["Content-Length"])
    except Exception:
        return None


def download(url: str, target: Path, connections: int) -> None:
    """Fetch `url` into `target`, resuming and using parallel byte ranges.

    Progress is recorded chunk by chunk in a sidecar file, so an interrupted
    run resumes without re-reading what it already has. A server that will not
    report a size or serve ranges falls back to one stream.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    ledger = target.with_name(target.name + ".progress")

    # The file is opened at its full length before anything is written into it,
    # so its size says nothing about how much of it has arrived. Only the
    # ledger does, and it is removed when the last segment lands — which makes
    # "right size and no ledger" the one safe definition of complete.
    if total and target.exists() and target.stat().st_size == total and not ledger.exists():
        print(f"already complete: {target}")
        return
    if total is None or connections == 1:
        stream(url, target)
        return

    segments = (total + CHUNK - 1) // CHUNK
    done: set[int] = set()
    if target.exists() and ledger.exists():
        done = {int(line) for line in ledger.read_text().split() if line.strip()}
    else:
        ledger.unlink(missing_ok=True)
    if len(done) >= segments:
        ledger.unlink(missing_ok=True)
        print(f"already complete: {target}")
        return

    handle = os.open(target, os.O_RDWR | os.O_CREAT)
    os.ftruncate(handle, total)
    # Created before the first byte arrives, not after the first segment lands:
    # an interrupt in the first seconds must still leave the mark that says
    # this file is a hole, not an archive.
    ledger.touch()
    pending = [index for index in range(segments) if index not in done]
    lock = threading.Lock()
    transferred = [len(done) * CHUNK]
    failures: list[str] = []

    def worker() -> None:
        while True:
            with lock:
                if not pending or failures:
                    return
                index = pending.pop(0)
            low = index * CHUNK
            high = min(low + CHUNK, total) - 1
            for attempt in range(6):
                try:
                    request = urllib.request.Request(url, headers={"Range": f"bytes={low}-{high}"})
                    position = low
                    with urllib.request.urlopen(request, timeout=120) as response:
                        while chunk := response.read(1 << 20):
                            os.pwrite(handle, chunk, position)
                            position += len(chunk)
                    if position != high + 1:
                        continue
                    with lock:
                        transferred[0] += high + 1 - low
                        with ledger.open("a") as record:
                            record.write(f"{index}\n")
                        share = min(transferred[0], total) / total
                        print(f"\r  {share:6.1%}  {transferred[0] / 2**30:.2f} GB", end="", flush=True)
                    break
                except Exception:
                    continue
            else:
                with lock:
                    failures.append(f"segment {index} failed after 6 attempts")

    threads = [threading.Thread(target=worker) for _ in range(connections)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    os.close(handle)
    print()
    if failures:
        raise SystemExit(failures[0] + " — re-run to resume from here")
    ledger.unlink(missing_ok=True)


def stream(url: str, target: Path) -> None:
    """One connection, via curl, resuming whatever is already on disk."""
    result = subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "5", "--progress-bar", "-o", str(target), url]
    )
    if result.returncode != 0:
        raise SystemExit(
            f"download failed (curl {result.returncode}). The partial file is kept, so "
            "re-running resumes. If nothing transfers at all, your network may be blocking "
            "the CDN that serves file content while allowing huggingface.co itself."
        )


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
            # `data` refuses absolute paths and links pointing outside the
            # target. It is the default from Python 3.14 and a warning before.
            t.extractall(target, filter="data")


def fetch_extras(dataset: str, root: Path, connections: int) -> None:
    """Label files that ship apart from the images, straight into the dataset.

    A URL may carry `#name` to say what the file should be called — AGIQA-3K's
    is published as a bare `data.csv`, which is also the name of an unrelated
    format `prepare_data.py` looks for.
    """
    target_dir = root / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    for extra in EXTRA_FILES.get(dataset, []):
        url, _, rename = extra.partition("#")
        name = rename or url.rsplit("/", 1)[-1]
        print(f"fetching {name}")
        destination = target_dir / name
        download(url, destination, connections)
        if name.endswith(".zip"):
            unpack(destination, target_dir)
            destination.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", choices=sorted({**DATASETS, **ELSEWHERE}))
    ap.add_argument("--data-root", default="~/iqa-data")
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument("--connections", type=int, default=8,
                    help="parallel byte-range connections; 1 falls back to a single stream")
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

    print(f"downloading {archive_name} (~{size:.1f} GB) -> {archive}")
    download(url, archive, max(1, args.connections))

    fetch_extras(args.dataset, root, max(1, args.connections))

    print(f"unpacking into {root / args.dataset}")
    unpack(archive, root / args.dataset)
    if not args.keep_archive:
        archive.unlink()
        # Leave no empty `archives/` behind: `prepare_data.py ~/iqa-data/*/`
        # would otherwise walk into it.
        if not any(archive.parent.iterdir()):
            archive.parent.rmdir()
    print(f"done: {root / args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
