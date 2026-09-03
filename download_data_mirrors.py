"""Download IQA datasets through Hugging Face mirrors usable from China.

    python download_dataset_mirrors.py --list
    python download_dataset_mirrors.py kadid10k --data-root ~/iqa-data
    python download_dataset_mirrors.py kadid10k --mirror https://my-hf-mirror.example

The downloader tries HF-Mirror's alternate and primary endpoints, then finally
the official Hugging Face Hub.  ``--mirror`` can be repeated to put custom
endpoints before those defaults.  Partial downloads and the progress ledger
are shared between attempts, so switching endpoints does not throw away
completed chunks.

Dataset metadata, resumable segmented downloads, archive extraction, and the
archive size constant all come from ``download_data.py``.  Non-Hugging-Face
sources (Zenodo, UHD-IQA's home, and GitHub label files) remain unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import download_data as original


HF_ENDPOINTS = (
    "https://alpha.hf-mirror.com",
    "https://hf-mirror.com",
    "https://huggingface.co",
)
IQA_REPOSITORY_PATH = "/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main"


def normalize_endpoint(endpoint: str) -> str:
    """Return a validated Hugging Face-compatible endpoint without a slash."""
    endpoint = endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise argparse.ArgumentTypeError(
            f"invalid mirror endpoint {endpoint!r}; expected http(s)://host"
        )
    if parsed.path or parsed.query or parsed.fragment:
        raise argparse.ArgumentTypeError(
            f"invalid mirror endpoint {endpoint!r}; do not include a path"
        )
    return endpoint


def endpoint_urls(hf_url: str, endpoints: list[str]) -> list[str]:
    """Make equivalent URLs at each endpoint for an official HF URL."""
    parsed = urlsplit(hf_url)
    if parsed.netloc.lower() not in {"huggingface.co", "www.huggingface.co"}:
        return [hf_url]

    urls = []
    for endpoint in endpoints:
        mirror = urlsplit(endpoint)
        urls.append(
            urlunsplit((mirror.scheme, mirror.netloc, parsed.path, parsed.query, parsed.fragment))
        )
    return urls


def download_from_candidates(
    urls: list[str], target: Path, connections: int
) -> None:
    """Try compatible sources in order while preserving resumable state."""
    failures: list[str] = []
    for number, url in enumerate(urls, start=1):
        host = urlsplit(url).netloc
        print(f"source {number}/{len(urls)}: {host}")
        try:
            original.download(url, target, connections)
            return
        except SystemExit as error:
            failures.append(f"{host}: {error}")
            if number < len(urls):
                print(f"  failed: {error}\n  trying the next source...")

    details = "\n  ".join(failures)
    raise SystemExit(
        "download failed at every source; partial data was kept for a later resume:\n"
        f"  {details}"
    )


def fetch_extras(
    dataset: str, root: Path, connections: int, endpoints: list[str]
) -> None:
    """Fetch separately published labels, mirroring only Hugging Face URLs."""
    target_dir = root / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    for extra in original.EXTRA_FILES.get(dataset, []):
        url, _, rename = extra.partition("#")
        name = rename or url.rsplit("/", 1)[-1]
        print(f"fetching {name}")
        destination = target_dir / name
        download_from_candidates(
            endpoint_urls(url, endpoints), destination, connections
        )
        if name.endswith(".zip"):
            original.unpack(destination, target_dir)
            destination.unlink()


def mirror_order(custom: list[str], official_fallback: bool) -> list[str]:
    """Put user endpoints first and remove duplicates without reordering."""
    endpoints = [*custom, *HF_ENDPOINTS]
    if not official_fallback:
        endpoints = [url for url in endpoints if url != "https://huggingface.co"]
    return list(dict.fromkeys(endpoints))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "dataset", nargs="?", choices=sorted({**original.DATASETS, **original.ELSEWHERE})
    )
    ap.add_argument("--data-root", default="~/iqa-data")
    ap.add_argument("--list", action="store_true", help="show what is available")
    ap.add_argument("--keep-archive", action="store_true")
    ap.add_argument(
        "--connections",
        type=int,
        default=8,
        help="parallel byte-range connections; 1 falls back to a single stream",
    )
    ap.add_argument(
        "--mirror",
        action="append",
        default=[],
        type=normalize_endpoint,
        metavar="URL",
        help="try a custom HF-compatible endpoint first; may be repeated",
    )
    ap.add_argument(
        "--no-official-fallback",
        action="store_true",
        help="do not try huggingface.co after the China-friendly mirrors",
    )
    args = ap.parse_args()

    if args.list or not args.dataset:
        for name, (archive, size, what, scale) in original.DATASETS.items():
            print(f"{name:12s} {size:5.1f} GB  {what}\n{'':12s} {scale}")
        print("\nnot on the shared IQA repository, fetched from their own homes:")
        for name, (url, archive, size, what, scale) in original.ELSEWHERE.items():
            print(f"{name:12s} {size:5.1f} GB  {what}\n{'':12s} {scale}")
        return 0

    endpoints = mirror_order(args.mirror, not args.no_official_fallback)
    connections = max(1, args.connections)
    root = Path(args.data_root).expanduser()

    if args.dataset in original.DATASETS:
        archive_name, size, _, _ = original.DATASETS[args.dataset]
        official_url = f"https://huggingface.co{IQA_REPOSITORY_PATH}/{archive_name}"
        urls = endpoint_urls(official_url, endpoints)
    else:
        official_url, archive_name, size, _, _ = original.ELSEWHERE[args.dataset]
        urls = endpoint_urls(official_url, endpoints)
    archive = root / "archives" / archive_name

    print(f"downloading {archive_name} (~{size:.1f} GB) -> {archive}")
    download_from_candidates(urls, archive, connections)

    fetch_extras(args.dataset, root, connections, endpoints)

    print(f"unpacking into {root / args.dataset}")
    original.unpack(archive, root / args.dataset)
    if not args.keep_archive:
        archive.unlink()
        if not any(archive.parent.iterdir()):
            archive.parent.rmdir()
    print(f"done: {root / args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
