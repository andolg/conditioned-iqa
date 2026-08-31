#!/usr/bin/env sh
# Download every training and held-out dataset through China-friendly mirrors.
# Usage: ./download_all_datasets_cn.sh [data-root]

set -eu

data_root=${1:-conditioned-iqa/data}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
downloader="$script_dir/download_data_mirrors.py"

pids=""

cleanup() {
    [ -z "$pids" ] || kill $pids 2>/dev/null || true
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# KADID-10k is already downloaded; start every other dataset concurrently.
for dataset in spaq gfiqa20k pipal aigciqa2023 \
    tid2013 csiq cid2013 koniq10k clive agiqa3k uhdiqa; do
    echo "starting: $dataset"
    python3 "$downloader" "$dataset" --data-root "$data_root" &
    pids="$pids $!"
done

status=0
for pid in $pids; do
    wait "$pid" || status=1
done

exit "$status"
