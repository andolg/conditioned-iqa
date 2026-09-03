#!/usr/bin/env sh
# Download every training and held-out dataset described in datasets.md.
# Usage: ./scripts/download_all_datasets_intellect.sh [data-root]

set -eu

data_root=${1:-conditioned-iqa/data}
downloader=./download_data.py

pids=""

cleanup() {
    [ -z "$pids" ] || kill $pids 2>/dev/null || true
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# Start every training and held-out dataset download concurrently.
for dataset in kadid10k spaq gfiqa20k pipal aigciqa2023 \
    tid2013 csiq cid2013 koniq10k clive agiqa3k uhdiqa; do
    echo "starting: $dataset"
    python "$downloader" "$dataset" --data-root "$data_root" &
    pids="$pids $!"
done

status=0
for pid in $pids; do
    wait "$pid" || status=1
done

exit "$status"
