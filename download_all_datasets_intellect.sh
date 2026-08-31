#!/usr/bin/env sh
# Download every training and held-out dataset described in datasets.md.
# Usage: ./download_all_datasets_intellect.sh [data-root]

set -eu

data_root=${1:-conditioned-iqa/data}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
downloader="$script_dir/download_data.py"

# Training datasets
for dataset in kadid10k spaq gfiqa20k pipal aigciqa2023; do
    python "$downloader" "$dataset" --data-root "$data_root"
done

# Held-out datasets
for dataset in tid2013 csiq cid2013 koniq10k clive agiqa3k uhdiqa; do
    python "$downloader" "$dataset" --data-root "$data_root"
done
