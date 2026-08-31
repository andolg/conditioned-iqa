#!/usr/bin/env sh
# Download every training and held-out dataset through China-friendly mirrors.
# Usage: ./download_all_datasets_cn.sh [data-root]

set -eu

data_root=${1:-conditioned-iqa/data}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
downloader="$script_dir/download_dataset_mirrors.py"

# Training datasets
for dataset in kadid10k spaq gfiqa20k pipal aigciqa2023; do
    python3 "$downloader" "$dataset" --data-root "$data_root"
done

# Held-out datasets
for dataset in tid2013 csiq cid2013 koniq10k clive agiqa3k uhdiqa; do
    python3 "$downloader" "$dataset" --data-root "$data_root"
done
