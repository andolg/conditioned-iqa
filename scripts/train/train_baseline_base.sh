CACHE_REPO="$HOME/.cache/huggingface/hub/models--openai--clip-vit-base-patch16"
REVISION="$(cat "$CACHE_REPO/refs/main")"

python3 train.py \
  --data ~/conditioned-iqa/data/kadid10k/labels.csv \
  --backbone clip-base \
  --device cuda:7 \
  --weights "$CACHE_REPO/snapshots/$REVISION" \
  --save-dir ./weights \
  --name baseline_clip_base \
  "$@"
