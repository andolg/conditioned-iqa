CACHE_REPO="$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336"
REVISION="$(cat "$CACHE_REPO/refs/main")"

python3 train.py \
  --data ~/conditioned-iqa/data/kadid10k/labels.csv \
  --backbone clip-large \
  --device cuda:7 \
  --weights "$CACHE_REPO/snapshots/$REVISION" \
  --save-dir ./weights \
  --name baseline_clip_large \
  "$@"
