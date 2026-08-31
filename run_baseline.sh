CACHE_REPO="$HOME/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336"
REVISION="$(cat "$CACHE_REPO/refs/main")"

python3 train.py \
  --data ~/conditioned-iqa/data/kadid10k/labels.csv \
  --backbone clip-large \
  --weights "$CACHE_REPO/snapshots/$REVISION"