CACHE_REPO="$HOME/.cache/huggingface/hub/models--openai--clip-vit-base-patch16"
REVISION="$(cat "$CACHE_REPO/refs/main")"

python3 train.py \
  --data ~/conditioned-iqa/data/kadid10k/labels.csv \
  --backbone clip-base \
  --device cuda:1 \
  --weights "$CACHE_REPO/snapshots/$REVISION" \
  --out ~/conditioned-iqa/28a_dol/conditioned-iqa/weights/baseline_clip_base.pth