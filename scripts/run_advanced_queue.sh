#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LOG="$ROOT/runs/logs/advanced-queue.log"
mkdir -p "$ROOT/runs/logs"
echo "[$(date -Is)] waiting for CLIP-L/SigLIP finalist runs" | tee -a "$LOG"
while pgrep -f 'best-config-(clip-large|siglip-large)-clean-interaction-s0' >/dev/null; do
  sleep 60
done
echo "[$(date -Is)] finalist runs released the GPUs; starting advanced queue" | tee -a "$LOG"

bash scripts/run_advanced_experiment.sh \
  configs/text_conditioning/57_clean_multi_multiview_interaction.yaml cuda:1 \
  multiview-clean-interaction-s0 \
  "CLIP-B/16 multi-view native interaction" "CLIP-B/16" variant \
  "Multi-view CLIP-B/16: one aspect-preserving global view plus four local tiles, learned quality-aware view weighting, and native text interaction; KADID-10k + SPAQ + AIGCIQA2023, seed 0." \
  | tee -a "$LOG"

bash scripts/run_advanced_experiment.sh \
  configs/text_conditioning/58_clean_multi_multiview_baseline.yaml cuda:1 \
  multiview-clean-baseline-s0 \
  "CLIP-B/16 multi-view image-only control" "CLIP-B/16" baseline \
  "Matched image-only multi-view CLIP-B/16 control with one aspect-preserving global view plus four local tiles and learned view weighting; same clean-mixture data, split, and seed as the conditioned run." \
  | tee -a "$LOG"

bash scripts/run_advanced_experiment.sh \
  configs/text_conditioning/59_clean_multi_adapter_interaction.yaml cuda:1 \
  adapter-clean-interaction-s0 \
  "CLIP-B/16 pooled visual adapter interaction" "CLIP-B/16" variant \
  "Current best native interaction head preceded by a zero-initialized trainable bottleneck residual adapter on pooled CLIP-B/16 features; clean mixture, seed 0." \
  | tee -a "$LOG"

bash scripts/run_advanced_experiment.sh \
  configs/text_conditioning/60_clean_multi_softmax_interaction.yaml cuda:1 \
  softmax-source-loss-interaction-s0 \
  "CLIP-B/16 softmax source-loss interaction" "CLIP-B/16" variant \
  "Native CLIP-B/16 interaction with MDTVSFA per-dataset scale alignment and softmax-adaptive source-loss weighting; clean mixture, seed 0." \
  | tee -a "$LOG"

echo "[$(date -Is)] advanced queue completed" | tee -a "$LOG"
