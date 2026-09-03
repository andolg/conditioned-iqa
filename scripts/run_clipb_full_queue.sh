#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
LOG="$ROOT/runs/logs/clipb-full-queue.log"
mkdir -p "$ROOT/runs/logs"

run_one() {
  local config=$1
  local run_name=$2
  local design=$3
  local description=$4
  local backbone=$5
  local baseline=$6
  echo "[$(date -Is)] starting $run_name" | tee -a "$LOG"
  bash scripts/run_advanced_experiment.sh \
    "$config" cuda:1 "$run_name" "$design" "$backbone" "$baseline" "$description" \
    | tee -a "$LOG"
  echo "[$(date -Is)] finished $run_name" | tee -a "$LOG"
}

run_one \
  configs/text_conditioning/66_clean_multi_multiview_interaction_s1.yaml \
  multiview-clean-interaction-s1 \
  "CLIP-B/16 five-view native interaction seed 1" \
  "Seed-1 confirmation of the five-view CLIP-B/16 native text interaction finalist; same clean mixture, multiscale global-plus-four-local views, reference split, and best-validation epoch." \
  "CLIP-B/16" variant

run_one \
  configs/text_conditioning/67_clean_multi_multiview_interaction_s2.yaml \
  multiview-clean-interaction-s2 \
  "CLIP-B/16 five-view native interaction seed 2" \
  "Seed-2 confirmation of the five-view CLIP-B/16 native text interaction finalist; same clean mixture, multiscale global-plus-four-local views, reference split, and best-validation epoch." \
  "CLIP-B/16" variant

run_one \
  configs/text_conditioning/63_clean_multi_multiview_capacity_baseline.yaml \
  multiview-clean-capacity-baseline-s0 \
  "CLIP-B/16 five-view capacity-matched image-only control" \
  "Capacity-matched image-only five-view CLIP-B/16 control; hidden width 1024 gives approximately 662k trainable head parameters, matching the conditioned head, with the same clean-mixture data, views, split, and seed." \
  "CLIP-B/16" baseline

run_one \
  configs/text_conditioning/64_clean_multi_multiview_uniform_interaction.yaml \
  multiview-clean-uniform-interaction-s0 \
  "CLIP-B/16 five-view uniform interaction" \
  "Uniform global-plus-four-local view pooling with native CLIP-B/16 text interaction; removes learned quality-aware view weighting while keeping the clean mixture, split, and seed fixed." \
  "CLIP-B/16" variant

run_one \
  configs/text_conditioning/65_clean_multi_multiview_mdtvsfa_interaction.yaml \
  multiview-clean-mdtvsfa-interaction-s0 \
  "CLIP-B/16 five-view calibrated interaction" \
  "Five-view CLIP-B/16 native interaction with the row-14 legacy MDTVSFA-style per-dataset calibration and ranking loss; same clean-mixture data, split, and seed." \
  "CLIP-B/16" variant

echo "[$(date -Is)] CLIP-B full queue completed" | tee -a "$LOG"
