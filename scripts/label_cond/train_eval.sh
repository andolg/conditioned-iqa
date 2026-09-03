#!/usr/bin/env bash
set -e

scripts/label_cond/train.sh

scripts/label_cond/eval.sh
