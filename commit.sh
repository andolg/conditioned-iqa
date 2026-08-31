#!/usr/bin/env bash
set -e

GIT_NAME="Your Name"
GIT_EMAIL="you@example.com"

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 \"commit message\""
  exit 1
fi

git \
  -c user.name="$GIT_NAME" \
  -c user.email="$GIT_EMAIL" \
  commit -m "$*"