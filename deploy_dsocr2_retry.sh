#!/usr/bin/env bash
# Create the DS-OCR2 vLLM pod, retrying every 10s while the namespace GPU
# quota is exhausted (e.g. "exceeded quota: ... nvidia.com/gpu=60,
# limited: ... 60"). Exits 0 as soon as kubectl reports the pod as created.
set -u

FILE="vllm_deepseek_ocr2.yaml"
SLEEP=10

while true; do
  if out="$(kubectl create -f "$FILE" 2>&1)"; then
    echo "SUCCESS: $out"
    exit 0
  fi
  if [[ "$out" == *"exceeded quota"* ]]; then
    echo "[$(date '+%H:%M:%S')] GPU quota exhausted - retrying in ${SLEEP}s"
  else
    echo "[$(date '+%H:%M:%S')] kubectl create failed - retrying in ${SLEEP}s"
    echo "$out" | sed 's/^/    /'
  fi
  sleep "$SLEEP"
done
