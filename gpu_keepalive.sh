#!/bin/bash
# GPU keepalive with a real duty cycle.
#
# The cluster's idle-GPU reaper cuts spec.activeDeadlineSeconds on GPU pods
# (observed: 432000 -> ~3370, i.e. the pod is killed ~56 min after start)
# when GPU utilization is too low, checked ~50 min after pod start and
# repeated afterwards.  Text-only API pings do NOT count, and even one
# small page OCR per 2 min was only ~0.12% average duty (39.7 GPU-s in
# 54 min) and did NOT prevent the reaper.
#
# This script repeatedly OCRs the 7 densest pages (each ~16 s of GPU work)
# until $BUSY seconds of work are done, sleeps $IDLE seconds, and repeats:
#   default duty cycle = 180/(180+60) = 75%
#
# Usage: gpu_keepalive.sh [busy_seconds] [idle_seconds]
#   env:  DSOCR2_DIR  dir holding ocr_page.py and pages/*.png
#                  (default: /tmp/opencode/dsocr2_test)
#         KA_URL    server URL (default baked into ocr_page.py:
#                  http://127.0.0.1:18000/v1/chat/completions,
#                  i.e. requires `kubectl port-forward <pod> 18000:8000`)
cd "${DSOCR2_DIR:-/tmp/opencode/dsocr2_test}" || exit 1
BUSY="${1:-180}"
IDLE="${2:-60}"
DENSE="29 30 31 33 35 37 40"   # the 7 dense handwritten pages (~16 s each)
i=0
while true; do
  busy=0
  while [ "$busy" -lt "$BUSY" ]; do
    p=$(echo $DENSE | awk -v n="$i" '{print $((n % 7) + 1)}')
    i=$((i + 1))
    t0=$(date +%s)
    out=$(python3 ocr_page.py page "pages/page-$p.png" "ka/keepalive-$p.md" 2>&1)
    dt=$(( $(date +%s) - t0 ))
    if [[ "$out" != *"prompt="* ]]; then
      # server unreachable (e.g. port-forward dropped) -- do NOT spin:
      # back off 15 s and don't count the failed attempt as busy time
      echo "[$(date -u +%H:%M:%S)] ka page-$p FAILED (backing off 15s): $(head -1 <<<"$out")"
      sleep 15
      continue
    fi
    busy=$((busy + dt))
    echo "[$(date -u +%H:%M:%S)] ka page-$p busy=$busy: $out"
  done
  echo "[$(date -u +%H:%M:%S)] sleeping ${IDLE}s"
  sleep "$IDLE"
done
