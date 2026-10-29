#!/usr/bin/env bash
# Benchmark driver: run both PDF->markdown pipelines over the testing set
# and lay out results for evaluation:
#
#   $OUT/qwen/<stem>.md   (+ $OUT/qwen/figures/fig_<stem>_pNNNN_K.png)
#   $OUT/dsocr/<stem>.md  (+ $OUT/dsocr/figures/<stem>-page-NN-fKK.png)
#
# The Qwen side is a single batch invocation of pdf_to_markdown_agentic_v2.py
# (it handles the whole input dir, page-file resume, doc-level skip).
# The DeepSeek-OCR2 side loops pdf_to_markdown_dsocr2.py per paper into
# $OUT/dsocr/.staging/<stem>/ and consolidates finished papers into the
# flat layout (final md renamed, figure files namespaced with the stem).
#
# Resumable: papers whose final $OUT/<side>/<stem>.md exists are skipped;
# an interrupted DeepSeek-OCR2 run resumes from its per-paper raw_pages/
# cache.  Re-run this script after an interruption to continue.
#
# Env overrides:
#   OUT, INPUT_DIR,
#   QWEN_URL, QWEN_MODEL, QWEN_WORKERS,
#   DSO_CR_URL, DSO_CR_MODEL, DSO_CR_WORKERS,
#   QWEN_ONLY=1, DSO_CR_ONLY=1
set -u
TESTING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONV_DIR="$(cd "$TESTING_DIR/../.." && pwd)/conversion"
OUT="${OUT:-$TESTING_DIR/eval_md}"
INPUT_DIR="${INPUT_DIR:-$TESTING_DIR/pdf}"
QWEN_URL="${QWEN_URL:-http://127.0.0.1:8008/v1}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}"
QWEN_WORKERS="${QWEN_WORKERS:-12}"
DSOCR_URL="${DSOCR_URL:-http://127.0.0.1:18000/v1}"
DSOCR_MODEL="${DSOCR_MODEL:-deepseek-ocr2}"
DSOCR_WORKERS="${DSOCR_WORKERS:-8}"
PDFINFO="$CONV_DIR/.pixi/envs/default/bin/pdfinfo"

mkdir -p "$OUT/qwen" "$OUT/dsocr/figures" "$OUT/dsocr/.staging" "$OUT/logs"

run_qwen() {
  cd "$CONV_DIR"
  pixi run python pdf_to_markdown_agentic_v2.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUT/qwen" \
    --llm-base-url "$QWEN_URL" \
    --model "$QWEN_MODEL" \
    --workers "$QWEN_WORKERS"
}

run_dsocr() {
  cd "$CONV_DIR"
  local f stem staging fig failures=0
  for f in "$INPUT_DIR"/*.pdf; do
    stem="$(basename "$f" .pdf)"
    if [ -f "$OUT/dsocr/$stem.md" ]; then
      echo "[$stem] done, skipping"
      continue
    fi
    if ! "$PDFINFO" "$f" >/dev/null 2>&1; then
      echo "[$stem] not a parseable PDF, skipping"
      continue
    fi
    staging="$OUT/dsocr/.staging/$stem"
    echo "=== [$stem] start $(date +%H:%M:%S) ==="
    if pixi run python pdf_to_markdown_dsocr2.py "$f" \
        -o "$staging" \
        --llm-base-url "$DSOCR_URL" \
        --model "$DSOCR_MODEL" \
        --npages-parallel "$DSOCR_WORKERS" \
        --no-cache-dir --continue-on-error \
        > "$OUT/logs/dsocr-$stem.log" 2>&1; then
      # consolidate into the flat benchmark layout
      sed "s#](../figures/#](figures/$stem-#g" \
        "$staging/md/${stem}_dsocr2.md" > "$OUT/dsocr/$stem.md"
      for fig in "$staging"/figures/*.png; do
        [ -f "$fig" ] && mv "$fig" "$OUT/dsocr/figures/$stem-$(basename "$fig")"
      done
      echo "=== [$stem] done $(date +%H:%M:%S): $(grep -c '!\[' "$OUT/dsocr/$stem.md") figure(s) ==="
    else
      failures=$((failures + 1))
      echo "=== [$stem] FAILED ($(tail -1 "$OUT/logs/dsocr-$stem.log")) -- retry on next run ==="
    fi
  done
  echo "=== dsocr sweep finished $(date +%H:%M:%S): $failures failure(s) ==="
  [ "$failures" -eq 0 ]
}

pids=()
if [ "${DSOCR_ONLY:-0}" != "1" ]; then
  run_qwen > "$OUT/logs/qwen.log" 2>&1 &
  pids+=($!)
  echo "qwen side started (pid $!) -> $OUT/logs/qwen.log"
fi
if [ "${QWEN_ONLY:-0}" != "1" ]; then
  run_dsocr > "$OUT/logs/dsocr.log" 2>&1 &
  pids+=($!)
  echo "dsocr side started (pid $!) -> $OUT/logs/dsocr.log"
fi

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done
echo "benchmark driver finished (rc=$rc) at $(date)"
exit $rc
