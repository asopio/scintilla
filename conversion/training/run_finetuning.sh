#!/bin/bash
# Fine-tuning configuration for figure-extraction

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TRAIN_DATASET_DIR=/global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex_recent/arxiv_figure_dataset
TRAIN_VXFILE=/global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex.vortex
VAL_DATASET_DIR=/global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ph_recent/arxiv_figure_dataset
VAL_VXFILE=/global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ph.vortex
OUTPUT_DIR=/global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/runs/qwen-figure-detector-v1
VAL_SAMPLE_COUNT=4

# Regenerate manifests/images if the dataset directory moved or image paths are stale:
# pixi exec -- python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/build_figure_detection_dataset.py \
#   --pdf-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex_recent \
#   --output-dir "$TRAIN_DATASET_DIR" \
#   --write-bbox-images
# pixi exec -- python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/build_figure_detection_dataset.py \
#   --pdf-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ph_recent \
#   --output-dir "$VAL_DATASET_DIR" \
#   --write-bbox-images

pixi run -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml \
    python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/train_qwen_figure_detector.py \
    --vortex-file "$TRAIN_VXFILE" \
    --val-vortex-file "$VAL_VXFILE" \
    --output-dir "$OUTPUT_DIR" \
    --model Qwen/Qwen3.6-35B-A3B \
    --epochs 2 \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.03 \
    --weight-decay 0.0 \
    --max-length 512 \
    --max-image-size 512 \
    --val-sample-count "$VAL_SAMPLE_COUNT" \
    --use-qlora \
    --gradient-checkpointing \
    --lora-r 4 \
    --lora-alpha 8 \
    --lora-dropout 0.05 \
    --lora-target-modules q_proj,k_proj,v_proj,o_proj,linear_attn.in_proj_qkv \
    --trackio-project qwen-figure-detector-v2 \
    --torch-profiling
    # --dataset-dir "$TRAIN_DATASET_DIR" \
    # --val-dataset-dir "$VAL_DATASET_DIR" \
