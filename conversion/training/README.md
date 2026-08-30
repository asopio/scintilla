# Training subproject

This directory isolates figure-detector fine-tuning dependencies from the main inference environment.

Note: QLoRA via bitsandbytes currently requires a Torch CUDA build that bitsandbytes supports. This training env uses Python 3.13 and pins Torch 2.8.0 / torchvision 0.23.0 to avoid pulling a newer cu130 build that bitsandbytes 0.47 cannot load.

## Install training environment

```bash
pixi install -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml
```

## Run training

If a dataset directory has moved or contains stale rendered-image paths, regenerate it before training:

```bash
pixi exec -- python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/build_figure_detection_dataset.py \
  --pdf-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex_recent \
  --output-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex_recent/arxiv_figure_dataset \
  --write-bbox-images

pixi exec -- python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/build_figure_detection_dataset.py \
  --pdf-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ph_recent \
  --output-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ph_recent/arxiv_figure_dataset \
  --write-bbox-images
```

Then run training:

```bash
pixi run -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml \
  python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/train_qwen_figure_detector.py \
  --dataset-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex_recent/arxiv_figure_dataset \
  --val-dataset-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ph_recent/arxiv_figure_dataset \
  --output-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/runs/qwen-figure-detector \
  --model Qwen/Qwen3.6-35B-A3B \
  --use-qlora \
  --gradient-checkpointing \
  --max-length 512 \
  --max-image-size 512 \
  --gradient-accumulation-steps 16 \
  --lora-r 4 \
  --lora-alpha 8 \
  --val-sample-count 4 \
  --lora-target-modules q_proj,k_proj,v_proj,o_proj,linear_attn.in_proj_qkv
```

Equivalent launcher:

```bash
bash /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/run_finetuning.sh
```

## Evaluation

To evaluate a LoRA adapter directly against a validation Vortex file, run:

```bash
pixi run -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml \
  python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/eval_qwen_figure_detector.py \
  --base-model Qwen/Qwen3.6-35B-A3B \
  --adapter-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/runs/qwen-figure-detector-v1/adapter \
  --vortex-file /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ph.vortex \
  --output-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/eval/qwen-figure-detector-v1 \
  --num-gpus 1 \
  --max-length 512 \
  --max-image-size 512
```

The evaluation writes `predictions.json` and `summary.json` under `ngpu-<N>/`.

Per page, the metric is:

```text
score = overlap_area / total_area
```

where:

```text
total_area = 0.5 * (sum(pred box areas) + sum(gt box areas))
```

If a page has no ground-truth boxes, the score is `1` when no boxes are predicted and `0` otherwise.

Kubernetes pod spec:

```text
training/qwen_figure_finetune_eval.yaml
```

## Merge adapter into a standalone checkpoint

The training run saves a LoRA adapter under `<output-dir>/adapter`. To produce a standalone merged checkpoint:

```bash
pixi run -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml \
  python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/merge_qwen_figure_adapter.py \
  --base-model Qwen/Qwen3.6-35B-A3B \
  --adapter-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/runs/qwen-figure-detector-v1/adapter \
  --output-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/merged/qwen-figure-detector-v1
```

This saves a merged Hugging Face checkpoint plus the matching processor.

## Quantization

The training pixi environment now includes NVIDIA Model Optimizer with Hugging Face support. The intended deployment flow is:

1. Merge the adapter into the base model.
2. Quantize the merged checkpoint with Model Optimizer PTQ.
3. Export a unified Hugging Face checkpoint for vLLM / TensorRT-LLM style serving.
4. Re-run `eval_qwen_figure_detector.py` on the quantized checkpoint to compare against the merged BF16 model.

Run PTQ with:

```bash
pixi run -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml \
  python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/quantize_qwen_figure_detector.py \
  --merged-model-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/merged/qwen-figure-detector-v1 \
  --export-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/quantized/qwen-figure-detector-v1-nvfp4 \
  --calib-vortex-file /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex.vortex \
  --qformat nvfp4_experts_only \
  --kv-cache-qformat fp8_cast \
  --calib-size 256 \
  --batch-size 1 \
  --max-length 512 \
  --max-image-size 512
```

Notes:
- The calibration set is task-specific and comes from the figure-detection Vortex file rather than a generic text corpus.
- The script reuses the same image+prompt formatting used in training and evaluation.
- For Qwen VLMs, the intent is to quantize the language-model / transformer linear layers while keeping the vision stack in high precision.
- Start with `nvfp4_experts_only` or `nvfp4_mlp_only` before trying broader NVFP4 presets.

Kubernetes pod spec to run merge + quantization in one pod:

```text
training/qwen_figure_merge_and_quantize.yaml
```

## Trackio

The trainer reports to Trackio via Hugging Face Transformers `report_to="trackio"`.

View the local dashboard with:

```bash
pixi run -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml trackio show
```

Optional remote sync:

```bash
pixi run -m /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/pixi.toml \
  python /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/train_qwen_figure_detector.py \
  --dataset-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/downloads/arxiv_hep_ex_recent/arxiv_figure_dataset \
  --output-dir /global/cfs/cdirs/atlas/asopio/llm-doc-converter/training/runs/qwen-figure-detector \
  --trackio-project qwen-figure-detector \
  --trackio-space-id your-org/your-space \
  --trackio-dataset-id your-org/your-dataset
```
