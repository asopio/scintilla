#!/usr/bin/env python3
"""Fine-tune a Qwen VLM for figure localization from page images."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import vortex as vx
from PIL import Image, ImageDraw
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


TRAINING_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from list_pdf_figures import render_pdf_page


DEFAULT_MODEL = "nvidia/Qwen3.6-35B-A3B-NVFP4"
SYSTEM_PROMPT = (
    "You are a vision-language model that localizes figures on PDF page images. "
    "Return only compact JSON with normalized coordinates in [0,1]."
)
USER_PROMPT = (
    "Detect all figures, plots, charts, diagrams, logos, and other non-body-text visual elements on this PDF page. "
    "Exclude ordinary body paragraphs, headings, equations, and tables unless they are visually embedded inside a figure. "
    "Return JSON exactly in the form: "
    '{"figures": [{"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0}]}'
)


def render_pdf_pages(pdf_path: str, output_dir: str, dpi: int) -> dict[int, Path]:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / "page"
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf_path, str(prefix)], check=True)
    pages: dict[int, Path] = {}
    for png in sorted(outdir.glob("page-*.png")):
        suffix = png.stem.split("-")[-1]
        try:
            pages[int(suffix)] = png
        except ValueError:
            continue
    return pages


def load_annotations(annotation_json: str) -> list[dict[str, Any]]:
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    return data["pages"]


def load_manifest(manifest_json: str) -> dict[str, Any]:
    return json.loads(Path(manifest_json).read_text(encoding="utf-8"))


def resolve_manifest_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def normalize_figures(page_record: dict[str, Any]) -> list[dict[str, float]]:
    figures: list[dict[str, float]] = []
    for fig in page_record.get("figures", []):
        for placement in fig.get("placements", []):
            box = placement.get("normalized_box")
            if not box:
                continue
            figures.append(
                {
                    "x1": round(float(box["x1"]), 6),
                    "y1": round(float(box["y1"]), 6),
                    "x2": round(float(box["x2"]), 6),
                    "y2": round(float(box["y2"]), 6),
                }
            )
    figures.sort(key=lambda b: (b["y1"], b["x1"], b["y2"], b["x2"]))
    deduped: list[dict[str, float]] = []
    seen = set()
    for box in figures:
        key = (box["x1"], box["y1"], box["x2"], box["y2"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(box)
    return deduped


@dataclass
class Example:
    image_path: Path | None
    target_text: str
    page_num: int
    page_label: Any
    source_pdf: str | None = None
    image_bytes: bytes | None = None


class FigureDetectionDataset(Dataset):
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def load_example_image(example: Example) -> Image.Image:
    if example.image_bytes is not None:
        return Image.open(io.BytesIO(example.image_bytes)).convert("RGB")
    if example.image_path is None:
        raise RuntimeError("Example does not have image_path or image_bytes")
    return Image.open(example.image_path).convert("RGB")


class QwenFigureDataCollator:
    def __init__(self, processor, max_length: int, max_image_size: int):
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size

    def __call__(self, features: list[Example]) -> dict[str, torch.Tensor]:
        texts = []
        prompt_texts = []
        target_token_lengths = []
        images = []
        for ex in features:
            image = load_example_image(ex)
            image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
            prompt_messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]
            messages = prompt_messages + [{"role": "assistant", "content": [{"type": "text", "text": ex.target_text}]}]
            prompt_text = self.processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            target_token_ids = self.processor.tokenizer(ex.target_text, add_special_tokens=False)["input_ids"]
            prompt_texts.append(prompt_text)
            texts.append(text)
            target_token_lengths.append(len(target_token_ids))
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = torch.full_like(batch["input_ids"], -100)
        attention_mask = batch.get("attention_mask")
        pad_token_id = self.processor.tokenizer.pad_token_id
        for i in range(labels.shape[0]):
            seq_len = int(attention_mask[i].sum().item()) if attention_mask is not None else labels.shape[1]
            prompt_len = int(prompt_batch["attention_mask"][i].sum().item()) if "attention_mask" in prompt_batch else seq_len
            target_len = min(target_token_lengths[i], max(0, seq_len - prompt_len))
            if target_len > 0:
                labels[i, seq_len - target_len : seq_len] = batch["input_ids"][i, seq_len - target_len : seq_len]
        if pad_token_id is not None:
            labels[batch["input_ids"] == pad_token_id] = -100
        batch["labels"] = labels
        return batch


def build_examples(pdf_path: str, annotations_path: str, rendered_dir: str, dpi: int) -> list[Example]:
    rendered_pages = render_pdf_pages(pdf_path, rendered_dir, dpi)
    page_records = load_annotations(annotations_path)
    examples: list[Example] = []
    for page_record in page_records:
        page_num = int(page_record["page"])
        image_path = rendered_pages.get(page_num)
        if image_path is None:
            continue
        target = {"figures": normalize_figures(page_record)}
        examples.append(
            Example(
                image_path=image_path,
                target_text=json.dumps(target, separators=(",", ":")),
                page_num=page_num,
                page_label=page_record.get("page_label"),
                source_pdf=Path(pdf_path).name,
                image_bytes=None,
            )
        )
    return examples


def build_examples_from_manifest(manifest_path: str) -> list[Example]:
    manifest = load_manifest(manifest_path)
    examples: list[Example] = []
    for record in manifest.get("records", []):
        image_path = resolve_manifest_path(record["image_path"])
        target = record.get("target", {"figures": []})
        examples.append(
            Example(
                image_path=image_path,
                target_text=json.dumps(target, separators=(",", ":")),
                page_num=int(record["page"]),
                page_label=record.get("page_label"),
                source_pdf=record.get("pdf_name"),
                image_bytes=None,
            )
        )
    return examples


def build_examples_from_vortex(vortex_path: str) -> list[Example]:
    table = vx.open(vortex_path).scan().read_all().to_arrow_table()
    rows = table.to_pylist()
    examples: list[Example] = []
    for row in rows:
        target_text = row.get("target_json")
        if not target_text:
            target = {"figures": json.loads(row.get("figures_json") or "[]")}
            target_text = json.dumps(target, separators=(",", ":"))
        examples.append(
            Example(
                image_path=Path(row["image_path"]) if row.get("image_path") else None,
                target_text=target_text,
                page_num=int(row["page"]),
                page_label=row.get("page_label"),
                source_pdf=row.get("pdf_name"),
                image_bytes=row.get("image_bytes"),
            )
        )
    return examples


def split_examples(examples: list[Example], val_ratio: float, seed: int) -> tuple[list[Example], list[Example]]:
    examples = examples[:]
    rng = random.Random(seed)
    rng.shuffle(examples)
    if not examples:
        return [], []
    val_size = max(1, int(math.ceil(len(examples) * val_ratio))) if len(examples) > 1 and val_ratio > 0 else 0
    val_examples = examples[:val_size]
    train_examples = examples[val_size:] if val_size < len(examples) else examples[:-1]
    if not train_examples and val_examples:
        train_examples = val_examples[:1]
        val_examples = val_examples[1:]
    return train_examples, val_examples


def parse_target_text(target_text: str) -> dict[str, Any]:
    return json.loads(target_text)


def resize_image_for_logging(image: Image.Image, max_image_size: int) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
    return resized


def draw_bbox_overlay(image: Image.Image, figures: list[dict[str, float]]) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    for idx, box in enumerate(figures, start=1):
        x1 = int(round(float(box["x1"]) * width))
        y1 = int(round((1.0 - float(box["y2"])) * height))
        x2 = int(round(float(box["x2"]) * width))
        y2 = int(round((1.0 - float(box["y1"])) * height))
        draw.rectangle((x1, y1, x2, y2), outline=(0, 128, 255), width=4)
        label = f"fig{idx}"
        tx1 = max(0, x1)
        ty1 = max(0, y1 - 18)
        tx2 = min(width, tx1 + max(60, 7 * len(label)))
        ty2 = min(height, ty1 + 18)
        draw.rectangle((tx1, ty1, tx2, ty2), fill=(255, 255, 255))
        draw.text((tx1 + 2, ty1 + 2), label, fill=(0, 128, 255))
    return overlay


class ValidationArtifactCallback(TrainerCallback):
    def __init__(self, output_dir: Path, samples: list[Example], max_image_size: int, enabled: bool):
        self.output_dir = output_dir
        self.samples = samples
        self.max_image_size = max_image_size
        self.enabled = enabled
        self.logged_steps: set[int] = set()

    def _write_validation_samples(self, step: int) -> Path | None:
        if not self.enabled or not self.samples:
            return None
        sample_dir = self.output_dir / "validation_samples" / f"step-{step}"
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict[str, Any]] = []
        for idx, example in enumerate(self.samples, start=1):
            target = parse_target_text(example.target_text)
            figures = target.get("figures", [])
            base_image = load_example_image(example)
            resized_image = resize_image_for_logging(base_image, self.max_image_size)
            overlay_image = draw_bbox_overlay(resized_image, figures)

            input_path = sample_dir / f"sample_{idx:02d}_input.png"
            overlay_path = sample_dir / f"sample_{idx:02d}_bbox.png"
            resized_image.save(input_path)
            overlay_image.save(overlay_path)

            manifest.append(
                {
                    "sample_index": idx,
                    "source_pdf": example.source_pdf,
                    "page_num": example.page_num,
                    "page_label": example.page_label,
                    "input_image": input_path.name,
                    "bbox_image": overlay_path.name,
                    "target": target,
                    "logged_resolution": {"width": resized_image.width, "height": resized_image.height},
                }
            )

        (sample_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return sample_dir

    def on_evaluate(self, args, state, control, **kwargs):
        if not self.enabled:
            return
        step = int(state.global_step)
        if step in self.logged_steps:
            return
        sample_dir = self._write_validation_samples(step)
        if sample_dir is None:
            return
        self.logged_steps.add(step)

        import trackio

        artifact = trackio.Artifact(
            name=f"validation-samples-step-{step}",
            type="validation-samples",
            metadata={"step": step},
        )
        artifact.add_dir(sample_dir)
        trackio.log_artifact(artifact, aliases=["latest-validation-samples"])

        for idx, example in enumerate(self.samples, start=1):
            input_path = sample_dir / f"sample_{idx:02d}_input.png"
            overlay_path = sample_dir / f"sample_{idx:02d}_bbox.png"
            caption_prefix = f"{example.source_pdf or 'unknown_pdf'} page {example.page_num}"
            trackio.log(
                {
                    f"validation/sample_{idx:02d}/input": trackio.Image(input_path, caption=f"{caption_prefix} input"),
                    f"validation/sample_{idx:02d}/bbox": trackio.Image(overlay_path, caption=f"{caption_prefix} bbox"),
                },
                step=step,
            )


class TorchProfilingCallback(TrainerCallback):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.events: list[dict[str, Any]] = []

    def _record(self, event: str, step: int | None = None) -> None:
        if not torch.cuda.is_available():
            return
        self.events.append(
            {
                "event": event,
                "step": step,
                "allocated_bytes": int(torch.cuda.memory_allocated()),
                "reserved_bytes": int(torch.cuda.memory_reserved()),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._record("train_begin", int(state.global_step))

    def on_step_begin(self, args, state, control, **kwargs):
        self._record("step_begin", int(state.global_step))

    def on_step_end(self, args, state, control, **kwargs):
        self._record("step_end", int(state.global_step))

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._record("log", int(state.global_step))

    def on_train_end(self, args, state, control, **kwargs):
        self._record("train_end", int(state.global_step))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "torch_memory_trace.json").write_text(json.dumps(self.events, indent=2), encoding="utf-8")
        if torch.cuda.is_available():
            (self.output_dir / "torch_memory_summary.txt").write_text(torch.cuda.memory_summary(), encoding="utf-8")


class TargetSpanLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, use_cache=False, return_dict=True)
        logits = outputs.logits if getattr(outputs, "logits", None) is not None else outputs[0]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        if not valid_mask.any():
            loss = shift_logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        target_logits = shift_logits[valid_mask]
        target_labels = shift_labels[valid_mask]
        loss = CrossEntropyLoss()(target_logits, target_labels)
        return (loss, outputs) if return_outputs else loss


def build_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }

    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    if args.use_qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["dtype"] = compute_dtype if torch.cuda.is_available() else torch.float32

    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen VLM for PDF figure detection with LoRA/QLoRA")
    parser.add_argument("--pdf", help="Source PDF used for single-document training")
    parser.add_argument("--annotations", help="JSON output from list_pdf_figures.py for single-document training")
    parser.add_argument("--dataset-dir", help="Dataset directory produced by build_figure_detection_dataset.py")
    parser.add_argument("--manifest", help="Explicit manifest.json path produced by build_figure_detection_dataset.py")
    parser.add_argument("--vortex-file", help="Training dataset stored as a .vortex file")
    parser.add_argument("--val-dataset-dir", help="Optional independent validation dataset directory produced by build_figure_detection_dataset.py")
    parser.add_argument("--val-manifest", help="Optional explicit validation manifest.json path")
    parser.add_argument("--val-vortex-file", help="Independent validation dataset stored as a .vortex file")
    parser.add_argument("--output-dir", required=True, help="Directory to save LoRA adapter/checkpoints")
    parser.add_argument("--rendered-pages-dir", default=None, help="Optional cache dir for rendered page PNGs (single-document mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model name or local path")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-image-size", type=int, default=896, help="Maximum image width/height passed to the processor")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-qlora", action="store_true", help="Enable 4-bit QLoRA with bitsandbytes NF4")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,linear_attn.in_proj_qkv",
        help="Comma-separated module names for LoRA injection",
    )
    parser.add_argument("--trackio-project", default=None, help="Optional Trackio project name")
    parser.add_argument("--trackio-space-id", default=None, help="Optional Hugging Face Space ID for Trackio sync")
    parser.add_argument("--trackio-dataset-id", default=None, help="Optional Hugging Face Dataset ID for Trackio backups")
    parser.add_argument("--val-sample-count", type=int, default=4, help="Number of validation sample images to log to Trackio during evaluation")
    parser.add_argument("--torch-profiling", action="store_true", help="Write per-step CUDA memory profiling artifacts to the output directory")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This training script requires a local CUDA GPU")

    if args.trackio_space_id:
        os.environ["TRACKIO_SPACE_ID"] = args.trackio_space_id
    if args.trackio_dataset_id:
        os.environ["TRACKIO_DATASET_ID"] = args.trackio_dataset_id

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiling_dir = output_dir / "torch_profiling"
    if args.torch_profiling and torch.cuda.is_available():
        torch.cuda.memory._record_memory_history()

    train_input_count = sum(bool(x) for x in [args.pdf or args.annotations, args.dataset_dir, args.manifest, args.vortex_file])
    if train_input_count != 1:
        raise RuntimeError("Specify exactly one training input source: single-document args, --dataset-dir, --manifest, or --vortex-file")
    if args.val_manifest and args.val_dataset_dir:
        raise RuntimeError("Use either --val-manifest or --val-dataset-dir, not both")
    if sum(bool(x) for x in [args.val_dataset_dir, args.val_manifest, args.val_vortex_file]) > 1:
        raise RuntimeError("Use at most one of --val-dataset-dir, --val-manifest, or --val-vortex-file")

    manifest_path = None
    if args.manifest:
        manifest_path = args.manifest
    elif args.dataset_dir:
        manifest_path = str(Path(args.dataset_dir) / "manifest.json")

    val_manifest_path = None
    if args.val_manifest:
        val_manifest_path = args.val_manifest
    elif args.val_dataset_dir:
        val_manifest_path = str(Path(args.val_dataset_dir) / "manifest.json")

    tmp_ctx = None
    if args.vortex_file:
        examples = build_examples_from_vortex(args.vortex_file)
    elif manifest_path:
        examples = build_examples_from_manifest(manifest_path)
    else:
        if not args.pdf or not args.annotations:
            raise RuntimeError("Single-document mode requires both --pdf and --annotations")
        if args.rendered_pages_dir:
            rendered_dir = args.rendered_pages_dir
            Path(rendered_dir).mkdir(parents=True, exist_ok=True)
        else:
            tmp_ctx = tempfile.TemporaryDirectory()
            rendered_dir = tmp_ctx.name
        examples = build_examples(args.pdf, args.annotations, rendered_dir, args.dpi)

    if not examples:
        raise RuntimeError("No training examples were built from the provided inputs")

    if args.val_vortex_file:
        train_examples = examples
        val_examples = build_examples_from_vortex(args.val_vortex_file)
    elif val_manifest_path:
        train_examples = examples
        val_examples = build_examples_from_manifest(val_manifest_path)
    else:
        train_examples, val_examples = split_examples(examples, args.val_ratio, args.seed)

    if not train_examples:
        raise RuntimeError("No training examples were available after dataset split")

    train_dataset = FigureDetectionDataset(train_examples)
    eval_dataset = FigureDetectionDataset(val_examples) if val_examples else None

    model, processor = build_model_and_processor(args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = QwenFigureDataCollator(
        processor=processor,
        max_length=args.max_length,
        max_image_size=args.max_image_size,
    )

    total_train_steps = max(
        1,
        math.ceil(len(train_dataset) / args.per_device_train_batch_size / args.gradient_accumulation_steps) * math.ceil(args.epochs),
    )
    warmup_steps = int(total_train_steps * args.warmup_ratio)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=args.trackio_project or output_dir.name,
        project=args.trackio_project or "huggingface",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="trackio",
        seed=args.seed,
    )

    callbacks: list[TrainerCallback] = []
    if args.torch_profiling:
        callbacks.append(TorchProfilingCallback(profiling_dir))
    if eval_dataset is not None and args.val_sample_count > 0:
        callbacks.append(
            ValidationArtifactCallback(
                output_dir=output_dir,
                samples=val_examples[: args.val_sample_count],
                max_image_size=args.max_image_size,
                enabled=True,
            )
        )

    trainer = TargetSpanLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks or None,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "adapter"))

    if args.torch_profiling and torch.cuda.is_available():
        torch.cuda.memory._dump_snapshot(str(profiling_dir / "torch_memory_snapshot.pickle"))
        torch.cuda.memory._record_memory_history(enabled=None)

    summary = {
        "model": args.model,
        "train_examples": len(train_examples),
        "eval_examples": len(val_examples),
        "use_qlora": args.use_qlora,
        "pdf": args.pdf,
        "annotations": args.annotations,
        "dataset_dir": args.dataset_dir,
        "manifest": manifest_path,
        "vortex_file": args.vortex_file,
        "val_dataset_dir": args.val_dataset_dir,
        "val_manifest": val_manifest_path,
        "val_vortex_file": args.val_vortex_file,
        "val_sample_count": args.val_sample_count,
        "trackio_project": args.trackio_project,
        "trackio_space_id": args.trackio_space_id,
        "trackio_dataset_id": args.trackio_dataset_id,
        "torch_profiling": args.torch_profiling,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if tmp_ctx is not None:
        tmp_ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
