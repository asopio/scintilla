#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_qwen_figure_detector import SYSTEM_PROMPT, USER_PROMPT, build_examples_from_vortex, load_example_image


def parse_prediction(text: str) -> tuple[dict[str, Any], str | None]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"figures": []}, "no_json_object_found"
    raw = text[start : end + 1]
    try:
        data = json.loads(raw)
    except Exception as exc:
        return {"figures": []}, f"json_decode_error: {exc}"
    figures = data.get("figures")
    if not isinstance(figures, list):
        return {"figures": []}, "figures_field_missing_or_not_list"
    cleaned = []
    for item in figures:
        if not isinstance(item, dict):
            continue
        try:
            box = {
                "x1": float(item["x1"]),
                "y1": float(item["y1"]),
                "x2": float(item["x2"]),
                "y2": float(item["y2"]),
            }
        except Exception:
            continue
        box = {k: min(1.0, max(0.0, v)) for k, v in box.items()}
        cleaned.append(box)
    cleaned.sort(key=lambda b: (b["y1"], b["x1"], b["y2"], b["x2"]))
    return {"figures": cleaned}, None


def box_area(box: dict[str, float]) -> float:
    return max(0.0, box["x2"] - box["x1"]) * max(0.0, box["y2"] - box["y1"])


def overlap_area(a: dict[str, float], b: dict[str, float]) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def page_score(pred_boxes: list[dict[str, float]], gt_boxes: list[dict[str, float]]) -> dict[str, float]:
    if not gt_boxes:
        score = 1.0 if not pred_boxes else 0.0
        return {
            "overlap_area": 0.0,
            "total_area": 0.0,
            "score": score,
        }

    overlap = 0.0
    for pred in pred_boxes:
        for gt in gt_boxes:
            overlap += overlap_area(pred, gt)

    total_area = 0.5 * (sum(box_area(box) for box in pred_boxes) + sum(box_area(box) for box in gt_boxes))
    score = overlap / total_area if total_area > 0 else 0.0
    return {
        "overlap_area": overlap,
        "total_area": total_area,
        "score": score,
    }


def load_model(base_model: str, adapter_dir: str, num_gpus: int):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device_map: str | dict[str, int] = {"": 0} if num_gpus == 1 else "auto"
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, processor


def main() -> int:
    parser = argparse.ArgumentParser(description="Run figure-detector inference on a vortex validation set")
    parser.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--vortex-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.num_gpus not in (1, 2, 3, 4):
        raise RuntimeError(f"--num-gpus must be one of 1,2,3,4, got {args.num_gpus}")
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation script requires CUDA")
    if torch.cuda.device_count() < args.num_gpus:
        raise RuntimeError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} visible")

    output_dir = Path(args.output_dir)
    run_dir = output_dir / f"ngpu-{args.num_gpus}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.base_model, args.adapter_dir, args.num_gpus)
    examples = build_examples_from_vortex(args.vortex_file)

    predictions = []
    total_overlap_area = 0.0
    total_area = 0.0
    total_score = 0.0
    parse_failures = 0

    for start in range(0, len(examples), args.batch_size):
        batch_examples = examples[start : start + args.batch_size]
        images = []
        prompt_texts = []
        for ex in batch_examples:
            image = load_example_image(ex)
            image.thumbnail((args.max_image_size, args.max_image_size), Image.Resampling.LANCZOS)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ]
            prompt_texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.append(image)

        inputs = processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        target_device = next(model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        decoded = []
        for row, prompt_len in zip(generated, prompt_lengths):
            decoded.append(processor.decode(row[int(prompt_len) :], skip_special_tokens=True))

        for ex, text in zip(batch_examples, decoded):
            pred, parse_error = parse_prediction(text)
            if parse_error is not None:
                parse_failures += 1
            gt = json.loads(ex.target_text)
            pred_boxes = pred.get("figures", [])
            gt_boxes = gt.get("figures", [])
            metrics = page_score(pred_boxes, gt_boxes)
            total_overlap_area += metrics["overlap_area"]
            total_area += metrics["total_area"]
            total_score += metrics["score"]
            predictions.append(
                {
                    "pdf_name": ex.source_pdf,
                    "page": ex.page_num,
                    "page_label": ex.page_label,
                    "ground_truth": gt,
                    "prediction": pred,
                    "raw_output": text,
                    "parse_error": parse_error,
                    "metrics": metrics,
                }
            )
            print(f"processed {len(predictions)}/{len(examples)}", flush=True)

    mean_score = total_score / len(examples) if examples else 0.0
    dataset_score = total_overlap_area / total_area if total_area > 0 else mean_score
    summary = {
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "vortex_file": args.vortex_file,
        "num_examples": len(examples),
        "num_gpus": args.num_gpus,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_image_size": args.max_image_size,
        "max_new_tokens": args.max_new_tokens,
        "parse_failures": parse_failures,
        "metrics": {
            "overlap_area": total_overlap_area,
            "total_area": total_area,
            "dataset_score": dataset_score,
            "mean_page_score": mean_score,
        },
    }

    (run_dir / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
