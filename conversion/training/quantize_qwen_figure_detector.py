#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.export.model_utils import get_language_model_from_vl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from train_qwen_figure_detector import SYSTEM_PROMPT, USER_PROMPT, build_examples_from_vortex, load_example_image


class CalibrationDataset(Dataset):
    def __init__(self, examples: list[Any], limit: int):
        self.examples = examples[:limit]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


class CalibrationCollator:
    def __init__(self, processor, max_length: int, max_image_size: int):
        self.processor = processor
        self.max_length = max_length
        self.max_image_size = max_image_size

    def __call__(self, features: list[Any]) -> dict[str, Any]:
        prompt_texts = []
        images = []
        for ex in features:
            image = load_example_image(ex)
            image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
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
            prompt_texts.append(self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
            images.append(image)

        batch = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize the merged Qwen figure detector with NVIDIA Model Optimizer")
    parser.add_argument("--merged-model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--calib-vortex-file", required=True)
    parser.add_argument("--qformat", default="nvfp4_experts_only")
    parser.add_argument("--kv-cache-qformat", default="fp8_cast")
    parser.add_argument("--calib-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def get_quant_cfg(qformat: str) -> dict[str, Any]:
    presets = {
        "nvfp4": mtq.NVFP4_DEFAULT_CFG,
        "nvfp4_mlp_only": mtq.NVFP4_MLP_ONLY_CFG,
        "nvfp4_experts_only": mtq.NVFP4_EXPERTS_ONLY_CFG,
        "nvfp4_omlp_only": mtq.NVFP4_OMLP_ONLY_CFG,
        "fp8": mtq.FP8_DEFAULT_CFG,
    }
    if qformat not in presets:
        raise ValueError(f"Unsupported qformat {qformat!r}. Supported: {sorted(presets)}")
    return presets[qformat]


def get_kv_cache_cfg(name: str) -> list[dict[str, Any]] | None:
    if name == "none":
        return None
    presets = {
        "fp8_cast": mtq.KV_CACHE_FP8_CAST_CFG,
        "fp8": mtq.KV_CACHE_FP8_DEFAULT_CFG,
        "nvfp4_cast": mtq.KV_CACHE_NVFP4_CAST_CFG,
        "nvfp4": mtq.KV_CACHE_NVFP4_DEFAULT_CFG,
    }
    if name not in presets:
        raise ValueError(f"Unsupported kv-cache-qformat {name!r}. Supported: {sorted(presets) + ['none']}")
    return presets[name]["quant_cfg"]


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def create_forward_loop(model, dataloader: DataLoader, is_vlm: bool):
    def forward_loop(_: torch.nn.Module) -> None:
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = move_to_device(batch, next(model.parameters()).device)
                if is_vlm:
                    model(**batch)
                else:
                    model(**batch)

    return forward_loop


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This quantization script requires CUDA")

    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.merged_model_dir, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageTextToText.from_pretrained(
        args.merged_model_dir,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map={"": 0},
    )
    model.eval()

    examples = build_examples_from_vortex(args.calib_vortex_file)
    if not examples:
        raise RuntimeError("No calibration examples were loaded from the vortex file")

    dataset = CalibrationDataset(examples, args.calib_size)
    collator = CalibrationCollator(processor, max_length=args.max_length, max_image_size=args.max_image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    quant_cfg = get_quant_cfg(args.qformat)
    kv_cache_cfg = get_kv_cache_cfg(args.kv_cache_qformat)
    if kv_cache_cfg is not None:
        quant_cfg = mtq.utils.update_quant_cfg_with_kv_cache_quant(quant_cfg, kv_cache_cfg)

    language_model_lineage = get_language_model_from_vl(model)
    language_model = language_model_lineage[-1] if language_model_lineage is not None else model

    forward_loop = create_forward_loop(model, dataloader, is_vlm=True)
    mtq.quantize(language_model, quant_cfg, forward_loop=forward_loop)

    metadata = {
        "merged_model_dir": args.merged_model_dir,
        "calib_vortex_file": args.calib_vortex_file,
        "num_calibration_examples": len(dataset),
        "qformat": args.qformat,
        "kv_cache_qformat": args.kv_cache_qformat,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_image_size": args.max_image_size,
    }
    (export_dir / "quantization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with torch.no_grad():
        export_hf_checkpoint(model, export_dir=str(export_dir))
    processor.save_pretrained(export_dir)
    print(f"Quantized model exported to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
