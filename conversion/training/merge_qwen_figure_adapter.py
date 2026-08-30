#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a Qwen figure-detector LoRA adapter into the base model")
    parser.add_argument("--base-model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = get_dtype(args.dtype)
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = {"": 0}

    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    base_model = AutoModelForImageTextToText.from_pretrained(args.base_model, **model_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
