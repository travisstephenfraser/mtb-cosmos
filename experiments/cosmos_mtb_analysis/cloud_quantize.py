"""
Quantize Cosmos-Reason2-8B to GPTQ INT8 on a cloud GPU (Vast.ai A100).

Usage:
    python cloud_quantize.py --bits 8
    python cloud_quantize.py --bits 4
    python cloud_quantize.py --bits 8 --upload
"""

import argparse
import os
import time
import torch
from pathlib import Path


def quantize_gptq(bits: int = 8, output_dir: str = None, group_size: int = 128):
    """Quantize using gptqmodel with proper Qwen3VL calibration format."""
    from gptqmodel import GPTQModel
    from gptqmodel.quantization import QuantizeConfig

    model_name = "nvidia/Cosmos-Reason2-8B"
    if output_dir is None:
        output_dir = f"/workspace/Cosmos-Reason2-8B-GPTQ-Int{bits}"

    print("=" * 60)
    print(f"Quantizing {model_name} -> GPTQ INT{bits}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({vram:.0f}GB)")

    # Step 1: Build calibration data in the format Qwen3VL expects
    print("\n[1/3] Preparing calibration data...")

    texts = [
        "Describe the terrain visible in this mountain bike trail footage.",
        "What obstacles are on the riding line? Identify rocks, roots, and drops.",
        "Assess the trail conditions: is the surface wet or dry?",
        "Rate the risk level of this trail segment from LOW to EXTREME.",
        "What is the trail surface material? Packed dirt, gravel, rock, or mixed?",
        "Is there tree canopy overhead? How is the lighting?",
        "Describe what you see to the left and right of the trail.",
        "How steep is the terrain beside the trail?",
        "Analyze the rider's speed and braking patterns in this footage.",
        "What technical skills would a rider need for this trail section?",
        "Describe the trail conditions including moisture, erosion, and maintenance.",
        "Provide a narration of this ride segment as a trail guide.",
    ]

    # Qwen3VL's prepare_dataset override calls apply_chat_template,
    # which expects message dicts with content as a list of typed items
    calibration_data = []
    for t in texts:
        # Repeat each text multiple times to hit the 256 sample minimum
        for _ in range(22):
            calibration_data.append([
                {"role": "user", "content": [{"type": "text", "text": t}]},
            ])

    print(f"  {len(calibration_data)} calibration samples")

    # Step 2: Load and quantize
    print(f"[2/3] Loading model and quantizing to INT{bits}...")
    start = time.time()

    quant_config = QuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
        sym=True,
    )

    model = GPTQModel.load(
        model_name,
        quant_config,
        torch_dtype=torch.float16,
        device="cpu",
    )

    # Pass calibration data
    model.quantize(calibration_data, batch_size=1)

    elapsed = time.time() - start
    print(f"  Quantized in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Step 3: Save
    print(f"[3/3] Saving to {output_dir}...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save(output_dir)

    # Also save the processor/tokenizer
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)

    total_size = sum(f.stat().st_size for f in Path(output_dir).rglob("*") if f.is_file())
    print(f"  Total size: {total_size / 1e9:.1f}GB")

    return output_dir


def validate(model_dir: str):
    """Quick smoke test."""
    from gptqmodel import GPTQModel
    from transformers import AutoProcessor

    print(f"\nValidating {model_dir}...")
    model = GPTQModel.load(model_dir, device="cuda:0")

    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM: {vram:.1f}GB")

    processor = AutoProcessor.from_pretrained(model_dir)
    messages = [{"role": "user", "content": "Describe a rocky trail."}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = processor.tokenizer(text, return_tensors="pt").to("cuda:0")
    output = model.generate(**tokens, max_new_tokens=50)
    response = processor.tokenizer.decode(output[0][tokens.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"  Response: {response[:150]}...")
    print("  PASSED")


def upload_to_hf(model_dir: str, repo_name: str = None, bits: int = 8):
    from huggingface_hub import HfApi
    if repo_name is None:
        repo_name = f"Cosmos-Reason2-8B-GPTQ-Int{bits}"
    api = HfApi()
    username = api.whoami()["name"]
    repo_id = f"{username}/{repo_name}"
    print(f"\nUploading to {repo_id}...")
    api.create_repo(repo_id, exist_ok=True)
    api.upload_folder(folder_path=model_dir, repo_id=repo_id)
    print(f"  Uploaded: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--validate-only", type=str, default=None)
    args = parser.parse_args()

    if args.validate_only:
        validate(args.validate_only)
    else:
        out = quantize_gptq(bits=args.bits, output_dir=args.output_dir, group_size=args.group_size)
        validate(out)
        if args.upload:
            upload_to_hf(out, bits=args.bits)

    print("\nDONE.")
