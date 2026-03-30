"""
Quantize Cosmos-Reason2-8B to GPTQ INT8 for local inference on RTX 5070 Ti (12GB).

BF16 8B = ~16GB VRAM (won't fit)
GPTQ INT8 = ~8GB VRAM (fits with ~4GB for context)
GPTQ INT4 = ~4.5GB VRAM (fallback if INT8 is too tight)

Requirements:
    pip install auto-gptq optimum

This script:
1. Downloads the full BF16 model
2. Runs GPTQ calibration with sample data
3. Saves the quantized model locally
4. Validates it loads and runs inference

Time estimate: ~20-40 min on RTX 5070 Ti for INT8
VRAM during quantization: needs ~20GB+ (uses CPU offload)

Usage:
    python quantize_8b.py --bits 8
    python quantize_8b.py --bits 4  # smaller, lower quality
"""

import argparse
import time
import torch
from pathlib import Path


def quantize(bits: int = 8, output_dir: str = None):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    model_name = "nvidia/Cosmos-Reason2-8B"
    if output_dir is None:
        output_dir = f"models/Cosmos-Reason2-8B-GPTQ-Int{bits}"

    print(f"Quantizing {model_name} to GPTQ INT{bits}")
    print(f"Output: {output_dir}")

    # Quantization config
    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=128,
        desc_act=True,  # activation ordering for better quality
        sym=True,       # symmetric quantization
    )

    print("\nStep 1: Loading model for quantization (CPU offload)...")
    start = time.time()

    # Load in float16 on CPU for quantization
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"  Loaded in {time.time() - start:.0f}s")

    # Calibration data -- simple text examples since we can't easily
    # create multimodal calibration data for GPTQ
    print("\nStep 2: Preparing calibration data...")
    calibration_texts = [
        "Describe the terrain visible in this mountain bike trail footage.",
        "What obstacles are on the riding line? Identify rocks, roots, and drops.",
        "Assess the trail conditions: is the surface wet or dry?",
        "Rate the risk level of this trail segment from LOW to EXTREME.",
        "What is the trail surface material? Packed dirt, gravel, rock, or mixed?",
        "Is there tree canopy overhead? How is the lighting?",
        "Describe what you see to the left and right of the trail.",
        "How steep is the terrain beside the trail?",
    ]

    calibration_data = []
    for text in calibration_texts:
        tokens = processor.tokenizer(text, return_tensors="pt")
        calibration_data.append(tokens.input_ids)

    print(f"  {len(calibration_data)} calibration samples")

    print(f"\nStep 3: Quantizing to INT{bits} (this takes a while)...")
    start = time.time()

    # Note: auto-gptq handles the actual quantization
    # For VLMs, we may need to use optimum's GPTQ integration instead
    try:
        model.quantize(calibration_data, quant_config=quantize_config)
        elapsed = time.time() - start
        print(f"  Quantized in {elapsed:.0f}s")

        print(f"\nStep 4: Saving quantized model to {output_dir}...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_quantized(output_dir)
        processor.save_pretrained(output_dir)
        print("  Saved.")

    except Exception as e:
        print(f"\n  auto-gptq direct quantization failed: {e}")
        print("  Trying optimum GPTQ pipeline instead...")

        # Fallback: use optimum's GPTQQuantizer
        from optimum.gptq import GPTQQuantizer

        quantizer = GPTQQuantizer(
            bits=bits,
            group_size=128,
            desc_act=True,
            sym=True,
            dataset="c4",  # standard calibration dataset
            num_samples=128,
        )

        # Reload fresh for optimum path
        del model
        torch.cuda.empty_cache()

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cpu",
        )

        start = time.time()
        quantized = quantizer.quantize_model(model, processor.tokenizer)
        elapsed = time.time() - start
        print(f"  Quantized in {elapsed:.0f}s")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        quantized.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"  Saved to {output_dir}")

    return output_dir


def validate(model_dir: str):
    """Quick validation that the quantized model loads and generates."""
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    print(f"\nValidating {model_dir}...")

    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map="auto",
    )

    # Check VRAM usage
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM used: {mem:.1f} GB")

    # Quick generation test
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello, what can you do?"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)
    response = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    print(f"  Test response: {response[:100]}...")
    print("  Validation passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Cosmos-Reason2-8B")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8],
                        help="Quantization bits (default: 8)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for quantized model")
    parser.add_argument("--validate-only", type=str, default=None,
                        help="Skip quantization, just validate an existing model dir")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.validate_only)
    else:
        out = quantize(bits=args.bits, output_dir=args.output_dir)
        validate(out)
