"""
GPU and environment validation for Cosmos Reason 2 MTB experiment.
Run this first to confirm your hardware and software stack is ready.
"""

import sys
import importlib


def check_python():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.minor < 12:
        print("  WARNING: Python 3.12+ recommended")
    else:
        print("  OK")


def check_torch():
    try:
        import torch
        print(f"PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  VRAM: {mem:.1f} GB")
            # Test BF16
            try:
                t = torch.tensor([1.0], dtype=torch.bfloat16, device="cuda")
                print("  BF16 support: OK")
            except Exception as e:
                print(f"  BF16 support: FAILED ({e})")
        else:
            print("  ERROR: No CUDA GPU detected")
    except ImportError:
        print("PyTorch: NOT INSTALLED")


def check_transformers():
    try:
        import transformers
        print(f"transformers {transformers.__version__}")
        # Check for Qwen3VL support
        try:
            from transformers import Qwen3VLForConditionalGeneration
            print("  Qwen3VLForConditionalGeneration: OK")
        except ImportError:
            print("  ERROR: Qwen3VLForConditionalGeneration not found (need transformers>=5.4.0)")
    except ImportError:
        print("transformers: NOT INSTALLED")


def check_dependencies():
    deps = [
        ("accelerate", "accelerate"),
        ("huggingface_hub", "huggingface-hub"),
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("cv2", "opencv-python"),
        ("ffmpeg", "ffmpeg-python"),
        ("PIL", "Pillow"),
        ("gpxpy", "gpxpy"),
        ("fitparse", "fitparse"),
        ("pandas", "pandas"),
    ]
    print("\nDependencies:")
    for module_name, pip_name in deps:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  {pip_name}: {version}")
        except ImportError:
            print(f"  {pip_name}: NOT INSTALLED")


def check_model_cache():
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cosmos_dirs = list(cache_dir.glob("models--nvidia--Cosmos-Reason2*"))
    if cosmos_dirs:
        print(f"\nCosmos model cache found: {cosmos_dirs[0].name}")
    else:
        print("\nCosmos model not cached yet (will download on first run)")


def estimate_vram():
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Cosmos-Reason2-2B in BF16 ~ 4.88 GB
            model_vram = 4.88
            free = total - model_vram
            print(f"\nVRAM estimate:")
            print(f"  Total: {total:.1f} GB")
            print(f"  Model (BF16): ~{model_vram} GB")
            print(f"  Available for context: ~{free:.1f} GB")
            if free < 4:
                print("  WARNING: Tight on VRAM. Keep clips under 20 seconds.")
            else:
                print("  OK: Should handle 30-second clips at 4 FPS")
    except ImportError:
        pass


if __name__ == "__main__":
    print("=" * 50)
    print("Cosmos MTB Experiment - Environment Check")
    print("=" * 50)
    print()
    check_python()
    check_torch()
    check_transformers()
    check_dependencies()
    check_model_cache()
    estimate_vram()
    print()
    print("=" * 50)
    print("Setup check complete")
    print("=" * 50)
