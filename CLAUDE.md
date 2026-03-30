# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Exploratory experiment testing whether NVIDIA Cosmos Reason 2 (a VLM trained for physical AI -- robots, AVs, warehouses) can produce meaningful analysis of mountain bike helmet cam footage. Tests WFM generalization far outside training distribution.

## Environment

- **Runtime:** WSL2 Ubuntu, Python 3.12.3, venv at `~/cosmos/bin/activate`
- **GPU:** RTX 5070 Ti 12GB GDDR7, requires PyTorch nightly cu128 for Blackwell SM 12.0
- **Model:** `nvidia/Cosmos-Reason2-2B` via `Qwen3VLForConditionalGeneration` (NOT Qwen2_5_VL)
- **Load with:** `dtype=torch.bfloat16` (NOT `torch_dtype=` which is deprecated)
- **Model cached at:** `~/.cache/huggingface/`
- **HF login:** Use `python -c "from huggingface_hub import login; login()"` instead of `huggingface-cli login` -- venv PATH issues make the CLI unreliable

## Commands

```bash
# Activate environment
source ~/cosmos/bin/activate

# Verify GPU/deps
python experiments/cosmos_mtb_analysis/setup_check.py

# Prep a clip (trim, downsample to 4fps, timestamp overlay)
python clip_prep.py test_data/video.mp4 --start 20 --max-duration 20

# Extract zone crops (left/right/center/top/bottom)
python clip_prep.py test_data/video.mp4 --start 20 --max-duration 20 --zones all --output-dir test_data/zones

# Run analysis (single category)
python run_experiment.py --clip test_data/prepared/clip.mp4 --categories exploratory_freeform

# Run analysis (all categories)
python run_experiment.py --clip test_data/prepared/clip.mp4

# Run on keyframes/zone crops
python run_experiment.py --frames test_data/zones/left/ --categories peripheral_exposure

# Compare with Strava GPS
python compare_with_strava.py --cosmos-results results/<run>/summary.json --gpx activity.gpx --video-start-offset 300
```

## Architecture

Pipeline: `clip_prep.py` → `cosmos_mtb.py` → `run_experiment.py` → `compare_with_strava.py`

- **clip_prep.py** -- Video preprocessing. Trim, resample to 4fps (matching Cosmos training), 720p resize, timestamp overlay. Also handles zone extraction (spatial frame crops) and keyframe extraction (interval or scene-change detection).
- **cosmos_mtb.py** -- `CosmosMTBAnalyzer` class. Lazy model loading, per-category inference, `<think>`/`<answer>` tag parsing. Custom `StoppingCriteria` halts generation on first `</answer>` (prevents repetition loops from 2B model). Each prompt category is an independent inference pass.
- **run_experiment.py** -- Orchestrates runs, saves timestamped results (`results/{timestamp}_{clip}/`), generates markdown reports with capability assessment scaffold.
- **compare_with_strava.py** -- Parses GPX/FIT files, aligns Cosmos timestamp references to GPS trackpoints, validates speed/gradient claims against sensor data.
- **cloud_quantize.py** -- GPTQ INT8 quantization of Cosmos-Reason2-8B on Vast.ai A100. Outputs ~8GB model that fits on local 12GB GPU.

## Prompt Design (prompts/)

Three tiers, evolved through experimentation:

1. **Open-ended** (7 prompts): `terrain_analysis`, `rider_dynamics`, `trail_conditions`, `segment_narration`, `risk_assessment`, `technical_skills`, `exploratory_freeform`. These produce verbose output but hallucinate environmental details (water, uphill sections) that don't exist.

2. **Constrained** (4 prompts): `constrained_terrain`, `constrained_risk`, `constrained_dynamics`, `constrained_conditions`. Binary/categorical forced-choice answers with "CANNOT DETERMINE" escape hatch. Reduced hallucination ~50% vs open-ended.

3. **Zone-specific** (5 prompts): `peripheral_exposure`, `zone_center_surface`, `zone_top_sightline`, `zone_bottom_texture`, `zone_side_vegetation`. Paired with spatial frame crops so the model analyzes one region at a time. Deterministic preprocessing replaces prompt engineering.

## Known Model Behaviors (2B)

- **Repetition loops:** Without StoppingCriteria, the model repeats `</think><answer>` cycles indefinitely (163s+ inference). The stop criteria in `_run_inference()` is critical.
- **Malformed tags:** Model emits `<answer>...</think>` instead of `<answer>...</answer>`. Parser handles this by terminating on `</answer>`, `</think>`, or next `<answer>`.
- **Template echoing:** Sometimes outputs "your answer" literally instead of generating content. Reasoning chain is usually substantive even when the answer fails.
- **Uphill hallucination:** Consistently misreads steepness transitions as direction changes (steep→less-steep reads as "uphill"). Likely from AV training where camera pitch = road gradient.
- **Environmental confabulation:** Fills in plausible but wrong details (puddles, clouds, water) when prompts ask about conditions. Constrained prompts mitigate this.
- **Pixel ignoring on crops:** Zone crops produce identical answers across all frames, suggesting the 2B may not perform genuine per-frame visual grounding on unfamiliar image types.

## Key Findings So Far

- **Risk assessment** was the strongest output category -- specific, structured, actionable
- **Structural recognition works:** downhill, technical, singletrack, rocks, roots
- **Environmental state fails:** moisture, weather, direction changes are hallucinated
- The boundary maps to training distribution: spatial-static (geometry) transfers, environmental-dynamic (conditions) doesn't
- 8B model (GPTQ INT8) is the next test to determine if issues are model-size or architecture problems
