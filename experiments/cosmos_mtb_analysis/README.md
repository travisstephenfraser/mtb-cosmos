# Cosmos Reason 2 x Mountain Bike Video Analysis

Exploring whether NVIDIA Cosmos Reason 2, a VLM trained for physical AI (robots, AVs, warehouses), can produce meaningful analysis of mountain bike footage. This tests the model's generalization to a domain far outside its training distribution.

## Why This Might Work

- **Egocentric training data.** Cosmos was trained on first-person perspective video. Helmet cam MTB footage is structurally identical.
- **Spatial-temporal reasoning.** The model understands geometry, motion dynamics, and physics. All relevant to trail riding.
- **Temporal localization.** "What happens next" and "when does the scene change" are core Cosmos capabilities.

## Why This Might Fail

- Training data is industrial/robotic. No cycling, trails, or outdoor recreation.
- MTB footage has extreme camera shake and motion blur that may break the model.
- Athletic motion dynamics are very different from robot kinematics.
- "Terrain analysis" from a model that has never seen singletrack is a big ask.

## Hardware

ASUS ROG Zephyrus G14: Ryzen AI 9 HX, RTX 5070 Ti (12GB GDDR7), 32GB RAM. Running Cosmos-Reason2-2B locally in BF16 (~4.88GB VRAM).

## Setup

```bash
# In WSL2
source ~/cosmos/bin/activate

# Verify environment
python setup_check.py

# Install deps (PyTorch nightly already installed in venv)
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Prep a clip (trim, downsample to 4fps, add timestamps)
python clip_prep.py your_mtb_clip.mp4 --output-dir test_data/prepared --max-duration 20

# 2. Start with freeform to see what Cosmos notices unprompted
python run_experiment.py --clip test_data/prepared/your_clip_prepared.mp4 --categories exploratory_freeform

# 3. Run all categories
python run_experiment.py --clip test_data/prepared/your_clip_prepared.mp4

# 4. Optional: compare with Strava GPS data
python compare_with_strava.py \
    --cosmos-results results/<run_dir>/summary.json \
    --gpx strava_exports/activity.gpx \
    --video-start-offset 300
```

## Experiment Design

### 7 Prompt Categories

| Category | Hypothesis | Expected Difficulty |
|----------|-----------|-------------------|
| `terrain_analysis` | Surface type, gradient, obstacles | Most likely to succeed (closest to spatial reasoning training) |
| `trail_conditions` | Moisture, erosion, maintenance, hazards | Likely moderate (visual pattern matching) |
| `segment_narration` | Natural language ride description | Tests coherent temporal narration |
| `risk_assessment` | Safety factors and hazard identification | Depends on terrain understanding |
| `rider_dynamics` | Speed, braking, body position from camera motion | Stretch goal (inferring biomechanics from egocentric video) |
| `technical_skills` | Skill identification and coaching feedback | Stretch goal (requires cycling domain knowledge) |
| `exploratory_freeform` | Unconstrained observation | **Most important.** Shows what the model's physics reasoning latches onto when unconstrained |

### Evaluation

Rate each category: **STRONG** / **MODERATE** / **WEAK** / **FAILED**

Key questions:
- What did Cosmos observe that was genuinely novel or non-obvious?
- What did Cosmos get wrong or hallucinate?
- Do observations correlate with GPS ground truth?

### Clip Selection Guide

For best coverage, test with:
- A smooth flow trail (berms, jumps, fast) -- speed/dynamics reasoning
- A technical rock garden -- obstacle recognition and terrain analysis
- A steep climb -- gradient estimation
- A fast descent -- model behavior under rapid visual change
- A transition section (flat to technical) -- detecting terrain changes
- A wet/muddy trail vs dry trail -- condition assessment

## What This Is NOT

This is not a product. This is a learning exercise: testing where spatial reasoning transfers and where it breaks down when a WFM trained on industrial video meets athletic outdoor content.
