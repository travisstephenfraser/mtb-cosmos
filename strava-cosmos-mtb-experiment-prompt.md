# Claude Code: Strava × Cosmos Reason 2 — Mountain Bike Video Analysis Experiment

## Context for Claude Code

This is an exploratory experiment to test whether NVIDIA Cosmos Reason 2 can extract meaningful insights from mountain bike footage. This is NOT a product — it's a learning exercise to understand WFM capabilities on athletic/outdoor video content that is far outside the model's primary training distribution (robots, AVs, warehouses).

The developer has 17,000+ miles and years of MTB footage from helmet cams (GoPro-style), chest mounts, and third-person trail shots. The hypothesis is that Cosmos Reason 2's spatial-temporal understanding and physics reasoning might transfer to analyzing terrain, rider dynamics, and trail conditions — even though it was never trained on cycling data.

**Hardware:** ASUS ROG Zephyrus G14 — Ryzen AI 9 HX, RTX 5070 Ti Laptop GPU (12GB GDDR7), 32GB RAM. Running Cosmos Reason 2-2B locally in BF16 (~4.88GB VRAM, ~7GB free for context).

**Validated environment (already working):**
- WSL2 Ubuntu, Python 3.12.3
- PyTorch nightly `2.12.0.dev+cu128` (required for Blackwell SM 12.0)
- CUDA 13.1 driver, toolkit 12.8 via PyTorch
- transformers 5.4.0 (native Qwen3-VL support)
- Model class: `Qwen3VLForConditionalGeneration` (NOT Qwen2_5_VL)
- Use `dtype=torch.bfloat16` (NOT `torch_dtype=` which is deprecated)
- HF authenticated, model cached at `~/.cache/huggingface/`
- Virtual environment: `source ~/cosmos/bin/activate`

**Important framing:** We genuinely don't know what Cosmos will be good at here. The experiment is designed to cast a wide net across different analysis types, find out what sticks, and document what works vs. what fails. The failure modes are as interesting as the successes.

## Instructions

### 1. Directory structure

Create this as a standalone experiment directory (not inside SiteProof):

```
experiments/
  cosmos_mtb_analysis/
    README.md
    requirements.txt
    setup_check.py            # Same GPU/env validation as SiteProof experiment (can symlink)
    clip_prep.py              # Prepare MTB video clips for analysis
    cosmos_mtb.py             # Core Cosmos inference module with MTB-specific prompts
    run_experiment.py          # Run analysis across multiple clips and prompt categories
    compare_with_strava.py    # Optional: compare Cosmos observations with Strava GPS/sensor data
    prompts/
      terrain_analysis.txt
      rider_dynamics.txt
      trail_conditions.txt
      segment_narration.txt
      risk_assessment.txt
      technical_skills.txt
      exploratory_freeform.txt
    test_data/
      .gitkeep               # Developer adds their own MTB clips here
    results/
      .gitkeep
    findings/
      .gitkeep               # Markdown summaries of what worked and what didn't
```

### 2. requirements.txt

```
# IMPORTANT: Install PyTorch FIRST with the nightly cu128 build (required for RTX 5070 Ti Blackwell GPU):
#   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
#
# Then install the rest:
#   pip install -r requirements.txt

transformers>=5.4.0
accelerate>=1.2.0
huggingface-hub>=1.8.0
qwen-vl-utils>=0.0.8
opencv-python>=4.10.0
ffmpeg-python>=0.2.0
Pillow>=10.0.0
gpxpy>=1.6.0             # For parsing Strava GPX exports
fitparse>=1.2.0          # For parsing .FIT files from bike computers
pandas>=2.0.0
```

### 3. clip_prep.py

Mountain bike footage has unique challenges — high camera shake, rapid scene changes, variable lighting (shade to sun), motion blur on technical sections. This module preps clips for Cosmos.

```python
"""
Prepare MTB video clips for Cosmos Reason 2 analysis.

MTB footage considerations:
- Helmet/chest cam = egocentric POV (similar to Cosmos training on egocentric robot data)
- High vibration and camera shake on rough terrain
- Rapid lighting changes (tree canopy shade <-> open sun)
- Typical clips: 30-120 seconds of a trail segment

This module:
1. Accepts raw MTB footage (MP4 from GoPro, Insta360, DJI Action, phone)
2. Optionally trims to a time window
3. Adds timestamp overlay (Cosmos uses these for temporal localization)
4. Optionally stabilizes (basic deshake — Cosmos may handle shake fine, test both)
5. Exports clips at a target resolution and frame rate
"""

import argparse
import cv2
import os
from datetime import datetime, timedelta

def prepare_clip(
    input_path: str,
    output_dir: str,
    start_sec: float = None,
    end_sec: float = None,
    target_fps: int = 4,        # Match Cosmos training FPS
    max_duration_sec: int = 30,  # Cosmos context window limit
    add_timestamps: bool = True,
    stabilize: bool = False,
    target_height: int = 720     # Downsample for faster inference
) -> dict:
    """
    Prepare a single MTB clip for Cosmos analysis.
    
    Returns:
        dict with:
        - output_path: path to processed clip
        - duration_sec: clip duration
        - frame_count: number of frames at target_fps
        - original_fps: source video FPS
        - resolution: (width, height) of output
    """
    pass

def extract_keyframes(
    input_path: str,
    output_dir: str,
    mode: str = "interval",  # "interval" or "scene_change"
    interval_sec: float = 1.0,
    scene_threshold: float = 30.0
) -> list:
    """
    Extract representative keyframes for individual image analysis.
    
    For MTB footage, scene_change mode may work well because:
    - Trail sections have consistent visual themes
    - Transitions (forest->clearing, flat->descent) create natural breakpoints
    
    Returns list of frame paths with timestamps.
    """
    pass

def batch_prepare(
    input_dir: str,
    output_dir: str,
    max_clips: int = None,
    **kwargs
) -> list:
    """Process all video files in a directory."""
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MTB clips for Cosmos analysis")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("--output-dir", default="test_data/prepared", help="Output directory")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument("--max-duration", type=int, default=30, help="Max clip duration (seconds)")
    parser.add_argument("--stabilize", action="store_true", help="Apply basic stabilization")
    parser.add_argument("--keyframes-only", action="store_true", help="Extract keyframes instead of video")
    args = parser.parse_args()
    # Implementation
```

### 4. Prompt Templates

These are the core of the experiment. Each targets a different analysis hypothesis. The goal is to find which prompts produce genuinely useful output vs. which ones Cosmos can't handle.

**prompts/terrain_analysis.txt:**
```
You are analyzing mountain bike trail footage from a first-person helmet camera perspective. Focus on the physical terrain and trail characteristics.

Analyze the terrain and trail surface visible in this footage:

1. SURFACE TYPE: Identify the trail surface (packed dirt, loose gravel, rock garden, roots, sand, mud, hardpack, loam, etc.)
2. SURFACE TRANSITIONS: Note where the surface type changes and how abruptly
3. GRADIENT: Estimate the trail gradient — flat, gradual climb/descent, steep climb/descent
4. TRAIL WIDTH: Estimate relative trail width (singletrack, doubletrack, fire road)
5. OBSTACLES: Identify specific obstacles — rocks, roots, drops, water crossings, berms, jumps
6. LINE CHOICE: If multiple viable riding lines are visible, describe them

For each observation, note the approximate timestamp or position in the clip.

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>
```

**prompts/rider_dynamics.txt:**
```
You are analyzing mountain bike footage from a first-person perspective. Focus on the rider's movement dynamics and bike handling as inferred from camera motion and visual cues.

Analyze the rider's dynamics visible through camera movement patterns:

1. SPEED VARIATION: Identify sections where the rider accelerates, maintains speed, or decelerates. What visual cues indicate speed changes? (motion blur, scenery passing rate, approach speed to objects)
2. BRAKING EVENTS: Identify moments that suggest braking — sudden deceleration before obstacles, corners, or technical sections
3. BODY POSITION: If visible (shadow, bike components in frame), describe body position changes — standing vs seated, weight forward vs back, leaning into turns
4. CORNERING: Describe corner entries — speed, line choice (inside/outside/apex), lean angle as suggested by camera tilt
5. IMPACT ABSORPTION: Identify moments where the camera shows significant vertical displacement suggesting rough terrain absorption
6. FLOW STATE: Are there sections where movement appears smooth and continuous vs. choppy and reactive?

Note: You are inferring dynamics from egocentric camera movement. State your confidence level for each observation.

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>
```

**prompts/trail_conditions.txt:**
```
You are analyzing mountain bike trail footage to assess current trail conditions. This information would be valuable for trail reports and ride planning.

Assess the trail conditions visible in this footage:

1. MOISTURE: Does the trail appear dry, damp, wet, or muddy? Look for puddles, wet rock, mud splatter, dust
2. EROSION: Are there signs of trail erosion — ruts, exposed roots, washed-out sections, loose surface material
3. MAINTENANCE: Does the trail appear well-maintained (clear sightlines, trimmed vegetation, maintained drainage) or neglected (overgrown, debris on trail, damaged features)
4. HAZARDS: Identify any safety hazards — blind corners, unexpected drops, loose rocks, fallen trees, exposure (cliff edges)
5. SEASONAL INDICATORS: What season or conditions do visual cues suggest? (leaf cover, vegetation state, sun angle, ground moisture)
6. TRAFFIC: Any signs of recent use — tire tracks, footprints, worn lines

Summarize with an overall trail condition rating: Excellent / Good / Fair / Poor / Hazardous

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>
```

**prompts/segment_narration.txt:**
```
You are a knowledgeable mountain bike trail guide watching first-person trail footage. Provide a real-time style narration of what is happening in this clip, as if you were describing the ride to another cyclist.

Narrate the ride segment covering:
- What terrain features appear and when
- Key moments (technical sections, fast flow sections, climbs, descents)
- The overall character and feel of this trail segment
- How challenging this section appears (beginner / intermediate / advanced / expert)
- What skills a rider would need for this section

Keep the narration engaging and informative, as if writing a trail review or ride report.

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>
```

**prompts/risk_assessment.txt:**
```
You are a trail safety analyst reviewing mountain bike footage from a first-person perspective. Assess risk factors visible in this clip.

Evaluate risk factors:

1. TERRAIN RISK: Rate the physical difficulty of the terrain (exposure, technical features, gradient)
2. SPEED RISK: Does the rider's apparent speed seem appropriate for the terrain and visibility?
3. VISIBILITY: Are there blind spots, blind corners, or limited sightlines?
4. ENVIRONMENTAL: Weather conditions, lighting (shadows, sun glare), time of day
5. FEATURE-SPECIFIC: Any individual features (drops, jumps, rock gardens) that carry elevated risk
6. ESCAPE ROUTES: If something goes wrong, is there runout space or is the trail constrained?

Rate overall segment risk: Low / Moderate / High / Extreme

This is NOT about discouraging riding — it is about helping riders make informed decisions about trail features.

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>
```

**prompts/technical_skills.txt:**
```
You are a mountain bike skills coach analyzing first-person riding footage. Identify the technical skills being demonstrated or required in this trail segment.

Analyze technical skill requirements:

1. SKILLS DEMONSTRATED: What bike handling skills are visibly in use? (pumping, manualing, bunny hopping, wheel placement, line selection, weight shifting, braking technique)
2. SKILLS REQUIRED: What skills would a rider need to clean this section?
3. TECHNIQUE OBSERVATIONS: Based on camera movement patterns, comment on the rider's technique — smooth or jerky inputs, consistent or inconsistent speed management, proactive or reactive riding
4. IMPROVEMENT OPPORTUNITIES: If the camera movement suggests areas where technique could improve, note them (e.g., "camera pitches forward sharply at 0:15 suggesting late braking into the rock garden")
5. DIFFICULTY PROGRESSION: Does the segment ramp in difficulty or maintain a consistent level?

Rate technical skill level required: Green (beginner) / Blue (intermediate) / Black (advanced) / Double Black (expert)

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>
```

**prompts/exploratory_freeform.txt:**
```
You are watching first-person mountain bike trail footage. You have expertise in physics, spatial reasoning, and understanding motion dynamics from video.

Watch this clip carefully and tell me everything interesting you observe. Do not limit yourself to any specific category. I want to understand what a physics-aware vision model notices about mountain biking footage that might not be obvious to a human viewer.

Consider: terrain physics, motion patterns, environmental factors, spatial geometry of the trail, anything about timing or rhythm, visual features that suggest forces acting on the rider, or any patterns across the clip that a human might miss.

Be specific and reference timestamps or positions in the clip where possible.

Answer the question in the following format:
<think>
your reasoning
</think>

<answer>
your answer
</answer>
```

### 5. cosmos_mtb.py

Core inference module:

```python
"""
Cosmos Reason 2 inference for mountain bike video analysis.

Design decisions:
- Run each prompt category independently on the same clip
- Support both video and individual frame analysis
- Log inference time per prompt (MTB clips may have more visual complexity than property interiors)
- Parse <think> and <answer> tags from Cosmos output
- Save raw responses alongside parsed results for analysis
"""

import os
import time
import json
import torch
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"

class CosmosMTBAnalyzer:
    """Analyze mountain bike footage with Cosmos Reason 2."""
    
    PROMPT_CATEGORIES = [
        "terrain_analysis",
        "rider_dynamics",
        "trail_conditions",
        "segment_narration",
        "risk_assessment",
        "technical_skills",
        "exploratory_freeform",
    ]
    
    def __init__(self, model_name="nvidia/Cosmos-Reason2-2B", device="cuda"):
        self.model = None
        self.processor = None
        self.model_name = model_name
        self.device = device
        self._prompts = {}
        self._load_prompts()
    
    def _load_prompts(self):
        """Load all prompt templates from prompts/ directory."""
        for category in self.PROMPT_CATEGORIES:
            prompt_path = PROMPT_DIR / f"{category}.txt"
            if prompt_path.exists():
                self._prompts[category] = prompt_path.read_text().strip()
            else:
                print(f"Warning: prompt file not found: {prompt_path}")
    
    def _load_model(self):
        """Lazy-load model on first inference call."""
        if self.model is not None:
            return
        
        from transformers import AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration
        
        print(f"Loading {self.model_name}...")
        start = time.time()
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print(f"Model loaded in {time.time() - start:.1f}s")
    
    def analyze_clip(
        self,
        video_path: str,
        categories: list = None,
        fps: float = 4.0,
        max_tokens: int = 4096
    ) -> dict:
        """
        Run one or more analysis categories on a single video clip.
        
        Args:
            video_path: Path to prepared video clip
            categories: List of prompt categories to run (default: all)
            fps: Frame rate for Cosmos input (default 4, matching training)
            max_tokens: Max output tokens (4096 recommended to avoid truncation)
        
        Returns:
            dict keyed by category, each containing:
            - raw_response: Full model output
            - reasoning: Content of <think> tags
            - answer: Content of <answer> tags
            - inference_time_sec: How long this analysis took
        """
        self._load_model()
        
        if categories is None:
            categories = self.PROMPT_CATEGORIES
        
        results = {}
        for category in categories:
            if category not in self._prompts:
                print(f"Skipping unknown category: {category}")
                continue
            
            prompt_text = self._prompts[category]
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert analyst reviewing first-person "
                        "mountain bike trail footage. You have deep knowledge "
                        "of physics, spatial reasoning, terrain dynamics, and "
                        "cycling biomechanics. "
                        "Answer the question in the following format: "
                        "<think>\nyour reasoning\n</think>\n\n"
                        "<answer>\nyour answer\n</answer>."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{os.path.abspath(video_path)}",
                            "fps": fps
                        },
                        {
                            "type": "text",
                            "text": prompt_text
                        }
                    ]
                }
            ]
            
            start = time.time()
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.3
            )
            
            response = self.processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            elapsed = time.time() - start
            
            # Parse <think> and <answer> tags
            reasoning, answer = self._parse_response(response)
            
            results[category] = {
                "raw_response": response,
                "reasoning": reasoning,
                "answer": answer,
                "inference_time_sec": round(elapsed, 2)
            }
            
            print(f"  [{category}] completed in {elapsed:.1f}s")
        
        return results
    
    def analyze_frame(self, image_path: str, categories: list = None, max_tokens: int = 4096) -> dict:
        """Analyze a single keyframe (same interface as analyze_clip but with image input)."""
        self._load_model()
        
        if categories is None:
            categories = self.PROMPT_CATEGORIES
        
        results = {}
        for category in categories:
            if category not in self._prompts:
                continue
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert analyst reviewing a still frame "
                        "from first-person mountain bike trail footage. "
                        "Answer the question in the following format: "
                        "<think>\nyour reasoning\n</think>\n\n"
                        "<answer>\nyour answer\n</answer>."
                    )
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                        {"type": "text", "text": self._prompts[category]}
                    ]
                }
            ]
            
            start = time.time()
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to(self.model.device)
            
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.3,
                top_p=0.3
            )
            
            response = self.processor.batch_decode(
                output[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            elapsed = time.time() - start
            reasoning, answer = self._parse_response(response)
            
            results[category] = {
                "raw_response": response,
                "reasoning": reasoning,
                "answer": answer,
                "inference_time_sec": round(elapsed, 2)
            }
        
        return results
    
    @staticmethod
    def _parse_response(response: str) -> tuple:
        """Extract content from <think> and <answer> tags."""
        import re
        
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        
        reasoning = think_match.group(1).strip() if think_match else ""
        answer = answer_match.group(1).strip() if answer_match else response.strip()
        
        return reasoning, answer
```

### 6. run_experiment.py

```python
"""
MTB Video Analysis Experiment Runner

Usage:
  # Analyze a single clip with all prompt categories
  python run_experiment.py --clip test_data/trail_segment.mp4

  # Analyze with specific categories only
  python run_experiment.py --clip test_data/trail_segment.mp4 --categories terrain_analysis rider_dynamics

  # Batch analyze all clips in a directory
  python run_experiment.py --clip-dir test_data/prepared/

  # Run only the exploratory freeform prompt (great for initial discovery)
  python run_experiment.py --clip test_data/trail_segment.mp4 --categories exploratory_freeform

  # Analyze keyframes instead of video
  python run_experiment.py --frames test_data/frames/

Output:
  - results/{timestamp}_{clip_name}/
    - per_category/ (individual JSON files per prompt category)
    - summary.json (all results combined)
    - report.md (human-readable analysis report)
    - meta.json (clip info, timing, model details)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

def generate_report(clip_name: str, results: dict, output_path: str):
    """
    Generate a human-readable markdown report from all category results.
    
    The report should:
    - Lead with the most interesting/novel findings
    - Group findings by what worked well vs. what was vague/unhelpful
    - Include a "Model Capability Assessment" section rating Cosmos on each category:
      * STRONG: Specific, novel, accurate observations
      * MODERATE: Some useful observations mixed with generic/obvious ones
      * WEAK: Vague, generic, or clearly hallucinated observations
      * FAILED: Could not produce relevant output for this category
    - Include selected reasoning chains (from <think> tags) that show how the model
      arrived at interesting conclusions
    - Note inference times to understand cost/latency profile
    """
    pass

def main():
    parser = argparse.ArgumentParser(description="MTB Cosmos Analysis Experiment")
    parser.add_argument("--clip", type=str, help="Single video clip to analyze")
    parser.add_argument("--clip-dir", type=str, help="Directory of prepared clips")
    parser.add_argument("--frames", type=str, help="Directory of keyframes")
    parser.add_argument("--categories", nargs="+", default=None,
                       help="Specific prompt categories to run")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="nvidia/Cosmos-Reason2-2B")
    args = parser.parse_args()

    # 1. Initialize CosmosMTBAnalyzer
    # 2. Process clip(s) or frames
    # 3. For each clip:
    #    a. Run all requested categories
    #    b. Save individual category results as JSON
    #    c. Generate summary and human-readable report
    #    d. Print highlights to terminal
    # 4. If batch: generate cross-clip comparison summary
    pass

if __name__ == "__main__":
    main()
```

### 7. compare_with_strava.py (Optional bonus module)

```python
"""
Compare Cosmos visual observations with Strava GPS/sensor data.

This is the most speculative part of the experiment. The idea:
- Cosmos says "steep descent starting around 0:12" from video
- Strava GPX/FIT data shows elevation drop of 50ft at the corresponding timestamp
- Do they agree?

If Cosmos visual terrain estimates correlate with actual GPS data, that's a
strong signal that the model genuinely understands terrain geometry from video.
If they don't correlate, the spatial reasoning is likely superficial.

Requires:
- Video clip with known recording start time
- Corresponding Strava activity GPX or FIT export
- Approximate time alignment between video start and activity timestamp

Usage:
  python compare_with_strava.py \
    --cosmos-results results/20260330_trail1/summary.json \
    --gpx strava_exports/activity_12345.gpx \
    --video-start-offset 300  # video starts 5 min into the activity
"""

import argparse
import json

def load_gpx(gpx_path: str) -> list:
    """
    Parse GPX file into list of trackpoints with:
    - timestamp, lat, lon, elevation
    - Derived: speed, gradient, distance from start
    """
    pass

def load_fit(fit_path: str) -> list:
    """
    Parse FIT file (richer data than GPX):
    - timestamp, lat, lon, elevation
    - heart_rate, cadence, power (if available)
    - speed, temperature
    """
    pass

def align_timestamps(cosmos_results: dict, gps_data: list, video_start_offset_sec: float) -> list:
    """
    Map Cosmos timestamp references (e.g., "at approximately 0:12") to GPS trackpoints.
    
    Returns list of aligned observation pairs:
    - cosmos_observation: what the model said
    - gps_data_at_time: what the sensors recorded
    """
    pass

def compare_terrain_estimates(aligned_data: list) -> dict:
    """
    Compare Cosmos terrain observations with GPS ground truth:
    
    - Gradient estimates vs actual elevation change
    - Speed observations (fast/slow) vs GPS speed
    - Climb/descent detection vs elevation profile
    - "Flat section" calls vs actual gradient near zero
    
    Returns correlation analysis and accuracy metrics.
    """
    pass

def generate_comparison_report(comparison: dict, output_path: str):
    """
    Markdown report showing:
    - Side-by-side: what Cosmos saw vs what GPS recorded
    - Accuracy of terrain gradient estimates
    - Whether speed observations match reality
    - Overall assessment: is the spatial reasoning grounded or superficial?
    """
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Cosmos analysis with Strava data")
    parser.add_argument("--cosmos-results", required=True, help="Path to Cosmos results JSON")
    parser.add_argument("--gpx", help="Strava GPX export")
    parser.add_argument("--fit", help="Strava FIT file")
    parser.add_argument("--video-start-offset", type=float, default=0,
                       help="Seconds into the activity when video recording started")
    parser.add_argument("--output", default="findings/strava_comparison.md")
    args = parser.parse_args()
    # Implementation
```

### 8. README.md

Write a README covering:

**Purpose:** Explore whether NVIDIA Cosmos Reason 2 — a VLM trained for physical AI (robots, AVs) — can produce meaningful analysis of mountain bike footage. This tests the model's generalization to a domain far outside its training distribution.

**Why this might work:**
- Cosmos was trained on egocentric video data (first-person perspective) — helmet cam MTB footage is structurally similar
- The model understands spatial geometry, motion dynamics, and physics — all relevant to trail riding
- Temporal reasoning (what happens next, speed changes, transitions) is a core Cosmos capability

**Why this might fail:**
- Training data is industrial/robotic — no cycling, trails, or outdoor recreation
- MTB footage has extreme camera shake and motion blur that may break the model
- Athletic motion dynamics are very different from robot kinematics
- "Terrain analysis" from a model that's never seen singletrack is a big ask

**Experiment design:**
- 7 prompt categories testing different analysis dimensions
- Start with `exploratory_freeform` to see what the model notices unprompted
- Use `terrain_analysis` and `trail_conditions` as the most likely to succeed (closest to spatial reasoning training)
- Use `rider_dynamics` and `technical_skills` as stretch goals (requires inferring human biomechanics from camera motion)
- Optional Strava GPS comparison as ground truth validation

**How to evaluate results:**
- Rate each prompt category: STRONG / MODERATE / WEAK / FAILED
- Specifically document: "What did Cosmos observe that was genuinely novel or non-obvious?"
- Specifically document: "What did Cosmos get wrong or hallucinate?"
- The freeform prompt is the most important — it shows what the model's physics reasoning latches onto when unconstrained

**Clip selection guidance:**
For best experiment coverage, include clips with variety:
- A smooth flow trail (berms, jumps, fast) — tests speed/dynamics reasoning
- A technical rock garden — tests obstacle recognition and terrain analysis  
- A steep climb — tests gradient estimation
- A fast descent — tests the model under rapid visual change
- A transition section (flat to technical) — tests ability to notice terrain changes
- A wet/muddy trail vs dry trail — tests condition assessment

**What success looks like:**
If Cosmos produces specific, accurate terrain descriptions that correlate with GPS data, or identifies trail features/conditions that a rider would recognize as correct, the model's spatial reasoning generalizes better than expected. Even partial success (e.g., terrain analysis works but rider dynamics doesn't) maps the boundary of WFM capability.

**What this is NOT:**
This is not a product. This is a learning exercise and a potential Strava PM interview talking point ("I tested world foundation models on athletic video content — here's what I learned about where spatial reasoning transfers and where it breaks down").

### 9. Implementation Notes

- **Start with the exploratory_freeform prompt.** Run it on 3 diverse clips before writing any other code. The freeform results tell you what direction to invest in.
- **Keep clips short.** 15-30 seconds is ideal. Cosmos at 4 FPS means a 30-second clip = 120 frames. Longer clips may hit context limits or dilute the analysis.
- **Log everything.** Raw responses, reasoning chains, inference times. The <think> tags are where the interesting stuff lives — they show HOW the model reasons about physical dynamics.
- **Don't over-engineer the Strava comparison.** It's a bonus validation. If the core analysis is garbage, GPS correlation won't help. If the core analysis is good, GPS correlation makes the experiment rigorous.
- **Document failure modes.** "Cosmos hallucinated a water crossing that doesn't exist" is just as valuable as "Cosmos correctly identified the rock garden." The failures map the boundary of WFM generalization.
