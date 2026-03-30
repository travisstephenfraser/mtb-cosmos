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
import re
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
        "constrained_terrain",
        "constrained_risk",
        "constrained_dynamics",
        "constrained_conditions",
        "peripheral_exposure",
        "zone_center_surface",
        "zone_top_sightline",
        "zone_bottom_texture",
        "zone_side_vegetation",
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

    def _run_inference(self, messages: list, max_tokens: int) -> tuple:
        """Run a single inference pass. Returns (response_text, elapsed_sec)."""
        start = time.time()

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.3,
            repetition_penalty=1.2,
        )

        # Stop generation once </answer> appears in decoded output.
        # Token-ID matching fails because tokenizers split </answer> unpredictably,
        # so we decode on every step and check the string directly.
        from transformers import StoppingCriteria, StoppingCriteriaList
        tokenizer = self.processor.tokenizer
        input_len = inputs.input_ids.shape[1]

        class StopOnAnswerClose(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                generated = input_ids[0, input_len:]
                n = len(generated)
                # Decode tail for fast </answer> check
                tail = tokenizer.decode(generated[-20:], skip_special_tokens=False)
                if "</answer>" in tail:
                    return True
                # Fallback: stop if model is looping (second <answer> tag)
                if n > 600:
                    raw = tokenizer.decode(generated, skip_special_tokens=False)
                    return raw.count("<answer>") >= 2
                return False

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnAnswerClose()])

        output = self.model.generate(**inputs, **gen_kwargs)

        response = self.processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        elapsed = time.time() - start
        return response, elapsed

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

            response, elapsed = self._run_inference(messages, max_tokens)
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

            response, elapsed = self._run_inference(messages, max_tokens)
            reasoning, answer = self._parse_response(response)

            results[category] = {
                "raw_response": response,
                "reasoning": reasoning,
                "answer": answer,
                "inference_time_sec": round(elapsed, 2)
            }

            print(f"  [{category}] completed in {elapsed:.1f}s")

        return results

    @staticmethod
    def _parse_response(response: str) -> tuple:
        """Extract content from <think> and <answer> tags.

        The 2B model often emits malformed tag nesting:
          <think>...</think><answer>...</think><answer>...
        instead of properly closing </answer>. So we grab the first
        <think> block (to </think>), then the first <answer> block
        terminated by </answer>, </think>, or the next <answer> tag.
        """
        think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        # Match <answer> content terminated by </answer>, </think>, or next <answer>
        answer_match = re.search(
            r"<answer>(.*?)(?:</answer>|</think>|<answer>)", response, re.DOTALL
        )

        reasoning = think_match.group(1).strip() if think_match else ""
        answer = answer_match.group(1).strip() if answer_match else response.strip()

        return reasoning, answer
