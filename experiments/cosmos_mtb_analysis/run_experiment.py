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
import sys
from datetime import datetime
from pathlib import Path

from cosmos_mtb import CosmosMTBAnalyzer


def generate_report(clip_name: str, results: dict, output_path: str):
    """
    Generate a human-readable markdown report from all category results.

    Rates each category: STRONG / MODERATE / WEAK / FAILED
    Highlights interesting reasoning chains and documents inference times.
    """
    lines = [
        f"# Cosmos MTB Analysis: {clip_name}",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
    ]

    # Summary table
    total_time = sum(r["inference_time_sec"] for r in results.values())
    lines.append(f"**Total inference time:** {total_time:.1f}s across {len(results)} categories")
    lines.append("")
    lines.append("| Category | Time (s) | Answer Length | Has Reasoning |")
    lines.append("|----------|----------|--------------|---------------|")

    for cat, r in results.items():
        answer_len = len(r["answer"])
        has_reasoning = "Yes" if r["reasoning"] else "No"
        lines.append(f"| {cat} | {r['inference_time_sec']} | {answer_len} chars | {has_reasoning} |")

    lines.append("")

    # Detailed results per category
    for cat, r in results.items():
        lines.append(f"---")
        lines.append(f"## {cat.replace('_', ' ').title()}")
        lines.append(f"*Inference: {r['inference_time_sec']}s*")
        lines.append("")

        if r["reasoning"]:
            lines.append("### Reasoning Chain")
            lines.append("```")
            # Truncate very long reasoning for readability
            reasoning = r["reasoning"]
            if len(reasoning) > 2000:
                reasoning = reasoning[:2000] + "\n... [truncated]"
            lines.append(reasoning)
            lines.append("```")
            lines.append("")

        lines.append("### Analysis")
        lines.append(r["answer"])
        lines.append("")

    # Capability assessment placeholder
    lines.append("---")
    lines.append("## Model Capability Assessment")
    lines.append("")
    lines.append("*Rate each category after reviewing the results:*")
    lines.append("")
    lines.append("| Category | Rating | Notes |")
    lines.append("|----------|--------|-------|")
    for cat in results:
        lines.append(f"| {cat} | _TODO_ | |")
    lines.append("")
    lines.append("**Rating key:** STRONG (specific, novel, accurate) / "
                  "MODERATE (some useful, some generic) / "
                  "WEAK (vague, generic, or hallucinated) / "
                  "FAILED (no relevant output)")

    report = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def run_single_clip(analyzer: CosmosMTBAnalyzer, clip_path: str, categories: list,
                    output_base: str) -> dict:
    """Analyze a single clip and save all outputs."""
    clip_name = Path(clip_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, f"{timestamp}_{clip_name}")
    per_cat_dir = os.path.join(run_dir, "per_category")
    os.makedirs(per_cat_dir, exist_ok=True)

    print(f"\nAnalyzing: {clip_name}")
    print(f"Categories: {categories or 'all'}")
    print(f"Output: {run_dir}")
    print("-" * 50)

    results = analyzer.analyze_clip(clip_path, categories=categories)

    # Save individual category results
    for cat, r in results.items():
        cat_path = os.path.join(per_cat_dir, f"{cat}.json")
        with open(cat_path, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2, ensure_ascii=False)

    # Save combined summary
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save metadata
    meta = {
        "clip_path": os.path.abspath(clip_path),
        "clip_name": clip_name,
        "model": analyzer.model_name,
        "categories_run": list(results.keys()),
        "total_inference_sec": round(sum(r["inference_time_sec"] for r in results.values()), 2),
        "timestamp": timestamp,
    }
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Generate report
    report_path = os.path.join(run_dir, "report.md")
    report = generate_report(clip_name, results, report_path)

    # Print highlights
    print("\n" + "=" * 50)
    print("HIGHLIGHTS")
    print("=" * 50)
    for cat, r in results.items():
        answer_preview = r["answer"][:200] + "..." if len(r["answer"]) > 200 else r["answer"]
        print(f"\n[{cat}] ({r['inference_time_sec']}s)")
        print(f"  {answer_preview}")

    print(f"\nFull report: {report_path}")
    return results


def run_frames(analyzer: CosmosMTBAnalyzer, frames_dir: str, categories: list,
               output_base: str) -> dict:
    """Analyze keyframes from a directory."""
    frame_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted([
        f for f in Path(frames_dir).iterdir()
        if f.suffix.lower() in frame_exts
    ])

    if not frames:
        print(f"No image files found in {frames_dir}")
        return {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_base, f"{timestamp}_frames")
    os.makedirs(run_dir, exist_ok=True)

    all_results = {}
    for i, frame_path in enumerate(frames):
        print(f"\n[Frame {i+1}/{len(frames)}] {frame_path.name}")
        results = analyzer.analyze_frame(str(frame_path), categories=categories)
        all_results[frame_path.name] = results

        # Save per-frame results
        frame_out = os.path.join(run_dir, f"{frame_path.stem}.json")
        with open(frame_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Save combined
    summary_path = os.path.join(run_dir, "frames_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nFrame results saved to: {run_dir}")
    return all_results


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

    if not any([args.clip, args.clip_dir, args.frames]):
        parser.error("Provide --clip, --clip-dir, or --frames")

    analyzer = CosmosMTBAnalyzer(model_name=args.model)

    if args.clip:
        run_single_clip(analyzer, args.clip, args.categories, args.output_dir)

    elif args.clip_dir:
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
        clips = sorted([
            f for f in Path(args.clip_dir).iterdir()
            if f.suffix.lower() in video_exts
        ])
        if not clips:
            print(f"No video files found in {args.clip_dir}")
            sys.exit(1)

        print(f"Found {len(clips)} clips to analyze")
        for clip in clips:
            run_single_clip(analyzer, str(clip), args.categories, args.output_dir)

        # Cross-clip summary
        print(f"\n{'=' * 50}")
        print(f"Batch complete: {len(clips)} clips analyzed")
        print(f"Results in: {args.output_dir}/")

    elif args.frames:
        run_frames(analyzer, args.frames, args.categories, args.output_dir)


if __name__ == "__main__":
    main()
