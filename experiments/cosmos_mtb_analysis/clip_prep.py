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
4. Optionally stabilizes (basic deshake -- Cosmos may handle shake fine, test both)
5. Exports clips at a target resolution and frame rate
"""

import argparse
import cv2
import os
from pathlib import Path
from datetime import datetime, timedelta


def prepare_clip(
    input_path: str,
    output_dir: str,
    start_sec: float = None,
    end_sec: float = None,
    target_fps: int = 4,
    max_duration_sec: int = 30,
    add_timestamps: bool = True,
    stabilize: bool = False,
    target_height: int = 720
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
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_duration = total_frames / original_fps if original_fps > 0 else 0

    # Determine time window
    start_sec = start_sec or 0.0
    end_sec = end_sec or total_duration
    end_sec = min(end_sec, start_sec + max_duration_sec)
    end_sec = min(end_sec, total_duration)
    duration = end_sec - start_sec

    # Calculate output dimensions preserving aspect ratio
    scale = target_height / orig_h
    out_w = int(orig_w * scale)
    # Ensure even dimensions for codec compatibility
    out_w = out_w + (out_w % 2)
    out_h = target_height + (target_height % 2)

    # Frame sampling: pick frames at target_fps intervals from the source
    frame_interval = original_fps / target_fps
    start_frame = int(start_sec * original_fps)
    end_frame = int(end_sec * original_fps)

    frames_to_extract = []
    f = start_frame
    while f < end_frame:
        frames_to_extract.append(int(f))
        f += frame_interval

    # Output path
    stem = Path(input_path).stem
    out_name = f"{stem}_prepared_{target_fps}fps_{int(duration)}s.mp4"
    output_path = os.path.join(output_dir, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (out_w, out_h))

    prev_gray = None
    frame_count = 0

    for frame_idx in frames_to_extract:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Resize
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

        # Basic stabilization using optical flow
        if stabilize:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                try:
                    transform = cv2.estimateRigidTransform(prev_gray, gray, False)
                    if transform is not None:
                        frame = cv2.warpAffine(frame, transform, (out_w, out_h))
                except Exception:
                    pass  # Skip stabilization on failure
            prev_gray = gray

        # Timestamp overlay
        if add_timestamps:
            elapsed = (frame_idx - start_frame) / original_fps
            ts_text = f"{int(elapsed // 60):02d}:{elapsed % 60:05.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(ts_text, font, font_scale, thickness)
            # Black background for readability
            cv2.rectangle(frame, (5, 5), (15 + tw, 15 + th), (0, 0, 0), -1)
            cv2.putText(frame, ts_text, (10, 10 + th), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        writer.write(frame)
        frame_count += 1

    writer.release()
    cap.release()

    return {
        "output_path": output_path,
        "duration_sec": round(duration, 2),
        "frame_count": frame_count,
        "original_fps": original_fps,
        "target_fps": target_fps,
        "resolution": (out_w, out_h),
        "source": input_path,
    }


def extract_keyframes(
    input_path: str,
    output_dir: str,
    mode: str = "interval",
    interval_sec: float = 1.0,
    scene_threshold: float = 30.0
) -> list:
    """
    Extract representative keyframes for individual image analysis.

    For MTB footage, scene_change mode may work well because:
    - Trail sections have consistent visual themes
    - Transitions (forest->clearing, flat->descent) create natural breakpoints

    Returns list of dicts with frame_path and timestamp_sec.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    stem = Path(input_path).stem
    keyframes = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps
        save = False

        if mode == "interval":
            # Save a frame every interval_sec seconds
            if frame_idx == 0 or (frame_idx % int(fps * interval_sec)) == 0:
                save = True

        elif mode == "scene_change":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = diff.mean()
                if mean_diff > scene_threshold:
                    save = True
            else:
                save = True  # Always save first frame
            prev_gray = gray

        if save:
            frame_path = os.path.join(output_dir, f"{stem}_frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            keyframes.append({
                "frame_path": frame_path,
                "timestamp_sec": round(timestamp, 3),
                "frame_index": frame_idx,
            })

        frame_idx += 1

    cap.release()
    print(f"Extracted {len(keyframes)} keyframes from {input_path} ({mode} mode)")
    return keyframes


def extract_peripheral_strips(
    input_path: str,
    output_dir: str,
    strip_width_pct: float = 0.30,
    interval_sec: float = 2.0,
) -> dict:
    """Extract left/right peripheral strips. Wrapper around extract_zones."""
    return extract_zones(
        input_path, output_dir,
        zones=["left", "right"],
        interval_sec=interval_sec,
        side_pct=strip_width_pct,
    )


# Zone definitions for frame splitting:
#   left   = left 30% of frame width (exposure, vegetation)
#   right  = right 30% of frame width (exposure, vegetation)
#   center = middle 40% of frame width (trail surface, obstacles)
#   top    = top 35% of frame height (sightlines, canopy, horizon)
#   bottom = bottom 30% of frame height (immediate surface texture)
ZONE_PRESETS = {
    "left":   lambda w, h, sp, tp, bp: (0, 0, int(w * sp), h),
    "right":  lambda w, h, sp, tp, bp: (w - int(w * sp), 0, w, h),
    "center": lambda w, h, sp, tp, bp: (int(w * sp), 0, w - int(w * sp), h),
    "top":    lambda w, h, sp, tp, bp: (0, 0, w, int(h * tp)),
    "bottom": lambda w, h, sp, tp, bp: (0, h - int(h * bp), w, h),
}


def extract_zones(
    input_path: str,
    output_dir: str,
    zones: list = None,
    interval_sec: float = 2.0,
    side_pct: float = 0.30,
    top_pct: float = 0.35,
    bottom_pct: float = 0.30,
) -> dict:
    """
    Extract cropped zone frames from video for focused analysis.

    Splits each frame into spatial zones so the model analyzes one
    region at a time. This is deterministic preprocessing -- the data
    does the work instead of the prompt.

    Zones:
        left    - left 30% width: drop-offs, vegetation, exposure
        right   - right 30% width: drop-offs, vegetation, exposure
        center  - middle 40% width: trail surface, obstacles, width
        top     - top 35% height: sightlines, canopy, blind corners
        bottom  - bottom 30% height: immediate surface texture

    Args:
        input_path: Source video file
        output_dir: Where to save zone crops
        zones: Which zones to extract (default: all five)
        interval_sec: Extract one set every N seconds
        side_pct: Width fraction for left/right strips
        top_pct: Height fraction for top strip
        bottom_pct: Height fraction for bottom strip

    Returns:
        dict keyed by zone name, each a list of {frame_path, timestamp_sec}
    """
    if zones is None:
        zones = list(ZONE_PRESETS.keys())

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = Path(input_path).stem
    frame_interval = int(fps * interval_sec)

    results = {z: [] for z in zones}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp = round(frame_idx / fps, 3)

            for zone_name in zones:
                x1, y1, x2, y2 = ZONE_PRESETS[zone_name](w, h, side_pct, top_pct, bottom_pct)
                crop = frame[y1:y2, x1:x2]
                zone_dir = os.path.join(output_dir, zone_name)
                os.makedirs(zone_dir, exist_ok=True)
                path = os.path.join(zone_dir, f"{stem}_{zone_name}_{frame_idx:06d}.jpg")
                cv2.imwrite(path, crop)
                results[zone_name].append({
                    "frame_path": path,
                    "timestamp_sec": timestamp,
                })

        frame_idx += 1

    cap.release()
    total = sum(len(v) for v in results.values())
    print(f"Extracted {total} zone crops across {len(zones)} zones from {input_path}")
    for z in zones:
        print(f"  {z}: {len(results[z])} frames")
    return results


def batch_prepare(
    input_dir: str,
    output_dir: str,
    max_clips: int = None,
    **kwargs
) -> list:
    """Process all video files in a directory."""
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV"}
    input_path = Path(input_dir)
    videos = [f for f in input_path.iterdir() if f.suffix in video_exts]
    videos.sort()

    if max_clips:
        videos = videos[:max_clips]

    results = []
    for i, video in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] Preparing {video.name}...")
        try:
            result = prepare_clip(str(video), output_dir, **kwargs)
            results.append(result)
            print(f"  -> {result['frame_count']} frames, {result['duration_sec']}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"source": str(video), "error": str(e)})

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MTB clips for Cosmos analysis")
    parser.add_argument("input", help="Video file or directory of videos")
    parser.add_argument("--output-dir", default="test_data/prepared", help="Output directory")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument("--fps", type=int, default=4, help="Target FPS (default: 4)")
    parser.add_argument("--max-duration", type=int, default=30, help="Max clip duration (seconds)")
    parser.add_argument("--stabilize", action="store_true", help="Apply basic stabilization")
    parser.add_argument("--keyframes-only", action="store_true", help="Extract keyframes instead of video")
    parser.add_argument("--keyframe-mode", choices=["interval", "scene_change"], default="interval")
    parser.add_argument("--keyframe-interval", type=float, default=1.0, help="Seconds between keyframes")
    parser.add_argument("--scene-threshold", type=float, default=30.0, help="Scene change sensitivity")
    parser.add_argument("--peripheral-strips", action="store_true",
                        help="Extract left/right edge strips for drop-off analysis")
    parser.add_argument("--zones", nargs="+",
                        choices=["left", "right", "center", "top", "bottom", "all"],
                        help="Extract zone crops (left, right, center, top, bottom, or all)")
    parser.add_argument("--strip-width", type=float, default=0.30,
                        help="Strip width as fraction of frame (default: 0.30)")
    parser.add_argument("--strip-interval", type=float, default=2.0,
                        help="Seconds between strip/zone extractions (default: 2.0)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.zones:
        zone_list = list(ZONE_PRESETS.keys()) if "all" in args.zones else args.zones
        extract_zones(
            str(input_path), args.output_dir,
            zones=zone_list,
            interval_sec=args.strip_interval,
            side_pct=args.strip_width,
        )
    elif args.peripheral_strips:
        extract_peripheral_strips(
            str(input_path), args.output_dir,
            strip_width_pct=args.strip_width,
            interval_sec=args.strip_interval
        )
    elif args.keyframes_only:
        if input_path.is_dir():
            video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
            for v in sorted(input_path.iterdir()):
                if v.suffix.lower() in video_exts:
                    extract_keyframes(
                        str(v), args.output_dir,
                        mode=args.keyframe_mode,
                        interval_sec=args.keyframe_interval,
                        scene_threshold=args.scene_threshold
                    )
        else:
            extract_keyframes(
                str(input_path), args.output_dir,
                mode=args.keyframe_mode,
                interval_sec=args.keyframe_interval,
                scene_threshold=args.scene_threshold
            )
    elif input_path.is_dir():
        batch_prepare(
            str(input_path), args.output_dir,
            start_sec=args.start, end_sec=args.end,
            target_fps=args.fps, max_duration_sec=args.max_duration,
            stabilize=args.stabilize
        )
    else:
        result = prepare_clip(
            str(input_path), args.output_dir,
            start_sec=args.start, end_sec=args.end,
            target_fps=args.fps, max_duration_sec=args.max_duration,
            stabilize=args.stabilize
        )
        print(f"\nPrepared: {result['output_path']}")
        print(f"  Duration: {result['duration_sec']}s")
        print(f"  Frames: {result['frame_count']} @ {result['target_fps']} FPS")
        print(f"  Resolution: {result['resolution']}")
