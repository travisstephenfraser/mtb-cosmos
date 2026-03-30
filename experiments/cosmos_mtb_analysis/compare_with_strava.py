"""
Compare Cosmos visual observations with Strava GPS/sensor data.

The idea:
- Cosmos says "steep descent starting around 0:12" from video
- Strava GPX/FIT data shows elevation drop of 50ft at the corresponding timestamp
- Do they agree?

If Cosmos visual terrain estimates correlate with actual GPS data, that's a
strong signal that the model genuinely understands terrain geometry from video.
If they don't correlate, the spatial reasoning is likely superficial.

Usage:
  python compare_with_strava.py \
    --cosmos-results results/20260330_trail1/summary.json \
    --gpx strava_exports/activity_12345.gpx \
    --video-start-offset 300  # video starts 5 min into the activity
"""

import argparse
import json
import re
from pathlib import Path

import gpxpy
import pandas as pd


def load_gpx(gpx_path: str) -> pd.DataFrame:
    """
    Parse GPX file into DataFrame with:
    - timestamp, lat, lon, elevation
    - Derived: speed_mps, gradient_pct, distance_m (cumulative)
    """
    with open(gpx_path, "r") as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append({
                    "timestamp": point.time,
                    "lat": point.latitude,
                    "lon": point.longitude,
                    "elevation_m": point.elevation,
                })

    df = pd.DataFrame(points)
    if df.empty:
        return df

    # Derive distance, speed, gradient
    from math import radians, sin, cos, sqrt, atan2

    distances = [0.0]
    for i in range(1, len(df)):
        lat1, lon1 = radians(df.iloc[i-1]["lat"]), radians(df.iloc[i-1]["lon"])
        lat2, lon2 = radians(df.iloc[i]["lat"]), radians(df.iloc[i]["lon"])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distances.append(6371000 * c)  # meters

    df["segment_distance_m"] = distances
    df["distance_m"] = df["segment_distance_m"].cumsum()

    # Time deltas
    df["dt_sec"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

    # Speed (m/s)
    df["speed_mps"] = df.apply(
        lambda r: r["segment_distance_m"] / r["dt_sec"] if r["dt_sec"] > 0 else 0, axis=1
    )
    df["speed_mph"] = df["speed_mps"] * 2.237

    # Gradient (%)
    df["elev_change_m"] = df["elevation_m"].diff().fillna(0)
    df["gradient_pct"] = df.apply(
        lambda r: (r["elev_change_m"] / r["segment_distance_m"] * 100)
        if r["segment_distance_m"] > 1 else 0,
        axis=1
    )

    # Elapsed seconds from start
    df["elapsed_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    return df


def load_fit(fit_path: str) -> pd.DataFrame:
    """
    Parse FIT file (richer data than GPX):
    - timestamp, lat, lon, elevation
    - heart_rate, cadence, power (if available)
    - speed, temperature
    """
    from fitparse import FitFile

    fit = FitFile(fit_path)
    records = []

    for record in fit.get_messages("record"):
        data = {}
        for field in record.fields:
            data[field.name] = field.value
        records.append(data)

    df = pd.DataFrame(records)

    # Normalize column names
    rename_map = {
        "position_lat": "lat",
        "position_long": "lon",
        "altitude": "elevation_m",
        "enhanced_altitude": "elevation_m",
        "enhanced_speed": "speed_mps",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # FIT stores lat/lon as semicircles
    if "lat" in df.columns and df["lat"].abs().max() > 180:
        df["lat"] = df["lat"] * (180 / 2**31)
        df["lon"] = df["lon"] * (180 / 2**31)

    if "timestamp" in df.columns:
        df["elapsed_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

    return df


def extract_timestamp_references(cosmos_results: dict) -> list:
    """
    Parse Cosmos output for timestamp references like "at 0:12", "around 0:15",
    "at approximately 0:20-0:25", etc.

    Returns list of dicts with:
    - category: which prompt produced this
    - timestamp_sec: parsed time in seconds
    - context: surrounding text for the reference
    - observation: what the model claimed at this time
    """
    refs = []
    # Match patterns like 0:12, 0:15-0:20, 00:12, etc.
    ts_pattern = re.compile(r'(\d{1,2}):(\d{2})(?:\s*[-–]\s*(\d{1,2}):(\d{2}))?')

    for category, result in cosmos_results.items():
        text = result.get("answer", "") + " " + result.get("reasoning", "")
        for match in ts_pattern.finditer(text):
            start_sec = int(match.group(1)) * 60 + int(match.group(2))
            end_sec = None
            if match.group(3) and match.group(4):
                end_sec = int(match.group(3)) * 60 + int(match.group(4))

            # Get surrounding context (50 chars each side)
            start_idx = max(0, match.start() - 50)
            end_idx = min(len(text), match.end() + 50)
            context = text[start_idx:end_idx].strip()

            refs.append({
                "category": category,
                "timestamp_sec": start_sec,
                "end_sec": end_sec,
                "context": context,
                "match": match.group(0),
            })

    return sorted(refs, key=lambda r: r["timestamp_sec"])


def align_timestamps(cosmos_results: dict, gps_data: pd.DataFrame,
                     video_start_offset_sec: float) -> list:
    """
    Map Cosmos timestamp references to GPS trackpoints.

    Returns list of aligned observation pairs.
    """
    refs = extract_timestamp_references(cosmos_results)
    aligned = []

    for ref in refs:
        # Convert video time to activity time
        activity_sec = ref["timestamp_sec"] + video_start_offset_sec

        # Find closest GPS point
        idx = (gps_data["elapsed_sec"] - activity_sec).abs().idxmin()
        gps_point = gps_data.iloc[idx]

        # Get a window of GPS data around this point (+/- 5 seconds)
        window = gps_data[
            (gps_data["elapsed_sec"] >= activity_sec - 5) &
            (gps_data["elapsed_sec"] <= activity_sec + 5)
        ]

        aligned.append({
            "cosmos_ref": ref,
            "gps_point": {
                "elapsed_sec": float(gps_point["elapsed_sec"]),
                "elevation_m": float(gps_point.get("elevation_m", 0)),
                "speed_mps": float(gps_point.get("speed_mps", 0)),
                "speed_mph": float(gps_point.get("speed_mph", 0)),
                "gradient_pct": float(gps_point.get("gradient_pct", 0)),
            },
            "gps_window": {
                "avg_speed_mph": float(window["speed_mph"].mean()) if "speed_mph" in window else None,
                "elevation_change_m": float(window["elevation_m"].iloc[-1] - window["elevation_m"].iloc[0])
                    if len(window) > 1 and "elevation_m" in window else None,
                "avg_gradient_pct": float(window["gradient_pct"].mean()) if "gradient_pct" in window else None,
            }
        })

    return aligned


def compare_terrain_estimates(aligned_data: list) -> dict:
    """
    Compare Cosmos terrain observations with GPS ground truth.

    Returns analysis of how well visual observations match sensor data.
    """
    comparisons = []

    for item in aligned_data:
        ref = item["cosmos_ref"]
        gps = item["gps_point"]
        window = item["gps_window"]
        context = ref["context"].lower()

        comparison = {
            "video_time": ref["match"],
            "category": ref["category"],
            "context": ref["context"],
            "gps_speed_mph": gps["speed_mph"],
            "gps_gradient_pct": gps["gradient_pct"],
            "gps_elevation_m": gps["elevation_m"],
        }

        # Check speed claims
        if any(w in context for w in ["fast", "accelerat", "high speed", "rapid"]):
            comparison["speed_claim"] = "fast"
            comparison["speed_match"] = gps["speed_mph"] > 10
        elif any(w in context for w in ["slow", "decelerat", "brak", "stop"]):
            comparison["speed_claim"] = "slow"
            comparison["speed_match"] = gps["speed_mph"] < 8
        else:
            comparison["speed_claim"] = None
            comparison["speed_match"] = None

        # Check gradient claims
        if any(w in context for w in ["steep descent", "steep downhill", "steep drop"]):
            comparison["gradient_claim"] = "steep_descent"
            comparison["gradient_match"] = gps["gradient_pct"] < -8
        elif any(w in context for w in ["descent", "downhill", "drop"]):
            comparison["gradient_claim"] = "descent"
            comparison["gradient_match"] = gps["gradient_pct"] < -2
        elif any(w in context for w in ["steep climb", "steep uphill"]):
            comparison["gradient_claim"] = "steep_climb"
            comparison["gradient_match"] = gps["gradient_pct"] > 8
        elif any(w in context for w in ["climb", "uphill", "ascen"]):
            comparison["gradient_claim"] = "climb"
            comparison["gradient_match"] = gps["gradient_pct"] > 2
        elif any(w in context for w in ["flat", "level"]):
            comparison["gradient_claim"] = "flat"
            comparison["gradient_match"] = abs(gps["gradient_pct"]) < 3
        else:
            comparison["gradient_claim"] = None
            comparison["gradient_match"] = None

        comparisons.append(comparison)

    # Aggregate accuracy
    speed_checks = [c for c in comparisons if c["speed_match"] is not None]
    gradient_checks = [c for c in comparisons if c["gradient_match"] is not None]

    return {
        "comparisons": comparisons,
        "speed_accuracy": {
            "total": len(speed_checks),
            "correct": sum(1 for c in speed_checks if c["speed_match"]),
            "rate": sum(1 for c in speed_checks if c["speed_match"]) / len(speed_checks)
                if speed_checks else None,
        },
        "gradient_accuracy": {
            "total": len(gradient_checks),
            "correct": sum(1 for c in gradient_checks if c["gradient_match"]),
            "rate": sum(1 for c in gradient_checks if c["gradient_match"]) / len(gradient_checks)
                if gradient_checks else None,
        },
        "total_timestamp_refs": len(aligned_data),
    }


def generate_comparison_report(comparison: dict, output_path: str):
    """Markdown report: Cosmos observations vs GPS ground truth."""
    lines = [
        "# Cosmos vs Strava GPS Comparison",
        f"*Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary",
        "",
        f"- **Timestamp references found:** {comparison['total_timestamp_refs']}",
    ]

    speed = comparison["speed_accuracy"]
    gradient = comparison["gradient_accuracy"]

    if speed["total"] > 0:
        lines.append(f"- **Speed claims:** {speed['correct']}/{speed['total']} correct "
                      f"({speed['rate']*100:.0f}%)")
    else:
        lines.append("- **Speed claims:** none detected")

    if gradient["total"] > 0:
        lines.append(f"- **Gradient claims:** {gradient['correct']}/{gradient['total']} correct "
                      f"({gradient['rate']*100:.0f}%)")
    else:
        lines.append("- **Gradient claims:** none detected")

    lines.extend(["", "## Detailed Comparisons", ""])
    lines.append("| Video Time | Category | Cosmos Says | GPS Speed (mph) | GPS Gradient (%) | Speed Match | Gradient Match |")
    lines.append("|------------|----------|-------------|-----------------|------------------|-------------|----------------|")

    for c in comparison["comparisons"]:
        speed_match = {True: "Yes", False: "**NO**", None: "-"}.get(c["speed_match"], "-")
        grad_match = {True: "Yes", False: "**NO**", None: "-"}.get(c["gradient_match"], "-")
        context_short = c["context"][:60] + "..." if len(c["context"]) > 60 else c["context"]
        lines.append(
            f"| {c['video_time']} | {c['category']} | {context_short} | "
            f"{c['gps_speed_mph']:.1f} | {c['gps_gradient_pct']:.1f} | "
            f"{speed_match} | {grad_match} |"
        )

    lines.extend([
        "",
        "## Assessment",
        "",
        "*Fill in after reviewing:*",
        "",
        "- Is the spatial reasoning **grounded** (correlates with GPS) or **superficial** (generic guesses)?",
        "- Which categories produce the most GPS-aligned observations?",
        "- Are gradient estimates directionally correct even if magnitude is off?",
    ])

    report = "\n".join(lines)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Comparison report saved to: {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Cosmos analysis with Strava data")
    parser.add_argument("--cosmos-results", required=True, help="Path to Cosmos results JSON")
    parser.add_argument("--gpx", help="Strava GPX export")
    parser.add_argument("--fit", help="Strava FIT file")
    parser.add_argument("--video-start-offset", type=float, default=0,
                        help="Seconds into the activity when video recording started")
    parser.add_argument("--output", default="findings/strava_comparison.md")
    args = parser.parse_args()

    if not args.gpx and not args.fit:
        parser.error("Provide --gpx or --fit file")

    # Load Cosmos results
    with open(args.cosmos_results, "r") as f:
        cosmos_results = json.load(f)

    # Load GPS data
    if args.gpx:
        gps_data = load_gpx(args.gpx)
    else:
        gps_data = load_fit(args.fit)

    print(f"GPS data: {len(gps_data)} trackpoints, "
          f"{gps_data['elapsed_sec'].max():.0f}s duration")

    # Align and compare
    aligned = align_timestamps(cosmos_results, gps_data, args.video_start_offset)
    print(f"Found {len(aligned)} timestamp references in Cosmos output")

    comparison = compare_terrain_estimates(aligned)
    generate_comparison_report(comparison, args.output)
