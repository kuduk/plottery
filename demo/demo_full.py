#!/usr/bin/env python3
"""
Demo: Full usage of Plottery with new API

This demo shows all features of Plottery:
1. Load a PDF and extract all charts
2. Load a single image and extract data
3. Configuration options

Usage:
    python demo/demo_full.py                    # Normal mode
    python demo/demo_full.py --debug            # Debug mode (save images, context, etc.)
    python demo/demo_full.py --parallel 4       # Parallel extraction with 4 workers
    python demo/demo_full.py --debug -p 4       # Both options
"""

import argparse
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from plottery import Paper, Chart, config

# Paths
PDF_PATH = Path(__file__).parent / "data/papers/frosini2010.pdf"
OUTPUT_DIR = Path(__file__).parent / "output/demo_full"


def save_debug_info(chart, index: int, output_dir: Path):
    """Save debug information for a chart."""
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True)

    prefix = f"chart_{index}_page{chart.page+1}"

    # Save chart image
    img_path = debug_dir / f"{prefix}.png"
    chart.save_image(img_path)

    # Save debug info as JSON
    debug_info = {
        "index": index,
        "page": chart.page + 1,
        "type": chart.type,
        "extracted": chart.is_extracted,
        "num_series": chart.num_series,
        "num_points": chart.num_points,
        "num_peaks": len(chart.peaks),
        "x_range": chart.x_range,
        "y_range": chart.y_range,
        "x_unit": chart.x_unit,
        "y_unit": chart.y_unit,
        "is_categorical": chart.is_categorical,
        "categories": chart.categories,
        "context": chart.context,
        "metadata": {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
                     for k, v in chart.metadata.items()},
        "series_info": [
            {
                "name": s.name,
                "color": list(s.color),
                "num_points": len(s.points),
                "x_min": min(p[0] for p in s.points) if s.points else None,
                "x_max": max(p[0] for p in s.points) if s.points else None,
                "y_min": min(p[1] for p in s.points) if s.points else None,
                "y_max": max(p[1] for p in s.points) if s.points else None,
            }
            for s in chart.series
        ],
        "peaks_info": [
            {
                "x": p.x,
                "y": p.y,
                "series": p.series,
                "prominence": p.prominence,
                "is_harmonic": p.is_harmonic,
                "harmonic_of": p.harmonic_of,
                "harmonic_order": p.harmonic_order,
            }
            for p in chart.peaks
        ],
    }

    info_path = debug_dir / f"{prefix}_debug.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(debug_info, f, indent=2, ensure_ascii=False)

    # Save context as text file (easier to read)
    if chart.context:
        ctx_path = debug_dir / f"{prefix}_context.txt"
        with open(ctx_path, "w", encoding="utf-8") as f:
            f.write(f"Chart {index} - Page {chart.page + 1}\n")
            f.write("=" * 50 + "\n\n")
            f.write("Generated Context:\n")
            f.write("-" * 30 + "\n")
            f.write(chart.context + "\n\n")
            f.write("Metadata:\n")
            f.write("-" * 30 + "\n")
            for k, v in chart.metadata.items():
                f.write(f"  {k}: {v}\n")

    return img_path, info_path


def demo_pdf(debug: bool = False, parallel: int = 1):
    """Demo: Extract charts from PDF."""
    print("\n" + "=" * 60)
    print("Demo 1: Extract charts from PDF")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load paper
    print(f"\nLoading: {PDF_PATH.name}")
    paper = Paper(PDF_PATH)

    print(f"Text extracted: {len(paper.text)} characters")
    print(f"Charts found: {paper.num_charts}")

    # Progress callback
    def on_progress(completed, total, chart):
        status = "✓" if chart.is_extracted else "✗"
        print(f"  [{completed}/{total}] Chart page {chart.page+1}: {chart.type or 'N/A'} {status}")

    # Extract all charts
    mode = f"parallel ({parallel} workers)" if parallel > 1 else "sequential"
    print(f"\nExtracting data from all charts ({mode})...")

    start_time = time.time()
    paper.extract_all(max_workers=parallel, on_progress=on_progress)
    elapsed = time.time() - start_time

    print(f"\n⏱ Extraction time: {elapsed:.1f}s")

    # Summary
    print(f"\n✓ Extracted charts: {len(paper.extracted_charts)}")
    print(f"✓ Total series: {paper.total_series}")
    print(f"✓ Total points: {paper.total_points}")

    # Show individual charts
    for i, chart in enumerate(paper.charts):
        print(f"\n  Chart {i+1} (page {chart.page+1}):")
        print(f"    Type: {chart.type}")
        print(f"    Series: {chart.num_series}")
        print(f"    Points: {chart.num_points}")
        if chart.x_range:
            print(f"    X range: {chart.x_range}")
        if chart.y_range:
            print(f"    Y range: {chart.y_range}")
        if chart.is_categorical:
            print(f"    Categories: {chart.categories}")

        # Save debug info if enabled
        if debug:
            img_path, info_path = save_debug_info(chart, i+1, OUTPUT_DIR)
            print(f"    [debug] Image: {img_path.name}")
            print(f"    [debug] Info: {info_path.name}")

    # Export
    print("\nExporting...")
    csv_files = paper.to_csv(OUTPUT_DIR)
    print(f"✓ CSV files: {len(csv_files)}")
    for f in csv_files:
        print(f"    - {f.name}")

    json_path = OUTPUT_DIR / "paper_data.json"
    paper.to_json(json_path)
    print(f"✓ JSON: {json_path.name}")

    if debug:
        print(f"✓ Debug files: {OUTPUT_DIR / 'debug'}")


def demo_single_image(debug: bool = False):
    """Demo: Extract data from single image."""
    print("\n" + "=" * 60)
    print("Demo 2: Extract data from single image")
    print("=" * 60)

    # Check for any chart image in output
    image_candidates = list((Path(__file__).parent / "output").glob("**/chart_*.png"))
    if not image_candidates:
        print("No chart images found. Run demo_pdf() first.")
        return

    image_path = image_candidates[0]
    print(f"\nLoading: {image_path.name}")

    # Load chart
    chart = Chart.from_image(image_path)

    # Extract with context
    print("Extracting data...")
    chart.extract(context="Scientific chart from motor fault analysis paper")

    print(f"\n✓ Type: {chart.type}")
    print(f"✓ Series: {chart.num_series}")
    print(f"✓ Points: {chart.num_points}")
    print(f"✓ Peaks: {len(chart.peaks)}")

    # Show series details
    for s in chart.series:
        print(f"\n  Series '{s.name}':")
        print(f"    Color: {s.color}")
        print(f"    Points: {len(s.points)}")
        if s.points:
            print(f"    X range: {min(p[0] for p in s.points):.2f} - {max(p[0] for p in s.points):.2f}")
            print(f"    Y range: {min(p[1] for p in s.points):.2f} - {max(p[1] for p in s.points):.2f}")

    # Export
    csv_path = OUTPUT_DIR / "single_chart.csv"
    chart.to_csv(csv_path)
    print(f"\n✓ Exported to: {csv_path.name}")

    # Save debug info if enabled
    if debug:
        img_path, info_path = save_debug_info(chart, 0, OUTPUT_DIR)
        print(f"✓ Debug image: {img_path.name}")
        print(f"✓ Debug info: {info_path.name}")


def demo_config():
    """Demo: Configuration options."""
    print("\n" + "=" * 60)
    print("Demo 3: Configuration options")
    print("=" * 60)

    print(f"\nCurrent configuration:")
    print(f"  Model: {config.model}")
    print(f"  Sample density: {config.sample_density}")
    print(f"  Detect peaks: {config.detect_peaks}")
    print(f"  API key set: {config.is_configured}")

    print("\nYou can modify configuration globally:")
    print('  config.sample_density = "high"  # More data points')
    print('  config.sample_density = "low"   # Fewer data points')
    print("  config.detect_peaks = False     # Disable peak detection")


def main():
    parser = argparse.ArgumentParser(description="Plottery Demo")
    parser.add_argument("--debug", "-d", action="store_true",
                        help="Enable debug mode: save images, context, and metadata")
    parser.add_argument("--parallel", "-p", type=int, default=1,
                        help="Number of parallel workers for extraction (default: 1 = sequential)")
    args = parser.parse_args()

    print("=" * 60)
    print("Plottery Demo: New Simple API")
    options = []
    if args.debug:
        options.append("debug")
    if args.parallel > 1:
        options.append(f"parallel={args.parallel}")
    if options:
        print(f"({', '.join(options)})")
    print("=" * 60)

    if not config.is_configured:
        print("\nWARNING: ANTHROPIC_API_KEY not set.")
        print("Set it in .env file or environment variable.")
        return

    demo_pdf(debug=args.debug, parallel=args.parallel)
    demo_single_image(debug=args.debug)
    demo_config()

    print("\n" + "=" * 60)
    print("Demo completed!")
    if args.debug:
        print(f"Debug files saved in: {OUTPUT_DIR / 'debug'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
