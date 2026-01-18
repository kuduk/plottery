#!/usr/bin/env python3
"""
Demo: Simple usage of Plottery with new API

This demo shows the basic usage of Plottery:
1. Load a PDF and extract all charts
2. Load a single image and extract data

Usage:
    python demo/demo_simple.py
"""

import os
import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load API key from .env
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.startswith("ANTHROPIC_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()

from plottery import Paper, Chart, config

# Paths
PDF_PATH = Path(__file__).parent / "data/papers/frosini2010.pdf"
OUTPUT_DIR = Path(__file__).parent / "output/demo_simple"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_pdf():
    """Demo: Extract charts from PDF."""
    print("\n" + "=" * 60)
    print("Demo 1: Extract charts from PDF")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"PDF not found: {PDF_PATH}")
        return

    # Load paper
    print(f"\nLoading: {PDF_PATH.name}")
    paper = Paper(PDF_PATH)

    print(f"Text extracted: {len(paper.text)} characters")
    print(f"Charts found: {paper.num_charts}")

    # Extract all charts
    print("\nExtracting data from all charts...")
    paper.extract_all()

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

    # Export
    print("\nExporting...")
    csv_files = paper.to_csv(OUTPUT_DIR)
    print(f"✓ CSV files: {len(csv_files)}")
    for f in csv_files:
        print(f"    - {f.name}")

    json_path = OUTPUT_DIR / "paper_data.json"
    paper.to_json(json_path)
    print(f"✓ JSON: {json_path.name}")


def demo_single_image():
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
    print("=" * 60)
    print("Plottery Demo: New Simple API")
    print("=" * 60)

    if not config.is_configured:
        print("\nWARNING: ANTHROPIC_API_KEY not set.")
        print("Set it in .env file or environment variable.")
        return

    demo_pdf()
    demo_single_image()
    demo_config()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
