#!/usr/bin/env python3
"""
Demo: Automatic context generation from paper text

This demo shows the complete workflow using the new Paper/Chart API:
1. Load a PDF with Paper class
2. Automatically extract text and find charts
3. Extract data with automatic context generation
4. Export to various formats

Usage:
    python demo/demo_auto_context.py
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from plottery import Paper, config

# Paths
PDF_PATH = Path(__file__).parent / "data/papers/frosini2010.pdf"
OUTPUT_DIR = Path(__file__).parent / "output/demo_auto_context"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("Plottery Demo: Automatic Context Generation")
    print("=" * 60)

    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found: {PDF_PATH}")
        return

    if not config.is_configured:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        return

    print(f"\nPDF: {PDF_PATH.name}")
    print(f"Model: {config.model}")
    print(f"Sample density: {config.sample_density}")

    # Step 1: Load paper
    print("\n" + "-" * 40)
    print("Step 1: Loading PDF...")
    print("-" * 40)

    paper = Paper(PDF_PATH, dpi=200)

    print(f"✓ Text extracted: {len(paper.text)} characters")
    print(f"✓ Charts found: {paper.num_charts}")

    if paper.text:
        print(f"\nText preview: {paper.text[:200]}...")

    # Step 2: Extract all charts with automatic context
    print("\n" + "-" * 40)
    print("Step 2: Extracting data with auto-context...")
    print("-" * 40)

    paper.extract_all(generate_context=True)

    # Step 3: Show results
    print("\n" + "-" * 40)
    print("Step 3: Results")
    print("-" * 40)

    for i, chart in enumerate(paper.charts):
        print(f"\n{'='*50}")
        print(f"Chart {i+1}/{len(paper.charts)} (page {chart.page+1})")
        print("=" * 50)

        # Save chart image
        img_path = OUTPUT_DIR / f"chart_{i+1}_page{chart.page+1}.png"
        chart.save_image(img_path)
        print(f"✓ Image saved: {img_path.name}")

        # Show extraction results
        if chart.is_extracted:
            print(f"✓ Type: {chart.type}")
            print(f"✓ Series: {chart.num_series}")
            print(f"✓ Points: {chart.num_points}")
            print(f"✓ Peaks: {len(chart.peaks)}")

            if chart.x_range and chart.x_range != (0.0, 1.0):
                print(f"✓ X range: {chart.x_range[0]} - {chart.x_range[1]} {chart.x_unit or ''}")
            if chart.y_range and chart.y_range != (0.0, 1.0):
                print(f"✓ Y range: {chart.y_range[0]} - {chart.y_range[1]} {chart.y_unit or ''}")

            if chart.is_categorical:
                print(f"✓ Categorical: {chart.categories}")

            if chart.context:
                print(f"\nContext (generated):")
                print(f"  {chart.context[:200]}...")

            # Save to CSV
            csv_path = OUTPUT_DIR / f"chart_{i+1}_data.csv"
            chart.to_csv(csv_path)
            print(f"\n✓ Data saved: {csv_path.name}")

            # Save context
            ctx_path = OUTPUT_DIR / f"chart_{i+1}_context.txt"
            with open(ctx_path, 'w') as f:
                f.write(f"Chart {i+1} - Page {chart.page+1}\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Type: {chart.type}\n")
                f.write(f"Series: {chart.num_series}\n")
                f.write(f"Points: {chart.num_points}\n")
                f.write(f"Peaks: {len(chart.peaks)}\n\n")
                f.write("Generated Context:\n")
                f.write(chart.context or "(none)")

        else:
            error = chart.metadata.get("error", "Unknown error")
            print(f"✗ Extraction failed: {error}")

    # Step 4: Summary and export
    print("\n" + "-" * 40)
    print("Step 4: Summary")
    print("-" * 40)

    print(f"\n✓ Charts processed: {len(paper.charts)}")
    print(f"✓ Successfully extracted: {len(paper.extracted_charts)}")
    print(f"✓ Total series: {paper.total_series}")
    print(f"✓ Total points: {paper.total_points}")

    # Export all to JSON
    json_path = OUTPUT_DIR / "all_charts.json"
    paper.to_json(json_path)
    print(f"\n✓ JSON export: {json_path.name}")

    # List output files
    print(f"\nOutput files in: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*")):
        print(f"  - {f.name}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
