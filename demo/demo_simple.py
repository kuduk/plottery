#!/usr/bin/env python3
"""
Plottery - Minimal Example

The simplest way to use Plottery: load a PDF and extract all charts.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from plottery import Paper

# Load PDF and extract all charts
paper = Paper("demo/data/papers/frosini2010.pdf")
paper.extract_all()

# Show results
print(f"Found {paper.num_charts} charts, extracted {paper.total_points} points")

for chart in paper.charts:
    print(f"  Page {chart.page+1}: {chart.type} - {chart.num_series} series, {chart.num_points} pts")

# Export
paper.to_csv("demo/output/simple/")
print("\nData exported to demo/output/simple/")
