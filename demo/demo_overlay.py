#!/usr/bin/env python3
"""
Demo: Visual Overlay Verification

Overlay extracted data points on the original chart image to verify accuracy.

Usage:
    python demo/demo_overlay.py                    # Use pre-calibrated or estimated bounds
    python demo/demo_overlay.py --interactive     # Click on plot corners to calibrate
    python demo/demo_overlay.py -i                # Short form
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from plottery import Chart

# Paths
IMAGE_PATH = Path(__file__).parent / "data/images/demo.png"
OUTPUT_DIR = Path(__file__).parent / "output/overlay"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def data_to_pixel(x_data, y_data, x_range, y_range, plot_bounds):
    """
    Convert data coordinates to pixel coordinates.

    Args:
        x_data, y_data: Data coordinates
        x_range: (x_min, x_max) in data units
        y_range: (y_min, y_max) in data units
        plot_bounds: (left, right, top, bottom) in pixels

    Returns:
        (x_pixel, y_pixel)
    """
    left, right, top, bottom = plot_bounds
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Normalize to 0-1
    x_norm = (x_data - x_min) / (x_max - x_min)
    y_norm = (y_data - y_min) / (y_max - y_min)

    # Convert to pixels
    x_pixel = left + x_norm * (right - left)
    y_pixel = bottom - y_norm * (bottom - top)  # Y is inverted in images

    return x_pixel, y_pixel


def find_plot_bounds_interactive(image):
    """
    Let user click on plot corners to define bounds.
    Only needs 2 clicks: top-left and bottom-right corners.
    """
    print("\n" + "=" * 50)
    print("INTERACTIVE CALIBRATION")
    print("=" * 50)
    print("\nClick on 2 points:")
    print("  1. TOP-LEFT corner of plot area (origin of axes)")
    print("  2. BOTTOM-RIGHT corner of plot area")
    print("\nThe window will close automatically after 2 clicks.")
    print("=" * 50 + "\n")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(image)
    ax.set_title("Click: 1) TOP-LEFT corner  2) BOTTOM-RIGHT corner", fontsize=14, fontweight='bold')

    clicks = []

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            clicks.append((event.xdata, event.ydata))
            color = 'green' if len(clicks) == 1 else 'red'
            label = 'TOP-LEFT' if len(clicks) == 1 else 'BOTTOM-RIGHT'
            ax.plot(event.xdata, event.ydata, 'o', markersize=12,
                   markerfacecolor=color, markeredgecolor='white', markeredgewidth=2)
            ax.annotate(f'{len(clicks)}: {label}', (event.xdata + 15, event.ydata),
                       color=color, fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            fig.canvas.draw()

            if len(clicks) >= 2:
                # Draw rectangle
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                     fill=False, edgecolor='yellow', linewidth=2)
                ax.add_patch(rect)
                ax.set_title("Plot area selected! Close window to continue.",
                           fontsize=14, fontweight='bold', color='green')
                fig.canvas.draw()

                # Auto-close after brief delay
                fig.canvas.flush_events()
                import time
                time.sleep(1)
                plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    if len(clicks) >= 2:
        x1, y1 = clicks[0]  # top-left
        x2, y2 = clicks[1]  # bottom-right
        bounds = (int(x1), int(x2), int(y1), int(y2))
        print(f"\nCalibrated bounds: {bounds}")
        print(f"  Left: {bounds[0]}, Right: {bounds[1]}")
        print(f"  Top: {bounds[2]}, Bottom: {bounds[3]}")
        return bounds

    return None


def estimate_plot_bounds(image):
    """
    Estimate plot bounds automatically (simple heuristic).
    Assumes plot area is roughly 10-90% of image.
    """
    h, w = image.shape[:2]

    # Typical matplotlib-style plot margins
    left = int(w * 0.12)
    right = int(w * 0.95)
    top = int(h * 0.08)
    bottom = int(h * 0.88)

    return (left, right, top, bottom)


# Pre-calibrated bounds for known images
KNOWN_BOUNDS = {
    # demo.png: 1234x910, X: 0-2000 Hz, Y: -180 to 0 dB
    "demo.png": (75, 1190, 35, 830),
}


def overlay_extraction(chart, plot_bounds=None, interactive=False):
    """
    Create overlay visualization of extracted data on original image.
    """
    if not chart.is_extracted:
        print("Chart not extracted yet!")
        return

    image = chart.image

    # Get or estimate plot bounds
    if interactive:
        plot_bounds = find_plot_bounds_interactive(image)
    elif plot_bounds is None:
        plot_bounds = estimate_plot_bounds(image)

    if plot_bounds is None:
        print("Could not determine plot bounds")
        return

    # Get data ranges from extraction
    x_range = chart.x_range or (0, 1)
    y_range = chart.y_range or (0, 1)

    print(f"\nPlot bounds (pixels): left={plot_bounds[0]}, right={plot_bounds[1]}, "
          f"top={plot_bounds[2]}, bottom={plot_bounds[3]}")
    print(f"X range (data): {x_range}")
    print(f"Y range (data): {y_range}")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Right: Image with overlay
    axes[1].imshow(image)
    axes[1].set_title("Extracted Data Overlay")

    # Draw plot bounds rectangle
    left, right, top, bottom = plot_bounds
    rect = mpatches.Rectangle((left, top), right-left, bottom-top,
                               fill=False, edgecolor='green', linewidth=2, linestyle='--')
    axes[1].add_patch(rect)

    # Colors for series
    colors = ['red', 'blue', 'orange', 'purple', 'cyan', 'magenta']

    # Overlay extracted points
    legend_handles = []
    for i, series in enumerate(chart.series):
        color = colors[i % len(colors)]

        x_pixels = []
        y_pixels = []

        for x_data, y_data in series.points:
            x_px, y_px = data_to_pixel(x_data, y_data, x_range, y_range, plot_bounds)
            x_pixels.append(x_px)
            y_pixels.append(y_px)

        # Plot points
        axes[1].scatter(x_pixels, y_pixels, c=color, s=10, alpha=0.7, marker='o')

        # Connect with line
        axes[1].plot(x_pixels, y_pixels, c=color, alpha=0.5, linewidth=1)

        # Legend
        handle = mpatches.Patch(color=color, label=f"{series.name} ({len(series.points)} pts)")
        legend_handles.append(handle)

    # Overlay peaks
    if chart.peaks:
        peak_x_pixels = []
        peak_y_pixels = []
        for peak in chart.peaks:
            x_px, y_px = data_to_pixel(peak.x, peak.y, x_range, y_range, plot_bounds)
            peak_x_pixels.append(x_px)
            peak_y_pixels.append(y_px)

        axes[1].scatter(peak_x_pixels, peak_y_pixels, c='yellow', s=100,
                       marker='*', edgecolors='black', linewidths=1, zorder=5)
        handle = mpatches.Patch(color='yellow', label=f"Peaks ({len(chart.peaks)})")
        legend_handles.append(handle)

    axes[1].legend(handles=legend_handles, loc='upper right')
    axes[1].axis('off')

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visual overlay verification for Plottery")
    parser.add_argument("-i", "--interactive", action="store_true",
                       help="Manually click on plot corners to calibrate")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to chart image (default: demo.png)")
    args = parser.parse_args()

    print("=" * 60)
    print("Plottery Demo: Visual Overlay Verification")
    if args.interactive:
        print("(Interactive mode)")
    print("=" * 60)

    # Use custom image or default
    image_path = Path(args.image) if args.image else IMAGE_PATH

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    # Load and extract
    print(f"\nLoading: {image_path.name}")
    chart = Chart.from_image(image_path)

    print("Extracting data...")
    chart.extract(context="Motor frequency spectrum with fault analysis")

    print(f"\nExtraction results:")
    print(f"  Type: {chart.type}")
    print(f"  Series: {chart.num_series}")
    print(f"  Points: {chart.num_points}")
    print(f"  Peaks: {len(chart.peaks)}")
    print(f"  X range: {chart.x_range}")
    print(f"  Y range: {chart.y_range}")

    # Show series details
    for series in chart.series:
        print(f"\n  Series '{series.name}':")
        print(f"    Points: {len(series.points)}")
        if series.points:
            x_vals = [p[0] for p in series.points]
            y_vals = [p[1] for p in series.points]
            print(f"    X: {min(x_vals):.1f} - {max(x_vals):.1f}")
            print(f"    Y: {min(y_vals):.1f} - {max(y_vals):.1f}")

    # Create overlay
    print("\n" + "-" * 40)
    print("Creating overlay visualization...")
    print("-" * 40)

    # Determine plot bounds
    if args.interactive:
        print("Interactive mode: click on plot corners...")
        plot_bounds = None
        interactive = True
    else:
        interactive = False
        image_name = image_path.name
        if image_name in KNOWN_BOUNDS:
            plot_bounds = KNOWN_BOUNDS[image_name]
            print(f"Using pre-calibrated bounds for {image_name}")
        else:
            plot_bounds = None
            print("Using estimated bounds (may need adjustment)")
            print("Tip: Use --interactive to calibrate manually")

    fig = overlay_extraction(chart, plot_bounds=plot_bounds, interactive=interactive)

    if fig:
        # Save
        output_path = OUTPUT_DIR / "overlay_result.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {output_path}")

        # Show
        plt.show()

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    print("\nTips:")
    print("  - If overlay doesn't match: python demo/demo_overlay.py --interactive")
    print("  - For custom image: python demo/demo_overlay.py --image path/to/chart.png -i")


if __name__ == "__main__":
    main()
