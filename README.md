# Plottery

**Extract numerical data from scientific chart images**

Plottery is a Python library for extracting numerical data from scientific charts and plots embedded in images or PDFs. It uses computer vision, OCR, and optionally VLM (Vision Language Models) to automatically detect chart types, calibrate axes, and extract data points.

## Features

- **Multi-format support**: Extract from PNG, JPG, PDF files
- **Chart type detection**: Automatically classifies line charts, bar charts, scatter plots, pie charts
- **OCR integration**: Reads axis labels and tick marks for automatic calibration
- **VLM support**: Uses Claude API for enhanced chart analysis (optional)
- **Peak detection**: Identifies peaks with harmonic analysis for spectrum data
- **Multiple series**: Detects and separates multiple data series by color
- **CLI included**: Command-line interface for quick extraction

## Installation

### Basic installation

```bash
pip install plottery
```

### With optional dependencies

```bash
# OCR support (for automatic axis calibration)
pip install plottery[ocr]

# PDF support
pip install plottery[pdf]

# VLM support (Claude API for chart classification)
pip install plottery[vlm]

# CLI tools
pip install plottery[cli]

# All optional dependencies
pip install plottery[all]
```

## Quick Start

### Python API

```python
import plottery

# Extract data from a chart image
result = plottery.extract("spectrum.png")

# Access extracted series
for series in result.series:
    print(f"Series: {series.name}, Points: {len(series)}")

# Convert to DataFrame
df = result.to_dataframe()
print(df.head())

# Export to CSV
result.to_csv("output.csv")

# Access peaks (for spectrum data)
for peak in result.peaks:
    print(f"Peak at x={peak.x:.2f}, y={peak.y:.2f}")
```

### With manual calibration

```python
result = plottery.extract(
    "chart.png",
    calibration={
        "x_range": (0, 2000),  # Hz
        "y_range": (-180, 0),   # dB
        "x_unit": "Hz",
        "y_unit": "dB",
    }
)
```

### CLI Usage

```bash
# Extract data from an image
plottery extract spectrum.png -o data.csv

# Analyze chart without extracting
plottery analyze chart.png -v

# Extract with manual calibration
plottery extract plot.png --x-min 0 --x-max 1000 --y-min -100 --y-max 0

# Extract images from PDF
plottery pdf-images paper.pdf -o ./images/

# Detect peaks in spectrum
plottery peaks spectrum.png -o peaks.csv
```

## Chart Types

Plottery supports extraction from:

- **Line charts** / Spectra / Time series
- **Bar charts** / Histograms
- **Scatter plots**
- **Pie charts**

## Architecture

```
plottery/
├── core/
│   ├── calibrator.py   # Axis calibration
│   ├── router.py       # Chart type classification
│   └── validator.py    # Result validation
├── extractors/
│   ├── base.py         # Base extractor class
│   ├── line_chart.py   # Line/spectrum extraction
│   ├── bar_chart.py    # Bar chart extraction
│   ├── scatter.py      # Scatter plot extraction
│   └── pie_chart.py    # Pie chart extraction
├── utils/
│   ├── colors.py       # Color segmentation
│   ├── ocr.py          # OCR utilities
│   ├── peaks.py        # Peak detection
│   └── pdf.py          # PDF processing
├── agents/
│   └── planner.py      # LLM-based planning
└── cli.py              # Command-line interface
```

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY`: API key for Claude VLM features

### Using VLM for Classification

When an Anthropic API key is available, Plottery uses Claude for:
- Accurate chart type classification
- Reading axis labels and units
- Identifying legend entries
- Suggesting calibration values

Without the API key, heuristic-based classification is used.

## Data Models

### Series

Represents a single data series:

```python
series.name      # Series name (e.g., "blue", "Signal A")
series.color     # RGB color tuple
series.points    # List of (x, y) tuples
series.x         # X values as numpy array
series.y         # Y values as numpy array
```

### Peak

Represents a detected peak:

```python
peak.x           # X coordinate
peak.y           # Y coordinate (amplitude)
peak.prominence  # Peak prominence
peak.harmonic_of # Fundamental frequency (if harmonic)
peak.harmonic_order  # Harmonic order (2nd, 3rd, etc.)
```

### ExtractionResult

Container for extraction output:

```python
result.series      # List of Series
result.peaks       # List of Peak
result.calibration # Calibrator used
result.metadata    # Additional info
result.to_dataframe()  # Convert to pandas DataFrame
result.to_csv(path)    # Export to CSV
```

## Development

### Setup

```bash
git clone https://github.com/plottery/plottery.git
cd plottery
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
plottery/
├── pyproject.toml
├── README.md
├── plottery/          # Main package
├── tests/             # Test suite
├── demo/              # Demo data
└── documentazione/    # Documentation (IT)
```

## License

MIT License

## Contributing

Contributions are welcome! Please open an issue or pull request.
