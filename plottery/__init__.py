"""Plottery: Extract numerical data from scientific chart images.

This library uses Claude's vision capabilities to extract numerical data
from various types of scientific charts including line charts, bar charts,
scatter plots, histograms, and stacked bar charts.

Basic usage with PDF:
    >>> from plottery import Paper
    >>> paper = Paper("paper.pdf")
    >>> paper.extract_all()
    >>> paper.to_excel("output.xlsx")

Basic usage with single image:
    >>> from plottery import Chart
    >>> chart = Chart.from_image("spectrum.png")
    >>> chart.extract(context="Motor frequency spectrum")
    >>> chart.to_csv("data.csv")

Configuration:
    >>> from plottery import config
    >>> config.sample_density = "high"
    >>> config.detect_peaks = False

"""

from .paper import Paper
from .chart import Chart
from .models import Series, Peak
from .config import config

__version__ = "0.2.0"
__all__ = [
    "Paper",
    "Chart",
    "Series",
    "Peak",
    "config",
]
