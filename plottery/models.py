"""Data models for plottery.

This module defines the core data structures used to represent
extracted data from charts.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Series:
    """Represents a single data series extracted from a chart.

    Attributes:
        name: Name or identifier of the series (e.g., from legend)
        color: RGB color tuple of the series line/markers
        points: List of (x, y) coordinate pairs

    Example:
        >>> series = Series(
        ...     name="Motor Current",
        ...     color=(255, 0, 0),
        ...     points=[(0, 1.2), (1, 2.3), (2, 1.8)]
        ... )
        >>> df = series.to_dataframe()
        >>> print(series.x)  # [0, 1, 2]
    """
    name: str
    color: Tuple[int, int, int]  # RGB
    points: List[Tuple[float, float]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert series to a pandas DataFrame.

        Returns:
            DataFrame with columns 'x' and 'y'
        """
        if not self.points:
            return pd.DataFrame(columns=["x", "y"])
        x_vals, y_vals = zip(*self.points)
        return pd.DataFrame({"x": x_vals, "y": y_vals})

    def to_numpy(self) -> np.ndarray:
        """Convert series to numpy array of shape (n, 2).

        Returns:
            Array where each row is [x, y]
        """
        if not self.points:
            return np.empty((0, 2))
        return np.array(self.points)

    @property
    def x(self) -> np.ndarray:
        """Get x coordinates as numpy array."""
        if not self.points:
            return np.array([])
        return np.array([p[0] for p in self.points])

    @property
    def y(self) -> np.ndarray:
        """Get y coordinates as numpy array."""
        if not self.points:
            return np.array([])
        return np.array([p[1] for p in self.points])

    def __len__(self) -> int:
        """Number of points in the series."""
        return len(self.points)

    def __repr__(self) -> str:
        return f"Series(name='{self.name}', points={len(self.points)})"


@dataclass
class Peak:
    """Represents a detected peak in a data series.

    Attributes:
        x: X coordinate of the peak
        y: Y coordinate (amplitude) of the peak
        series: Name of the series this peak belongs to
        prominence: Prominence of the peak (height relative to surrounding)
        harmonic_of: If this is a harmonic, the fundamental frequency
        harmonic_order: If this is a harmonic, the order (2nd, 3rd, etc.)

    Example:
        >>> peak = Peak(x=50.0, y=-20.5, series="Spectrum", prominence=15.3)
        >>> print(f"Peak at {peak.x} Hz: {peak.y} dB")
    """
    x: float
    y: float
    series: str
    prominence: float = 0.0
    harmonic_of: Optional[float] = None
    harmonic_order: Optional[int] = None

    @property
    def is_harmonic(self) -> bool:
        """Check if this peak is a harmonic of another frequency."""
        return self.harmonic_of is not None

    def __repr__(self) -> str:
        if self.is_harmonic:
            return f"Peak(x={self.x}, y={self.y}, harmonic={self.harmonic_order})"
        return f"Peak(x={self.x}, y={self.y})"
