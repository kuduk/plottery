"""Tests for data models."""

import numpy as np
import pandas as pd
import pytest

from plottery.models import Peak, Series


class TestSeries:
    """Tests for Series data model."""

    def test_empty_series(self):
        """Test creation of empty series."""
        series = Series(name="test", color=(255, 0, 0), points=[])
        assert len(series) == 0
        assert series.name == "test"
        assert series.color == (255, 0, 0)

    def test_series_with_points(self):
        """Test series with data points."""
        points = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
        series = Series(name="line1", color=(0, 0, 255), points=points)

        assert len(series) == 3
        assert series.points == points

    def test_series_x_property(self):
        """Test x coordinate extraction."""
        points = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
        series = Series(name="test", color=(0, 0, 0), points=points)

        x = series.x
        assert isinstance(x, np.ndarray)
        np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])

    def test_series_y_property(self):
        """Test y coordinate extraction."""
        points = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
        series = Series(name="test", color=(0, 0, 0), points=points)

        y = series.y
        assert isinstance(y, np.ndarray)
        np.testing.assert_array_equal(y, [10.0, 20.0, 30.0])

    def test_series_to_dataframe(self):
        """Test DataFrame conversion."""
        points = [(0.0, 1.0), (1.0, 2.0)]
        series = Series(name="test", color=(0, 0, 0), points=points)

        df = series.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 2

    def test_series_to_numpy(self):
        """Test numpy array conversion."""
        points = [(0.0, 1.0), (1.0, 2.0)]
        series = Series(name="test", color=(0, 0, 0), points=points)

        arr = series.to_numpy()
        assert arr.shape == (2, 2)

    def test_series_repr(self):
        """Test Series string representation."""
        points = [(0.0, 1.0), (1.0, 2.0)]
        series = Series(name="test", color=(0, 0, 0), points=points)

        assert "test" in repr(series)
        assert "2" in repr(series)  # num points


class TestPeak:
    """Tests for Peak data model."""

    def test_basic_peak(self):
        """Test basic peak creation."""
        peak = Peak(x=100.0, y=-20.0, series="blue", prominence=5.0)

        assert peak.x == 100.0
        assert peak.y == -20.0
        assert peak.series == "blue"
        assert peak.prominence == 5.0
        assert not peak.is_harmonic

    def test_harmonic_peak(self):
        """Test peak with harmonic info."""
        peak = Peak(
            x=200.0, y=-25.0, series="blue",
            prominence=3.0, harmonic_of=100.0, harmonic_order=2
        )

        assert peak.is_harmonic
        assert peak.harmonic_of == 100.0
        assert peak.harmonic_order == 2

    def test_peak_repr(self):
        """Test Peak string representation."""
        peak = Peak(x=100.0, y=-20.0, series="blue")
        assert "100.0" in repr(peak)
        assert "-20.0" in repr(peak)

    def test_harmonic_peak_repr(self):
        """Test harmonic Peak string representation."""
        peak = Peak(
            x=200.0, y=-25.0, series="blue",
            harmonic_of=100.0, harmonic_order=2
        )
        assert "harmonic" in repr(peak)
        assert "2" in repr(peak)
