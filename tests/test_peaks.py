"""Tests for peak detection utilities."""

import numpy as np
import pytest

from plottery.models import Peak, Series
from plottery.utils.peaks import (
    analyze_spectrum,
    detect_peaks,
    filter_noise_peaks,
    find_harmonics,
    find_peaks_in_series,
    interpolate_peak,
)


class TestPeakDetection:
    """Tests for peak detection functions."""

    def test_detect_single_peak(self):
        """Test detection of a single peak."""
        # Create series with one clear peak
        x = np.linspace(0, 100, 200)
        y = np.exp(-((x - 50) ** 2) / 20)  # Gaussian peak at x=50

        series = Series(name="test", color=(0, 0, 0), points=list(zip(x, y)))

        peaks = find_peaks_in_series(series, prominence=0.1)

        assert len(peaks) >= 1
        # Peak should be near x=50
        assert any(45 < p.x < 55 for p in peaks)

    def test_detect_multiple_peaks(self):
        """Test detection of multiple peaks."""
        x = np.linspace(0, 100, 500)
        y = (np.exp(-((x - 25) ** 2) / 10) +
             np.exp(-((x - 50) ** 2) / 10) +
             np.exp(-((x - 75) ** 2) / 10))

        series = Series(name="test", color=(0, 0, 0), points=list(zip(x, y)))

        peaks = find_peaks_in_series(series, prominence=0.3, min_distance=10)

        assert len(peaks) >= 3

    def test_detect_peaks_multiple_series(self):
        """Test peak detection across multiple series."""
        x = np.linspace(0, 100, 200)

        series1 = Series(
            name="s1", color=(255, 0, 0),
            points=list(zip(x, np.exp(-((x - 30) ** 2) / 20)))
        )
        series2 = Series(
            name="s2", color=(0, 0, 255),
            points=list(zip(x, np.exp(-((x - 70) ** 2) / 20)))
        )

        peaks = detect_peaks([series1, series2], prominence=0.1)

        # Should find peaks in both series
        s1_peaks = [p for p in peaks if p.series == "s1"]
        s2_peaks = [p for p in peaks if p.series == "s2"]

        assert len(s1_peaks) >= 1
        assert len(s2_peaks) >= 1

    def test_empty_series(self):
        """Test with empty series."""
        series = Series(name="empty", color=(0, 0, 0), points=[])

        peaks = find_peaks_in_series(series)

        assert len(peaks) == 0

    def test_short_series(self):
        """Test with very short series."""
        series = Series(name="short", color=(0, 0, 0), points=[(0, 1), (1, 2)])

        peaks = find_peaks_in_series(series)

        assert len(peaks) == 0


class TestHarmonicDetection:
    """Tests for harmonic detection."""

    def test_find_harmonics(self):
        """Test harmonic relationship detection."""
        # Create peaks at fundamental and harmonics
        peaks = [
            Peak(x=100, y=1.0, series="test", prominence=1.0),   # Fundamental
            Peak(x=200, y=0.5, series="test", prominence=0.5),   # 2nd harmonic
            Peak(x=300, y=0.3, series="test", prominence=0.3),   # 3rd harmonic
            Peak(x=400, y=0.2, series="test", prominence=0.2),   # 4th harmonic
        ]

        result = find_harmonics(peaks, fundamental_range=(50, 150))

        # Should identify harmonics
        harmonics = [p for p in result if p.is_harmonic]
        assert len(harmonics) >= 2

    def test_no_harmonics(self):
        """Test with unrelated peaks."""
        peaks = [
            Peak(x=100, y=1.0, series="test", prominence=1.0),
            Peak(x=137, y=0.5, series="test", prominence=0.5),  # Not a harmonic
            Peak(x=251, y=0.3, series="test", prominence=0.3),  # Not a harmonic
        ]

        result = find_harmonics(peaks, tolerance=0.02)

        harmonics = [p for p in result if p.is_harmonic]
        assert len(harmonics) == 0

    def test_harmonic_tolerance(self):
        """Test harmonic detection with tolerance."""
        # Peaks slightly off from exact harmonics
        peaks = [
            Peak(x=100, y=1.0, series="test", prominence=1.0),
            Peak(x=202, y=0.5, series="test", prominence=0.5),  # 2% off
            Peak(x=297, y=0.3, series="test", prominence=0.3),  # 1% off
        ]

        # With 5% tolerance, should find harmonics
        result = find_harmonics(peaks, tolerance=0.05)
        harmonics = [p for p in result if p.is_harmonic]
        assert len(harmonics) >= 1

        # Create fresh peaks for strict test (find_harmonics modifies peaks in-place)
        peaks_strict = [
            Peak(x=100, y=1.0, series="test", prominence=1.0),
            Peak(x=202, y=0.5, series="test", prominence=0.5),
            Peak(x=297, y=0.3, series="test", prominence=0.3),
        ]

        # With 0.5% tolerance, should miss them (202 is 1% off, 297 is 1% off)
        result_strict = find_harmonics(peaks_strict, tolerance=0.005)
        harmonics_strict = [p for p in result_strict if p.is_harmonic]
        assert len(harmonics_strict) == 0


class TestSpectrumAnalysis:
    """Tests for spectrum analysis."""

    def test_analyze_spectrum_with_harmonics(self):
        """Test full spectrum analysis."""
        # Create spectrum with fundamental and harmonics
        x = np.linspace(0, 1000, 2000)
        y = np.zeros_like(x)

        # Add peaks at 100Hz and harmonics
        for harmonic in [1, 2, 3, 4, 5]:
            freq = 100 * harmonic
            amplitude = 1.0 / harmonic
            y += amplitude * np.exp(-((x - freq) ** 2) / 50)

        # Add noise floor
        y += np.random.randn(len(y)) * 0.01

        series = Series(name="spectrum", color=(0, 0, 0), points=list(zip(x, y)))

        analysis = analyze_spectrum(series, fundamental_hint=100)

        assert analysis["peak_count"] >= 3
        assert analysis["fundamental"] is not None or analysis["harmonic_count"] >= 1

    def test_analyze_empty_spectrum(self):
        """Test with empty spectrum."""
        series = Series(name="empty", color=(0, 0, 0), points=[])

        analysis = analyze_spectrum(series)

        assert analysis["peaks"] == []
        assert analysis["fundamental"] is None


class TestPeakInterpolation:
    """Tests for peak interpolation."""

    def test_interpolate_peak(self):
        """Test sub-sample peak interpolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 0.5, 1.0, 0.5, 0.0])  # Peak at index 2

        x_interp, y_interp = interpolate_peak(x, y, 2)

        # Should be close to x=2, y=1
        assert x_interp == pytest.approx(2.0, abs=0.1)
        assert y_interp == pytest.approx(1.0, abs=0.1)

    def test_interpolate_asymmetric_peak(self):
        """Test interpolation of asymmetric peak."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 0.8, 1.0, 0.3, 0.0])  # Asymmetric peak

        x_interp, y_interp = interpolate_peak(x, y, 2)

        # Peak should be slightly left of center
        assert x_interp < 2.0

    def test_interpolate_edge_peak(self):
        """Test interpolation at array edge."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.5, 0.0])  # Peak at edge

        # Should handle edge case gracefully
        x_interp, y_interp = interpolate_peak(x, y, 0)
        assert x_interp == 0.0


class TestNoiseFiltering:
    """Tests for noise peak filtering."""

    def test_filter_noise_peaks(self):
        """Test filtering of noise peaks."""
        peaks = [
            Peak(x=100, y=10.0, series="test", prominence=1.0),   # Signal
            Peak(x=150, y=0.1, series="test", prominence=0.1),    # Noise
            Peak(x=200, y=8.0, series="test", prominence=0.8),    # Signal
            Peak(x=250, y=0.2, series="test", prominence=0.1),    # Noise
        ]

        noise_floor = 1.0
        filtered = filter_noise_peaks(peaks, noise_floor, min_snr=3.0)

        assert len(filtered) == 2
        assert all(p.y >= 3.0 for p in filtered)

    def test_filter_all_noise(self):
        """Test when all peaks are below threshold."""
        peaks = [
            Peak(x=100, y=0.5, series="test", prominence=0.1),
            Peak(x=200, y=0.3, series="test", prominence=0.1),
        ]

        filtered = filter_noise_peaks(peaks, noise_floor=1.0, min_snr=2.0)

        assert len(filtered) == 0
