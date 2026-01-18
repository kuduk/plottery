"""Peak detection and harmonic analysis utilities."""

from typing import List, Optional, Tuple

import numpy as np
from scipy import signal

from ..models import Peak, Series


def detect_peaks(
    series_list: List[Series],
    prominence: float = 0.1,
    min_distance: int = 5,
    relative_height: float = 0.5,
) -> List[Peak]:
    """Detect peaks in all data series.

    Args:
        series_list: List of Series objects to analyze
        prominence: Minimum prominence relative to data range
        min_distance: Minimum number of samples between peaks
        relative_height: Height at which to measure peak width

    Returns:
        List of Peak objects
    """
    all_peaks = []

    for series in series_list:
        if len(series) < 3:
            continue

        peaks = find_peaks_in_series(
            series,
            prominence=prominence,
            min_distance=min_distance,
            relative_height=relative_height,
        )
        all_peaks.extend(peaks)

    return all_peaks


def find_peaks_in_series(
    series: Series,
    prominence: float = 0.1,
    min_distance: int = 5,
    relative_height: float = 0.5,
    max_peaks: int = 50,
) -> List[Peak]:
    """Find peaks in a single data series.

    Args:
        series: Series to analyze
        prominence: Minimum prominence relative to data range
        min_distance: Minimum samples between peaks
        relative_height: Height for width measurement
        max_peaks: Maximum number of peaks to return

    Returns:
        List of Peak objects
    """
    if len(series) < 3:
        return []

    x_vals = series.x
    y_vals = series.y

    # Calculate prominence threshold
    data_range = y_vals.max() - y_vals.min()
    if data_range == 0:
        return []

    prominence_abs = prominence * data_range

    # Find peaks
    peak_indices, properties = signal.find_peaks(
        y_vals,
        prominence=prominence_abs,
        distance=min_distance,
        rel_height=relative_height,
    )

    if len(peak_indices) == 0:
        return []

    # Get prominences for found peaks
    prominences = properties.get('prominences', np.ones(len(peak_indices)))

    # Sort by prominence and limit
    sorted_indices = np.argsort(-prominences)[:max_peaks]

    peaks = []
    for idx in sorted_indices:
        peak_idx = peak_indices[idx]
        peak = Peak(
            x=float(x_vals[peak_idx]),
            y=float(y_vals[peak_idx]),
            series=series.name,
            prominence=float(prominences[idx]),
        )
        peaks.append(peak)

    return peaks


def find_harmonics(
    peaks: List[Peak],
    fundamental_range: Tuple[float, float] = None,
    tolerance: float = 0.05,
    max_harmonic: int = 10,
) -> List[Peak]:
    """Identify harmonic relationships between peaks.

    This function finds peaks that are integer multiples of a fundamental
    frequency, which is common in vibration and acoustic spectra.

    Args:
        peaks: List of peaks to analyze
        fundamental_range: Optional (min, max) range to search for fundamental
        tolerance: Relative tolerance for harmonic matching
        max_harmonic: Maximum harmonic order to detect

    Returns:
        List of peaks with harmonic relationships identified
    """
    if len(peaks) < 2:
        return peaks

    # Sort peaks by x (frequency)
    peaks_sorted = sorted(peaks, key=lambda p: p.x)

    # Get x values
    x_values = [p.x for p in peaks_sorted]

    # Find potential fundamental frequencies
    if fundamental_range is not None:
        f_min, f_max = fundamental_range
        fundamental_candidates = [
            p for p in peaks_sorted if f_min <= p.x <= f_max
        ]
    else:
        # Use lowest frequency peaks as candidates
        fundamental_candidates = peaks_sorted[:5]

    # For each fundamental candidate, check for harmonics
    best_fundamental = None
    best_harmonic_count = 0

    for candidate in fundamental_candidates:
        f0 = candidate.x
        if f0 == 0:
            continue

        harmonic_count = 0
        for peak in peaks_sorted:
            if peak.x == f0:
                continue

            # Check if peak is a harmonic
            ratio = peak.x / f0
            nearest_int = round(ratio)

            if 2 <= nearest_int <= max_harmonic:
                relative_error = abs(ratio - nearest_int) / nearest_int
                if relative_error < tolerance:
                    harmonic_count += 1

        if harmonic_count > best_harmonic_count:
            best_harmonic_count = harmonic_count
            best_fundamental = f0

    # Mark harmonics
    if best_fundamental is not None and best_harmonic_count >= 2:
        for peak in peaks_sorted:
            ratio = peak.x / best_fundamental
            nearest_int = round(ratio)

            if 2 <= nearest_int <= max_harmonic:
                relative_error = abs(ratio - nearest_int) / nearest_int
                if relative_error < tolerance:
                    peak.harmonic_of = best_fundamental
                    peak.harmonic_order = nearest_int

    return peaks_sorted


def analyze_spectrum(
    series: Series,
    fundamental_hint: Optional[float] = None,
) -> dict:
    """Analyze a frequency spectrum for peaks and harmonics.

    Args:
        series: Series containing spectrum data
        fundamental_hint: Optional hint for fundamental frequency

    Returns:
        Dictionary with analysis results
    """
    # Find peaks
    peaks = find_peaks_in_series(
        series,
        prominence=0.05,
        min_distance=3,
    )

    if len(peaks) == 0:
        return {
            "peaks": [],
            "fundamental": None,
            "harmonics": [],
            "noise_floor": None,
        }

    # Estimate noise floor (median of lower values)
    y_vals = series.y
    noise_floor = float(np.percentile(y_vals, 20))

    # Find harmonics
    if fundamental_hint is not None:
        fundamental_range = (
            fundamental_hint * 0.8,
            fundamental_hint * 1.2,
        )
    else:
        # Look in lower 20% of frequency range
        x_range = series.x.max() - series.x.min()
        fundamental_range = (
            series.x.min(),
            series.x.min() + x_range * 0.2,
        )

    peaks_with_harmonics = find_harmonics(
        peaks,
        fundamental_range=fundamental_range,
    )

    # Identify fundamental
    fundamental = None
    harmonics = []

    for peak in peaks_with_harmonics:
        if peak.harmonic_of is not None:
            harmonics.append(peak)
            if fundamental is None or peak.harmonic_of < fundamental:
                fundamental = peak.harmonic_of

    return {
        "peaks": peaks_with_harmonics,
        "fundamental": fundamental,
        "harmonics": harmonics,
        "noise_floor": noise_floor,
        "peak_count": len(peaks),
        "harmonic_count": len(harmonics),
    }


def interpolate_peak(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    peak_idx: int,
) -> Tuple[float, float]:
    """Interpolate peak position for sub-sample accuracy.

    Uses parabolic interpolation on the peak and its neighbors.

    Args:
        x_vals: X coordinate array
        y_vals: Y coordinate array
        peak_idx: Index of the peak

    Returns:
        Tuple of (interpolated_x, interpolated_y)
    """
    if peak_idx == 0 or peak_idx >= len(y_vals) - 1:
        return (float(x_vals[peak_idx]), float(y_vals[peak_idx]))

    # Get three points around peak
    y0 = y_vals[peak_idx - 1]
    y1 = y_vals[peak_idx]
    y2 = y_vals[peak_idx + 1]

    # Parabolic interpolation
    denominator = y0 - 2 * y1 + y2
    if abs(denominator) < 1e-10:
        return (float(x_vals[peak_idx]), float(y_vals[peak_idx]))

    delta = 0.5 * (y0 - y2) / denominator

    # Interpolate x position
    x_step = x_vals[peak_idx] - x_vals[peak_idx - 1]
    x_interp = x_vals[peak_idx] + delta * x_step

    # Interpolate y value
    y_interp = y1 - 0.25 * (y0 - y2) * delta

    return (float(x_interp), float(y_interp))


def filter_noise_peaks(
    peaks: List[Peak],
    noise_floor: float,
    min_snr: float = 3.0,
) -> List[Peak]:
    """Filter out peaks that are likely noise.

    Args:
        peaks: List of peaks
        noise_floor: Estimated noise floor level
        min_snr: Minimum signal-to-noise ratio

    Returns:
        Filtered list of peaks
    """
    if noise_floor == 0:
        return peaks

    return [
        p for p in peaks
        if abs(p.y) / abs(noise_floor) >= min_snr
    ]
