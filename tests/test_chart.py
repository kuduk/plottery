"""Tests for Chart class."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from plottery.chart import Chart
from plottery.models import Series, Peak


class TestChartCreation:
    """Tests for Chart creation methods."""

    def test_chart_from_array(self):
        """Test creating chart from numpy array."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        assert chart.image.shape == (100, 100, 3)
        assert chart.page == 0
        assert not chart.is_extracted

    def test_chart_from_array_with_page(self):
        """Test creating chart with page number."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image, page=5)

        assert chart.page == 5

    def test_chart_from_image_not_found(self):
        """Test error when image file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Chart.from_image("/nonexistent/path/image.png")


class TestChartProperties:
    """Tests for Chart properties."""

    def test_num_points_empty(self):
        """Test num_points with no series."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        assert chart.num_points == 0
        assert chart.num_series == 0

    def test_num_points_with_series(self):
        """Test num_points with series data."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        chart.series = [
            Series(name="s1", color=(255, 0, 0), points=[(0, 1), (1, 2), (2, 3)]),
            Series(name="s2", color=(0, 0, 255), points=[(0, 0.5), (1, 1.5)]),
        ]

        assert chart.num_points == 5
        assert chart.num_series == 2

    def test_is_categorical_false_by_default(self):
        """Test is_categorical property."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        assert not chart.is_categorical
        assert chart.categories == []
        assert chart.category_mapping == {}

    def test_is_categorical_with_metadata(self):
        """Test is_categorical with metadata set."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        chart.metadata = {
            "is_categorical": True,
            "categories": ["A", "B", "C"],
            "category_mapping": {0: "A", 1: "B", 2: "C"},
        }

        assert chart.is_categorical
        assert chart.categories == ["A", "B", "C"]
        assert chart.category_mapping == {0: "A", 1: "B", 2: "C"}


class TestChartDataExport:
    """Tests for Chart data export methods."""

    def test_to_dataframe_empty(self):
        """Test DataFrame export with no data."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        df = chart.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["x", "y", "series"]

    def test_to_dataframe_with_series(self):
        """Test DataFrame export with series."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        chart.series = [
            Series(name="s1", color=(255, 0, 0), points=[(0, 1), (1, 2)]),
            Series(name="s2", color=(0, 0, 255), points=[(0, 0.5)]),
        ]

        df = chart.to_dataframe()

        assert len(df) == 3
        assert "series" in df.columns
        assert set(df["series"].unique()) == {"s1", "s2"}

    def test_to_csv(self, tmp_path):
        """Test CSV export."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        chart.series = [
            Series(name="test", color=(0, 0, 0), points=[(0, 1), (1, 2)]),
        ]

        csv_path = tmp_path / "output.csv"
        chart.to_csv(csv_path)

        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert len(df) == 2


class TestChartSeriesAccess:
    """Tests for Chart series access methods."""

    def test_get_series_found(self):
        """Test getting series by name."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        series = Series(name="test", color=(255, 0, 0), points=[(0, 1)])
        chart.series = [series]

        found = chart.get_series("test")

        assert found is series

    def test_get_series_not_found(self):
        """Test getting non-existent series."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        found = chart.get_series("nonexistent")

        assert found is None

    def test_get_peaks_for_series(self):
        """Test getting peaks for specific series."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        chart.peaks = [
            Peak(x=100, y=10, series="s1", prominence=1.0),
            Peak(x=200, y=20, series="s2", prominence=1.0),
            Peak(x=300, y=30, series="s1", prominence=1.0),
        ]

        s1_peaks = chart.get_peaks_for_series("s1")

        assert len(s1_peaks) == 2
        assert all(p.series == "s1" for p in s1_peaks)


class TestChartRepr:
    """Tests for Chart string representation."""

    def test_repr_not_extracted(self):
        """Test repr for non-extracted chart."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        repr_str = repr(chart)

        assert "Chart" in repr_str
        assert "not extracted" in repr_str

    def test_repr_extracted(self):
        """Test repr for extracted chart."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        chart._extracted = True
        chart.type = "line"
        chart.series = [
            Series(name="s1", color=(0, 0, 0), points=[(0, 1), (1, 2)]),
        ]

        repr_str = repr(chart)

        assert "extracted" in repr_str
        assert "line" in repr_str


class TestChartExtraction:
    """Tests for Chart extraction (requires mocking LLM)."""

    def test_extract_without_api_key(self):
        """Test that extract raises error without API key."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)
        chart._client = None  # Ensure no client

        with pytest.raises(RuntimeError) as exc_info:
            chart.extract()

        assert "not available" in str(exc_info.value)

    def test_build_prompt_low_density(self):
        """Test prompt building with low density."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        # Mock config
        with patch('plottery.chart.config') as mock_config:
            mock_config.sample_density = "low"
            mock_config.detect_peaks = False

            prompt = chart._build_prompt(context="Test context")

            assert "10-20" in prompt
            assert "Test context" in prompt

    def test_build_prompt_high_density(self):
        """Test prompt building with high density."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        with patch('plottery.chart.config') as mock_config:
            mock_config.sample_density = "high"
            mock_config.detect_peaks = True

            prompt = chart._build_prompt()

            assert "50-100" in prompt
            assert "peaks" in prompt.lower()

    def test_parse_response(self):
        """Test parsing LLM response."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        response = '''```json
        {
            "chart_info": {"type": "line"},
            "series": [{"name": "test", "points": [{"x": 0, "y": 1}]}]
        }
        ```'''

        data = chart._parse_response(response)

        assert data["chart_info"]["type"] == "line"
        assert len(data["series"]) == 1

    def test_apply_extraction(self):
        """Test applying extracted data to chart."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        chart = Chart.from_array(image)

        data = {
            "chart_info": {
                "type": "bar",
                "x_range": [0, 10],
                "y_range": [0, 100],
                "x_unit": "Hz",
                "y_unit": "dB",
            },
            "series": [
                {
                    "name": "Data",
                    "color": "blue",
                    "points": [
                        {"x": 1, "y": 50},
                        {"x": 2, "y": 75},
                    ],
                }
            ],
            "peaks": [
                {"x": 2, "y": 75, "series": "Data"},
            ],
            "notes": ["Test note"],
        }

        chart._apply_extraction(data)

        assert chart.type == "bar"
        assert chart.x_range == (0.0, 10.0)
        assert chart.y_range == (0.0, 100.0)
        assert chart.x_unit == "Hz"
        assert chart.y_unit == "dB"
        assert len(chart.series) == 1
        assert chart.series[0].name == "Data"
        assert len(chart.series[0].points) == 2
        assert len(chart.peaks) == 1
