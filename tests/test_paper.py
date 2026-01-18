"""Tests for Paper class."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from plottery.paper import Paper
from plottery.chart import Chart
from plottery.models import Series


class TestPaperCreation:
    """Tests for Paper creation."""

    def test_paper_file_not_found(self):
        """Test error when PDF doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Paper("/nonexistent/path/paper.pdf")

    def test_paper_creation_with_mock(self, tmp_path):
        """Test Paper creation with mocked PDF functions."""
        # Create a dummy PDF file
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = "Test paper text"
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                assert paper.path == pdf_path
                assert paper.text == "Test paper text"
                assert len(paper.charts) == 0


class TestPaperProperties:
    """Tests for Paper properties."""

    def test_num_charts(self, tmp_path):
        """Test num_charts property."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                assert paper.num_charts == 0

    def test_total_points_empty(self, tmp_path):
        """Test total_points with no charts."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                assert paper.total_points == 0
                assert paper.total_series == 0

    def test_total_points_with_charts(self, tmp_path):
        """Test total_points with extracted charts."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                # Add mock charts with data
                chart1 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                chart1.series = [
                    Series(name="s1", color=(0, 0, 0), points=[(0, 1), (1, 2)]),
                ]
                chart1._extracted = True

                chart2 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                chart2.series = [
                    Series(name="s2", color=(0, 0, 0), points=[(0, 0.5)]),
                ]
                chart2._extracted = True

                paper.charts = [chart1, chart2]

                assert paper.total_points == 3
                assert paper.total_series == 2

    def test_extracted_charts(self, tmp_path):
        """Test extracted_charts property."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                # Add mix of extracted and non-extracted charts
                chart1 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                chart1._extracted = True

                chart2 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                chart2._extracted = False

                chart3 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                chart3._extracted = True

                paper.charts = [chart1, chart2, chart3]

                assert len(paper.extracted_charts) == 2


class TestPaperAccess:
    """Tests for Paper chart access methods."""

    def test_get_chart_valid_index(self, tmp_path):
        """Test get_chart with valid index."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)
                chart = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                paper.charts = [chart]

                assert paper.get_chart(0) is chart

    def test_get_chart_invalid_index(self, tmp_path):
        """Test get_chart with invalid index."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                assert paper.get_chart(5) is None
                assert paper.get_chart(-1) is None

    def test_get_charts_on_page(self, tmp_path):
        """Test get_charts_on_page."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                chart1 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8), page=0)
                chart2 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8), page=1)
                chart3 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8), page=0)

                paper.charts = [chart1, chart2, chart3]

                page0_charts = paper.get_charts_on_page(0)
                page1_charts = paper.get_charts_on_page(1)

                assert len(page0_charts) == 2
                assert len(page1_charts) == 1


class TestPaperIteration:
    """Tests for Paper iteration."""

    def test_len(self, tmp_path):
        """Test __len__."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)
                paper.charts = [
                    Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8)),
                    Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8)),
                ]

                assert len(paper) == 2

    def test_iter(self, tmp_path):
        """Test __iter__."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)
                chart1 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                chart2 = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                paper.charts = [chart1, chart2]

                charts_list = list(paper)

                assert len(charts_list) == 2
                assert charts_list[0] is chart1
                assert charts_list[1] is chart2

    def test_getitem(self, tmp_path):
        """Test __getitem__."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)
                chart = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8))
                paper.charts = [chart]

                assert paper[0] is chart


class TestPaperExport:
    """Tests for Paper export methods."""

    def test_to_csv(self, tmp_path):
        """Test CSV export."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")
        output_dir = tmp_path / "output"

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                chart = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8), page=0)
                chart.series = [
                    Series(name="test", color=(0, 0, 0), points=[(0, 1), (1, 2)]),
                ]
                paper.charts = [chart]

                files = paper.to_csv(output_dir)

                assert len(files) == 1
                assert files[0].exists()

    def test_to_json(self, tmp_path):
        """Test JSON export."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")
        json_path = tmp_path / "output.json"

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                chart = Chart.from_array(np.zeros((100, 100, 3), dtype=np.uint8), page=0)
                chart.type = "line"
                chart.series = [
                    Series(name="test", color=(255, 0, 0), points=[(0, 1)]),
                ]
                paper.charts = [chart]

                paper.to_json(json_path)

                assert json_path.exists()

                with open(json_path) as f:
                    data = json.load(f)

                assert data["num_charts"] == 1
                assert len(data["charts"]) == 1
                assert data["charts"][0]["type"] == "line"


class TestPaperRepr:
    """Tests for Paper string representation."""

    def test_repr(self, tmp_path):
        """Test __repr__."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_text("dummy")

        with patch('plottery.paper.extract_text_from_pdf') as mock_text:
            with patch('plottery.paper.find_charts_in_pdf') as mock_charts:
                mock_text.return_value = ""
                mock_charts.return_value = []

                paper = Paper(pdf_path)

                repr_str = repr(paper)

                assert "Paper" in repr_str
                assert "test.pdf" in repr_str
