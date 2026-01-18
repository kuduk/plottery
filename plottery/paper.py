"""Paper class for plottery.

This module provides the Paper class which represents a PDF document
containing charts that can be extracted and analyzed.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Union

import pandas as pd

from .chart import Chart
from .utils.pdf import extract_text_from_pdf, find_charts_in_pdf


class Paper:
    """Represents a PDF document with charts.

    A Paper loads a PDF file, extracts its text content and chart images,
    and provides methods to extract data from all charts.

    Attributes:
        path: Path to the PDF file
        text: Extracted text content from the PDF
        charts: List of Chart objects found in the PDF

    Example:
        >>> paper = Paper("paper.pdf")
        >>> paper.extract_all()
        >>> print(f"Found {len(paper.charts)} charts")
        >>> paper.to_excel("output.xlsx")
    """

    def __init__(
        self,
        path: Union[str, Path],
        dpi: int = 200,
        pages: Optional[List[int]] = None,
    ):
        """Initialize a Paper from a PDF file.

        Args:
            path: Path to the PDF file
            dpi: Resolution for chart image extraction
            pages: Optional list of pages to process (0-indexed).
                   If None, processes all pages.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

        self._dpi = dpi
        self._pages = pages

        self.text: str = ""
        self.charts: List[Chart] = []

        self._load()

    def _load(self) -> None:
        """Load text and find charts in the PDF."""
        # Extract text
        try:
            self.text = extract_text_from_pdf(self.path, pages=self._pages)
        except Exception:
            self.text = ""

        # Find charts
        self._find_charts()

    def _find_charts(self) -> None:
        """Find chart images in the PDF."""
        try:
            pdf_images = find_charts_in_pdf(
                self.path,
                pages=self._pages,
                dpi=self._dpi,
            )

            for pdf_image in pdf_images:
                chart = Chart(
                    image=pdf_image.image,
                    page=pdf_image.page_number,
                    paper_text=self.text,
                )
                self.charts.append(chart)

        except Exception as e:
            # If chart extraction fails, leave charts empty
            pass

    def extract_all(
        self,
        context: Optional[str] = None,
        generate_context: bool = True,
        max_workers: int = 1,
        on_progress: Optional[Callable[[int, int, Chart], None]] = None,
    ) -> "Paper":
        """Extract data from all charts.

        Args:
            context: Optional context to use for all charts.
                     If not provided and generate_context is True,
                     context will be generated from paper text.
            generate_context: Whether to generate context from paper
                              text for each chart (default True).
            max_workers: Number of parallel workers for extraction.
                         1 = sequential (default), >1 = parallel.
            on_progress: Optional callback called after each chart extraction.
                         Signature: (completed_count, total_count, chart) -> None

        Returns:
            Self, for method chaining

        Raises:
            RuntimeError: If LLM is not available
        """
        if max_workers <= 1:
            # Sequential extraction
            for i, chart in enumerate(self.charts):
                self._extract_single_chart(chart, context, generate_context)
                if on_progress:
                    on_progress(i + 1, len(self.charts), chart)
        else:
            # Parallel extraction
            self._extract_parallel(context, generate_context, max_workers, on_progress)

        return self

    def _extract_single_chart(
        self,
        chart: Chart,
        context: Optional[str],
        generate_context: bool,
    ) -> None:
        """Extract data from a single chart."""
        try:
            if context:
                chart.extract(context=context)
            elif generate_context and self.text:
                chart.extract()
            else:
                chart.extract(context=f"Scientific chart from {self.path.name}")
        except Exception as e:
            chart.metadata["error"] = str(e)

    def _extract_parallel(
        self,
        context: Optional[str],
        generate_context: bool,
        max_workers: int,
        on_progress: Optional[Callable[[int, int, Chart], None]],
    ) -> None:
        """Extract data from all charts in parallel."""
        total = len(self.charts)
        completed = 0

        def extract_task(chart: Chart) -> Chart:
            self._extract_single_chart(chart, context, generate_context)
            return chart

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chart = {
                executor.submit(extract_task, chart): chart
                for chart in self.charts
            }

            # Process completed tasks
            for future in as_completed(future_to_chart):
                chart = future_to_chart[future]
                completed += 1
                try:
                    future.result()  # Raise exception if any
                except Exception as e:
                    chart.metadata["error"] = str(e)

                if on_progress:
                    on_progress(completed, total, chart)

    def to_csv(self, output_dir: Union[str, Path]) -> List[Path]:
        """Export each chart to a separate CSV file.

        Args:
            output_dir: Directory to save CSV files

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for i, chart in enumerate(self.charts):
            if chart.num_points > 0:
                filename = f"chart_{i+1}_page{chart.page+1}.csv"
                filepath = output_dir / filename
                chart.to_csv(filepath)
                files.append(filepath)

        return files

    def to_excel(self, path: Union[str, Path]) -> None:
        """Export all charts to an Excel file (one sheet per chart).

        Args:
            path: Output Excel file path

        Raises:
            ImportError: If openpyxl is not installed
        """
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                for i, chart in enumerate(self.charts):
                    if chart.num_points > 0:
                        sheet_name = f"Chart_{i+1}_p{chart.page+1}"
                        # Excel sheet names limited to 31 chars
                        sheet_name = sheet_name[:31]
                        df = chart.to_dataframe()
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

                        # Add metadata sheet
                        meta_sheet = f"Meta_{i+1}"[:31]
                        meta_df = pd.DataFrame([{
                            "chart_type": chart.type,
                            "page": chart.page + 1,
                            "num_series": chart.num_series,
                            "num_points": chart.num_points,
                            "x_range": str(chart.x_range),
                            "y_range": str(chart.y_range),
                            "x_unit": chart.x_unit,
                            "y_unit": chart.y_unit,
                            "context": chart.context[:500] if chart.context else "",
                        }])
                        meta_df.to_excel(writer, sheet_name=meta_sheet, index=False)

        except ImportError:
            raise ImportError(
                "Excel export requires openpyxl. Install with: pip install openpyxl"
            )

    def to_json(self, path: Union[str, Path]) -> None:
        """Export all charts to a JSON file.

        Args:
            path: Output JSON file path
        """
        data = {
            "source": str(self.path),
            "num_charts": len(self.charts),
            "charts": []
        }

        for i, chart in enumerate(self.charts):
            chart_data = {
                "index": i + 1,
                "page": chart.page + 1,
                "type": chart.type,
                "context": chart.context,
                "x_range": chart.x_range,
                "y_range": chart.y_range,
                "x_unit": chart.x_unit,
                "y_unit": chart.y_unit,
                "is_categorical": chart.is_categorical,
                "categories": chart.categories,
                "series": [],
                "peaks": [],
                "metadata": chart.metadata,
            }

            for s in chart.series:
                chart_data["series"].append({
                    "name": s.name,
                    "color": list(s.color),
                    "points": s.points,
                })

            for p in chart.peaks:
                chart_data["peaks"].append({
                    "x": p.x,
                    "y": p.y,
                    "series": p.series,
                    "harmonic_of": p.harmonic_of,
                    "harmonic_order": p.harmonic_order,
                })

            data["charts"].append(chart_data)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_images(self, output_dir: Union[str, Path]) -> List[Path]:
        """Save all chart images to a directory.

        Args:
            output_dir: Directory to save images

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for i, chart in enumerate(self.charts):
            filename = f"chart_{i+1}_page{chart.page+1}.png"
            filepath = output_dir / filename
            chart.save_image(filepath)
            files.append(filepath)

        return files

    @property
    def num_charts(self) -> int:
        """Number of charts found in the paper."""
        return len(self.charts)

    @property
    def extracted_charts(self) -> List[Chart]:
        """Charts that have been successfully extracted."""
        return [c for c in self.charts if c.is_extracted]

    @property
    def total_points(self) -> int:
        """Total number of data points across all charts."""
        return sum(c.num_points for c in self.charts)

    @property
    def total_series(self) -> int:
        """Total number of series across all charts."""
        return sum(c.num_series for c in self.charts)

    def get_chart(self, index: int) -> Optional[Chart]:
        """Get a chart by index.

        Args:
            index: Chart index (0-indexed)

        Returns:
            Chart if index is valid, None otherwise
        """
        if 0 <= index < len(self.charts):
            return self.charts[index]
        return None

    def get_charts_on_page(self, page: int) -> List[Chart]:
        """Get all charts on a specific page.

        Args:
            page: Page number (0-indexed)

        Returns:
            List of charts on that page
        """
        return [c for c in self.charts if c.page == page]

    def __len__(self) -> int:
        """Number of charts in the paper."""
        return len(self.charts)

    def __iter__(self) -> Iterator[Chart]:
        """Iterate over charts."""
        return iter(self.charts)

    def __getitem__(self, index: int) -> Chart:
        """Get chart by index."""
        return self.charts[index]

    def __repr__(self) -> str:
        extracted = len(self.extracted_charts)
        return (
            f"Paper('{self.path.name}', "
            f"charts={self.num_charts}, "
            f"extracted={extracted}, "
            f"points={self.total_points})"
        )
