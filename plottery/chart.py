"""Chart class for plottery.

This module provides the Chart class which represents a single chart
and provides methods for data extraction using LLM vision capabilities.
"""

import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .models import Series, Peak
from .config import config


# Color name to RGB mapping
COLOR_MAP = {
    "blue": (0, 0, 255),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "black": (0, 0, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "brown": (139, 69, 19),
    "pink": (255, 192, 203),
}


@dataclass
class Chart:
    """Represents a single chart with extraction capabilities.

    A Chart can be created from an image file or numpy array. Data extraction
    is performed using Claude's vision capabilities.

    Attributes:
        image: The chart image as numpy array (RGB)
        page: Page number if from PDF (0-indexed)
        paper_text: Text from the source paper (for context generation)
        type: Detected chart type (line, bar, scatter, etc.)
        context: Context used for extraction
        series: Extracted data series
        peaks: Detected peaks
        metadata: Additional extraction metadata
        x_range: X-axis range tuple (min, max)
        y_range: Y-axis range tuple (min, max)
        x_unit: X-axis unit string
        y_unit: Y-axis unit string

    Example:
        >>> chart = Chart.from_image("spectrum.png")
        >>> chart.extract(context="Motor frequency spectrum")
        >>> print(f"Extracted {chart.num_points} points")
        >>> chart.to_csv("data.csv")
    """

    image: np.ndarray
    page: int = 0
    paper_text: str = ""

    # Populated after extraction
    type: str = ""
    context: str = ""
    series: List[Series] = field(default_factory=list)
    peaks: List[Peak] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Calibration info
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    x_unit: Optional[str] = None
    y_unit: Optional[str] = None

    _extracted: bool = field(default=False, repr=False)
    _client: Any = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the Anthropic client if available."""
        if config.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=config.api_key)
            except ImportError:
                pass

    @classmethod
    def from_image(cls, path: Union[str, Path]) -> "Chart":
        """Create a Chart from an image file.

        Args:
            path: Path to the image file (PNG, JPG, etc.)

        Returns:
            Chart instance with the loaded image

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be loaded
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        try:
            pil_image = Image.open(path)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            image = np.array(pil_image)
            return cls(image=image)
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

    @classmethod
    def from_array(cls, image: np.ndarray, page: int = 0) -> "Chart":
        """Create a Chart from a numpy array.

        Args:
            image: RGB image as numpy array
            page: Page number (for PDF sources)

        Returns:
            Chart instance
        """
        return cls(image=image, page=page)

    @property
    def is_extracted(self) -> bool:
        """Check if data has been extracted."""
        return self._extracted

    @property
    def is_categorical(self) -> bool:
        """Check if the chart has a categorical X-axis."""
        return self.metadata.get("is_categorical", False)

    @property
    def categories(self) -> List[str]:
        """Get category names for categorical X-axis."""
        return self.metadata.get("categories", [])

    @property
    def category_mapping(self) -> Dict[int, str]:
        """Get mapping from index to category name."""
        return self.metadata.get("category_mapping", {})

    @property
    def num_points(self) -> int:
        """Total number of data points across all series."""
        return sum(len(s.points) for s in self.series)

    @property
    def num_series(self) -> int:
        """Number of extracted series."""
        return len(self.series)

    def extract(
        self,
        context: Optional[str] = None,
        x_unit: Optional[str] = None,
        y_unit: Optional[str] = None,
        chart_type_hint: Optional[str] = None,
    ) -> "Chart":
        """Extract data from the chart using LLM.

        Args:
            context: Description of the chart to improve accuracy
            x_unit: Expected X-axis unit
            y_unit: Expected Y-axis unit
            chart_type_hint: Hint about chart type (line, bar, etc.)

        Returns:
            Self, for method chaining

        Raises:
            RuntimeError: If LLM is not available or extraction fails
        """
        if not self._client:
            raise RuntimeError(
                "LLM extraction not available. Install anthropic package "
                "and set ANTHROPIC_API_KEY environment variable."
            )

        # Use provided context or generate from paper_text
        if context:
            self.context = context
        elif self.paper_text and not self.context:
            self.context = self._generate_context()

        # Convert image to base64
        pil_image = Image.fromarray(self.image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Build prompt
        prompt = self._build_prompt(
            context=self.context,
            x_unit=x_unit,
            y_unit=y_unit,
            chart_type_hint=chart_type_hint,
        )

        try:
            response = self._client.messages.create(
                model=config.model,
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            # Parse response
            response_text = response.content[0].text
            data = self._parse_response(response_text)

            # Update chart with extracted data
            self._apply_extraction(data)
            self._extracted = True

            return self

        except Exception as e:
            raise RuntimeError(f"LLM extraction failed: {e}")

    def _generate_context(self) -> str:
        """Generate context from paper_text using LLM."""
        if not self._client or not self.paper_text:
            return ""

        # Convert image to base64
        pil_image = Image.fromarray(self.image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Truncate paper text if too long
        text = self.paper_text
        max_len = 15000
        if len(text) > max_len:
            half = max_len // 2
            text = text[:half] + "\n\n[...]\n\n" + text[-half:]

        prompt = f"""I have a scientific paper and a chart image from page {self.page + 1}.

Here is the paper text:

{text}

Based on the paper text and the chart image, provide a concise context description for this chart that would help with data extraction. Include:

1. What the chart shows (type of data, what's being measured)
2. The axis information if mentioned (units, ranges)
3. What the different series/lines represent
4. Any specific values mentioned in the text

Keep the context to 3-5 sentences. Return only the context description."""

        try:
            response = self._client.messages.create(
                model=config.model,
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception:
            return ""

    def _build_prompt(
        self,
        context: str = "",
        x_unit: Optional[str] = None,
        y_unit: Optional[str] = None,
        chart_type_hint: Optional[str] = None,
    ) -> str:
        """Build the extraction prompt."""
        # Sampling instructions based on density
        density = config.sample_density
        if density == "low":
            sampling = "Extract approximately 10-20 key data points per series, focusing on peaks, valleys, and significant changes."
        elif density == "high":
            sampling = "Extract as many data points as you can accurately read from the chart (50-100+ points per series if visible)."
        else:
            sampling = "Extract approximately 30-50 data points per series, capturing the overall shape and key features."

        # Context section
        context_section = ""
        if context:
            context_section = f"\n\nContext about this chart:\n{context}"

        # Hints section
        hints = []
        if chart_type_hint:
            hints.append(f"- Chart type: {chart_type_hint}")
        if x_unit:
            hints.append(f"- X-axis unit: {x_unit}")
        if y_unit:
            hints.append(f"- Y-axis unit: {y_unit}")

        hints_section = ""
        if hints:
            hints_section = "\n\nHints:\n" + "\n".join(hints)

        # Peak detection
        peak_section = ""
        if config.detect_peaks:
            peak_section = """

Also identify significant peaks in the data:
- For each peak, provide its x and y coordinates
- Note if peaks appear to be harmonics (multiples of a fundamental frequency)
- Describe what the peak might represent if apparent from context"""

        return f"""Analyze this chart image and extract the numerical data.{context_section}{hints_section}

Instructions:
1. First, identify all data series in the chart (different colored lines, bars, or point sets)
2. For each series, extract the (x, y) data points by reading values from the axes
3. {sampling}
4. Read axis labels and determine the actual value ranges
5. Be as accurate as possible when reading values from the axes
6. For BAR CHARTS with categorical x-axis: use category names as x values (strings)
7. For STACKED BAR CHARTS: create a separate series for each stacked component{peak_section}

Return the data as JSON with this structure:
{{
    "chart_info": {{
        "type": "line|bar|scatter|spectrum|histogram|pie|stacked_bar",
        "title": "chart title if visible",
        "x_label": "x-axis label",
        "y_label": "y-axis label",
        "x_range": [min, max] or ["category1", "category2", ...] for categorical,
        "y_range": [min, max],
        "x_unit": "unit string",
        "y_unit": "unit string",
        "is_categorical": true/false
    }},
    "series": [
        {{
            "name": "series name or color description",
            "color": "color description (e.g., 'blue', 'red dashed')",
            "points": [
                {{"x": value_or_category, "y": value}},
                ...
            ]
        }}
    ],
    "peaks": [
        {{
            "x": value,
            "y": value,
            "series": "series name",
            "description": "what this peak represents",
            "is_harmonic": false,
            "harmonic_of": null,
            "harmonic_order": null
        }}
    ],
    "notes": ["any observations about the data or extraction quality"]
}}

IMPORTANT:
- Return ONLY the JSON, no other text
- Use null for unknown values, not empty strings
- For numerical axes: ensure numbers are valid JSON numbers
- For categorical axes: use strings for x values
- Points should be sorted by x value (or category order)
- For stacked bars: each component should be a separate series"""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response into structured data."""
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(response_text)

    def _apply_extraction(self, data: Dict[str, Any]) -> None:
        """Apply extracted data to this chart."""
        chart_info = data.get("chart_info", {})

        # Update chart info
        self.type = chart_info.get("type", "unknown")
        self.x_unit = chart_info.get("x_unit")
        self.y_unit = chart_info.get("y_unit")

        if chart_info.get("x_range"):
            x_range = chart_info["x_range"]
            if isinstance(x_range, list) and len(x_range) == 2:
                try:
                    self.x_range = (float(x_range[0]), float(x_range[1]))
                except (ValueError, TypeError):
                    pass  # Categorical range

        if chart_info.get("y_range"):
            y_range = chart_info["y_range"]
            if isinstance(y_range, list) and len(y_range) == 2:
                try:
                    self.y_range = (float(y_range[0]), float(y_range[1]))
                except (ValueError, TypeError):
                    pass

        # Convert series
        self.series = []
        all_categories = []

        for i, series_data in enumerate(data.get("series", [])):
            name = series_data.get("name", f"Series {i+1}")
            color_desc = series_data.get("color", "").lower()

            # Determine RGB color
            color = (0, 0, 0)
            for color_name, rgb in COLOR_MAP.items():
                if color_name in color_desc:
                    color = rgb
                    break

            # Extract points
            points = []
            is_categorical = chart_info.get("is_categorical", False)

            for pt in series_data.get("points", []):
                x = pt.get("x")
                y = pt.get("y")
                if x is not None and y is not None:
                    try:
                        y_val = float(y)
                        try:
                            x_val = float(x)
                        except (ValueError, TypeError):
                            # Categorical x
                            is_categorical = True
                            if x not in all_categories:
                                all_categories.append(x)
                            x_val = float(all_categories.index(x))
                        points.append((x_val, y_val))
                    except (ValueError, TypeError):
                        continue

            if points:
                points.sort(key=lambda p: p[0])
                self.series.append(Series(name=name, color=color, points=points))

        # Convert peaks
        self.peaks = []
        for peak_data in data.get("peaks", []):
            x = peak_data.get("x")
            y = peak_data.get("y")
            if x is not None and y is not None:
                try:
                    peak = Peak(
                        x=float(x),
                        y=float(y),
                        series=peak_data.get("series", ""),
                        prominence=0.0,
                        harmonic_of=peak_data.get("harmonic_of"),
                        harmonic_order=peak_data.get("harmonic_order"),
                    )
                    self.peaks.append(peak)
                except (ValueError, TypeError):
                    continue

        # Build metadata
        self.metadata = {
            "extraction_method": "llm",
            "model": config.model,
            "chart_type": self.type,
            "title": chart_info.get("title"),
            "x_label": chart_info.get("x_label"),
            "y_label": chart_info.get("y_label"),
            "notes": data.get("notes", []),
        }

        if self.context:
            self.metadata["context"] = self.context

        # Add category mapping
        if all_categories:
            self.metadata["is_categorical"] = True
            self.metadata["categories"] = all_categories
            self.metadata["category_mapping"] = {i: cat for i, cat in enumerate(all_categories)}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all series to a pandas DataFrame.

        Returns:
            DataFrame with columns: x, y, series
        """
        if not self.series:
            return pd.DataFrame(columns=["x", "y", "series"])

        dfs = []
        for s in self.series:
            df = s.to_dataframe()
            df["series"] = s.name
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def to_csv(self, path: Union[str, Path], sep: str = ",") -> None:
        """Export all series to a CSV file.

        Args:
            path: Output file path
            sep: Column separator (default: comma)
        """
        df = self.to_dataframe()
        df.to_csv(path, index=False, sep=sep)

    def save_image(self, path: Union[str, Path]) -> None:
        """Save the chart image to a file.

        Args:
            path: Output file path (PNG, JPG, etc.)
        """
        pil_image = Image.fromarray(self.image)
        pil_image.save(path)

    def get_series(self, name: str) -> Optional[Series]:
        """Get a series by name.

        Args:
            name: Series name

        Returns:
            Series if found, None otherwise
        """
        for s in self.series:
            if s.name == name:
                return s
        return None

    def get_peaks_for_series(self, name: str) -> List[Peak]:
        """Get all peaks for a specific series.

        Args:
            name: Series name

        Returns:
            List of peaks for the series
        """
        return [p for p in self.peaks if p.series == name]

    def __repr__(self) -> str:
        status = "extracted" if self._extracted else "not extracted"
        return f"Chart(page={self.page}, type='{self.type}', series={self.num_series}, points={self.num_points}, {status})"
