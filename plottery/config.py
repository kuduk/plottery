"""Plottery configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Centralized configuration for Plottery.

    Attributes:
        api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
        model: Claude model to use.
        sample_density: Point sampling density ("low", "medium", "high").
        detect_peaks: Whether to detect peaks in extracted data.

    Example:
        >>> from plottery import config
        >>> config.sample_density = "high"
        >>> config.detect_peaks = False
    """

    api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    model: str = "claude-opus-4-5-20251101"
    sample_density: str = "medium"
    detect_peaks: bool = True

    def __post_init__(self):
        if self.sample_density not in ("low", "medium", "high"):
            raise ValueError(f"sample_density must be 'low', 'medium', or 'high', got '{self.sample_density}'")

    @property
    def is_configured(self) -> bool:
        """Check if API key is set."""
        return bool(self.api_key)


# Global configuration instance
config = Config()
