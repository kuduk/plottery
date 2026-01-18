"""PDF processing utilities for extracting chart images."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class PDFImage:
    """Represents an image extracted from a PDF."""

    image: np.ndarray  # RGB image
    page_number: int
    position: Tuple[float, float, float, float]  # x0, y0, x1, y1
    dpi: int = 150
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def extract_images_from_pdf(
    pdf_path: Union[str, Path],
    pages: Optional[List[int]] = None,
    min_size: Tuple[int, int] = (100, 100),
    dpi: int = 150,
) -> List[np.ndarray]:
    """Extract images from a PDF file.

    Tries PyMuPDF first, then falls back to pdfplumber.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page numbers (0-indexed) to process
        min_size: Minimum (width, height) for images
        dpi: Resolution for rasterizing

    Returns:
        List of RGB images as numpy arrays
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Try PyMuPDF first
    try:
        return _extract_with_pymupdf(pdf_path, pages, min_size, dpi)
    except ImportError:
        pass

    # Try pdfplumber
    try:
        return _extract_with_pdfplumber(pdf_path, pages, min_size, dpi)
    except ImportError:
        pass

    raise ImportError(
        "No PDF library available. Install pymupdf or pdfplumber."
    )


def _extract_with_pymupdf(
    pdf_path: Path,
    pages: Optional[List[int]],
    min_size: Tuple[int, int],
    dpi: int,
) -> List[np.ndarray]:
    """Extract images using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    images = []
    doc = fitz.open(str(pdf_path))

    try:
        page_range = pages if pages is not None else range(len(doc))

        for page_num in page_range:
            if page_num >= len(doc):
                continue

            page = doc[page_num]

            # Method 1: Extract embedded images
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                base_image = doc.extract_image(xref)

                if base_image is None:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Convert to numpy array
                from PIL import Image
                import io

                try:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    rgb_image = pil_image.convert("RGB")
                    np_image = np.array(rgb_image)

                    # Check minimum size
                    if np_image.shape[1] >= min_size[0] and np_image.shape[0] >= min_size[1]:
                        images.append(np_image)
                except Exception:
                    continue

            # Method 2: Render page as image (catches vector graphics)
            if not images or len(image_list) == 0:
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Convert to numpy array
                np_image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                # Convert to RGB if necessary
                if pix.n == 4:  # RGBA
                    from PIL import Image
                    pil_image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
                    rgb_image = pil_image.convert("RGB")
                    np_image = np.array(rgb_image)
                elif pix.n == 1:  # Grayscale
                    np_image = np.stack([np_image.squeeze()] * 3, axis=-1)

                if np_image.shape[1] >= min_size[0] and np_image.shape[0] >= min_size[1]:
                    images.append(np_image)

    finally:
        doc.close()

    return images


def _extract_with_pdfplumber(
    pdf_path: Path,
    pages: Optional[List[int]],
    min_size: Tuple[int, int],
    dpi: int,
) -> List[np.ndarray]:
    """Extract images using pdfplumber."""
    import pdfplumber
    from PIL import Image

    images = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_range = pages if pages is not None else range(len(pdf.pages))

        for page_num in page_range:
            if page_num >= len(pdf.pages):
                continue

            page = pdf.pages[page_num]

            # Method 1: Extract embedded images
            if hasattr(page, 'images'):
                for img in page.images:
                    try:
                        # pdfplumber image extraction
                        if 'stream' in img:
                            from io import BytesIO
                            pil_image = Image.open(BytesIO(img['stream'].get_data()))
                            rgb_image = pil_image.convert("RGB")
                            np_image = np.array(rgb_image)

                            if np_image.shape[1] >= min_size[0] and np_image.shape[0] >= min_size[1]:
                                images.append(np_image)
                    except Exception:
                        continue

            # Method 2: Convert page to image
            if not images:
                try:
                    pil_image = page.to_image(resolution=dpi).original
                    rgb_image = pil_image.convert("RGB")
                    np_image = np.array(rgb_image)

                    if np_image.shape[1] >= min_size[0] and np_image.shape[0] >= min_size[1]:
                        images.append(np_image)
                except Exception:
                    continue

    return images


def find_charts_in_pdf(
    pdf_path: Union[str, Path],
    pages: Optional[List[int]] = None,
    dpi: int = 150,
) -> List[PDFImage]:
    """Find chart images in a PDF with position information.

    This function attempts to identify which images in a PDF
    are likely to be charts/plots.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page numbers to process
        dpi: Resolution for image extraction

    Returns:
        List of PDFImage objects containing chart candidates
    """
    pdf_path = Path(pdf_path)

    try:
        import fitz
        return _find_charts_pymupdf(pdf_path, pages, dpi)
    except ImportError:
        pass

    # Fallback: just extract all images
    images = extract_images_from_pdf(pdf_path, pages, dpi=dpi)
    return [
        PDFImage(
            image=img,
            page_number=0,
            position=(0, 0, img.shape[1], img.shape[0]),
            dpi=dpi,
        )
        for img in images
    ]


def _find_charts_pymupdf(
    pdf_path: Path,
    pages: Optional[List[int]],
    dpi: int,
) -> List[PDFImage]:
    """Find charts using PyMuPDF with position info."""
    import fitz

    charts = []
    doc = fitz.open(str(pdf_path))

    try:
        page_range = pages if pages is not None else range(len(doc))

        for page_num in page_range:
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            page_rect = page.rect

            # Get images with positions
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]

                try:
                    base_image = doc.extract_image(xref)
                    if base_image is None:
                        continue

                    image_bytes = base_image["image"]

                    # Get image position on page
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        rect = img_rects[0]
                        position = (rect.x0, rect.y0, rect.x1, rect.y1)
                    else:
                        position = (0, 0, page_rect.width, page_rect.height)

                    # Convert to numpy
                    from PIL import Image
                    import io

                    pil_image = Image.open(io.BytesIO(image_bytes))
                    rgb_image = pil_image.convert("RGB")
                    np_image = np.array(rgb_image)

                    # Check if likely a chart
                    if _is_likely_chart(np_image):
                        charts.append(PDFImage(
                            image=np_image,
                            page_number=page_num,
                            position=position,
                            dpi=dpi,
                            metadata={
                                "xref": xref,
                                "original_size": (base_image.get("width"), base_image.get("height")),
                            },
                        ))
                except Exception:
                    continue

            # If no charts found in images, render the page
            if not charts:
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                from PIL import Image
                pil_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                np_image = np.array(pil_image)

                charts.append(PDFImage(
                    image=np_image,
                    page_number=page_num,
                    position=(0, 0, page_rect.width, page_rect.height),
                    dpi=dpi,
                    metadata={"type": "page_render"},
                ))

    finally:
        doc.close()

    return charts


def _is_likely_chart(image: np.ndarray) -> bool:
    """Heuristic check if an image is likely a chart.

    Uses basic size and aspect ratio checks. The LLM will determine
    if the image is actually a chart during extraction.

    Args:
        image: RGB image array

    Returns:
        True if image appears to be a chart based on size/aspect ratio
    """
    h, w = image.shape[:2]

    # Minimum size check
    if w < 100 or h < 100:
        return False

    # Maximum size check (probably a full page, not a single chart)
    if w > 3000 and h > 3000:
        return False

    # Check aspect ratio (charts are usually not too tall or too wide)
    aspect_ratio = w / h
    if aspect_ratio < 0.2 or aspect_ratio > 5:
        return False

    return True


def extract_text_from_pdf(
    pdf_path: Union[str, Path],
    pages: Optional[List[int]] = None,
) -> str:
    """Extract text content from a PDF file.

    Useful for getting paper context to improve chart extraction.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page numbers (0-indexed) to process.
               If None, extracts text from all pages.

    Returns:
        Extracted text as a single string
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Try PyMuPDF first
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            text_parts = []
            page_range = pages if pages is not None else range(len(doc))

            for page_num in page_range:
                if page_num >= len(doc):
                    continue
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            return "\n\n".join(text_parts)
        finally:
            doc.close()

    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            text_parts = []
            page_range = pages if pages is not None else range(len(pdf.pages))

            for page_num in page_range:
                if page_num >= len(pdf.pages):
                    continue
                page = pdf.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")

            return "\n\n".join(text_parts)

    except ImportError:
        pass

    raise ImportError(
        "No PDF library available. Install pymupdf or pdfplumber."
    )


def extract_page_as_image(
    pdf_path: Union[str, Path],
    page_number: int = 0,
    dpi: int = 150,
) -> np.ndarray:
    """Render a single PDF page as an image.

    Args:
        pdf_path: Path to PDF file
        page_number: Page to render (0-indexed)
        dpi: Resolution for rendering

    Returns:
        RGB image as numpy array
    """
    pdf_path = Path(pdf_path)

    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        try:
            if page_number >= len(doc):
                raise ValueError(f"Page {page_number} does not exist")

            page = doc[page_number]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            from PIL import Image
            if pix.n == 4:
                pil_image = Image.frombytes("RGBA", (pix.width, pix.height), pix.samples)
                rgb_image = pil_image.convert("RGB")
            else:
                rgb_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            return np.array(rgb_image)
        finally:
            doc.close()

    except ImportError:
        pass

    # Try pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_number >= len(pdf.pages):
                raise ValueError(f"Page {page_number} does not exist")

            page = pdf.pages[page_number]
            pil_image = page.to_image(resolution=dpi).original
            return np.array(pil_image.convert("RGB"))

    except ImportError:
        raise ImportError("No PDF library available. Install pymupdf or pdfplumber.")
