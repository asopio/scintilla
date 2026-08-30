"""Extract a figure from a PDF page image using relative coordinates."""

import os
import tempfile
from pathlib import Path
from PIL import Image


def extract_figure(
    pdf_path: str,
    output_path: str,
    page_number: int = 0,
    bbox: tuple[float, float, float, float] = (0.33, 0.49, 0.74, 0.71),
    dpi: int = 300,
) -> None:
    """Crop a figure region from a rendered PDF page.

    Parameters
    ----------
    pdf_path : path to the source PDF
    output_path : where to save the cropped PNG
    page_number : 0-indexed page to extract from
    bbox : relative coordinates (x1, y1, x2, y2) in [0, 1]
    dpi : DPI used when rendering the page (must match the source image)
    """
    from subprocess import check_output

    # Render the page to PNG using pdftoppm.
    # pdftoppm appends the page number to the prefix, so we use "-f" and "-l" to get one page.
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "page")
        check_output(
            ["pdftoppm", "-png", "-r", str(dpi),
             "-f", str(page_number + 1), "-l", str(page_number + 1),
             pdf_path, prefix]
        )
        # Find the actual file (e.g. page-1.png)
        rendered = Path(tmpdir) / f"{prefix}-{page_number + 1}.png"
        if not rendered.exists():
            # Fallback: list and pick the first .png
            pngs = list(Path(tmpdir).glob("*.png"))
            if not pngs:
                raise FileNotFoundError(f"No PNG rendered from {pdf_path}")
            rendered = pngs[0]

        page_img = Image.open(str(rendered))

    x1, y1, x2, y2 = bbox
    w, h = page_img.size
    box = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))

    cropped = page_img.crop(box)
    cropped.save(output_path)
    print(f"Saved figure to {output_path}  ({cropped.size[0]}x{cropped.size[1]} px)")


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "test_input/cheng_jiang_full_revised_thesis_v12.pdf"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "outfig.png"
    page = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    # Coordinates from the previous analysis
    bbox = (0.33, 0.49, 0.74, 0.71)

    extract_figure(pdf_path, output_path, page, bbox, dpi=300)
