"""Text PDF rendering with reportlab; ``invariant=1`` keeps the output byte-stable."""

import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from llmeval.dataset.fake_data import Document

# The standard PDF fonts lack Czech letters (č, ř, ů …), so a TrueType font gets embedded.
# Debian/Ubuntu package fonts-dejavu-core (first path); the other two are common locations on
# other distributions (not tried here). Elsewhere point LLMEVAL_FONT_DIR at the font.
FONT_DIRS = (
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/usr/share/fonts/TTF"),
    Path("/usr/share/fonts/dejavu"),
)
REGULAR, BOLD = "DejaVuSans", "DejaVuSans-Bold"

_STYLES = {  # kind: (font, size, space before, line height)
    "title": (BOLD, 16, 0, 20),
    "subtitle": (REGULAR, 10, 0, 14),
    "heading": (BOLD, 12, 10, 16),
    "field": (REGULAR, 10, 0, 15),
    "para": (REGULAR, 10, 6, 13),
}
MARGIN = 20 * mm


def find_font_dir(dirs: tuple[Path, ...] | None = None) -> Path:
    """Directory with DejaVu Sans; ``LLMEVAL_FONT_DIR`` overrides the usual locations."""
    if dirs is None:
        override = os.environ.get("LLMEVAL_FONT_DIR")
        dirs = (Path(override),) if override else FONT_DIRS
    for directory in dirs:
        if (directory / f"{REGULAR}.ttf").exists() and (directory / f"{BOLD}.ttf").exists():
            return directory
    raise FileNotFoundError(
        f"DejaVu Sans not found in {', '.join(map(str, dirs))}; set LLMEVAL_FONT_DIR"
    )


def _register_fonts() -> None:
    if REGULAR in pdfmetrics.getRegisteredFontNames():
        return
    directory = find_font_dir()
    pdfmetrics.registerFont(TTFont(REGULAR, str(directory / f"{REGULAR}.ttf")))
    pdfmetrics.registerFont(TTFont(BOLD, str(directory / f"{BOLD}.ttf")))


def render_text_pdf(document: Document, path: Path) -> None:
    _register_fonts()
    width, height = A4
    canvas = Canvas(str(path), pagesize=A4, invariant=1)
    y = height - MARGIN
    for block in document.blocks:
        if block.kind == "note":
            canvas.setFont(REGULAR, 8)
            canvas.drawString(MARGIN, MARGIN, block.text)
            continue
        font, size, before, leading = _STYLES[block.kind]
        y -= before
        canvas.setFont(font, size)
        # Fields stay on one line: in a skewed scan a separate value column drifts half a row
        # towards the neighbouring label, which makes the pairing ambiguous.
        for line in simpleSplit(block.line(), font, size, width - 2 * MARGIN):
            canvas.drawString(MARGIN, y, line)
            y -= leading
    if y < MARGIN + 20:
        raise ValueError("document does not fit on one page")
    canvas.showPage()
    canvas.save()
