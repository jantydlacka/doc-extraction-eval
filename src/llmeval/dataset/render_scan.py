"""Fake scans: text PDF -> pdftoppm -> Pillow damage from a seeded generator -> img2pdf.

Scan bytes may change with other poppler, Pillow/libjpeg or img2pdf versions, so tests check
scan properties, not bytes; the expected values do not depend on the scans.
"""

import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from llmeval.dataset.fake_data import ScanParams


def degrade(image: Image.Image, params: ScanParams, rng: random.Random) -> Image.Image:
    image = image.convert("L")
    if params.downscale > 1:
        small = (max(1, image.width // params.downscale), max(1, image.height // params.downscale))
        image = image.resize(small, Image.Resampling.BILINEAR).resize(
            (image.width, image.height), Image.Resampling.BILINEAR
        )
    if params.blur:
        image = image.filter(ImageFilter.GaussianBlur(params.blur))
    if params.contrast != 1.0:
        image = ImageEnhance.Contrast(image).enhance(params.contrast)
    if params.angle:
        angle = params.angle + rng.uniform(-0.3, 0.3)
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
    if params.noise:
        # randbytes comes from the seeded generator; Image.effect_noise would not be reproducible.
        raw = Image.frombytes("L", image.size, rng.randbytes(image.width * image.height))
        noise = raw.point(lambda v: 128 + (v - 128) * params.noise // 128)
        image = ImageChops.add(image, noise, scale=1.0, offset=-128)
    return image


def render_scan_pdf(
    text_pdf: Path, out_pdf: Path, params: ScanParams, seed: str, dpi: int | None = None
) -> None:
    dpi = dpi or params.dpi
    rng = random.Random(f"scan:{seed}")
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-gray", "-png", "-singlefile", str(text_pdf), str(page)],
            check=True,
        )
        image = degrade(Image.open(f"{page}.png"), params, rng)
        jpeg = Path(tmp) / "page.jpg"
        image.save(jpeg, "JPEG", quality=params.jpeg_quality, dpi=(dpi, dpi))
        subprocess.run(["img2pdf", "--nodate", str(jpeg), "-o", str(out_pdf)], check=True)
