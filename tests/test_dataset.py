import csv
import json
import random
import re
import subprocess
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageFilter, ImageStat

from llmeval.dataset import build, render_pdf, render_scan
from llmeval.dataset.fake_data import (
    SMLOUVA,
    VYPIS,
    Block,
    Document,
    ScanParams,
    build_cases,
    format_amount,
)
from llmeval.dataset.isin import is_valid_isin
from llmeval.dataset.render_pdf import render_text_pdf
from llmeval.schema import FIELDS, validate_expected

REPO = Path(__file__).resolve().parent.parent
CASES_DIR = REPO / "data" / "cases"
CASES = build_cases()
BY_ID = {case.id: case for case in CASES}
TEXT_CASES = [case for case in CASES if case.forma == "text"]
SCAN_CASES = [case for case in CASES if case.forma == "sken"]


def pdf_text(path: Path) -> str:
    out = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return " ".join(out.split())


def squash(text: str) -> str:
    return " ".join(text.split())


AMOUNT = r"(\d{1,3}(?:[ \u00a0]\d{3})*,(?:\d{2}|-))"
# reportlab invariant mode pins the dates to this value.
FIXED_DATE = b"/CreationDate (D:20000101000000"


def parse_printed_amount(text: str) -> Decimal:
    match = re.fullmatch(AMOUNT + r"\s*(?:CZK|EUR|Kč)", text.strip())
    assert match, text
    number = match.group(1).replace(" ", "").replace("\u00a0", "")
    return Decimal(number.replace(",-", ",00").replace(",", "."))


def read_index() -> list[dict[str, str]]:
    with (CASES_DIR / "index.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_generator_is_deterministic():
    again = build_cases()
    assert [c.expected for c in again] == [c.expected for c in CASES]
    assert [c.document.lines() for c in again] == [c.document.lines() for c in CASES]


def test_case_ids_are_unique():
    assert len(BY_ID) == len(CASES)


def test_index_matches_case_folders():
    rows = read_index()
    assert list(rows[0]) == build.INDEX_COLUMNS
    assert [r["id"] for r in rows] == [c.id for c in CASES]
    folders = sorted(p.name for p in CASES_DIR.iterdir() if p.is_dir())
    assert folders == sorted(BY_ID)
    for row in rows:
        case = BY_ID[row["id"]]
        assert row == build.index_row(case)
        assert sorted(p.name for p in (CASES_DIR / case.id).iterdir()) == [
            "document.pdf",
            "expected.json",
        ]


def test_committed_expected_matches_generator_byte_for_byte():
    for case in CASES:
        committed = (CASES_DIR / case.id / "expected.json").read_text(encoding="utf-8")
        assert committed == build.expected_json(case), case.id


def test_every_expected_passes_schema():
    for case in CASES:
        data = json.loads((CASES_DIR / case.id / "expected.json").read_text(encoding="utf-8"))
        assert validate_expected(data) == [], case.id


def test_dataset_size_types_and_scans():
    assert 18 <= len(CASES) <= 22
    assert {c.typ for c in CASES} == {"smlouva_o_investici", "vypis_z_uctu", "potvrzeni_pokynu"}
    assert 3 <= len(SCAN_CASES) <= 4
    assert all(c.scan is not None for c in SCAN_CASES)
    assert all(c.scan is None for c in TEXT_CASES)


@pytest.mark.parametrize(
    "tag",
    [
        "neplatna_kontrolni_cislice",
        "castka_mezery_kc",
        "dve_castky",
        "chybi_cislo_smlouvy",
        "sken_rozmazany",
        "sken_natoceny",
        "sken_necitelny",
    ],
)
def test_tricky_case_is_present(tag):
    assert any(tag in c.stitky for c in CASES)


def test_tricky_cases_carry_their_meaning():
    for case in CASES:
        e = case.expected
        if "neplatna_kontrolni_cislice" in case.stitky:
            assert e["isin_platny"] is False and not is_valid_isin(e["isin"])
        lines = case.document.lines()
        if "chybi_cislo_smlouvy" in case.stitky:
            assert e["cislo_smlouvy"] is None
            if case.typ == SMLOUVA:
                # An unsigned draft has no conclusion date either.
                assert not any(line.startswith("Datum uzavření") for line in lines)
        if "isin_s_mezerami" in case.stitky:
            printed = [line.removeprefix("ISIN: ") for line in lines if line.startswith("ISIN:")]
            assert len(printed) == 1 and " " in printed[0]
            assert printed[0].replace(" ", "") == e["isin"] and " " not in e["isin"]
        if "bez_isin" in case.stitky:
            assert e["isin"] is None and e["isin_platny"] is None
            assert not any("ISIN" in line for line in lines)
        if "castka_mezery_kc" in case.stitky:
            assert e["mena"] == "CZK"
            assert "Kč" in case.castka_text and " " in case.castka_text
        if case.stitky:
            assert case.zaludnost
        else:
            assert not case.zaludnost


def test_unreadable_means_all_other_fields_null():
    unreadable = [c for c in CASES if c.expected["citelne"] is False]
    assert [c.id for c in unreadable] == [c.id for c in CASES if "sken_necitelny" in c.stitky]
    assert len(unreadable) == 1 and unreadable[0].forma == "sken"
    for case in unreadable:
        assert all(case.expected[f] is None for f in FIELDS if f != "citelne")


def test_readable_cases_have_amount_and_currency():
    for case in CASES:
        if case.expected["citelne"]:
            assert case.expected["castka"] is not None, case.id


def test_expected_values_are_printed_in_document():
    for case in CASES:
        if not case.expected["citelne"]:
            continue
        text = squash("\n".join(case.document.lines()))
        e = case.expected
        if e["cislo_smlouvy"] is not None:
            assert e["cislo_smlouvy"] in text, case.id
        if e["isin"] is not None:
            assert e["isin"] in text.replace(" ", ""), case.id
        assert case.castka_text in text, case.id


def test_expected_amount_is_the_printed_amount():
    for case in CASES:
        if case.expected["citelne"]:
            assert parse_printed_amount(case.castka_text) == Decimal(case.expected["castka"]), (
                case.id
            )


TWO_AMOUNTS_ANCHOR = {
    VYPIS: r"Konečný zůstatek k [^:]*: ",
    "potvrzeni_pokynu": r"Objem pokynu \(bez poplatku\): ",
    SMLOUVA: r"mění na ",
}


def test_two_amounts_expected_is_the_one_that_counts():
    cases = [c for c in TEXT_CASES if "dve_castky" in c.stitky]
    assert {c.typ for c in cases} == set(TWO_AMOUNTS_ANCHOR)
    for case in cases:
        text = pdf_text(CASES_DIR / case.id / "document.pdf")
        amounts = {parse_printed_amount(m + " CZK") for m in re.findall(AMOUNT, text)}
        assert len(amounts) >= 2, case.id
        match = re.search(TWO_AMOUNTS_ANCHOR[case.typ] + AMOUNT, text)
        assert match, case.id
        assert parse_printed_amount(match.group(1) + " CZK") == Decimal(case.expected["castka"])


def test_missing_contract_number_is_not_printed():
    for case in CASES:
        if case.expected["cislo_smlouvy"] is None and case.expected["citelne"]:
            assert not re.search(r"SM-\d{4}-\d{6}", "\n".join(case.document.lines())), case.id


def test_committed_text_pdfs_match_generator(tmp_path):
    for case in TEXT_CASES:
        fresh = tmp_path / f"{case.id}.pdf"
        render_text_pdf(case.document, fresh)
        committed_text = pdf_text(CASES_DIR / case.id / "document.pdf")
        assert committed_text == pdf_text(fresh), case.id
        for line in case.document.lines():
            assert squash(line) in committed_text, (case.id, line)


def test_text_pdf_rendering_is_byte_stable(tmp_path):
    case = TEXT_CASES[0]
    render_text_pdf(case.document, tmp_path / "a.pdf")
    render_text_pdf(case.document, tmp_path / "b.pdf")
    assert (tmp_path / "a.pdf").read_bytes() == (tmp_path / "b.pdf").read_bytes()
    # Two renders within one second match even without invariant mode; the fixed date does not.
    assert FIXED_DATE in (tmp_path / "a.pdf").read_bytes()
    for text_case in TEXT_CASES:
        assert FIXED_DATE in (CASES_DIR / text_case.id / "document.pdf").read_bytes(), text_case.id


def test_font_dir_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("LLMEVAL_FONT_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="LLMEVAL_FONT_DIR"):
        render_pdf.find_font_dir()
    real = render_pdf.find_font_dir(render_pdf.FONT_DIRS)
    for name in (render_pdf.REGULAR, render_pdf.BOLD):
        (tmp_path / f"{name}.ttf").symlink_to(real / f"{name}.ttf")
    assert render_pdf.find_font_dir() == tmp_path


def test_missing_font_is_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("LLMEVAL_FONT_DIR", raising=False)
    monkeypatch.setattr(render_pdf, "FONT_DIRS", (tmp_path,))
    with pytest.raises(FileNotFoundError, match="DejaVu Sans"):
        render_pdf.find_font_dir()


def test_document_longer_than_a_page_is_refused(tmp_path):
    long = Document(tuple(Block("field", "Řádek", str(i)) for i in range(80)))
    with pytest.raises(ValueError, match="one page"):
        render_text_pdf(long, tmp_path / "long.pdf")


def test_committed_scans_have_no_text_layer():
    for case in SCAN_CASES:
        assert pdf_text(CASES_DIR / case.id / "document.pdf") == "", case.id


@pytest.mark.parametrize("case_id", [c.id for c in SCAN_CASES])
def test_degrade_is_seeded(case_id):
    image = Image.new("L", (64, 40), 255)
    for x in range(10, 50):
        image.putpixel((x, 20), 0)
    params = BY_ID[case_id].scan
    a = render_scan.degrade(image, params, random.Random(7))
    b = render_scan.degrade(image, params, random.Random(7))
    c = render_scan.degrade(image, params, random.Random(8))
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()


def test_scan_pipeline_produces_image_only_pdf(tmp_path):
    case = SCAN_CASES[0]
    text_pdf = tmp_path / "text.pdf"
    render_text_pdf(case.document, text_pdf)
    out = tmp_path / "scan.pdf"
    render_scan.render_scan_pdf(text_pdf, out, case.scan, seed=case.id, dpi=30)
    assert out.read_bytes().startswith(b"%PDF")
    assert pdf_text(out) == ""
    info = subprocess.run(["pdfinfo", str(out)], capture_output=True, text=True, check=True)
    assert re.search(r"^Pages:\s+1$", info.stdout, re.M)


@pytest.fixture(scope="module")
def unreadable_text_pdf(tmp_path_factory):
    case = next(c for c in SCAN_CASES if "sken_necitelny" in c.stitky)
    pdf = tmp_path_factory.mktemp("unreadable") / "text.pdf"
    render_text_pdf(case.document, pdf)
    return case, pdf


def test_scan_pipeline_applies_damage(unreadable_text_pdf, tmp_path):
    case, text_pdf = unreadable_text_pdf
    out = tmp_path / "scan.pdf"
    render_scan.render_scan_pdf(text_pdf, out, case.scan, seed=case.id, dpi=30)
    clean = _page_image(text_pdf, 30, tmp_path / "clean")
    # Measured 0.08 with the damage, 0.93 for the same pipeline without it.
    assert _correlation(clean, _page_image(out, 30, tmp_path / "scan")) < SIMILAR


def test_scan_pipeline_honours_jpeg_quality(unreadable_text_pdf, tmp_path):
    case, text_pdf = unreadable_text_pdf
    sizes = {}
    for quality in (20, 95):
        out = tmp_path / f"q{quality}.pdf"
        params = ScanParams(jpeg_quality=quality)
        render_scan.render_scan_pdf(text_pdf, out, params, seed=case.id, dpi=30)
        sizes[quality] = out.stat().st_size
    assert sizes[95] > 1.5 * sizes[20], sizes


def test_build_cli_writes_selected_case(tmp_path):
    case = TEXT_CASES[0]
    assert build.main(["--out", str(tmp_path), "--only", case.id]) == 0
    assert (tmp_path / case.id / "expected.json").read_text(encoding="utf-8") == (
        build.expected_json(case)
    )
    assert pdf_text(tmp_path / case.id / "document.pdf") == pdf_text(
        CASES_DIR / case.id / "document.pdf"
    )
    with (tmp_path / "index.csv").open(newline="", encoding="utf-8") as f:
        # Partial regeneration must not shrink the index of an existing dataset.
        assert [r["id"] for r in csv.DictReader(f)] == [c.id for c in CASES]
    assert [p.name for p in tmp_path.iterdir() if p.is_dir()] == [case.id]


def test_build_cli_rejects_unknown_case(tmp_path):
    with pytest.raises(SystemExit):
        build.main(["--out", str(tmp_path), "--only", "neexistuje"])


@pytest.mark.parametrize(
    "value, styl, text",
    [
        ("1250000.50", "kod", "1 250 000,50 CZK"),
        ("1250000.50", "kc", "1 250 000,50 Kč"),
        ("35000", "kc_pomlcka", "35 000,- Kč"),
        ("999.05", "kod", "999,05 CZK"),
    ],
)
def test_format_amount(value, styl, text):
    assert format_amount(Decimal(value), "CZK", styl) == text


@pytest.mark.parametrize("value, styl", [("35000.50", "kc_pomlcka"), ("1.00", "neznamy")])
def test_format_amount_rejects(value, styl):
    with pytest.raises(ValueError):
        format_amount(Decimal(value), "CZK", styl)


def _correlation(a: Image.Image, b: Image.Image) -> float:
    pa, pb = a.tobytes(), b.tobytes()
    n = len(pa)
    ma, mb = sum(pa) / n, sum(pb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(pa, pb, strict=True))
    va = sum((x - ma) ** 2 for x in pa)
    vb = sum((y - mb) ** 2 for y in pb)
    return cov / (va * vb) ** 0.5


def _edge_energy(image: Image.Image) -> float:
    return ImageStat.Stat(image.filter(ImageFilter.FIND_EDGES)).mean[0]


STRIP = ("-x", "60", "-y", "200", "-W", "500", "-H", "120")  # field lines at 100 dpi


def _page_image(pdf: Path, dpi: int, out: Path, crop: tuple[str, ...] = ()) -> Image.Image:
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-gray", "-png", "-singlefile", *crop, str(pdf), str(out)],
        check=True,
    )
    return Image.open(f"{out}.png").convert("L")


@pytest.fixture(scope="module")
def clean_crops(tmp_path_factory):
    """A strip of field lines from each scan case, rendered at the scan resolution."""
    tmp = tmp_path_factory.mktemp("crops")
    crops = {}
    for case in SCAN_CASES:
        pdf = tmp / f"{case.id}.pdf"
        render_text_pdf(case.document, pdf)
        crops[case.id] = _page_image(pdf, case.scan.dpi, tmp / case.id, STRIP)
    return crops


# Pearson similarity to the clean page is a guard against too little or too much damage, not
# a readability measure: it ignores contrast loss and still gives about 0.56 for a blur that
# makes the text illegible. Rotation is left out where noted, it moves every pixel.
# Measured on these strips: damage without rotation c17-c19 above 0.8, c20 below 0.1;
# committed scans c17 0.79, c18 and c19 below 0.1 (rotated), c20 0.14.
SIMILAR = 0.5


def test_scan_damage_similarity_to_clean_page(clean_crops):
    for case in SCAN_CASES:
        flat = replace(case.scan, angle=0.0)
        damaged = render_scan.degrade(clean_crops[case.id], flat, random.Random(1))
        similarity = _correlation(clean_crops[case.id], damaged)
        if "sken_necitelny" in case.stitky:
            assert similarity < SIMILAR, (case.id, similarity)
        else:
            assert similarity > SIMILAR, (case.id, similarity)


def test_committed_scans_are_damaged(clean_crops, tmp_path):
    for case in SCAN_CASES:
        pdf = CASES_DIR / case.id / "document.pdf"
        committed = _page_image(pdf, case.scan.dpi, tmp_path / case.id, STRIP)
        similarity = _correlation(clean_crops[case.id], committed)
        assert similarity < 0.95, (case.id, similarity)
        if "sken_necitelny" in case.stitky:
            assert similarity < SIMILAR, (case.id, similarity)
        elif not case.scan.angle:
            assert similarity > SIMILAR, (case.id, similarity)


def test_each_damage_step_has_its_effect(clean_crops):
    # Each step runs alone against an undamaged copy, so one step cannot mask another.
    for case in SCAN_CASES:
        clean = clean_crops[case.id]

        def only(clean=clean, **step):
            return render_scan.degrade(clean, ScanParams(**step), random.Random(1))

        def spread(image):
            return ImageStat.Stat(image).stddev[0]

        base = only()
        p = case.scan
        if p.blur:
            assert spread(only(blur=p.blur)) < 0.95 * spread(base), case.id
        if p.downscale > 1:
            assert _edge_energy(only(downscale=p.downscale)) < 0.5 * _edge_energy(base), case.id
        if p.contrast != 1.0:
            flat = only(contrast=p.contrast)
            assert spread(flat) < 0.95 * spread(base) and _correlation(base, flat) > 0.99, case.id
        if p.angle:
            assert _correlation(base, only(angle=p.angle)) < 0.9, case.id
        if p.noise:
            difference = ImageChops.difference(only(noise=p.noise), base)
            assert ImageStat.Stat(difference).mean[0] > p.noise / 8, case.id


def test_scan_pdf_is_seeded_at_the_case_resolution(unreadable_text_pdf, tmp_path):
    # No dpi argument: the case's own resolution is used.
    case, text_pdf = unreadable_text_pdf
    outputs = {}
    for name, seed in (("a", case.id), ("b", case.id), ("other", "another-seed")):
        outputs[name] = tmp_path / f"{name}.pdf"
        render_scan.render_scan_pdf(text_pdf, outputs[name], case.scan, seed=seed)
    a, b, other = (outputs[n].read_bytes() for n in ("a", "b", "other"))
    assert a == b
    assert a != other


def test_build_cli_defaults_to_every_case_in_data_cases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert build.main([]) == 0
    out = tmp_path / "data" / "cases"
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [c.id for c in CASES]
    for case in CASES:
        assert (out / case.id / "expected.json").read_text(encoding="utf-8") == (
            build.expected_json(case)
        )
