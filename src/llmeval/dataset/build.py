"""Regenerate the dataset: ``python -m llmeval.dataset.build [--out DIR] [--only ID ...]``."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

from llmeval.dataset.fake_data import Case, build_cases
from llmeval.dataset.render_pdf import render_text_pdf
from llmeval.dataset.render_scan import render_scan_pdf

INDEX_COLUMNS = ["id", "typ", "forma", "stitky", "zaludnost"]


def expected_json(case: Case) -> str:
    return json.dumps(case.expected, ensure_ascii=False, indent=2) + "\n"


def index_row(case: Case) -> dict[str, str]:
    return {
        "id": case.id,
        "typ": case.typ,
        "forma": case.forma,
        "stitky": ";".join(case.stitky),
        "zaludnost": case.zaludnost,
    }


def write_case(case: Case, out_dir: Path) -> None:
    case_dir = out_dir / case.id
    case_dir.mkdir(parents=True, exist_ok=True)
    pdf = case_dir / "document.pdf"
    if case.scan is None:
        render_text_pdf(case.document, pdf)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            text_pdf = Path(tmp) / "text.pdf"
            render_text_pdf(case.document, text_pdf)
            render_scan_pdf(text_pdf, pdf, case.scan, seed=case.id)
    (case_dir / "expected.json").write_text(expected_json(case), encoding="utf-8")


def build(out_dir: Path, only: list[str] | None = None) -> list[Case]:
    """Render the selected cases (all by default); the index always lists every case."""
    cases = build_cases()
    selected = [c for c in cases if only is None or c.id in only]
    for case in selected:
        write_case(case, out_dir)
    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(index_row(c) for c in cases)
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/cases"))
    parser.add_argument("--only", nargs="+", metavar="ID")
    args = parser.parse_args(argv)
    known = {c.id for c in build_cases()}
    unknown = sorted(set(args.only or ()) - known)
    if unknown:
        parser.error(f"unknown case id: {', '.join(unknown)}")
    args.out.mkdir(parents=True, exist_ok=True)
    cases = build(args.out, args.only)
    print(f"{len(cases)} cases written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
