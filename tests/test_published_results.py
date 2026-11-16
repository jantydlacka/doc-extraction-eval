import csv
import json
import re
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "results"
FILES = sorted(p for p in RESULTS.rglob("*") if p.is_file())
RUN_DIRS = sorted(p for p in RESULTS.iterdir() if p.is_dir()) if RESULTS.is_dir() else []
RUN_FILES = {"summary.md", "runs.csv", "runs.jsonl"}

# Local paths, request ids and anything shaped like an API key ("sk-ant-…", "sk-proj-…").
FORBIDDEN = re.compile(r"/home/|/Users/|[A-Za-z]:\\\\?Users|req_|\bsk-[A-Za-z0-9_-]{8,}")


def test_published_results_exist():
    assert RUN_DIRS


@pytest.mark.parametrize("run", RUN_DIRS, ids=lambda p: p.name)
def test_run_directory_is_complete(run):
    assert {p.name for p in run.iterdir() if p.is_file()} == RUN_FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(RESULTS)))
def test_no_local_paths_or_secrets(path):
    match = FORBIDDEN.search(path.read_text(encoding="utf-8"))
    assert match is None, match


@pytest.mark.parametrize(
    "path", [p for p in FILES if p.name == "runs.csv"], ids=lambda p: p.parent.name
)
def test_csv_is_published_as_written(path):
    # The tool writes CRLF; a checkout that converted line endings would no longer match it.
    data = path.read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n") > 0


@pytest.mark.parametrize(
    "path", [p for p in FILES if p.name == "runs.csv"], ids=lambda p: p.parent.name
)
def test_csv_request_id_is_empty(path):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert all(row["request_id"] == "" for row in rows)


@pytest.mark.parametrize(
    "path", [p for p in FILES if p.name == "runs.jsonl"], ids=lambda p: p.parent.name
)
def test_jsonl_request_id_is_empty(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert all("request_id" in row and row["request_id"] is None for row in rows)
