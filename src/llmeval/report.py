"""Run outputs: ``runs.jsonl`` (written as the run goes), ``runs.csv`` and ``summary.md``."""

import csv
import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from llmeval.budget import LedgerBudget
from llmeval.models.base import Status
from llmeval.run import Row, RunOutcome
from llmeval.schema import FIELDS

CSV_COLUMNS = (
    "case",
    "prompt",
    "model",
    "rep",
    "status",
    "passed",
    "reason",
    "raw",
    # false when the call did not end normally or the output is not a JSON object: the summary
    # then counts every field as failed, whatever the *_status columns (the grader's verdict
    # on the text) say.
    "fields_counted",
    *(f"{name}_{part}" for name in FIELDS for part in ("expected", "actual", "status")),
    "extra_keys",
    "input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
    "thinking_tokens",
    "cost_usd",
    "latency_s",
    "stop_reason",
    "response_model",
    "effort",
    "attempts",
    "prompt_tokens_estimate",
    "request_id",
    # Space-separated: response_model_differs (priced as the requested model anyway),
    # prompt_over_estimate (billed prompt above count_tokens with its margin).
    "warnings",
)

PACKAGE_DIR = Path(__file__).resolve().parent


def _cost(row: Row) -> Decimal:
    return row.result.cost.total_usd if row.result.cost is not None else Decimal(0)


def _priced(rows: list[Row]) -> bool:
    return any(r.result.cost is not None for r in rows)


def _usd(amount: Decimal) -> str:
    return f"{amount:.6f}"


def _json(value: Any) -> str:
    # JSON keeps null, "" and a string apart; an empty cell means the key was missing.
    return json.dumps(value, ensure_ascii=False)


def csv_row(row: Row) -> dict[str, Any]:
    r = row.result
    tokens = r.tokens
    out: dict[str, Any] = {
        "case": row.case_id,
        "prompt": row.prompt,
        "model": row.model,
        "rep": row.rep,
        "status": str(r.status),
        "passed": "true" if row.passed else "false",
        "reason": row.reason,
        "raw": r.raw or "",
        "fields_counted": "true" if row.fields_counted else "false",
        "extra_keys": " ".join(row.grade.extra_keys),
        "input_tokens": "" if tokens is None else tokens.input_tokens,
        "cache_read_input_tokens": "" if tokens is None else tokens.cache_read_input_tokens,
        "output_tokens": "" if tokens is None else tokens.output_tokens,
        "thinking_tokens": ""
        if tokens is None or tokens.thinking_tokens is None
        else tokens.thinking_tokens,
        "cost_usd": "" if r.cost is None else str(r.cost.total_usd),
        "latency_s": "" if r.latency_s is None else f"{r.latency_s:.3f}",
        "stop_reason": r.stop_reason or "",
        "response_model": r.response_model or "",
        "effort": r.effort or "",
        "attempts": r.attempts,
        "prompt_tokens_estimate": ""
        if r.prompt_tokens_estimate is None
        else r.prompt_tokens_estimate,
        "request_id": r.request_id or "",
        "warnings": " ".join(row.warnings),
    }
    for name in FIELDS:
        field = row.grade.fields.get(name)
        if field is None:
            expected = _json(row.expected[name]) if name in row.expected else ""
            out |= {f"{name}_expected": expected, f"{name}_actual": "", f"{name}_status": ""}
            continue
        missing = field.status.value == "chybi_klic"
        out |= {
            f"{name}_expected": _json(field.expected),
            f"{name}_actual": "" if missing else _json(field.actual),
            f"{name}_status": field.status.value,
        }
    return out


def jsonl_row(row: Row) -> dict[str, Any]:
    """The CSV row with JSON types: booleans, numbers, and null for a missing value."""
    r = row.result
    tokens = r.tokens
    out = csv_row(row)

    def count(name: str) -> int | None:
        return None if tokens is None else getattr(tokens, name)

    return out | {
        "passed": row.passed,
        "fields_counted": row.fields_counted,
        "raw": r.raw,
        "input_tokens": count("input_tokens"),
        "cache_read_input_tokens": count("cache_read_input_tokens"),
        "output_tokens": count("output_tokens"),
        "thinking_tokens": count("thinking_tokens"),
        # A decimal string, like cost_usd in the ledger: a float would round it.
        "cost_usd": None if r.cost is None else str(r.cost.total_usd),
        "latency_s": None if r.latency_s is None else round(r.latency_s, 3),
        "stop_reason": r.stop_reason,
        "response_model": r.response_model,
        "effort": r.effort,
        "prompt_tokens_estimate": r.prompt_tokens_estimate,
        "request_id": r.request_id,
        "warnings": row.warnings,
    }


def _tilde(text: str) -> str:
    """The home directory as ~ in paths, so a summary can be shared as it is."""
    home = str(Path.home()).rstrip(os.sep)
    return text.replace(home + os.sep, "~" + os.sep) if home else text


def tool_commit(repo_root: Path = PACKAGE_DIR.parents[1]) -> str:
    """Commit of the checkout the code runs from, or "unknown" (no git, not its own checkout)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    git = ["git", "-C", str(repo_root)]
    try:
        head = subprocess.run(
            [*git, "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        if head.returncode != 0:
            return "unknown"
        toplevel, sha = head.stdout.splitlines()
        # A copy inside another repository would otherwise report that repository's commit.
        if Path(toplevel).resolve() != Path(repo_root).resolve():
            return "unknown"
        # Files unpacked at the root of another repository: the top level matches, but the
        # repository does not track them.
        tracked = subprocess.run(
            [*git, "ls-files", "--error-unmatch", "src/llmeval/report.py"],
            capture_output=True,
            env=env,
            timeout=10,
        )
        if tracked.returncode != 0:
            return "unknown"
        status = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown"
    if status.returncode != 0 or status.stdout.strip():
        return f"{sha} (with uncommitted changes)"
    return sha


@dataclass(frozen=True)
class Paired:
    improved: list[str]
    worse: list[str]
    unchanged: list[str]


def _rate_by_case(rows: list[Row], model: str, prompt: str) -> dict[str, float]:
    passes: dict[str, list[bool]] = {}
    for row in rows:
        if row.model == model and row.prompt == prompt:
            passes.setdefault(row.case_id, []).append(row.passed)
    return {case: sum(p) / len(p) for case, p in passes.items()}


def paired(rows: list[Row], model: str, base: str, other: str) -> Paired:
    """Per case, ``other`` prompt against ``base``: share of passing repetitions compared."""
    a = _rate_by_case(rows, model, base)
    b = _rate_by_case(rows, model, other)
    improved, worse, unchanged = [], [], []
    for case in a:
        if case not in b:
            continue
        target = improved if b[case] > a[case] else worse if b[case] < a[case] else unchanged
        target.append(case)
    return Paired(improved, worse, unchanged)


def _ratio(ok: int, n: int) -> str:
    return f"{ok}/{n} ({100 * ok / n:.0f}%)" if n else "0/0"


def _groups(rows: list[Row]) -> list[tuple[str, str]]:
    return list(dict.fromkeys((r.model, r.prompt) for r in rows))


def _select(rows: list[Row], model: str, prompt: str) -> list[Row]:
    return [r for r in rows if r.model == model and r.prompt == prompt]


def summarize(
    outcome: RunOutcome,
    *,
    budget: LedgerBudget | None = None,
    baselines: list[Row] | None = None,
    meta: dict[str, Any] | None = None,
    fake: bool = False,
) -> str:
    """Markdown summary; ``fake`` marks canned outputs that ignore the prompt and cost nothing."""
    rows = outcome.rows
    groups = _groups(rows)
    lines = ["# Run summary", ""]
    for key, value in (meta or {}).items():
        lines.append(f"- {key}: {value}")
    lines.append(
        f"- rows (case × prompt × model × rep): {len(rows)}; wall time: {outcome.wall_s:.1f} s"
    )
    if outcome.stopped is not None:
        error = outcome.stopped
        lines += [
            "",
            _tilde(
                f"**STOPPED: {type(error).__name__}: {error}** – the results below are partial."
            ),
        ]
        lines += [f"  - {_tilde(note)}" for note in getattr(error, "__notes__", [])]

    lines += [
        "",
        "## Pass rate",
        "",
        'Counts are rows (case × rep). Cost is "-" where no call in the group was priced.',
        "",
        "| model | prompt | passed rows | cost USD | cost per passed row | mean cost per row "
        "| median latency s |",
        "|---|---|---|---|---|---|---|",
    ]
    for model, prompt in groups:
        sel = _select(rows, model, prompt)
        passed = sum(r.passed for r in sel)
        cost = sum((_cost(r) for r in sel), Decimal(0))
        latencies = [r.result.latency_s for r in sel if r.result.latency_s is not None]
        median = f"{statistics.median(latencies):.2f}" if latencies else "-"
        if _priced(sel):
            costs = [_usd(cost), _usd(cost / passed) if passed else "-", _usd(cost / len(sel))]
        else:
            costs = ["-", "-", "-"]
        lines.append(
            f"| {model} | {prompt} | {_ratio(passed, len(sel))} | {' | '.join(costs)} | {median} |"
        )

    lines += [
        "",
        "## Tokens per row",
        "",
        "Means over the rows whose call returned usage; thinking tokens are part of output.",
        "",
        "| model | prompt | rows with usage | mean input tokens | mean output tokens |",
        "|---|---|---|---|---|",
    ]
    for model, prompt in groups:
        sel = _select(rows, model, prompt)
        used = [r.result.tokens for r in sel if r.result.tokens is not None]
        if used:
            means = [
                f"{statistics.fmean(t.input_tokens for t in used):.1f}",
                f"{statistics.fmean(t.output_tokens for t in used):.1f}",
            ]
        else:
            means = ["-", "-"]
        lines.append(f"| {model} | {prompt} | {len(used)}/{len(sel)} | {' | '.join(means)} |")

    header = " | ".join(f"{m} {p}" for m, p in groups)
    lines += [
        "",
        "## Pass rate per field",
        "",
        "Rows (case × rep) with the field ok. A row whose call did not end normally (max_tokens, "
        "refusal, no response) or whose output is not a JSON object counts as failed on every "
        "field; runs.csv marks such rows with fields_counted=false.",
        "",
        f"| field | {header} |",
    ]
    lines.append("|---" * (len(groups) + 1) + "|")
    for name in FIELDS:
        cells = []
        for model, prompt in groups:
            sel = _select(rows, model, prompt)
            cells.append(_ratio(sum(r.field_ok[name] for r in sel), len(sel)))
        lines.append(f"| {name} | {' | '.join(cells)} |")

    lines += ["", "## Per case (passing repetitions)", "", f"| case | {header} |"]
    lines.append("|---" * (len(groups) + 1) + "|")
    for case in dict.fromkeys(r.case_id for r in rows):
        cells = []
        for model, prompt in groups:
            sel = [r for r in _select(rows, model, prompt) if r.case_id == case]
            cells.append(f"{sum(r.passed for r in sel)}/{len(sel)}")
        lines.append(f"| {case} | {' | '.join(cells)} |")

    if baselines:
        lines += ["", "## Baselines", "", "| baseline | passed |", "|---|---|"]
        for name in dict.fromkeys(r.model for r in baselines):
            sel = [r for r in baselines if r.model == name]
            lines.append(f"| {name} | {_ratio(sum(r.passed for r in sel), len(sel))} |")

    prompts = list(dict.fromkeys(r.prompt for r in rows))
    if len(prompts) > 1:
        base = prompts[0]
        lines += ["", f"## Prompts against {base}, paired per case", ""]
        for model in dict.fromkeys(r.model for r in rows):
            for other in prompts[1:]:
                p = paired(rows, model, base, other)
                lines.append(
                    f"- {model}, {other} vs {base}: improved {len(p.improved)} "
                    f"({', '.join(p.improved) or '-'}), worse {len(p.worse)} "
                    f"({', '.join(p.worse) or '-'}), unchanged {len(p.unchanged)}"
                )
        reps = max(r.rep for r in rows)
        cases = len(dict.fromkeys(r.case_id for r in rows))
        if fake:
            lines.append(
                "- Note: fake outputs ignore the prompt, so this comparison only exercises the "
                "report."
            )
        else:
            lines.append(
                f"- Note: {cases} cases × {reps} repetition(s) and non-deterministic output; a "
                "difference of a case or two may be noise, not an effect of the prompt."
            )

    cut = [r for r in rows if r.result.status is Status.MAX_TOKENS]
    lines += ["", "## Stopped at max_tokens", ""]
    lines += [f"- {_label(r)}: {r.result.reason}" for r in cut] or ["- none"]

    flagged = [r for r in rows if r.warnings]
    lines += [
        "",
        "## Warnings",
        "",
        "response_model_differs: the response names another model than requested; the row is "
        "priced as the requested one. prompt_over_estimate: the billed prompt is above "
        "count_tokens with its margin, so the bound the budget checked was too low.",
        "",
    ]
    lines += [f"- {_label(r)}: {' '.join(r.warnings)}" for r in flagged] or ["- none"]

    failures = [r for r in rows if not r.passed]
    lines += ["", "## Failures", ""]
    lines += [f"- {_label(r)}: {r.reason}" for r in failures] or ["- none"]

    lines += ["", "## Spend", ""]
    if _priced(rows):
        total = sum((_cost(r) for r in rows), Decimal(0))
        lines.append(f"- cost of the graded calls: {_usd(total)} USD")
    else:
        lines.append("- cost of the graded calls: - (no priced call)")
    if budget is None:
        lines.append("- no ledger: nothing was paid")
    else:
        lines.append(
            f"- written to the ledger by this run (cost_usd): {_usd(budget.recorded_usd)} USD"
        )
        lines.append(
            "- possible unrecorded spend (attempts without a response, recorded at 0): up to "
            f"{_usd(budget.possible_unrecorded_usd)} USD; counted at this bound toward the cap "
            f"and the run limit ({budget.run_limit} USD)"
        )
        try:
            totals = budget.totals()
            lines.append(
                f"- ledger now: {_usd(totals.counted)} USD counted toward the {budget.cap} USD cap "
                f"(cost_usd sum {_usd(totals.cost)} USD; {totals.unresolved} line(s) with "
                'billed "unknown" counted at their bound_usd)'
            )
        except Exception as exc:
            lines.append(_tilde(f"- ledger total cannot be read: {exc}"))
    return "\n".join(lines) + "\n"


def _label(row: Row) -> str:
    return f"{row.case_id} {row.prompt} {row.model} #{row.rep}"


def new_run_dir(root: Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    for n in range(1, 1000):
        path = root / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise FileExistsError(f"no free run folder for {stamp} in {root}")


def _write_atomic(path: Path, write: Any) -> None:
    # Written beside and renamed, so an interrupt never leaves a half-written file.
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class ResultsWriter:
    """Appends each row to ``runs.jsonl`` as it is done, then ``runs.csv``, then ``summary.md``.

    The CSV goes first because it needs nothing but the rows; an interrupt while the summary is
    built still leaves runs.jsonl and runs.csv complete.
    """

    def __init__(self, root: Path):
        self.dir = new_run_dir(root)
        self._jsonl = self.dir / "runs.jsonl"
        self._jsonl.touch()

    def add(self, row: Row) -> None:
        with self._jsonl.open("a", encoding="utf-8") as f:
            # ASCII-escaped: raw model text may hold U+2028 and similar, which some readers
            # take for line breaks.
            f.write(json.dumps(jsonl_row(row)) + "\n")

    def write_csv(self, outcome: RunOutcome) -> None:
        def write(f: Any) -> None:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(csv_row(r) for r in outcome.rows)

        _write_atomic(self.dir / "runs.csv", write)

    def write_summary(self, summary: str) -> None:
        _write_atomic(self.dir / "summary.md", lambda f: f.write(summary))
