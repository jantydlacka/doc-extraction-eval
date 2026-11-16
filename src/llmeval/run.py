"""The run matrix: cases x prompts x models x repetitions, one call at a time.

Calls are sequential on purpose: the budget check before each call is only sound when no
other call of this run is in flight.
"""

import csv
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import ROUND_UP, Decimal
from pathlib import Path

from llmeval.budget import DEFAULT_RUN_LIMIT_USD, prompt_with_margin, upper_bound
from llmeval.grade import Grade, grade_text
from llmeval.grade import Status as FieldStatus
from llmeval.models.base import Extractor, Result, Status
from llmeval.models.claude import ClaudeExtractor
from llmeval.models.reference import AllNullExtractor, OracleExtractor
from llmeval.pricing import PriceTable
from llmeval.schema import FIELDS

DEFAULT_CASES_DIR = Path(__file__).resolve().parents[2] / "data" / "cases"
# Also the run limit of a run that needed no confirmation.
CONFIRM_ABOVE_USD = DEFAULT_RUN_LIMIT_USD


@dataclass(frozen=True)
class Case:
    id: str
    document: Path
    expected: dict


def load_cases(cases_dir: Path = DEFAULT_CASES_DIR, only: list[str] | None = None) -> list[Case]:
    """Cases in ``index.csv`` order, optionally only the given ids."""
    cases_dir = Path(cases_dir)
    with (cases_dir / "index.csv").open(encoding="utf-8", newline="") as f:
        ids = [row["id"] for row in csv.DictReader(f)]
    if only is not None:
        unknown = sorted(set(only) - set(ids))
        if unknown:
            raise ValueError(f"unknown case id: {', '.join(unknown)}")
        ids = [i for i in ids if i in only]
    return [
        Case(
            id=i,
            document=cases_dir / i / "document.pdf",
            expected=json.loads((cases_dir / i / "expected.json").read_text(encoding="utf-8")),
        )
        for i in ids
    ]


def grade_result(case: Case, result: Result) -> Grade:
    if result.raw is None:
        return Grade(False, reason=result.reason or f"no output ({result.status})")
    return grade_text(case.expected, result.raw)


@dataclass(frozen=True)
class Row:
    case_id: str
    prompt: str
    model: str
    rep: int
    result: Result
    grade: Grade
    # The case's expected values, for reporting rows whose output has no field results.
    expected: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        # A cut-off or refused answer does not count even if its text happens to grade.
        return self.result.status is Status.OK and self.grade.passed

    @property
    def fields_counted(self) -> bool:
        """False when every field counts as failed: the call did not end normally or the
        output is not a JSON object."""
        return self.result.status is Status.OK and bool(self.grade.fields)

    @property
    def field_ok(self) -> dict[str, bool]:
        return {
            name: self.fields_counted and self.grade.fields[name].status is FieldStatus.OK
            for name in FIELDS
        }

    @property
    def warnings(self) -> list[str]:
        """Things to check by hand; they do not change the verdict."""
        r = self.result
        found = []
        # Pricing always uses the requested model id.
        if r.response_model is not None and r.response_model != r.requested_model:
            found.append("response_model_differs")
        # The bound the budget checked assumed at most this many prompt tokens.
        if (
            r.tokens is not None
            and r.prompt_tokens_estimate is not None
            and r.tokens.prompt_tokens > prompt_with_margin(r.prompt_tokens_estimate)
        ):
            found.append("prompt_over_estimate")
        return found

    @property
    def reason(self) -> str:
        parts = [] if self.result.status is Status.OK else [self.result.reason]
        if self.grade.reason and self.grade.reason not in parts:
            parts.append(self.grade.reason)
        return "; ".join(p for p in parts if p)


@dataclass
class RunOutcome:
    rows: list[Row] = field(default_factory=list)
    # The exception that ended the run early, if any.
    stopped: BaseException | None = None
    wall_s: float = 0.0


def run_matrix(
    cases: list[Case],
    prompts: Mapping[str, str],
    extractors: Mapping[str, Extractor],
    reps: int,
    on_row: Callable[[Row], None] | None = None,
) -> RunOutcome:
    """Run every combination; any exception from an extractor stops the run.

    The rows done before the stop are kept, so partial results can still be written.
    """
    outcome = RunOutcome()
    start = time.perf_counter()
    try:
        for case in cases:
            for prompt_name, prompt in prompts.items():
                for model, extractor in extractors.items():
                    for rep in range(1, reps + 1):
                        result = extractor.extract(case.document, prompt)
                        grade = grade_result(case, result)
                        row = Row(case.id, prompt_name, model, rep, result, grade, case.expected)
                        outcome.rows.append(row)
                        if on_row is not None:
                            on_row(row)
    except (Exception, KeyboardInterrupt) as exc:
        outcome.stopped = exc
    outcome.wall_s = time.perf_counter() - start
    return outcome


def estimate_run(
    extractors: Mapping[str, ClaudeExtractor],
    cases: list[Case],
    prompts: Mapping[str, str],
    reps: int,
    prices: PriceTable,
) -> Decimal:
    """Upper bound of the whole run, from ``count_tokens`` per case x prompt x model.

    Retries are not included: a failed attempt is recorded at 0 (it may still be billed).
    """
    total = Decimal(0)
    for case in cases:
        for prompt in prompts.values():
            for extractor in extractors.values():
                total += reps * upper_bound(extractor.estimate(case.document, prompt), prices)
    return total


def run_baselines(cases: list[Case]) -> list[Row]:
    """Oracle (ceiling) and AllNull (floor) over the same cases; they ignore the prompt."""
    extractors = {"oracle": OracleExtractor(), "all-null": AllNullExtractor()}
    return run_matrix(cases, {"-": ""}, extractors, reps=1).rows


def usd_up(amount: Decimal) -> Decimal:
    """Rounded up to 0.0001, so that confirming the shown amount always covers the estimate."""
    return amount.quantize(Decimal("0.0001"), rounding=ROUND_UP)


class ConfirmationError(Exception):
    pass


def check_confirmation(estimate: Decimal, confirmed: Decimal | None) -> None:
    if estimate <= CONFIRM_ABOVE_USD:
        return
    if confirmed is None:
        raise ConfirmationError(
            f"estimated cost up to {usd_up(estimate)} USD is over {CONFIRM_ABOVE_USD} USD; "
            f"rerun with --potvrzuji <amount> where amount >= the estimate"
        )
    if confirmed < estimate:
        raise ConfirmationError(
            f"--potvrzuji {confirmed} is below the estimated {usd_up(estimate)} USD; refused"
        )
