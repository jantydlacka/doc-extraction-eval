"""Command line: ``python -m llmeval run``."""

import argparse
import sys
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from llmeval.budget import LedgerBudget, LedgerError, default_ledger_path, init_ledger
from llmeval.models.claude import DEFAULT_EFFORT, DEFAULT_MAX_TOKENS, ClaudeExtractor
from llmeval.models.fake import FakeExtractor
from llmeval.pricing import load_prices
from llmeval.prompts import PROMPT_NAMES, load_prompt
from llmeval.report import ResultsWriter, summarize, tool_commit
from llmeval.run import (
    CONFIRM_ABOVE_USD,
    Case,
    ConfirmationError,
    check_confirmation,
    estimate_run,
    load_cases,
    run_baselines,
    run_matrix,
    usd_up,
)

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}


def _amount(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}") from None
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError(f"not an amount: {text!r}")
    return value


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llmeval")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser(
        "run",
        help="run the evaluation",
        description=(
            "Run cases x prompts x models x repetitions, one call at a time, and write "
            "results/<run id>/{runs.jsonl,runs.csv,summary.md}. A real run needs the "
            "spending ledger; every call is checked against the 5 USD cap and the run limit "
            "first."
        ),
    )
    run.add_argument(
        "--fake",
        action="store_true",
        help="canned outputs, no API, no ledger; refuses the options of a real run",
    )
    run.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=f"{', '.join(MODELS)} or a model id (default: all)",
    )
    run.add_argument("--prompts", nargs="+", default=list(PROMPT_NAMES), choices=PROMPT_NAMES)
    run.add_argument("--cases", nargs="+", metavar="ID", help="case ids (default: all)")
    run.add_argument("--reps", type=_positive, default=1, help="repetitions per combination")
    run.add_argument(
        "--max-tokens", type=_positive, default=None, help=f"default: {DEFAULT_MAX_TOKENS}"
    )
    run.add_argument("--results-dir", type=Path, default=Path("results"))
    run.add_argument(
        "--ledger",
        type=Path,
        default=None,
        help=f"spending ledger (default: {default_ledger_path()})",
    )
    run.add_argument(
        "--potvrzuji",
        type=_amount,
        metavar="USD",
        help=(
            "confirm a run estimated over 1 USD; must be at least the estimate; the run limit "
            "is max(1 USD, this amount)"
        ),
    )
    run.add_argument(
        "--init-ledger",
        action="store_true",
        help="create an empty ledger and exit; refuses an existing file and other run options",
    )
    return parser


def _default_client() -> Any:
    import anthropic

    return anthropic.Anthropic(max_retries=0)


def _err(message: str) -> None:
    print(f"llmeval: {message}", file=sys.stderr)


def _shown(path: Path) -> str:
    """The path with the home directory as ~, so a summary can be shared as it is."""
    path = Path(path).absolute()
    try:
        return str(Path("~") / path.relative_to(Path.home()))
    except ValueError:
        return str(path)


_REAL_RUN_ONLY = ("--models", "--max-tokens", "--potvrzuji", "--ledger", "--init-ledger")


def main(
    argv: list[str] | None = None,
    *,
    client_factory: Callable[[], Any] = _default_client,
) -> int:
    args = _parser().parse_args(argv)
    if args.fake:
        given = [
            option
            for option in _REAL_RUN_ONLY
            if getattr(args, option[2:].replace("-", "_")) not in (None, False)
        ]
        if given:
            _err(f"--fake does not use {', '.join(given)}; refused")
            return 2
    if args.init_ledger:
        defaults = vars(_parser().parse_args(["run"]))
        given = [
            f"--{name.replace('_', '-')}"
            for name, value in vars(args).items()
            if name not in ("init_ledger", "ledger") and value != defaults[name]
        ]
        if given:
            _err(f"--init-ledger does not use {', '.join(given)}; refused")
            return 2
    ledger_path = args.ledger or default_ledger_path()
    max_tokens = args.max_tokens or DEFAULT_MAX_TOKENS

    if args.init_ledger:
        try:
            init_ledger(ledger_path)
        except LedgerError as exc:
            _err(str(exc))
            return 2
        print(f"ledger created: {ledger_path}")
        return 0

    try:
        cases = load_cases(only=args.cases)
    except ValueError as exc:
        _err(str(exc))
        return 2
    prompts = {name: load_prompt(name) for name in args.prompts}
    prices = load_prices()
    case_ids = [c.id for c in cases]
    meta: dict[str, Any] = {
        "run": None,  # the name of the results folder, set when it is made
        "tool commit": tool_commit(),
        "mode": "fake" if args.fake else "real",
        "prompts": ", ".join(prompts),
        "cases": f"{len(cases)}: {', '.join(case_ids)}" if args.cases else f"{len(cases)} (all)",
        "reps": args.reps,
        "prices verified": f"{prices.verified.isoformat()} (data/prices.json)",
    }

    if args.fake:
        return _execute(args, cases, prompts, {"fake": FakeExtractor()}, None, meta)

    models = [MODELS.get(m, m) for m in args.models or MODELS]
    try:
        # Holds the ledger lock until closed: taken before the estimate, kept for the run.
        run_limit = max(CONFIRM_ABOVE_USD, args.potvrzuji or Decimal(0))
        budget = LedgerBudget(ledger_path, prices, run_limit=run_limit)
    except LedgerError as exc:
        _err(str(exc))
        return 2
    with budget:
        try:
            client = client_factory()
            extractors = {
                m: ClaudeExtractor(m, budget, client=client, max_tokens=max_tokens, prices=prices)
                for m in models
            }
            estimate = estimate_run(extractors, cases, prompts, args.reps, prices)
            spent = budget.total()
        except Exception as exc:
            _err(f"estimate failed, nothing was run: {type(exc).__name__}: {exc}")
            return 2
        print(f"estimated cost up to {usd_up(estimate)} USD; ledger so far {spent} USD")
        if spent + estimate > budget.cap:
            _err(
                f"refused: ledger {spent} + estimate {usd_up(estimate)} USD is over the "
                f"{budget.cap} USD cap"
            )
            return 2
        try:
            check_confirmation(estimate, args.potvrzuji)
        except ConfirmationError as exc:
            _err(str(exc))
            return 2
        meta |= {
            "models": ", ".join(models),
            "effort": DEFAULT_EFFORT,
            "max_tokens": max_tokens,
            "estimate": f"up to {usd_up(estimate)} USD",
            "run limit": f"{budget.run_limit} USD",
            "ledger": _shown(ledger_path),
        }
        return _execute(args, cases, prompts, extractors, budget, meta)


def _execute(
    args: argparse.Namespace,
    cases: list[Case],
    prompts: dict[str, str],
    extractors: dict[str, Any],
    budget: LedgerBudget | None,
    meta: dict[str, Any],
) -> int:
    writer = ResultsWriter(args.results_dir)
    # One id for the folder, the ledger lines and the summary; a taken folder name gets a
    # suffix, and the suffix is part of the id.
    meta["run"] = writer.dir.name
    if budget is not None:
        budget.run_id = writer.dir.name
    outcome = run_matrix(cases, prompts, extractors, args.reps, on_row=writer.add)
    # An interrupt from here on propagates; runs.jsonl and the ledger are already complete.
    writer.write_csv(outcome)
    summary = summarize(
        outcome, budget=budget, baselines=run_baselines(cases), meta=meta, fake=args.fake
    )
    writer.write_summary(summary)
    print(summary)
    print(f"results: {writer.dir}")
    flagged: dict[str, list[str]] = {}
    for row in outcome.rows:
        for warning in row.warnings:
            flagged.setdefault(warning, []).append(
                f"{row.case_id} {row.prompt} {row.model} #{row.rep}"
            )
    for warning, labels in flagged.items():
        _err(f"warning: {warning} in {len(labels)} row(s): {', '.join(labels)}")
    if isinstance(outcome.stopped, KeyboardInterrupt):
        return 130
    return 1 if outcome.stopped is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
