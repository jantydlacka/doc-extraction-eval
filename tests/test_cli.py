import csv
import json
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message, MessageTokensCount

from llmeval.__main__ import main
from llmeval.budget import init_ledger, read_ledger
from llmeval.run import load_cases

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"
CASES = {c.id: c for c in load_cases()}


def message(case_id, model):
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": json.dumps(CASES[case_id].expected)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1000, "output_tokens": 200},
        }
    )


class Factory:
    """Stands in for creating the real Anthropic client; records whether it was asked to."""

    def __init__(self, responses=(), count=1000):
        self.calls = 0
        self.client = MagicMock()
        self.client.max_retries = 0
        self.client.messages.create.side_effect = list(responses)
        self.client.messages.count_tokens.return_value = MessageTokensCount.model_validate(
            {"input_tokens": count}
        )

    def __call__(self):
        self.calls += 1
        return self.client


@pytest.fixture
def paths(tmp_path):
    return {"ledger": tmp_path / "ledger" / "spend.jsonl", "results": tmp_path / "results"}


def run(paths, *args, factory=None):
    argv = ["run", "--results-dir", str(paths["results"])]
    if "--fake" not in args:
        argv += ["--ledger", str(paths["ledger"])]
    return main([*argv, *args], client_factory=factory or Factory())


def only_run_dir(paths):
    (run_dir,) = list(paths["results"].iterdir())
    return run_dir


def test_fake_run_writes_outputs_without_a_ledger_or_client(paths, capsys):
    factory = Factory()
    assert run(paths, "--fake", factory=factory) == 0
    assert factory.calls == 0
    assert not paths["ledger"].exists()
    run_dir = only_run_dir(paths)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20 * 2
    assert sum(r["passed"] == "true" for r in rows) == 24
    out = capsys.readouterr().out
    assert "fake" in out and "## Failures" in out and str(run_dir) in out


def test_fake_run_leaves_an_existing_ledger_untouched(paths):
    from llmeval.budget import default_ledger_path

    ledger = default_ledger_path()
    init_ledger(ledger)
    ledger.write_text('{"cost_usd": "1.5"}\n', encoding="utf-8")
    before = ledger.read_bytes()
    assert run(paths, "--fake", "--reps", "2") == 0
    assert ledger.read_bytes() == before


def init_run(paths, *args):
    return main(["run", "--ledger", str(paths["ledger"]), "--init-ledger", *args])


def test_init_ledger_creates_it_and_refuses_an_existing_one(paths, capsys):
    assert init_run(paths) == 0
    assert paths["ledger"].read_bytes() == b""
    paths["ledger"].write_text('{"cost_usd": "0.3"}\n', encoding="utf-8")
    assert init_run(paths) == 2
    assert paths["ledger"].read_text(encoding="utf-8") == '{"cost_usd": "0.3"}\n'
    assert "exists" in capsys.readouterr().err


def test_real_run_without_a_ledger_is_refused_before_any_client(paths, capsys):
    factory = Factory()
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 2
    assert factory.calls == 0
    assert not paths["ledger"].exists()
    assert "--init-ledger" in capsys.readouterr().err
    assert not paths["results"].exists()


def test_estimate_over_one_dollar_needs_confirmation(paths, capsys):
    init_ledger(paths["ledger"])
    factory = Factory()
    # 20 cases x 2 prompts x Sonnet, each bounded at 1200*2/1M + 4096*10/1M = 0.04336.
    assert run(paths, "--models", "sonnet", factory=factory) == 2
    err = capsys.readouterr().err
    assert "1.7344" in err and "--potvrzuji" in err
    assert factory.client.messages.create.call_count == 0
    assert read_ledger(paths["ledger"]) == []
    assert not paths["results"].exists()


def test_confirmation_below_the_estimate_is_refused(paths, capsys):
    init_ledger(paths["ledger"])
    factory = Factory()
    assert run(paths, "--models", "sonnet", "--potvrzuji", "1.70", factory=factory) == 2
    assert "1.70" in capsys.readouterr().err
    assert factory.client.messages.create.call_count == 0
    assert read_ledger(paths["ledger"]) == []


def test_run_is_refused_when_the_estimate_would_pass_the_cap(paths, capsys):
    init_ledger(paths["ledger"])
    paths["ledger"].write_text('{"cost_usd": "4.999"}\n', encoding="utf-8")
    factory = Factory()
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 2
    assert "cap" in capsys.readouterr().err
    assert factory.client.messages.create.call_count == 0


def test_real_run_with_a_mocked_client_records_every_call(paths, capsys):
    init_ledger(paths["ledger"])
    responses = [message(c, HAIKU) for c in ("c01", "c01", "c02", "c02")]
    factory = Factory(responses)
    assert run(paths, "--models", "haiku", "--cases", "c01", "c02", factory=factory) == 0
    assert factory.calls == 1
    lines = read_ledger(paths["ledger"])
    assert len(lines) == 4 == factory.client.messages.create.call_count
    assert all(line["model"] == HAIKU and line["outcome"] == "response" for line in lines)
    total = sum(Decimal(line["cost_usd"]) for line in lines)
    run_dir = only_run_dir(paths)
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert f"{total:.6f}" in summary
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["passed"] for r in rows] == ["true"] * 4
    assert {r["effort"] for r in rows} == {"medium"}
    assert {r["response_model"] for r in rows} == {HAIKU}


def test_stopped_run_exits_nonzero_and_still_writes_results(paths):
    init_ledger(paths["ledger"])
    factory = Factory([message("c01", HAIKU)])
    factory.client.messages.create.side_effect = [message("c01", HAIKU), RuntimeError("boom")]
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 1
    summary = (only_run_dir(paths) / "summary.md").read_text(encoding="utf-8")
    assert "STOPPED" in summary and "boom" in summary
    assert len(read_ledger(paths["ledger"])) == 2


def test_help_lists_the_options(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--help"])
    out = capsys.readouterr().out
    for option in ("--fake", "--reps", "--potvrzuji", "--init-ledger", "--ledger", "--models"):
        assert option in out


def test_unknown_model_is_refused_before_any_call(paths, capsys):
    init_ledger(paths["ledger"])
    factory = Factory()
    assert run(paths, "--models", "claude-unknown", "--cases", "c01", factory=factory) == 2
    assert "no price" in capsys.readouterr().err
    assert factory.client.messages.count_tokens.call_count == 0
    assert read_ledger(paths["ledger"]) == []


def test_interrupt_writes_partial_results_and_exits_130(paths):
    init_ledger(paths["ledger"])
    factory = Factory()
    factory.client.messages.create.side_effect = [message("c01", HAIKU), KeyboardInterrupt()]
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 130
    run_dir = only_run_dir(paths)
    assert "KeyboardInterrupt" in (run_dir / "summary.md").read_text(encoding="utf-8")
    assert len((run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len(read_ledger(paths["ledger"])) == 2


# --- fix round ----------------------------------------------------------------------


def test_real_run_is_refused_while_another_run_holds_the_ledger(paths, capsys):
    from llmeval.budget import LedgerBudget
    from llmeval.pricing import load_prices

    init_ledger(paths["ledger"])
    factory = Factory()
    with LedgerBudget(paths["ledger"], load_prices()):
        assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 2
    assert "in use" in capsys.readouterr().err
    assert factory.calls == 0
    assert factory.client.messages.count_tokens.call_count == 0


@pytest.mark.parametrize("ending", ["ok", "stop", "interrupt", "refused"])
def test_real_run_releases_the_ledger_lock(paths, ending):
    from llmeval.budget import LedgerBudget
    from llmeval.pricing import load_prices

    init_ledger(paths["ledger"])
    responses = {
        "ok": [message("c01", HAIKU), message("c01", HAIKU)],
        "stop": [message("c01", HAIKU), RuntimeError("boom")],
        "interrupt": [KeyboardInterrupt()],
        "refused": [],
    }[ending]
    factory = Factory(responses)
    args = ["--models", "haiku", "--cases", "c01"]
    if ending == "refused":
        args = ["--models", "sonnet"]  # over 1 USD without confirmation
    run(paths, *args, factory=factory)
    LedgerBudget(paths["ledger"], load_prices()).close()


def test_fake_run_takes_no_lock(paths):
    from llmeval.budget import LedgerBudget, default_ledger_path
    from llmeval.pricing import load_prices

    init_ledger(default_ledger_path())
    with LedgerBudget(default_ledger_path(), load_prices()):
        assert run(paths, "--fake") == 0


def test_confirmed_run_over_one_dollar_goes_ahead(paths, capsys):
    init_ledger(paths["ledger"])
    cases = list(CASES)
    responses = [message(c, SONNET) for c in cases for _ in ("v1", "v2")]
    factory = Factory(responses)
    assert run(paths, "--models", "sonnet", "--potvrzuji", "1.7344", factory=factory) == 0
    assert factory.client.messages.create.call_count == 40
    assert len(read_ledger(paths["ledger"])) == 40


def test_interrupt_while_writing_the_summary_leaves_csv_jsonl_and_ledger(paths, monkeypatch):
    import llmeval.__main__ as cli

    init_ledger(paths["ledger"])
    factory = Factory([message("c01", HAIKU), message("c01", HAIKU)])

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "summarize", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(paths, "--models", "haiku", "--cases", "c01", factory=factory)
    run_dir = only_run_dir(paths)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 2
    assert len((run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert not (run_dir / "summary.md").exists()
    assert len(read_ledger(paths["ledger"])) == 2


def test_unknown_case_id_is_refused(paths, capsys):
    assert run(paths, "--fake", "--cases", "c99") == 2
    assert "c99" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args", [["--potvrzuji", "-1"], ["--potvrzuji", "abc"], ["--potvrzuji", "nan"], ["--reps", "0"]]
)
def test_bad_numbers_are_refused_by_the_parser(paths, args):
    with pytest.raises(SystemExit) as exc:
        run(paths, "--fake", *args)
    assert exc.value.code == 2


def test_count_tokens_api_error_refuses_the_run_without_any_create(paths, capsys):
    import anthropic
    import httpx2

    init_ledger(paths["ledger"])
    factory = Factory()
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages/count_tokens")
    response = httpx2.Response(400, request=request)
    factory.client.messages.count_tokens.side_effect = anthropic.BadRequestError(
        "bad", response=response, body=None
    )
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 2
    assert "estimate failed" in capsys.readouterr().err
    assert factory.client.messages.create.call_count == 0
    assert read_ledger(paths["ledger"]) == []


def test_default_client_has_no_sdk_retries(monkeypatch):
    from llmeval.__main__ import _default_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    client = _default_client()
    assert client.max_retries == 0


def test_interrupt_with_a_changed_ledger_exits_130(paths):
    init_ledger(paths["ledger"])
    paths["ledger"].write_text('{"cost_usd": "2.00"}\n', encoding="utf-8")
    factory = Factory()

    def shrink_and_interrupt(**kwargs):
        paths["ledger"].write_text('{"cost_usd": "1.00"}\n', encoding="utf-8")
        raise KeyboardInterrupt

    factory.client.messages.create.side_effect = shrink_and_interrupt
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 130
    summary = (only_run_dir(paths) / "summary.md").read_text(encoding="utf-8")
    assert "KeyboardInterrupt" in summary and "written to the ledger" in summary


def _server_error():
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(500, request=request)
    return anthropic.InternalServerError("status 500", response=response, body=None)


def test_confirmed_amount_is_the_run_limit_for_possibly_billed_attempts(paths, monkeypatch):
    import functools

    import llmeval.__main__ as cli

    no_backoff = functools.partial(cli.ClaudeExtractor, sleep=lambda s: None)
    monkeypatch.setattr(cli, "ClaudeExtractor", no_backoff)
    init_ledger(paths["ledger"])
    factory = Factory([_server_error() for _ in range(200)])
    assert run(paths, "--models", "sonnet", "--potvrzuji", "1.7344", factory=factory) == 1
    # Every attempt fails and counts at its bound 0.04336: 40 fit under 1.7344, a 41st not.
    assert factory.client.messages.create.call_count == 40
    lines = read_ledger(paths["ledger"])
    assert len(lines) == 40 and {line["billed"] for line in lines} == {"unknown"}
    summary = (only_run_dir(paths) / "summary.md").read_text(encoding="utf-8")
    assert "run limit 1.7344 USD" in summary


# --- run options and the summary header ------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [
        ["--models", "sonnet"],
        ["--max-tokens", "5"],
        ["--potvrzuji", "0.5"],
        ["--ledger", "spend.jsonl"],
        ["--init-ledger"],
    ],
    ids=["models", "max-tokens", "potvrzuji", "ledger", "init-ledger"],
)
def test_fake_run_refuses_options_of_a_real_run(paths, capsys, args):
    assert run(paths, "--fake", *args) == 2
    err = capsys.readouterr().err
    assert args[0] in err and "--fake" in err
    assert not paths["results"].exists()
    assert not (paths["results"].parent / "spend.jsonl").exists()


def _summary_meta(run_dir):
    text = (run_dir / "summary.md").read_text(encoding="utf-8")
    head = text.split("\n## ")[0]
    return dict(
        line[2:].split(": ", 1)
        for line in head.splitlines()
        if line.startswith("- ") and ": " in line
    )


def test_summary_header_names_the_tool_commit_prices_date_and_all_cases(paths, monkeypatch):
    import llmeval.__main__ as cli

    monkeypatch.setattr(cli, "tool_commit", lambda: "abc1234")
    assert run(paths, "--fake") == 0
    meta = _summary_meta(only_run_dir(paths))
    assert meta["tool commit"] == "abc1234"
    assert meta["prices verified"] == "2026-11-16 (data/prices.json)"
    assert meta["cases"] == "20 (all)"


def test_summary_header_lists_the_selected_cases(paths):
    assert run(paths, "--fake", "--cases", "c17", "c01") == 0
    assert _summary_meta(only_run_dir(paths))["cases"] == "2: c01, c17"


def test_summary_shows_the_ledger_under_home_with_a_tilde(paths, capsys):
    from llmeval.budget import default_ledger_path

    init_ledger(default_ledger_path())
    argv = ["run", "--results-dir", str(paths["results"]), "--models", "haiku", "--cases", "c01"]
    factory = Factory([message("c01", HAIKU), message("c01", HAIKU)])
    assert main(argv, client_factory=factory) == 0
    meta = _summary_meta(only_run_dir(paths))
    assert meta["ledger"] == "~/.local/share/llmeval/spend.jsonl"


def test_summary_shows_a_ledger_outside_home_in_full(paths):
    init_ledger(paths["ledger"])
    factory = Factory([message("c01", HAIKU), message("c01", HAIKU)])
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 0
    assert _summary_meta(only_run_dir(paths))["ledger"] == str(paths["ledger"])


@pytest.mark.parametrize("collision", [False, True], ids=["fresh", "name-taken"])
def test_results_folder_ledger_and_summary_share_the_run_id(paths, monkeypatch, collision):
    import datetime as real_datetime

    import llmeval.report as report

    class Fixed(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 16, 12, 0, 0)

    monkeypatch.setattr(report, "datetime", Fixed)
    if collision:
        (paths["results"] / "2026-11-16T12-00-00").mkdir(parents=True)
    init_ledger(paths["ledger"])
    factory = Factory([message("c01", HAIKU), message("c01", HAIKU)])
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 0
    expected = "2026-11-16T12-00-00" + ("-2" if collision else "")
    run_dir = paths["results"] / expected
    assert {line["run_id"] for line in read_ledger(paths["ledger"])} == {expected}
    assert _summary_meta(run_dir)["run"] == expected


def test_billed_prompt_above_the_estimate_is_reported(paths, capsys):
    init_ledger(paths["ledger"])
    big = message("c01", HAIKU)
    # count_tokens says 1000; with the margin 1000 * 1.1 + 100 = 1200 may be billed.
    big.usage.input_tokens = 1201
    factory = Factory([big, message("c01", HAIKU)])
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 0
    err = capsys.readouterr().err
    assert "warning" in err and "c01 v1 claude-haiku-5-5 #1" in err
    run_dir = only_run_dir(paths)
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    warnings = summary.split("## Warnings")[1].split("\n## ")[0]
    assert "c01 v1 claude-haiku-5-5 #1: prompt_over_estimate" in warnings
    assert "c01 v2" not in warnings
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["warnings"] for r in rows] == ["prompt_over_estimate", ""]


def test_prompt_exactly_at_the_margin_is_no_warning(paths, capsys):
    init_ledger(paths["ledger"])
    at = message("c01", HAIKU)
    at.usage.input_tokens = 1200
    factory = Factory([at, message("c01", HAIKU)])
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 0
    assert "warning" not in capsys.readouterr().err


def test_response_from_another_model_is_reported(paths, capsys):
    init_ledger(paths["ledger"])
    factory = Factory([message("c01", SONNET), message("c01", HAIKU)])
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 0
    assert "response_model_differs" in capsys.readouterr().err
    summary = (only_run_dir(paths) / "summary.md").read_text(encoding="utf-8")
    assert "c01 v1 claude-haiku-5-5 #1: response_model_differs" in summary


@pytest.mark.parametrize(
    "args",
    [
        ["--models", "sonnet"],
        ["--prompts", "v1"],
        ["--cases", "c01"],
        ["--reps", "2"],
        ["--max-tokens", "5"],
        ["--results-dir", "elsewhere"],
        ["--potvrzuji", "0.5"],
    ],
    ids=["models", "prompts", "cases", "reps", "max-tokens", "results-dir", "potvrzuji"],
)
def test_init_ledger_refuses_options_of_a_run(paths, capsys, args):
    assert init_run(paths, *args) == 2
    err = capsys.readouterr().err
    assert args[0] in err and "--init-ledger" in err
    assert not paths["ledger"].exists()


def test_confirmation_below_one_dollar_leaves_the_run_limit_at_one_dollar(paths, monkeypatch):
    import llmeval.__main__ as cli

    limits = []

    class Spy(cli.LedgerBudget):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            limits.append(self.run_limit)

    monkeypatch.setattr(cli, "LedgerBudget", Spy)
    init_ledger(paths["ledger"])
    factory = Factory([message("c01", HAIKU), message("c01", HAIKU)])
    args = ("--models", "haiku", "--cases", "c01", "--potvrzuji", "0.5")
    assert run(paths, *args, factory=factory) == 0
    assert limits == [Decimal(1)]
    assert _summary_meta(only_run_dir(paths))["run limit"] == "1 USD"


def test_summary_counts_the_unresolved_lines_of_the_ledger(paths):
    init_ledger(paths["ledger"])
    unknown = {"cost_usd": "0", "billed": "unknown", "bound_usd": "0.05"}
    lines = [unknown, {"cost_usd": "0.10"}, unknown, {"cost_usd": "0", "billed": "no"}]
    paths["ledger"].write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    factory = Factory([message("c01", HAIKU), message("c01", HAIKU)])
    assert run(paths, "--models", "haiku", "--cases", "c01", factory=factory) == 0
    summary = (only_run_dir(paths) / "summary.md").read_text(encoding="utf-8")
    assert '2 line(s) with billed "unknown"' in summary


def test_stopped_run_shows_the_home_directory_as_a_tilde(paths, capsys):
    from pathlib import Path

    from llmeval.budget import default_ledger_path

    ledger = default_ledger_path()
    init_ledger(ledger)
    ledger.write_text('{"cost_usd": "2.00"}\n', encoding="utf-8")
    factory = Factory()

    def shrink(**kwargs):
        ledger.write_text('{"cost_usd": "1.00"}\n', encoding="utf-8")
        return message("c01", HAIKU)

    factory.client.messages.create.side_effect = shrink
    argv = ["run", "--results-dir", str(paths["results"]), "--models", "haiku", "--cases", "c01"]
    assert main(argv, client_factory=factory) == 1
    summary = (only_run_dir(paths) / "summary.md").read_text(encoding="utf-8")
    home = str(Path.home())
    assert "STOPPED" in summary and "shrank" in summary
    assert "~/.local/share/llmeval/spend.jsonl" in summary
    assert home not in summary
    captured = capsys.readouterr()
    assert home not in captured.out and home not in captured.err
