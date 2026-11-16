import csv
import json
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest
from anthropic.types import Message, MessageTokensCount

from llmeval.budget import (
    LedgerBudget,
    LedgerError,
    init_ledger,
    ledger_total,
    read_ledger,
    upper_bound,
)
from llmeval.models.base import (
    BudgetExceededError,
    CallEstimate,
    LedgerWriteError,
    Result,
    Status,
)
from llmeval.models.claude import ClaudeExtractor
from llmeval.models.fake import FakeExtractor
from llmeval.models.reference import AllNullExtractor, OracleExtractor
from llmeval.pricing import CacheWriteError, PromptTooLongError, load_prices
from llmeval.report import CSV_COLUMNS, ResultsWriter, paired, summarize
from llmeval.run import (
    ConfirmationError,
    Row,
    check_confirmation,
    estimate_run,
    grade_result,
    load_cases,
    run_baselines,
    run_matrix,
)
from llmeval.schema import FIELDS

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
PRICES = load_prices()
CASES = load_cases()
PROMPTS = {"v1": "prompt one", "v2": "prompt two"}


def message(text, *, model=SONNET, usage=None, stop_reason="end_turn"):
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 1000, "output_tokens": 200},
        }
    )


def server_error(status=500):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    return anthropic.InternalServerError(f"status {status}", response=response, body=None)


def mock_client(*responses, count=1000):
    client = MagicMock()
    client.max_retries = 0
    client.messages.create.side_effect = list(responses)
    client.messages.count_tokens.return_value = MessageTokensCount.model_validate(
        {"input_tokens": count}
    )
    return client


def claude(client, budget, model=SONNET, **kwargs):
    kwargs.setdefault("sleep", lambda s: None)
    return ClaudeExtractor(model, budget, client=client, prices=PRICES, **kwargs)


def expected_text(case_id):
    case = next(c for c in CASES if c.id == case_id)
    return json.dumps(case.expected)


def ledger(tmp_path, *amounts):
    path = tmp_path / "spend.jsonl"
    init_ledger(path)
    with path.open("a", encoding="utf-8") as f:
        for amount in amounts:
            f.write(json.dumps({"model": SONNET, "outcome": "response", "cost_usd": amount}) + "\n")
    return path


class Scripted:
    """Returns given raw outputs per (case, prompt); for paired comparisons."""

    def __init__(self, outputs, name="scripted"):
        self.outputs = outputs
        self.name = name

    def extract(self, document, prompt):
        raw = self.outputs[(Path(document).parent.name, prompt)]
        return Result(raw=raw, status=Status.OK, requested_model=self.name, latency_s=0.5)


def write_outputs(root, outcome, *, budget=None, baselines=None):
    writer = ResultsWriter(root)
    for row in outcome.rows:
        writer.add(row)
    writer.write_csv(outcome)
    writer.write_summary(summarize(outcome, budget=budget, baselines=baselines))
    return writer.dir


# --- dataset and grading ------------------------------------------------------


def test_cases_are_loaded_in_index_order():
    assert [c.id for c in CASES] == [f"c{i:02d}" for i in range(1, 21)]
    assert all(c.document.name == "document.pdf" and c.document.exists() for c in CASES)
    assert set(CASES[0].expected) == set(FIELDS)


def test_load_cases_can_select_and_refuses_unknown_ids():
    assert [c.id for c in load_cases(only=["c03", "c01"])] == ["c01", "c03"]
    with pytest.raises(ValueError, match="c99"):
        load_cases(only=["c99"])


def test_invalid_json_fails_every_field():
    case = CASES[0]
    result = Result(raw="{not json", status=Status.OK, requested_model="m")
    grade = grade_result(case, result)
    row = Row(case.id, "v1", "m", 1, result, grade)
    assert not row.passed
    assert row.field_ok == dict.fromkeys(FIELDS, False)
    assert "invalid JSON" in row.reason


def test_no_response_fails_with_the_extractor_reason():
    case = CASES[0]
    result = Result(raw=None, status=Status.NO_RESPONSE, requested_model="m", reason="timeout x3")
    row = Row(case.id, "v1", "m", 1, result, grade_result(case, result))
    assert not row.passed
    assert row.field_ok == dict.fromkeys(FIELDS, False)
    assert "timeout x3" in row.reason


def test_max_tokens_never_passes_even_with_valid_json():
    case = CASES[0]
    result = Result(
        raw=expected_text(case.id),
        status=Status.MAX_TOKENS,
        requested_model="m",
        reason="output cut at max_tokens=4096",
    )
    row = Row(case.id, "v1", "m", 1, result, grade_result(case, result))
    assert not row.passed
    assert "max_tokens" in row.reason


# --- matrix ---------------------------------------------------------------------


def test_matrix_covers_cases_prompts_models_and_reps_in_order():
    outcome = run_matrix(
        CASES[:2], PROMPTS, {"oracle": OracleExtractor(), "all-null": AllNullExtractor()}, reps=2
    )
    assert outcome.stopped is None
    keys = [(r.case_id, r.prompt, r.model, r.rep) for r in outcome.rows]
    assert len(keys) == 2 * 2 * 2 * 2
    assert len(set(keys)) == len(keys)
    assert keys[0] == ("c01", "v1", "oracle", 1)


def test_baselines_bracket_the_score():
    outcome = run_matrix(
        CASES, {"v1": "x"}, {"oracle": OracleExtractor(), "all-null": AllNullExtractor()}, reps=1
    )
    oracle = [r.passed for r in outcome.rows if r.model == "oracle"]
    all_null = [r.passed for r in outcome.rows if r.model == "all-null"]
    assert all(oracle) and len(oracle) == 20
    assert sum(all_null) == 0


def test_any_exception_from_extract_stops_the_run_with_partial_rows():
    class Boom:
        calls = 0

        def extract(self, document, prompt):
            Boom.calls += 1
            if Boom.calls == 3:
                raise RuntimeError("boom")
            return OracleExtractor().extract(document, prompt)

    seen = []
    outcome = run_matrix(CASES[:5], {"v1": "x"}, {"b": Boom()}, reps=1, on_row=seen.append)
    assert isinstance(outcome.stopped, RuntimeError)
    assert len(outcome.rows) == 2 and seen == outcome.rows
    assert Boom.calls == 3


# --- budget in a run ---------------------------------------------------------------


def test_cumulative_spend_stops_mid_run_and_ledger_has_every_executed_call(tmp_path):
    path = ledger(tmp_path, "4.80")
    budget = LedgerBudget(path, PRICES)
    # Bound per call: 1200 * 2/1M + 19000 * 10/1M = 0.1924; actual first call 0.102.
    usage = {"input_tokens": 1000, "output_tokens": 10_000}
    client = mock_client(*(message(expected_text("c01"), usage=usage) for _ in range(3)))
    ex = claude(client, budget, max_tokens=19_000)
    outcome = run_matrix(CASES[:3], {"v1": "x"}, {SONNET: ex}, reps=1)
    assert isinstance(outcome.stopped, BudgetExceededError)
    assert len(outcome.rows) == 1
    assert client.messages.create.call_count == 1
    lines = read_ledger(path)
    assert len(lines) == 2
    assert ledger_total(path) == Decimal("4.902")


def test_retry_rechecks_the_budget(tmp_path):
    path = ledger(tmp_path)
    budget = LedgerBudget(path, PRICES)

    def first_call_fails_and_spending_happens_elsewhere(**kwargs):
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"cost_usd": "4.99"}) + "\n")
        raise server_error(500)

    client = mock_client()
    client.messages.create.side_effect = first_call_fails_and_spending_happens_elsewhere
    outcome = run_matrix(CASES[:1], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert isinstance(outcome.stopped, BudgetExceededError)
    assert client.messages.create.call_count == 1
    lines = read_ledger(path)
    assert [line.get("outcome") for line in lines] == [None, "no_response"]


def test_failed_attempts_are_flagged_and_reported_as_possible_unrecorded_spend(tmp_path):
    path = ledger(tmp_path)
    budget = LedgerBudget(path, PRICES)
    client = mock_client(server_error(), server_error(), server_error())
    outcome = run_matrix(CASES[:1], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert outcome.stopped is None
    (row,) = outcome.rows
    assert row.result.status is Status.NO_RESPONSE and not row.passed
    lines = read_ledger(path)
    assert [line["billed"] for line in lines] == ["unknown"] * 3
    bound = upper_bound(CallEstimate(SONNET, 1000, 4096), PRICES)
    assert budget.possible_unrecorded_usd == 3 * bound
    text = summarize(outcome, budget=budget)
    assert "possible unrecorded spend" in text.lower()
    assert f"{3 * bound:.6f}" in text


def test_ledger_write_error_stops_the_run_and_partial_results_are_written(tmp_path):
    path = ledger(tmp_path)
    budget = LedgerBudget(path, PRICES)
    client = mock_client(message(expected_text("c01")), message(expected_text("c02")))
    original = budget.record
    calls = []

    def record(attempt):
        calls.append(attempt)
        if len(calls) == 2:
            path.unlink()
        original(attempt)

    budget.record = record
    out = tmp_path / "results"
    outcome = run_matrix(CASES[:3], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert isinstance(outcome.stopped, LedgerWriteError)
    assert len(outcome.rows) == 1
    run_dir = write_outputs(out, outcome, budget=budget)
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "STOPPED" in summary and "LedgerWriteError" in summary
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 1


def test_pricing_error_with_a_cost_is_recorded_once_and_stops_the_run(tmp_path):
    path = ledger(tmp_path)
    budget = LedgerBudget(path, PRICES)
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 50}
    client = mock_client(message(expected_text("c01"), usage=usage), message("{}"))
    outcome = run_matrix(CASES[:2], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert isinstance(outcome.stopped, CacheWriteError)
    assert outcome.rows == []
    (line,) = read_ledger(path)
    assert line["outcome"] == "pricing_error"
    assert Decimal(line["cost_usd"]) == outcome.stopped.cost.total_usd > 0
    assert client.messages.create.call_count == 1


# --- estimate and confirmation ------------------------------------------------------


def test_estimate_counts_with_the_extractor_parameters_for_every_call(tmp_path):
    budget = LedgerBudget(ledger(tmp_path), PRICES)
    client = mock_client(count=1000)
    ex = claude(client, budget, max_tokens=4096)
    total = estimate_run({SONNET: ex}, CASES[:2], PROMPTS, reps=3, prices=PRICES)
    assert total == 2 * 2 * 3 * upper_bound(CallEstimate(SONNET, 1000, 4096), PRICES)
    assert client.messages.count_tokens.call_count == 4
    counted = client.messages.count_tokens.call_args.kwargs
    sent = ex.request(CASES[1].document, PROMPTS["v2"])
    assert counted == {k: v for k, v in sent.items() if k != "max_tokens"}
    assert client.messages.create.call_count == 0
    assert read_ledger(budget.path) == []


def test_estimate_refuses_a_prompt_over_the_haiku_limit(tmp_path):
    budget = LedgerBudget(ledger(tmp_path), PRICES)
    ex = claude(mock_client(count=100_001), budget, model=HAIKU)
    with pytest.raises(PromptTooLongError):
        estimate_run({HAIKU: ex}, CASES[:1], {"v1": "x"}, reps=1, prices=PRICES)


@pytest.mark.parametrize(
    ("estimate", "confirmed", "ok"),
    [
        ("0.80", None, True),
        ("1.00", None, True),
        ("1.01", None, False),
        ("1.50", "1.49", False),
        ("1.50", "1.50", True),
        ("1.50", "2", True),
    ],
)
def test_confirmation_is_needed_above_one_dollar(estimate, confirmed, ok):
    confirmed = None if confirmed is None else Decimal(confirmed)
    if ok:
        check_confirmation(Decimal(estimate), confirmed)
    else:
        with pytest.raises(ConfirmationError):
            check_confirmation(Decimal(estimate), confirmed)


# --- report ---------------------------------------------------------------------------


def test_paired_comparison_per_case():
    good = {c: expected_text(c) for c in ("c01", "c02", "c03", "c04")}
    outputs = {
        ("c01", "v1"): good["c01"],
        ("c01", "v2"): good["c01"],  # unchanged (pass)
        ("c02", "v1"): "{}",
        ("c02", "v2"): good["c02"],  # improved
        ("c03", "v1"): good["c03"],
        ("c03", "v2"): "{}",  # worse
        ("c04", "v1"): "{}",
        ("c04", "v2"): "{}",  # unchanged (fail)
    }
    prompts = {"v1": "v1", "v2": "v2"}
    outcome = run_matrix(CASES[:4], prompts, {"m": Scripted(outputs)}, reps=1)
    result = paired(outcome.rows, "m", "v1", "v2")
    assert result.improved == ["c02"]
    assert result.worse == ["c03"]
    assert result.unchanged == ["c01", "c04"]
    text = summarize(outcome)
    assert "improved 1" in text and "worse 1" in text and "unchanged 2" in text
    assert "noise" in text


def test_per_field_pass_rate_counts_invalid_json_as_failed_on_every_field():
    outputs = {("c01", "v1"): expected_text("c01"), ("c02", "v1"): "not json"}
    outcome = run_matrix(CASES[:2], {"v1": "v1"}, {"m": Scripted(outputs)}, reps=1)
    text = summarize(outcome)
    for name in FIELDS:
        assert f"| {name} | 1/2 (50%) |" in text


def test_csv_has_a_row_per_call_and_survives_commas_quotes_and_newlines(tmp_path):
    raw = json.dumps({**json.loads(expected_text("c01")), "cislo_smlouvy": 'a, "b"\nc'})
    outcome = run_matrix(CASES[:1], {"v1": "v1"}, {"m": Scripted({("c01", "v1"): raw})}, reps=1)
    run_dir = write_outputs(tmp_path, outcome)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(CSV_COLUMNS)
        (row,) = list(reader)
    assert json.loads(row["cislo_smlouvy_actual"]) == 'a, "b"\nc'
    assert row["cislo_smlouvy_status"] == "spatne"
    assert 'a, \\"b\\"' in row["reason"] or 'a, "b"' in row["reason"]
    assert row["passed"] == "false"
    assert row["case"] == "c01" and row["prompt"] == "v1" and row["model"] == "m"
    jsonl = (run_dir / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(jsonl[0])["raw"] == raw


def test_csv_distinguishes_null_empty_and_missing_key(tmp_path):
    output = json.loads(expected_text("c16"))  # isin is null there
    output["cislo_smlouvy"] = ""
    del output["mena"]
    outcome = run_matrix(
        CASES[15:16], {"v1": "v1"}, {"m": Scripted({("c16", "v1"): json.dumps(output)})}, reps=1
    )
    run_dir = write_outputs(tmp_path, outcome)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["isin_actual"] == "null"
    assert row["cislo_smlouvy_actual"] == '""'
    assert row["mena_actual"] == ""
    assert row["mena_status"] == "chybi_klic"


def test_summary_lists_failures_with_reasons_and_max_tokens_separately():
    outcome = run_matrix(CASES, {"v1": "x"}, {"fake": FakeExtractor()}, reps=1)
    text = summarize(outcome, baselines=run_baselines(CASES))
    assert "## Failures" in text
    assert "c06 v1 fake #1: isin: format" in text
    assert "## Stopped at max_tokens" in text
    max_section = text.split("## Stopped at max_tokens")[1].split("##")[0]
    assert "c13" in max_section and "c06" not in max_section
    assert "| oracle | 20/20 (100%) |" in text
    assert "| all-null | 0/20 (0%) |" in text
    assert "12/20" in text  # fake passes 12 of 20


def test_write_outputs_creates_a_new_timestamped_folder(tmp_path):
    outcome = run_matrix(CASES[:1], {"v1": "x"}, {"oracle": OracleExtractor()}, reps=1)
    first = write_outputs(tmp_path, outcome)
    second = write_outputs(tmp_path, outcome)
    assert first != second
    assert {p.name for p in first.iterdir()} == {"runs.jsonl", "runs.csv", "summary.md"}


# --- fix round ----------------------------------------------------------------------


def test_ledger_truncated_between_two_calls_stops_before_the_next_create(tmp_path):
    path = ledger(tmp_path, "1.00")
    budget = LedgerBudget(path, PRICES)
    client = mock_client(message(expected_text("c01")), message(expected_text("c02")))

    def truncate_after_first(row):
        path.write_text("", encoding="utf-8")

    outcome = run_matrix(
        CASES[:2],
        {"v1": "x"},
        {SONNET: claude(client, budget)},
        reps=1,
        on_row=truncate_after_first,
    )
    assert isinstance(outcome.stopped, LedgerError)
    assert "shrank" in str(outcome.stopped)
    assert client.messages.create.call_count == 1


def priced(case_id, cost_tokens):
    from llmeval.pricing import call_cost

    cost = call_cost(SONNET, {"input_tokens": cost_tokens, "output_tokens": 0}, PRICES)
    return cost


class Priced:
    """Scripted outputs carrying a cost, to check the cost columns of the summary."""

    def __init__(self, outputs):
        self.outputs = outputs

    def extract(self, document, prompt):
        raw, tokens = self.outputs[Path(document).parent.name]
        cost = priced(None, tokens)
        return Result(
            raw=raw,
            status=Status.OK,
            requested_model=SONNET,
            tokens=cost.tokens,
            cost=cost,
            latency_s=1.0,
        )


def test_cost_per_passed_case_divides_all_spend_by_passing_rows():
    outputs = {"c01": (expected_text("c01"), 1_000_000), "c02": ("{}", 500_000)}
    outcome = run_matrix(CASES[:2], {"v1": "x"}, {"m": Priced(outputs)}, reps=1)
    text = summarize(outcome)
    # 2 USD + 1 USD over one passing row; mean over two rows.
    assert "| m | v1 | 1/2 (50%) | 3.000000 | 3.000000 | 1.500000 | 1.00 |" in text


def test_non_ok_row_counts_as_failed_per_field_and_the_csv_says_so(tmp_path):
    class CutOff:
        def extract(self, document, prompt):
            return Result(
                raw=expected_text("c01"),
                status=Status.MAX_TOKENS,
                requested_model="m",
                reason="output cut at max_tokens=4096",
            )

    outcome = run_matrix(CASES[:1], {"v1": "x"}, {"m": CutOff()}, reps=1)
    text = summarize(outcome)
    assert "| castka | 0/1 (0%) |" in text
    run_dir = write_outputs(tmp_path, outcome)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["castka_status"] == "ok"
    assert row["fields_counted"] == "false"
    assert row["passed"] == "false"


def test_expected_values_are_json_encoded_in_the_csv(tmp_path):
    outcome = run_matrix(CASES[15:16], {"v1": "x"}, {"oracle": OracleExtractor()}, reps=1)
    run_dir = write_outputs(tmp_path, outcome)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        (row,) = list(csv.DictReader(f))
    expected = CASES[15].expected
    assert row["isin_expected"] == "null"
    assert row["castka_expected"] == json.dumps(expected["castka"])
    assert row["citelne_expected"] == "true"
    assert row["fields_counted"] == "true"


def test_output_that_is_not_an_object_keeps_expected_and_raw_in_the_csv(tmp_path):
    raw = 'oops, "not" json\nat all'
    outcome = run_matrix(CASES[:1], {"v1": "v1"}, {"m": Scripted({("c01", "v1"): raw})}, reps=1)
    run_dir = write_outputs(tmp_path, outcome)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["raw"] == raw
    for name in FIELDS:
        assert row[f"{name}_expected"] == json.dumps(CASES[0].expected[name])
        assert row[f"{name}_actual"] == "" and row[f"{name}_status"] == ""


def test_summary_counts_rows_and_fake_note_does_not_claim_noise():
    outcome = run_matrix(CASES[:2], PROMPTS, {"fake": FakeExtractor()}, reps=1)
    fake = summarize(outcome, fake=True)
    assert "rows (case × rep)" in fake
    assert "non-deterministic" not in fake
    assert "ignore the prompt" in fake
    real = summarize(outcome)
    assert "non-deterministic" in real


def test_unpriced_rows_show_no_cost_in_the_summary_as_in_the_csv(tmp_path):
    outcome = run_matrix(CASES[:1], {"v1": "x"}, {"fake": FakeExtractor()}, reps=1)
    text = summarize(outcome, fake=True)
    assert "| fake | v1 | 1/1 (100%) | - | - | - |" in text
    assert "0.000000" not in text
    run_dir = write_outputs(tmp_path, outcome)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        (row,) = list(csv.DictReader(f))
    assert row["cost_usd"] == ""


# --- fix round 2 ----------------------------------------------------------------------


def test_ledger_replaced_mid_run_stops_before_the_next_create(tmp_path):
    path = ledger(tmp_path, "1.00")
    budget = LedgerBudget(path, PRICES)
    client = mock_client(message(expected_text("c01")), message(expected_text("c02")))

    def replace_after_first(row):
        tmp = path.with_name("spend.new")
        tmp.write_bytes(path.read_bytes())
        tmp.replace(path)

    outcome = run_matrix(
        CASES[:2], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1, on_row=replace_after_first
    )
    assert "replaced" in str(outcome.stopped)
    assert client.messages.create.call_count == 1


def test_ledger_changed_during_record_stops_the_run_with_a_truthful_message(tmp_path):
    from llmeval.models.base import RecordedStopError

    path = ledger(tmp_path, "2.00")
    budget = LedgerBudget(path, PRICES)
    client = mock_client(message(expected_text("c01")), message(expected_text("c02")))

    def shrink_then_create(**kwargs):
        path.write_text('{"cost_usd": "1.00"}\n', encoding="utf-8")
        return message(expected_text("c01"))

    client.messages.create.side_effect = shrink_then_create
    outcome = run_matrix(CASES[:2], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert isinstance(outcome.stopped, RecordedStopError)
    assert not isinstance(outcome.stopped, LedgerWriteError)
    text = summarize(outcome, budget=budget)
    assert "written to the ledger" in text
    assert "unrecorded attempt" not in text
    assert client.messages.create.call_count == 1
    assert read_ledger(path)[-1]["outcome"] == "response"


def test_fields_are_not_counted_for_an_ok_call_whose_output_is_not_an_object(tmp_path):
    outcome = run_matrix(
        CASES[:1], {"v1": "v1"}, {"m": Scripted({("c01", "v1"): "not json"})}, reps=1
    )
    (row,) = outcome.rows
    assert row.result.status is Status.OK and not row.fields_counted
    run_dir = write_outputs(tmp_path, outcome)
    with (run_dir / "runs.csv").open(encoding="utf-8", newline="") as f:
        (csv_row,) = list(csv.DictReader(f))
    assert csv_row["fields_counted"] == "false"


def test_outputs_are_written_through_a_temporary_file_and_replace(tmp_path, monkeypatch):
    import llmeval.report as report

    replaced = []

    def no_replace(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        raise KeyboardInterrupt

    monkeypatch.setattr(report.os, "replace", no_replace)
    outcome = run_matrix(CASES[:1], {"v1": "x"}, {"oracle": OracleExtractor()}, reps=1)
    with pytest.raises(KeyboardInterrupt):
        write_outputs(tmp_path, outcome)
    assert replaced == [("runs.csv.tmp", "runs.csv")]
    (run_dir,) = list(tmp_path.iterdir())
    assert not (run_dir / "runs.csv").exists()


def test_runs_jsonl_is_ascii_escaped(tmp_path):
    raw = json.dumps({**json.loads(expected_text("c01")), "cislo_smlouvy": "č x\x85"})
    outcome = run_matrix(CASES[:1], {"v1": "v1"}, {"m": Scripted({("c01", "v1"): raw})}, reps=1)
    run_dir = write_outputs(tmp_path, outcome)
    data = (run_dir / "runs.jsonl").read_bytes()
    assert data.isascii()
    (line,) = data.decode("ascii").splitlines()
    assert json.loads(line)["raw"] == raw


# --- fix round 3: changed ledger seen through the client ----------------------------------


def test_interrupt_during_create_with_a_changed_ledger_stays_an_interrupt(tmp_path):
    path = ledger(tmp_path, "2.00")
    budget = LedgerBudget(path, PRICES)
    client = mock_client()

    def shrink_and_interrupt(**kwargs):
        path.write_text('{"cost_usd": "1.00"}\n', encoding="utf-8")
        raise KeyboardInterrupt

    client.messages.create.side_effect = shrink_and_interrupt
    with pytest.raises(KeyboardInterrupt) as exc:
        claude(client, budget).extract(CASES[0].document, "x")
    notes = "\n".join(exc.value.__notes__)
    assert "written to the ledger" in notes and "do not add it by hand" in notes
    assert read_ledger(path)[-1]["outcome"] == "no_response"


def test_recorded_stop_carries_the_call_error_note(tmp_path):
    from llmeval.models.base import RecordedStopError

    path = ledger(tmp_path, "2.00")
    budget = LedgerBudget(path, PRICES)
    client = mock_client()

    def shrink_and_fail(**kwargs):
        path.write_text('{"cost_usd": "1.00"}\n', encoding="utf-8")
        raise server_error(500)

    client.messages.create.side_effect = shrink_and_fail
    with pytest.raises(RecordedStopError) as exc:
        claude(client, budget).extract(CASES[0].document, "x")
    assert any(n.startswith("call error: InternalServerError") for n in exc.value.__notes__)
    assert client.messages.create.call_count == 1


def test_interrupt_during_the_recheck_is_not_reported_as_unrecorded(tmp_path):
    path = ledger(tmp_path)
    budget = LedgerBudget(path, PRICES)
    calls = []
    real_total = budget.total

    def total():
        calls.append(1)
        if len(calls) == 2:  # the first is check, the second the re-check in record
            raise KeyboardInterrupt
        return real_total()

    budget.total = total
    client = mock_client(message(expected_text("c01")))
    with pytest.raises(KeyboardInterrupt) as exc:
        claude(client, budget).extract(CASES[0].document, "x")
    notes = "\n".join(exc.value.__notes__)
    assert "appended to the ledger before the interrupt" in notes
    assert "unrecorded attempt" not in notes
    assert len(read_ledger(path)) == 1


def test_line_lost_after_append_reaches_the_runner_as_unrecorded(tmp_path):
    path = ledger(tmp_path, "1.00")
    budget = LedgerBudget(path, PRICES)
    original = budget._append

    def append_then_restore(line):
        snapshot = path.read_bytes()
        original(line)
        path.write_bytes(snapshot)

    budget._append = append_then_restore
    client = mock_client(message(expected_text("c01")), message(expected_text("c02")))
    outcome = run_matrix(CASES[:2], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert isinstance(outcome.stopped, LedgerWriteError)
    text = summarize(outcome, budget=budget)
    assert "unrecorded" in text and '"attempt_id": "' in text
    assert budget.recorded_usd == 0
    assert "do not add it by hand" not in text
    assert client.messages.create.call_count == 1


# --- branches without a test so far ------------------------------------------------


def _row(case_id, prompt, passed, rep=1, model="m", tokens=None):
    case = next(c for c in CASES if c.id == case_id)
    raw = json.dumps(case.expected) if passed else "{}"
    result = Result(raw=raw, status=Status.OK, requested_model=model, tokens=tokens)
    return Row(case_id, prompt, model, rep, result, grade_result(case, result), case.expected)


@pytest.mark.parametrize("thinking, cell, value", [(42, 42, 42), (None, "", None)])
def test_thinking_tokens_reach_the_csv_and_jsonl_rows(thinking, cell, value):
    from llmeval.pricing import TokenCounts
    from llmeval.report import csv_row, jsonl_row

    tokens = TokenCounts(1000, 0, 0, 200, thinking_tokens=thinking)
    row = _row("c01", "v1", True, tokens=tokens)
    assert csv_row(row)["thinking_tokens"] == cell
    assert jsonl_row(row)["thinking_tokens"] == value


def test_noise_note_states_cases_and_repetitions():
    rows = [
        _row(c, p, True, rep) for c in ("c01", "c02") for p in ("v1", "v2") for rep in (1, 2, 3)
    ]
    text = summarize(run_outcome(rows))
    assert "2 cases × 3 repetition(s)" in text


def run_outcome(rows):
    from llmeval.run import RunOutcome

    return RunOutcome(rows=rows)


def test_paired_comparison_uses_the_share_of_passing_repetitions():
    rows = [
        _row("c01", "v1", True, 1),
        _row("c01", "v1", False, 2),
        _row("c01", "v2", True, 1),
        _row("c01", "v2", True, 2),
        _row("c02", "v1", True, 1),
        _row("c02", "v1", True, 2),
        _row("c02", "v2", True, 1),
        _row("c02", "v2", False, 2),
        _row("c03", "v1", True, 1),
        _row("c03", "v1", False, 2),
        _row("c03", "v2", False, 1),
        _row("c03", "v2", True, 2),
    ]
    p = paired(rows, "m", "v1", "v2")
    assert (p.improved, p.worse, p.unchanged) == (["c01"], ["c02"], ["c03"])


def test_paired_comparison_skips_a_case_missing_from_one_prompt():
    rows = [_row("c01", "v1", True), _row("c01", "v2", False), _row("c04", "v1", True)]
    p = paired(rows, "m", "v1", "v2")
    assert (p.improved, p.worse, p.unchanged) == ([], ["c01"], [])


def test_three_prompts_are_each_compared_with_the_first():
    rows = [_row("c01", p, p != "v2") for p in ("v1", "v2", "v3")]
    text = summarize(run_outcome(rows))
    assert "- m, v2 vs v1: improved 0 (-), worse 1 (c01), unchanged 0" in text
    assert "- m, v3 vs v1: improved 0 (-), worse 0 (-), unchanged 1" in text


def test_summary_says_when_the_ledger_cannot_be_read_after_the_run(tmp_path):
    path = ledger(tmp_path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        path.unlink()
        text = summarize(run_outcome([_row("c01", "v1", True)]), budget=budget)
    assert "- ledger total cannot be read:" in text


def test_repeated_timeouts_stop_the_run_at_the_run_limit(tmp_path):
    from llmeval.budget import read_ledger as read

    path = ledger(tmp_path)
    budget = LedgerBudget(path, PRICES, run_limit=Decimal("0.1"))
    client = mock_client(*(server_error() for _ in range(6)))
    outcome = run_matrix(CASES[:2], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert isinstance(outcome.stopped, BudgetExceededError)
    assert "run limit" in str(outcome.stopped)
    # 2 x 0.04336 fit under 0.1, a third would not.
    assert client.messages.create.call_count == 2
    assert [line["billed"] for line in read(path)] == ["unknown"] * 2


def test_ledger_deleted_during_a_call_reports_the_attempt_id(tmp_path):
    path = ledger(tmp_path)
    budget = LedgerBudget(path, PRICES)

    def delete_and_answer(**kwargs):
        path.unlink()
        return message(expected_text("c01"))

    client = mock_client()
    client.messages.create.side_effect = delete_and_answer
    outcome = run_matrix(CASES[:1], {"v1": "x"}, {SONNET: claude(client, budget)}, reps=1)
    assert isinstance(outcome.stopped, LedgerWriteError)
    assert "attempt_id=" in str(outcome.stopped)


# --- output details ------------------------------------------------------------------


def test_runs_jsonl_uses_json_types(tmp_path):
    cost = priced(None, 1000)
    rows = [
        Result(
            raw=expected_text("c01"),
            status=Status.OK,
            requested_model=SONNET,
            response_model=SONNET,
            tokens=cost.tokens,
            cost=cost,
            latency_s=1.25,
            prompt_tokens_estimate=900,
        ),
        Result(raw=None, status=Status.NO_RESPONSE, requested_model=SONNET, reason="timeout"),
    ]

    class Two:
        def __init__(self):
            self.results = iter(rows)

        def extract(self, document, prompt):
            return next(self.results)

    outcome = run_matrix(CASES[:2], {"v1": "x"}, {SONNET: Two()}, reps=1)
    run_dir = write_outputs(tmp_path, outcome)
    first, second = (json.loads(line) for line in (run_dir / "runs.jsonl").read_text().splitlines())
    assert first["passed"] is True and first["fields_counted"] is True
    assert second["passed"] is False and second["fields_counted"] is False
    assert first["input_tokens"] == 1000 and first["thinking_tokens"] is None
    assert first["latency_s"] == 1.25 and first["prompt_tokens_estimate"] == 900
    assert first["cost_usd"] == "0.002" and first["rep"] == 1 and first["attempts"] == 1
    assert first["warnings"] == []
    for key in (
        "input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_s",
        "prompt_tokens_estimate",
        "stop_reason",
        "response_model",
        "effort",
        "request_id",
        "raw",
    ):
        assert second[key] is None, key


def test_summary_shows_mean_tokens_per_row_for_each_model_and_prompt():
    from llmeval.pricing import TokenCounts

    rows = [
        _row("c01", "v1", True, tokens=TokenCounts(1000, 0, 0, 100)),
        _row("c02", "v1", True, tokens=TokenCounts(2001, 0, 0, 301)),
        _row("c01", "v2", True, tokens=TokenCounts(1500, 0, 0, 200)),
        _row("c02", "v2", True),  # no usage: not in the mean
        _row("c01", "v1", True, model="other"),
    ]
    text = summarize(run_outcome(rows))
    section = text.split("## Tokens per row")[1].split("\n## ")[0]
    assert "| m | v1 | 2/2 | 1500.5 | 200.5 |" in section
    assert "| m | v2 | 1/2 | 1500.0 | 200.0 |" in section
    assert "| other | v1 | 0/1 | - | - |" in section


def test_interrupted_output_write_leaves_no_temporary_file(tmp_path, monkeypatch):
    import llmeval.report as report

    def interrupted(row):
        raise KeyboardInterrupt

    outcome = run_matrix(CASES[:1], {"v1": "x"}, {"oracle": OracleExtractor()}, reps=1)
    writer = ResultsWriter(tmp_path)
    monkeypatch.setattr(report, "csv_row", interrupted)
    with pytest.raises(KeyboardInterrupt):
        writer.write_csv(outcome)
    assert sorted(p.name for p in writer.dir.iterdir()) == ["runs.jsonl"]


def test_results_folder_gets_a_suffix_when_its_name_is_taken(tmp_path, monkeypatch):
    import datetime as real_datetime

    import llmeval.report as report

    class Fixed(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 16, 12, 0, 0)

    monkeypatch.setattr(report, "datetime", Fixed)
    names = [report.new_run_dir(tmp_path).name for _ in range(3)]
    assert names == ["2026-11-16T12-00-00", "2026-11-16T12-00-00-2", "2026-11-16T12-00-00-3"]


def test_extract_reads_and_encodes_the_document_once(tmp_path, monkeypatch):
    import llmeval.models.claude as claude_module

    encoded = []
    real = claude_module.base64.standard_b64encode

    class Spy:
        @staticmethod
        def standard_b64encode(data):
            encoded.append(len(data))
            return real(data)

    monkeypatch.setattr(claude_module, "base64", Spy)
    path = ledger(tmp_path)
    with LedgerBudget(path, PRICES) as budget:
        ex = claude(mock_client(message(expected_text("c01"))), budget)
        ex.extract(CASES[0].document, "x")
    assert len(encoded) == 1
    params = ex.client.messages.create.call_args.kwargs
    counted = ex.client.messages.count_tokens.call_args.kwargs
    assert counted == {k: v for k, v in params.items() if k != "max_tokens"}
