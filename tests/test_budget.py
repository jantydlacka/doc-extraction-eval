import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

from llmeval.budget import (
    LedgerBudget,
    LedgerError,
    default_ledger_path,
    init_ledger,
    ledger_total,
    read_ledger,
    upper_bound,
)
from llmeval.models.base import (
    Attempt,
    AttemptOutcome,
    BudgetExceededError,
    CallEstimate,
)
from llmeval.pricing import call_cost, load_prices

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
PRICES = load_prices()


def seed(path: Path, *amounts: str) -> None:
    init_ledger(path)
    with path.open("a", encoding="utf-8") as f:
        for amount in amounts:
            f.write(json.dumps({"model": SONNET, "outcome": "response", "cost_usd": amount}) + "\n")


def response_attempt(model=SONNET, input_tokens=1000, output_tokens=200, estimate=None):
    cost = call_cost(model, {"input_tokens": input_tokens, "output_tokens": output_tokens}, PRICES)
    return Attempt(model, AttemptOutcome.RESPONSE, cost, estimate=estimate, request_id="req_1")


def test_default_ledger_is_outside_the_repository():
    assert default_ledger_path() == Path.home() / ".local/share/llmeval/spend.jsonl"


def test_init_ledger_creates_an_empty_file_and_parents(tmp_path):
    path = tmp_path / "a" / "b" / "spend.jsonl"
    init_ledger(path)
    assert path.read_bytes() == b""
    assert ledger_total(path) == Decimal(0)


def test_init_ledger_refuses_an_existing_file_and_leaves_it_unchanged(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "1.25")
    before = path.read_bytes()
    with pytest.raises(LedgerError, match="exists"):
        init_ledger(path)
    assert path.read_bytes() == before


def test_missing_ledger_is_refused(tmp_path):
    with pytest.raises(LedgerError, match="--init-ledger"):
        LedgerBudget(tmp_path / "spend.jsonl", PRICES)


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[1, 2]",
        '{"cost_usd": 1.5}',
        '{"cost_usd": "abc"}',
        '{"cost_usd": "-1"}',
        '{"cost_usd": "NaN"}',
        '{"cost_usd": "Infinity"}',
        '{"model": "x"}',
    ],
)
def test_corrupt_ledger_line_is_refused(tmp_path, line):
    path = tmp_path / "spend.jsonl"
    path.write_text('{"cost_usd": "0.5"}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(LedgerError, match="line 2"):
        read_ledger(path)
    with pytest.raises(LedgerError):
        LedgerBudget(path, PRICES)


def test_ledger_without_final_newline_is_refused(tmp_path):
    # A line cut short by a crash mid-write must not be read as a smaller amount.
    path = tmp_path / "spend.jsonl"
    path.write_text('{"cost_usd": "0.5"}\n{"cost_usd": "0.2"}', encoding="utf-8")
    with pytest.raises(LedgerError, match="newline"):
        read_ledger(path)


def test_ledger_total_is_an_exact_decimal_sum(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "0.1", "0.2", "0.0000001")
    assert ledger_total(path) == Decimal("0.3000001")


def test_upper_bound_adds_a_margin_to_the_counted_prompt():
    # Sonnet: (ceil(1000 * 1.10) + 100) * 2 / 1M + 4096 * 10 / 1M
    bound = upper_bound(CallEstimate(SONNET, 1000, 4096), PRICES)
    assert bound == Decimal(1200 * 2 + 4096 * 10) / Decimal(1_000_000)


def test_upper_bound_uses_the_dearer_card_near_the_haiku_limit():
    below = upper_bound(CallEstimate(HAIKU, 50_000, 1000), PRICES)
    assert (
        below == Decimal(55_100) * Decimal("0.10") / 10**6 + Decimal(1000) * Decimal("0.50") / 10**6
    )
    # 95 000 counted is under the 100k limit, but with the margin it may be billed over it.
    near = upper_bound(CallEstimate(HAIKU, 95_000, 1000), PRICES)
    assert (
        near == Decimal(104_600) * Decimal("0.50") / 10**6 + Decimal(1000) * Decimal("2.50") / 10**6
    )


def test_upper_bound_of_an_unknown_model_is_an_error():
    with pytest.raises(Exception, match="no price"):
        upper_bound(CallEstimate("claude-unknown", 10, 10), PRICES)


def test_check_passes_under_the_cap_and_refuses_over_it(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "4.80")
    budget = LedgerBudget(path, PRICES)
    budget.check(CallEstimate(SONNET, 1000, 19_000))  # 4.80 + 0.1924
    with pytest.raises(BudgetExceededError, match="5"):
        budget.check(CallEstimate(SONNET, 1000, 21_000))  # 4.80 + 0.2124
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_check_rereads_the_ledger_every_time(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    budget.check(CallEstimate(SONNET, 1000, 1000))
    seed_more = json.dumps({"cost_usd": "4.999"})
    with path.open("a", encoding="utf-8") as f:
        f.write(seed_more + "\n")
    with pytest.raises(BudgetExceededError):
        budget.check(CallEstimate(SONNET, 1000, 1000))


def test_check_refuses_when_the_ledger_disappears(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    path.unlink()
    with pytest.raises(LedgerError):
        budget.check(CallEstimate(SONNET, 1000, 1000))


def test_record_appends_one_line_with_the_exact_cost(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "1")
    budget = LedgerBudget(path, PRICES, run_id="run-x")
    estimate = CallEstimate(SONNET, 1000, 4096)
    attempt = response_attempt(estimate=estimate)
    budget.record(attempt)
    lines = read_ledger(path)
    assert len(lines) == 2
    line = lines[-1]
    assert line["cost_usd"] == str(attempt.cost.total_usd)
    assert line["run_id"] == "run-x"
    assert line["model"] == SONNET
    assert line["outcome"] == "response"
    assert line["request_id"] == "req_1"
    assert line["input_tokens"] == 1000 and line["output_tokens"] == 200
    assert line["bound_usd"] == str(upper_bound(estimate, PRICES))
    assert ledger_total(path) == Decimal(1) + attempt.cost.total_usd
    assert budget.recorded_usd == attempt.cost.total_usd


def test_failed_call_goes_to_the_ledger_with_a_flag_and_counts_as_possible_spend(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    estimate = CallEstimate(SONNET, 1000, 4096)
    budget.record(
        Attempt(
            SONNET, AttemptOutcome.NO_RESPONSE, None, error="APITimeoutError", estimate=estimate
        )
    )
    (line,) = read_ledger(path)
    assert line["outcome"] == "no_response"
    assert line["cost_usd"] == "0"
    assert line["billed"] == "unknown"
    assert line["error"] == "APITimeoutError"
    assert budget.possible_unrecorded_usd == upper_bound(estimate, PRICES)
    assert budget.recorded_usd == 0


def test_pricing_error_with_a_cost_is_recorded_once_with_that_cost(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    cost = call_cost(SONNET, {"input_tokens": 1000, "output_tokens": 200}, PRICES)
    budget.record(
        Attempt(
            SONNET,
            AttemptOutcome.PRICING_ERROR,
            cost,
            error="cache write",
            estimate=CallEstimate(SONNET, 1000, 4096),
        )
    )
    (line,) = read_ledger(path)
    assert line["outcome"] == "pricing_error"
    assert line["cost_usd"] == str(cost.total_usd)
    assert line["cost_known"] is True


def test_paid_call_of_unknown_cost_records_its_bound_and_blocks_further_calls(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    estimate = CallEstimate(SONNET, 1000, 4096)
    budget.record(Attempt(SONNET, AttemptOutcome.PRICING_ERROR, None, estimate=estimate))
    (line,) = read_ledger(path)
    assert line["cost_known"] is False
    assert Decimal(line["cost_usd"]) == upper_bound(estimate, PRICES)
    with pytest.raises(BudgetExceededError, match="unknown cost"):
        budget.check(estimate)


def test_paid_call_of_unknown_cost_without_a_bound_raises(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    with pytest.raises(LedgerError, match="unknown cost"):
        budget.record(Attempt(SONNET, AttemptOutcome.PRICING_ERROR, None))
    assert read_ledger(path) == []


def test_record_never_recreates_a_deleted_ledger(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    path.unlink()
    with pytest.raises(LedgerError):
        budget.record(response_attempt())
    assert not path.exists()


# --- fix round: lock, shrinking ledger, line separators, boundaries -----------------


def test_second_budget_on_the_same_ledger_is_refused_while_the_first_holds_it(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    first = LedgerBudget(path, PRICES)
    with pytest.raises(LedgerError, match="in use"):
        LedgerBudget(path, PRICES)
    first.close()
    second = LedgerBudget(path, PRICES)
    second.close()


def test_budget_releases_the_lock_as_a_context_manager_even_on_error(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with pytest.raises(RuntimeError), LedgerBudget(path, PRICES):
        raise RuntimeError("stop")
    with LedgerBudget(path, PRICES):
        pass


def test_failed_budget_construction_leaves_no_lock(tmp_path):
    path = tmp_path / "spend.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(LedgerError):
        LedgerBudget(path, PRICES)
    path.write_text("", encoding="utf-8")
    LedgerBudget(path, PRICES).close()


def test_ledger_shrinking_mid_run_stops_check(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "2.00")
    budget = LedgerBudget(path, PRICES)
    budget.record(response_attempt())
    path.write_text('{"cost_usd": "0.01"}\n', encoding="utf-8")
    with pytest.raises(LedgerError, match="shrank"):
        budget.check(CallEstimate(SONNET, 1000, 1000))


def test_ledger_edited_down_mid_run_stops_record_after_writing(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "2.00")
    budget = LedgerBudget(path, PRICES)
    path.write_text('{"cost_usd": "1.00"}\n', encoding="utf-8")
    attempt = response_attempt()
    with pytest.raises(LedgerError, match="shrank"):
        budget.record(attempt)
    # The spend itself is still written before the run is stopped.
    assert read_ledger(path)[-1]["cost_usd"] == str(attempt.cost.total_usd)


def test_unicode_line_separators_in_a_record_keep_the_ledger_readable(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    estimate = CallEstimate(SONNET, 1000, 4096)
    budget.record(
        Attempt(
            SONNET,
            AttemptOutcome.NO_RESPONSE,
            None,
            error="a b c\x85d",
            estimate=estimate,
        )
    )
    budget.record(response_attempt())
    assert b"\xe2\x80\xa8" not in path.read_bytes()
    lines = read_ledger(path)
    assert lines[0]["error"] == "a b c\x85d"
    assert ledger_total(path) == budget.run_counted_usd


def test_reader_splits_only_on_newline(tmp_path):
    # A ledger written by an older version could hold raw separators inside strings.
    path = tmp_path / "spend.jsonl"
    path.write_text('{"cost_usd": "0.5", "error": "x y\x85z"}\n', encoding="utf-8")
    assert ledger_total(path) == Decimal("0.5")


def test_check_passes_exactly_at_the_cap_and_refuses_just_over_it(tmp_path):
    estimate = CallEstimate(SONNET, 1000, 4096)
    bound = upper_bound(estimate, PRICES)
    at_cap = tmp_path / "at.jsonl"
    seed(at_cap, str(Decimal(5) - bound))
    with LedgerBudget(at_cap, PRICES) as budget:
        budget.check(estimate)
    over = tmp_path / "over.jsonl"
    seed(over, str(Decimal(5) - bound + Decimal("0.0000001")))
    with LedgerBudget(over, PRICES) as budget, pytest.raises(BudgetExceededError):
        budget.check(estimate)


def test_record_fsyncs_the_ledger(tmp_path, monkeypatch):
    import llmeval.budget as budget_module

    path = tmp_path / "spend.jsonl"
    seed(path)
    synced = []
    real_fsync = budget_module.os.fsync
    monkeypatch.setattr(budget_module.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd)))
    with LedgerBudget(path, PRICES) as budget:
        budget.record(response_attempt())
    assert len(synced) == 1


def test_short_write_is_an_error(tmp_path, monkeypatch):
    import llmeval.budget as budget_module

    path = tmp_path / "spend.jsonl"
    seed(path)
    real_write = budget_module.os.write
    monkeypatch.setattr(budget_module.os, "write", lambda fd, data: real_write(fd, data[:-1]))
    with LedgerBudget(path, PRICES) as budget, pytest.raises(LedgerError, match="short write"):
        budget.record(response_attempt())


# --- fix round 2: replaced ledger, truthful stop after record, descriptors ------------


def _replace_by_rename(path: Path) -> None:
    # What sed -i, editors and cp+mv do: same content, new file under the same name.
    tmp = path.with_name(path.name + ".new")
    tmp.write_bytes(path.read_bytes())
    os.replace(tmp, path)


def test_ledger_replaced_by_rename_stops_check(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        _replace_by_rename(path)
        with pytest.raises(LedgerError, match="replaced"):
            budget.check(CallEstimate(SONNET, 1000, 1000))


def test_ledger_replaced_by_rename_lets_a_second_run_lock_it_but_the_first_stops(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES) as first:
        _replace_by_rename(path)
        with LedgerBudget(path, PRICES), pytest.raises(LedgerError):
            first.check(CallEstimate(SONNET, 1000, 1000))


def test_check_after_close_is_refused(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    budget = LedgerBudget(path, PRICES)
    budget.close()
    with pytest.raises(LedgerError, match="closed"):
        budget.check(CallEstimate(SONNET, 1000, 1000))


def test_record_into_a_changed_ledger_says_the_attempt_was_written(tmp_path):
    from llmeval.models.base import RecordedStopError

    path = tmp_path / "spend.jsonl"
    seed(path, "2.00")
    with LedgerBudget(path, PRICES) as budget:
        path.write_text('{"cost_usd": "1.00"}\n', encoding="utf-8")
        attempt = response_attempt()
        with pytest.raises(RecordedStopError) as exc:
            budget.record(attempt)
    message = str(exc.value)
    assert "written to the ledger" in message
    assert "do not add it by hand" in message
    assert str(attempt.cost.total_usd) in message and "req_1" in message
    assert "unrecorded" not in message
    assert read_ledger(path)[-1]["request_id"] == "req_1"


def _open_fds() -> set[str]:
    return set(os.listdir("/proc/self/fd"))


def test_refused_lock_closes_its_descriptor(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES):
        before = _open_fds()
        with pytest.raises(LedgerError, match="in use"):
            LedgerBudget(path, PRICES)
        assert _open_fds() == before


def test_os_error_opening_the_ledger_for_the_lock_is_a_ledger_error(tmp_path, monkeypatch):
    import llmeval.budget as budget_module

    path = tmp_path / "spend.jsonl"
    seed(path)

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(budget_module.os, "open", denied)
    with pytest.raises(LedgerError, match="cannot be opened"):
        LedgerBudget(path, PRICES)


# --- fix round 3: prove our own line is in the ledger before saying so -----------------


def _full_text(exc: BaseException) -> str:
    return "\n".join([str(exc), *getattr(exc, "__notes__", [])])


def _with_append_hook(budget, after):
    """Make record's append also run ``after(snapshot)``, the ledger bytes before the append."""
    original = budget._append

    def append(line):
        snapshot = budget.path.read_bytes()
        original(line)
        after(snapshot)

    budget._append = append


def _replace_with(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".new")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def test_every_entry_has_a_unique_attempt_id(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES) as budget:
        budget.record(response_attempt())
        budget.record(response_attempt())
    ids = [line["attempt_id"] for line in read_ledger(path)]
    assert len(ids) == 2 and len(set(ids)) == 2 and all(len(i) == 32 for i in ids)


@pytest.mark.parametrize("how", ["replace", "rewrite_in_place"])
def test_line_lost_after_append_is_reported_as_unrecorded_with_the_line(tmp_path, how):
    from llmeval.models.base import LedgerWriteError, RecordedStopError

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        if how == "replace":
            _with_append_hook(budget, lambda snap: _replace_with(path, snap))
        else:
            _with_append_hook(budget, lambda snap: path.write_bytes(snap))
        with pytest.raises(LedgerWriteError) as exc:
            budget.record(response_attempt())
    assert not isinstance(exc.value, RecordedStopError)
    text = _full_text(exc.value)
    assert "unrecorded" in text
    assert "do not add it by hand" not in text
    assert all("attempt_id" not in line for line in read_ledger(path))
    assert '"attempt_id": "' in text and '"cost_usd": "' in text


def test_line_duplicated_after_append_is_reported_as_uncertain(tmp_path):
    from llmeval.models.base import RecordedStopError, RecordUncertainError

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")

    def duplicate(snapshot):
        data = path.read_bytes()
        ours = data[len(snapshot) :]
        _replace_with(path, data + ours)

    with LedgerBudget(path, PRICES) as budget:
        _with_append_hook(budget, duplicate)
        with pytest.raises(RecordUncertainError) as exc:
            budget.record(response_attempt())
    assert not isinstance(exc.value, RecordedStopError)
    text = _full_text(exc.value)
    assert "unknown whether" in text and "2 times" in text
    assert '"attempt_id": "' in text


def test_ledger_unreadable_after_append_is_reported_as_uncertain(tmp_path):
    from llmeval.models.base import RecordUncertainError

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        _with_append_hook(budget, lambda snap: _replace_with(path, b"garbage\n"))
        with pytest.raises(RecordUncertainError) as exc:
            budget.record(response_attempt())
    assert "unknown whether" in _full_text(exc.value)


def test_interrupt_during_the_recheck_notes_that_the_line_was_written(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES) as budget:

        def interrupted():
            raise KeyboardInterrupt

        budget.total = interrupted
        with pytest.raises(KeyboardInterrupt) as exc:
            budget.record(response_attempt())
    assert "appended to the ledger before the interrupt" in _full_text(exc.value)
    assert exc.value.ledger_line_written is True
    assert len(read_ledger(path)) == 1


def test_os_error_from_stat_in_total_is_a_ledger_error(tmp_path, monkeypatch):
    import llmeval.budget as budget_module

    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES) as budget:

        def denied(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(budget_module.os, "stat", denied)
        with pytest.raises(LedgerError, match="cannot be read"):
            budget.check(CallEstimate(SONNET, 1000, 1000))


def test_shrinking_is_a_ledger_changed_error(tmp_path):
    from llmeval.budget import LedgerChangedError

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        path.write_text("", encoding="utf-8")
        with pytest.raises(LedgerChangedError, match="shrank"):
            budget.check(CallEstimate(SONNET, 1000, 1000))


def test_ledger_that_is_not_utf8_is_a_ledger_error(tmp_path):
    path = tmp_path / "spend.jsonl"
    path.write_bytes(b'{"cost_usd": "\xff"}\n')
    with pytest.raises(LedgerError, match="cannot be read"):
        read_ledger(path)


def test_ledger_path_that_cannot_be_read_is_a_ledger_error(tmp_path):
    with pytest.raises(LedgerError, match="cannot be read"):
        read_ledger(tmp_path)  # a directory: IsADirectoryError, an OSError


def test_failed_reread_under_the_lock_releases_the_lock(tmp_path, monkeypatch):
    import llmeval.budget as budget_module

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    real = budget_module.ledger_total
    calls = []

    def second_read_fails(p):
        calls.append(p)
        if len(calls) == 2:
            raise LedgerError("changed while locking")
        return real(p)

    monkeypatch.setattr(budget_module, "ledger_total", second_read_fails)
    with pytest.raises(LedgerError, match="changed while locking"):
        LedgerBudget(path, PRICES)
    monkeypatch.setattr(budget_module, "ledger_total", real)
    with LedgerBudget(path, PRICES) as budget:
        assert budget.start_total == Decimal("1.00")


# --- possibly billed attempts and the run limit --------------------------------------

TIMEOUT_ESTIMATE = CallEstimate(SONNET, 1000, 4096)
# 1200 * 2/1M + 4096 * 10/1M
TIMEOUT_BOUND = Decimal("0.04336")


def timeout_attempt():
    return Attempt(
        SONNET,
        AttemptOutcome.NO_RESPONSE,
        None,
        error="APITimeoutError",
        estimate=TIMEOUT_ESTIMATE,
    )


def write_lines(path: Path, *entries: dict) -> None:
    init_ledger(path)
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_timeout_bound_is_what_the_tests_assume():
    assert upper_bound(TIMEOUT_ESTIMATE, PRICES) == TIMEOUT_BOUND


def test_line_of_unknown_billing_counts_at_its_bound(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_lines(
        path,
        {"cost_usd": "1.00", "billed": "yes"},
        {"cost_usd": "0", "billed": "unknown", "bound_usd": "0.25"},
        {"cost_usd": "0.50"},
    )
    assert ledger_total(path) == Decimal("1.75")


@pytest.mark.parametrize(
    "line",
    [
        {"cost_usd": "0", "billed": "unknown"},
        {"cost_usd": "0", "billed": "unknown", "bound_usd": None},
        {"cost_usd": "0", "billed": "unknown", "bound_usd": "abc"},
        {"cost_usd": "0", "billed": "unknown", "bound_usd": "-0.1"},
        {"cost_usd": "0", "billed": "unknown", "bound_usd": 0.25},
        {"cost_usd": "0", "billed": "maybe"},
        {"cost_usd": "0", "billed": True},
        {"cost_usd": "0.10", "billed": "no"},
        {"cost_usd": "0", "billed": None},
        {"cost_usd": "0", "billed": "unknown", "bound_usd": "0"},
        {"cost_usd": "0", "billed": "yes", "outcome": "no_response"},
    ],
    ids=[
        "unknown-without-bound",
        "unknown-null-bound",
        "unknown-text-bound",
        "unknown-negative-bound",
        "unknown-float-bound",
        "other-billed-value",
        "billed-bool",
        "not-billed-with-a-cost",
        "billed-null",
        "unknown-zero-bound",
        "no-response-billed-without-a-cost",
    ],
)
def test_line_whose_billing_cannot_be_counted_stops_the_run(tmp_path, line):
    path = tmp_path / "spend.jsonl"
    write_lines(path, {"cost_usd": "1.00"}, line)
    with pytest.raises(LedgerError, match="line 2"):
        ledger_total(path)
    with pytest.raises(LedgerError, match="line 2"):
        LedgerBudget(path, PRICES)


@pytest.mark.parametrize(
    "resolved, counted",
    [
        ({"billed": "no"}, Decimal("1.00")),
        ({"billed": "yes", "cost_usd": "0.031"}, Decimal("1.031")),
    ],
    ids=["not-billed", "billed-with-real-cost"],
)
def test_resolved_line_counts_as_stated(tmp_path, resolved, counted):
    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES, run_limit=Decimal(5)) as budget:
        budget.record(timeout_attempt())
    assert ledger_total(path) == Decimal("1.00") + TIMEOUT_BOUND
    lines = read_ledger(path)
    lines[-1] |= resolved
    path.write_text("".join(json.dumps(e) + "\n" for e in lines), encoding="utf-8")
    assert ledger_total(path) == counted


def test_timeouts_of_this_run_count_toward_the_cap(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "4.85")
    with LedgerBudget(path, PRICES, run_limit=Decimal(5)) as budget:
        budget.check(TIMEOUT_ESTIMATE)
        for _ in range(3):
            budget.record(timeout_attempt())
        # 4.85 + 3 x 0.04336 + 0.04336 > 5; the cost_usd sum alone (4.85) would pass.
        with pytest.raises(BudgetExceededError, match="spending cap"):
            budget.check(TIMEOUT_ESTIMATE)


def test_timeouts_of_earlier_runs_count_toward_the_cap(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "4.85")
    for _ in range(3):
        with LedgerBudget(path, PRICES) as budget:
            budget.record(timeout_attempt())
    with LedgerBudget(path, PRICES) as budget, pytest.raises(BudgetExceededError, match="cap"):
        budget.check(TIMEOUT_ESTIMATE)


def test_repeated_timeouts_hit_the_run_limit_before_the_cap(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES, run_limit=Decimal("0.1")) as budget:
        budget.check(TIMEOUT_ESTIMATE)
        budget.record(timeout_attempt())
        budget.check(TIMEOUT_ESTIMATE)  # 0.04336 + 0.04336 <= 0.1
        budget.record(timeout_attempt())
        with pytest.raises(BudgetExceededError, match="run limit 0.1 USD"):
            budget.check(TIMEOUT_ESTIMATE)  # 3 x 0.04336 > 0.1


def test_run_limit_counts_recorded_spend_of_this_run_only(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "3.00")  # earlier runs: only the cap applies to them
    attempt = response_attempt(input_tokens=1000, output_tokens=200)  # 0.004
    with LedgerBudget(path, PRICES, run_limit=Decimal("0.045")) as budget:
        budget.check(TIMEOUT_ESTIMATE)  # 0.04336 <= 0.045
        budget.record(attempt)
        with pytest.raises(BudgetExceededError, match="run limit"):
            budget.check(TIMEOUT_ESTIMATE)  # 0.004 + 0.04336 > 0.045


def test_default_run_limit_is_one_dollar(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES) as budget:
        assert budget.run_limit == Decimal("1")
        budget.check(CallEstimate(SONNET, 1000, 99_000))  # 0.9924
        with pytest.raises(BudgetExceededError, match="run limit 1 USD"):
            budget.check(CallEstimate(SONNET, 1000, 101_000))  # 1.0124


def test_interrupt_inside_the_append_names_the_attempt_id(tmp_path, monkeypatch):
    import llmeval.budget as budget_module

    path = tmp_path / "spend.jsonl"
    seed(path)
    real_write = os.write

    def write_then_interrupt(fd, data):
        real_write(fd, data)
        raise KeyboardInterrupt

    with LedgerBudget(path, PRICES) as budget:
        monkeypatch.setattr(budget_module.os, "write", write_then_interrupt)
        with pytest.raises(KeyboardInterrupt) as exc:
            budget.record(response_attempt())
        monkeypatch.setattr(budget_module.os, "write", real_write)
    (line,) = read_ledger(path)
    assert exc.value.attempt_id == line["attempt_id"]
    assert exc.value.ledger_line_uncertain is True
    text = _full_text(exc.value)
    assert f"look for attempt_id {line['attempt_id']}" in text


def test_failed_append_carries_the_attempt_id(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path)
    with LedgerBudget(path, PRICES) as budget:
        path.unlink()
        with pytest.raises(LedgerError) as exc:
            budget.record(response_attempt())
    assert len(exc.value.attempt_id) == 32


def test_interrupt_while_rereading_a_changed_ledger_is_uncertain(tmp_path, monkeypatch):
    import llmeval.budget as budget_module

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")

    def interrupted(p):
        raise KeyboardInterrupt

    def replace_then_break_reading(snapshot):
        _replace_with(path, path.read_bytes())
        monkeypatch.setattr(budget_module, "read_ledger", interrupted)

    with LedgerBudget(path, PRICES) as budget:
        _with_append_hook(budget, replace_then_break_reading)
        with pytest.raises(KeyboardInterrupt) as exc:
            budget.record(response_attempt())
    assert exc.value.ledger_line_uncertain is True
    assert not getattr(exc.value, "ledger_line_written", False)
    text = _full_text(exc.value)
    assert "look for attempt_id" in text and "replaced" in text


def test_line_with_a_duplicate_key_stops_the_run(tmp_path):
    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"cost_usd": "2.00", "cost_usd": "0"}\n')
    with pytest.raises(LedgerError, match="line 2.*duplicate key 'cost_usd'"):
        ledger_total(path)


def test_no_response_line_billed_with_its_real_cost_counts_that_cost(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_lines(path, {"cost_usd": "0.031", "billed": "yes", "outcome": "no_response"})
    assert ledger_total(path) == Decimal("0.031")


def test_unknown_line_counts_its_cost_when_that_is_above_the_bound(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_lines(path, {"cost_usd": "0.40", "billed": "unknown", "bound_usd": "0.25"})
    assert ledger_total(path) == Decimal("0.40")


def test_possibly_billed_attempt_resolved_mid_run_stops_the_run(tmp_path):
    from llmeval.budget import LedgerChangedError

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        budget.record(timeout_attempt())
        inode = path.stat().st_ino
        lines = read_ledger(path)
        lines[-1] |= {"billed": "no"}
        with path.open("r+", encoding="utf-8") as f:  # in place: same inode
            f.truncate(0)
            f.write("".join(json.dumps(e) + "\n" for e in lines))
        assert path.stat().st_ino == inode
        with pytest.raises(LedgerChangedError, match="shrank"):
            budget.check(TIMEOUT_ESTIMATE)


def test_lost_line_of_a_timeout_no_longer_counts_toward_the_run(tmp_path):
    from llmeval.models.base import LedgerWriteError

    path = tmp_path / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        _with_append_hook(budget, lambda snap: path.write_bytes(snap))
        with pytest.raises(LedgerWriteError):
            budget.record(timeout_attempt())
        assert budget.possible_unrecorded_usd == 0
        assert budget.run_counted_usd == 0
        budget.check(TIMEOUT_ESTIMATE)  # the ledger did not shrink below what is expected


def test_response_line_billed_at_cost_zero_is_accepted(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_lines(path, {"cost_usd": "0", "billed": "yes", "outcome": "response"}, {"cost_usd": "1"})
    assert ledger_total(path) == Decimal("1")


def test_no_response_line_without_billed_at_cost_zero_names_the_missing_key(tmp_path):
    path = tmp_path / "spend.jsonl"
    write_lines(path, {"cost_usd": "0", "outcome": "no_response"})
    with pytest.raises(LedgerError, match="line 1: no billed key") as exc:
        ledger_total(path)
    assert 'billed "yes"' not in str(exc.value)


def test_summary_of_a_lost_line_shows_the_ledger_under_home_with_a_tilde(tmp_path):
    from llmeval.models.base import LedgerWriteError
    from llmeval.report import summarize
    from llmeval.run import RunOutcome

    path = Path.home() / "spend.jsonl"
    seed(path, "1.00")
    with LedgerBudget(path, PRICES) as budget:
        _with_append_hook(budget, lambda snap: path.write_bytes(snap))
        with pytest.raises(LedgerWriteError) as exc:
            budget.record(response_attempt())
    assert str(path) in _full_text(exc.value)
    text = summarize(RunOutcome(stopped=exc.value))
    assert "add this line to ~/spend.jsonl by hand" in text
    assert str(Path.home()) not in text
