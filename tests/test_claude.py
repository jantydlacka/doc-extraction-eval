import base64
import json
import logging
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest
from anthropic.types import Message, MessageTokensCount

from llmeval.models.base import (
    AttemptOutcome,
    BudgetExceededError,
    CallEstimate,
    LedgerWriteError,
    Status,
)
from llmeval.models.claude import ClaudeExtractor
from llmeval.pricing import (
    CacheWriteError,
    PriceModifierError,
    PricingError,
    PromptTooLongError,
    UnknownModelError,
)
from llmeval.schema import JSON_SCHEMA

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
DOC = Path(__file__).resolve().parent.parent / "data" / "cases" / "c01" / "document.pdf"
PROMPT = "Vytěž údaje z dokumentu."
OUTPUT = '{"cislo_smlouvy": null}'
FORBIDDEN = (
    "fallbacks",
    "thinking",
    "temperature",
    "top_p",
    "top_k",
    "inference_geo",
    "cache_control",
    "betas",
    "speed",
    "service_tier",
    "stream",
)


def message(
    text=OUTPUT,
    *,
    stop_reason="end_turn",
    model=HAIKU,
    usage=None,
    stop_details=None,
    thinking_tokens=None,
):
    usage = dict(usage or {"input_tokens": 1000, "output_tokens": 200})
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    return Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": usage,
        }
    )


def tokens(n):
    return MessageTokensCount.model_validate({"input_tokens": n})


def server_error(status=500):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    cls = {
        400: anthropic.BadRequestError,
        408: anthropic.RequestTimeoutError,
        409: anthropic.ConflictError,
        429: anthropic.RateLimitError,
        529: anthropic.OverloadedError,
    }.get(status, anthropic.InternalServerError)
    return cls(f"status {status}", response=response, body=None)


def connection_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIConnectionError(request=request)


def timeout_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APITimeoutError(request=request)


class StubBudget:
    def __init__(self, refuse_at=None):
        self.events = []
        self.refuse_at = refuse_at

    def check(self, estimate):
        self.events.append(("check", estimate))
        if self.refuse_at is not None and self.checks == self.refuse_at:
            raise BudgetExceededError("over the cap")

    def record(self, attempt):
        self.events.append(("record", attempt))

    @property
    def checks(self):
        return sum(1 for kind, _ in self.events if kind == "check")

    @property
    def records(self):
        return [a for kind, a in self.events if kind == "record"]

    @property
    def kinds(self):
        return [kind for kind, _ in self.events]


def mock_client(*responses, count=1500):
    client = MagicMock()
    client.max_retries = 0
    client.messages.create.side_effect = list(responses)
    client.messages.count_tokens.return_value = tokens(count)
    return client


def extractor(client, budget=None, model=HAIKU, **kwargs):
    kwargs.setdefault("sleep", lambda s: None)
    return ClaudeExtractor(model, budget or StubBudget(), client=client, **kwargs)


# --- request ---------------------------------------------------------------


def test_request_has_model_document_schema_and_effort():
    params = extractor(mock_client()).request(DOC, PROMPT)
    assert params["model"] == HAIKU
    assert params["max_tokens"] == 4096
    [msg] = params["messages"]
    assert msg["role"] == "user"
    document, text = msg["content"]
    assert document == {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.standard_b64encode(DOC.read_bytes()).decode("ascii"),
        },
    }
    assert text == {"type": "text", "text": PROMPT}
    assert params["output_config"] == {
        "format": {"type": "json_schema", "schema": JSON_SCHEMA},
        "effort": "medium",
    }


@pytest.mark.parametrize("key", FORBIDDEN)
def test_request_has_no_forbidden_parameter(key):
    params = extractor(mock_client()).request(DOC, PROMPT)
    assert key not in params
    assert f'"{key}"' not in json.dumps(params)


def test_request_reads_the_given_file(tmp_path):
    other = tmp_path / "x.pdf"
    other.write_bytes(b"%PDF-1.4 other")
    params = extractor(mock_client()).request(other, PROMPT)
    data = params["messages"][0]["content"][0]["source"]["data"]
    assert base64.standard_b64decode(data) == b"%PDF-1.4 other"


def test_count_tokens_and_create_get_identical_parameters():
    client = mock_client(message())
    ex = extractor(client)
    ex.extract(DOC, PROMPT)
    created = client.messages.create.call_args.kwargs
    counted = client.messages.count_tokens.call_args.kwargs
    assert created == ex.request(DOC, PROMPT)
    assert counted == {k: v for k, v in created.items() if k != "max_tokens"}


def test_effort_and_max_tokens_are_configurable():
    params = extractor(mock_client(), effort="low", max_tokens=2000).request(DOC, PROMPT)
    assert params["output_config"]["effort"] == "low"
    assert params["max_tokens"] == 2000


# --- construction ----------------------------------------------------------


def test_budget_is_required():
    with pytest.raises(TypeError):
        ClaudeExtractor(HAIKU)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="Budget"):
        ClaudeExtractor(HAIKU, None, client=mock_client())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Budget"):
        ClaudeExtractor(HAIKU, object(), client=mock_client())  # type: ignore[arg-type]


def test_builds_its_own_client_without_sdk_retries():
    ex = ClaudeExtractor(HAIKU, StubBudget())
    assert isinstance(ex.client, anthropic.Anthropic)
    assert ex.client.max_retries == 0


def test_injected_client_with_sdk_retries_is_refused():
    client = mock_client()
    client.max_retries = 2
    with pytest.raises(ValueError, match="max_retries"):
        extractor(client)


def test_unknown_model_is_refused_before_any_call():
    client = mock_client(message())
    with pytest.raises(UnknownModelError):
        extractor(client, model="claude-unknown")
    client.messages.count_tokens.assert_not_called()


# --- response --------------------------------------------------------------


def test_reads_text_usage_cost_and_metadata():
    client = mock_client(message(model="claude-haiku-5-5-20261101"))
    result = extractor(client).extract(DOC, PROMPT)
    assert result.raw == OUTPUT
    assert result.status is Status.OK
    assert result.reason == ""
    assert result.requested_model == HAIKU
    # Priced by the requested id: the dated response id is not in the price table.
    assert result.response_model == "claude-haiku-5-5-20261101"
    assert result.effort == "medium"
    assert result.stop_reason == "end_turn"
    assert result.tokens.input_tokens == 1000
    assert result.tokens.output_tokens == 200
    # 1000 * 0.10 / 1e6 + 200 * 0.50 / 1e6
    assert result.cost.total_usd == Decimal("0.0002")
    assert result.prompt_tokens_estimate == 1500
    assert result.attempts == 1
    assert result.latency_s >= 0


def test_sonnet_is_priced_on_its_own_rate():
    client = mock_client(message(model=SONNET))
    result = extractor(client, model=SONNET).extract(DOC, PROMPT)
    # 1000 * 2 / 1e6 + 200 * 10 / 1e6
    assert result.cost.total_usd == Decimal("0.004")


def test_invalid_json_is_passed_through_as_raw():
    result = extractor(mock_client(message("not json {"))).extract(DOC, PROMPT)
    assert result.raw == "not json {"
    assert result.status is Status.OK


def test_refusal_is_a_failed_case_with_reason():
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    client = mock_client(message(None, stop_reason="refusal", stop_details=details))
    budget = StubBudget()
    result = extractor(client, budget).extract(DOC, PROMPT)
    assert result.status is Status.REFUSAL
    assert "cyber" in result.reason and "declined" in result.reason
    assert result.raw == ""
    [attempt] = budget.records
    assert attempt.outcome is AttemptOutcome.RESPONSE
    assert attempt.cost.total_usd == Decimal("0.0002")


def test_refusal_without_details_still_has_reason():
    client = mock_client(message(None, stop_reason="refusal"))
    result = extractor(client).extract(DOC, PROMPT)
    assert result.status is Status.REFUSAL
    assert result.reason


def test_max_tokens_is_reported_separately():
    client = mock_client(message('{"cislo_smlo', stop_reason="max_tokens"))
    result = extractor(client).extract(DOC, PROMPT)
    assert result.status is Status.MAX_TOKENS
    assert "4096" in result.reason
    assert result.raw == '{"cislo_smlo'


def test_other_stop_reason_is_flagged():
    client = mock_client(message(stop_reason="pause_turn"))
    result = extractor(client).extract(DOC, PROMPT)
    assert result.status is Status.UNEXPECTED_STOP
    assert "pause_turn" in result.reason


def test_thinking_tokens_are_logged(caplog):
    client = mock_client(message(thinking_tokens=150))
    with caplog.at_level(logging.INFO, logger="llmeval.models.claude"):
        result = extractor(client).extract(DOC, PROMPT)
    assert result.tokens.thinking_tokens == 150
    assert "thinking_tokens=150" in caplog.text


# --- budget and retries ----------------------------------------------------


def test_budget_check_before_and_record_after_the_call():
    budget = StubBudget()
    extractor(mock_client(message()), budget).extract(DOC, PROMPT)
    assert budget.kinds == ["check", "record"]
    estimate = budget.events[0][1]
    assert estimate == CallEstimate(model=HAIKU, prompt_tokens=1500, max_tokens=4096)
    attempt = budget.records[0]
    assert attempt.outcome is AttemptOutcome.RESPONSE
    assert attempt.requested_model == HAIKU
    assert attempt.cost_usd == Decimal("0.0002")


def test_every_retry_passes_the_budget_again():
    sleeps = []
    client = mock_client(server_error(500), connection_error(), message())
    budget = StubBudget()
    result = extractor(client, budget, sleep=sleeps.append).extract(DOC, PROMPT)
    assert budget.kinds == ["check", "record"] * 3
    first, second, third = budget.records
    assert first.outcome is AttemptOutcome.NO_RESPONSE
    assert first.cost is None and first.cost_usd == Decimal(0)
    assert "500" in first.error
    assert second.outcome is AttemptOutcome.NO_RESPONSE
    assert third.outcome is AttemptOutcome.RESPONSE
    assert result.status is Status.OK
    assert result.attempts == 3
    assert len(sleeps) == 2
    client.messages.count_tokens.assert_called_once()


def test_retries_exhausted_give_a_failed_result():
    client = mock_client(server_error(429), server_error(500), connection_error())
    budget = StubBudget()
    result = extractor(client, budget, max_attempts=3).extract(DOC, PROMPT)
    assert result.status is Status.NO_RESPONSE
    assert result.raw is None
    assert result.tokens is None and result.cost is None
    assert result.attempts == 3
    assert "Connection error" in result.reason
    assert [a.outcome for a in budget.records] == [AttemptOutcome.NO_RESPONSE] * 3


def test_refused_budget_stops_before_the_call():
    client = mock_client(message())
    budget = StubBudget(refuse_at=1)
    with pytest.raises(BudgetExceededError):
        extractor(client, budget).extract(DOC, PROMPT)
    client.messages.create.assert_not_called()
    assert budget.records == []


def test_refused_budget_on_a_retry_stops_the_retries():
    client = mock_client(server_error(500), message())
    budget = StubBudget(refuse_at=2)
    with pytest.raises(BudgetExceededError):
        extractor(client, budget).extract(DOC, PROMPT)
    assert client.messages.create.call_count == 1
    assert budget.kinds == ["check", "record", "check"]


def test_non_retryable_error_is_recorded_once_and_raised():
    client = mock_client(server_error(400), message())
    budget = StubBudget()
    with pytest.raises(anthropic.BadRequestError):
        extractor(client, budget).extract(DOC, PROMPT)
    assert client.messages.create.call_count == 1
    [attempt] = budget.records
    assert attempt.outcome is AttemptOutcome.NO_RESPONSE


def test_count_tokens_failure_is_retried_without_touching_the_budget():
    client = mock_client(message())
    client.messages.count_tokens.side_effect = [connection_error(), tokens(1500)]
    budget = StubBudget()
    result = extractor(client, budget).extract(DOC, PROMPT)
    assert result.status is Status.OK
    assert budget.kinds == ["check", "record"]


def test_count_tokens_failing_every_time_is_raised():
    client = mock_client(message())
    client.messages.count_tokens.side_effect = [connection_error()] * 3
    budget = StubBudget()
    with pytest.raises(anthropic.APIConnectionError):
        extractor(client, budget).extract(DOC, PROMPT)
    client.messages.create.assert_not_called()
    assert budget.events == []


def test_count_tokens_non_retryable_error_is_raised_at_once():
    client = mock_client(message())
    client.messages.count_tokens.side_effect = [server_error(400), tokens(1500)]
    with pytest.raises(anthropic.BadRequestError):
        extractor(client).extract(DOC, PROMPT)
    assert client.messages.count_tokens.call_count == 1


def test_long_haiku_prompt_is_refused_before_paying():
    client = mock_client(message(), count=100_001)
    budget = StubBudget()
    with pytest.raises(PromptTooLongError) as info:
        extractor(client, budget).extract(DOC, PROMPT)
    assert info.value.cost is None
    client.messages.create.assert_not_called()
    assert budget.events == []


def test_long_sonnet_prompt_is_not_refused():
    client = mock_client(message(model=SONNET), count=150_000)
    result = extractor(client, model=SONNET).extract(DOC, PROMPT)
    assert result.status is Status.OK


@pytest.mark.parametrize(
    ("usage", "error"),
    [
        (
            {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 50},
            CacheWriteError,
        ),
        ({"input_tokens": 1000, "output_tokens": 200, "inference_geo": "us"}, PriceModifierError),
        # count_tokens said it fits, the billed usage says it does not.
        ({"input_tokens": 100_500, "output_tokens": 200}, PromptTooLongError),
    ],
)
def test_paid_pricing_error_is_recorded_once_then_raised(usage, error):
    client = mock_client(message(usage=usage), message())
    budget = StubBudget()
    with pytest.raises(error) as info:
        extractor(client, budget).extract(DOC, PROMPT)
    [attempt] = budget.records
    assert attempt.outcome is AttemptOutcome.PRICING_ERROR
    assert attempt.cost is info.value.cost
    assert attempt.cost_usd == info.value.cost.total_usd > 0
    assert attempt.error == str(info.value)
    assert attempt.estimate == CallEstimate(model=HAIKU, prompt_tokens=1500, max_tokens=4096)
    assert client.messages.create.call_count == 1


def test_unpriceable_usage_is_recorded_with_unknown_cost_then_raised():
    client = mock_client(message(usage={"input_tokens": -1, "output_tokens": 200}))
    budget = StubBudget()
    with pytest.raises(PricingError):
        extractor(client, budget).extract(DOC, PROMPT)
    [attempt] = budget.records
    assert attempt.outcome is AttemptOutcome.PRICING_ERROR
    assert attempt.cost is None
    assert attempt.cost_usd is None


# --- response blocks, retries, interrupts ------------------------------------


def test_raw_is_only_the_text_block_not_thinking():
    msg = message('{"citelne": true}')
    msg.content[0].thinking = "Přemýšlím o dokumentu."
    redacted = Message.model_validate(
        {**msg.model_dump(), "content": [{"type": "redacted_thinking", "data": "opaque"}]}
    ).content[0]
    msg.content.insert(1, redacted)
    assert [b.type for b in msg.content] == ["thinking", "redacted_thinking", "text"]
    result = extractor(mock_client(msg)).extract(DOC, PROMPT)
    assert result.raw == '{"citelne": true}'


def test_timeout_is_retried():
    estimate = CallEstimate(model=HAIKU, prompt_tokens=1500, max_tokens=4096)
    client = mock_client(timeout_error(), message())
    budget = StubBudget()
    result = extractor(client, budget).extract(DOC, PROMPT)
    assert budget.kinds == ["check", "record"] * 2
    first, second = budget.records
    assert first.outcome is AttemptOutcome.NO_RESPONSE
    assert "APITimeoutError" in first.error
    assert first.estimate == estimate
    assert second.outcome is AttemptOutcome.RESPONSE
    assert result.status is Status.OK
    assert result.attempts == 2


@pytest.mark.parametrize("status", [408, 409, 529])
def test_retryable_status_is_retried(status):
    client = mock_client(server_error(status), message())
    budget = StubBudget()
    result = extractor(client, budget).extract(DOC, PROMPT)
    assert budget.kinds == ["check", "record"] * 2
    assert str(status) in budget.records[0].error
    assert result.status is Status.OK
    assert result.attempts == 2


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), RuntimeError("boom")])
def test_any_exception_during_the_call_is_recorded_then_raised(exc):
    client = mock_client(exc, message())
    budget = StubBudget()
    with pytest.raises(type(exc)):
        extractor(client, budget).extract(DOC, PROMPT)
    assert budget.kinds == ["check", "record"]
    [attempt] = budget.records
    assert attempt.outcome is AttemptOutcome.NO_RESPONSE
    assert type(exc).__name__ in attempt.error
    assert client.messages.create.call_count == 1


class FailingRecordBudget(StubBudget):
    def record(self, attempt):
        super().record(attempt)
        raise OSError("ledger not writable")


class InterruptedRecordBudget(StubBudget):
    def record(self, attempt):
        super().record(attempt)
        raise KeyboardInterrupt


def _assert_unrecorded(error, attempt):
    text = "\n".join([str(error), *getattr(error, "__notes__", [])])
    assert f"cost_usd={attempt.cost_usd}" in text
    assert f"request_id={attempt.request_id}" in text


@pytest.mark.parametrize("exc", [KeyboardInterrupt(), connection_error()])
def test_failed_record_after_a_failed_call_is_a_ledger_write_error(exc):
    client = mock_client(exc, message())
    budget = FailingRecordBudget()
    with pytest.raises(LedgerWriteError) as info:
        extractor(client, budget).extract(DOC, PROMPT)
    # No retry once the ledger cannot be written.
    assert client.messages.create.call_count == 1
    [attempt] = budget.records
    error = info.value
    assert error.attempt is attempt
    assert attempt.outcome is AttemptOutcome.NO_RESPONSE
    assert attempt.estimate == CallEstimate(model=HAIKU, prompt_tokens=1500, max_tokens=4096)
    assert error.call_error is exc
    assert isinstance(error.__cause__, OSError)
    assert any(type(exc).__name__ in note for note in error.__notes__)
    _assert_unrecorded(error, attempt)
    assert not isinstance(error, (anthropic.APIError, PricingError))


def test_interrupt_inside_record_stays_an_interrupt():
    exc = connection_error()
    client = mock_client(exc, message())
    budget = InterruptedRecordBudget()
    with pytest.raises(KeyboardInterrupt) as info:
        extractor(client, budget).extract(DOC, PROMPT)
    assert client.messages.create.call_count == 1
    [attempt] = budget.records
    assert info.value.__cause__ is exc
    _assert_unrecorded(info.value, attempt)


def test_failed_record_after_a_response_is_a_ledger_write_error():
    msg = message()
    msg._request_id = "req_ok"
    client = mock_client(msg, message())
    budget = FailingRecordBudget()
    with pytest.raises(LedgerWriteError) as info:
        extractor(client, budget).extract(DOC, PROMPT)
    assert client.messages.create.call_count == 1
    [attempt] = budget.records
    assert info.value.attempt is attempt
    assert info.value.call_error is None
    assert attempt.cost_usd == Decimal("0.0002")
    assert attempt.request_id == "req_ok"
    _assert_unrecorded(info.value, attempt)


def test_interrupt_inside_record_after_a_response_stays_an_interrupt():
    msg = message()
    msg._request_id = "req_ok"
    budget = InterruptedRecordBudget()
    with pytest.raises(KeyboardInterrupt) as info:
        extractor(mock_client(msg), budget).extract(DOC, PROMPT)
    _assert_unrecorded(info.value, budget.records[0])


def test_failed_record_after_a_paid_pricing_error_is_a_ledger_write_error():
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 50}
    msg = message(usage=usage)
    msg._request_id = "req_paid"
    client = mock_client(msg, message())
    budget = FailingRecordBudget()
    with pytest.raises(LedgerWriteError) as info:
        extractor(client, budget).extract(DOC, PROMPT)
    assert client.messages.create.call_count == 1
    [attempt] = budget.records
    error = info.value
    assert error.attempt is attempt
    assert attempt.outcome is AttemptOutcome.PRICING_ERROR
    assert isinstance(error.call_error, CacheWriteError)
    assert attempt.cost is error.call_error.cost
    assert attempt.cost_usd > 0
    assert any("CacheWriteError" in note for note in error.__notes__)
    _assert_unrecorded(error, attempt)


def test_backoff_sleeps_between_attempts_only():
    sleeps = []
    client = mock_client(server_error(500), server_error(500), server_error(500))
    extractor(client, sleep=sleeps.append, max_attempts=3).extract(DOC, PROMPT)
    assert sleeps == [1.0, 2.0]


def test_request_id_is_carried_from_the_response():
    msg = message()
    msg._request_id = "req_test123"
    budget = StubBudget()
    result = extractor(mock_client(msg), budget).extract(DOC, PROMPT)
    assert result.request_id == "req_test123"
    assert budget.records[0].request_id == "req_test123"


@pytest.mark.parametrize("attempts", [0, -1])
def test_max_attempts_below_one_is_refused(attempts):
    with pytest.raises(ValueError, match="max_attempts"):
        extractor(mock_client(), max_attempts=attempts)


def test_every_attempt_carries_its_estimate():
    estimate = CallEstimate(model=HAIKU, prompt_tokens=1500, max_tokens=4096)
    client = mock_client(server_error(500), connection_error(), message())
    budget = StubBudget()
    extractor(client, budget).extract(DOC, PROMPT)
    assert [a.estimate for a in budget.records] == [estimate] * 3


# --- branches without a test so far ------------------------------------------------


@pytest.mark.parametrize("exc", [SystemExit(3), GeneratorExit()])
def test_system_exit_and_generator_exit_during_the_call_are_recorded_then_raised(exc):
    client = mock_client(exc, message())
    budget = StubBudget()
    with pytest.raises(type(exc)):
        extractor(client, budget).extract(DOC, PROMPT)
    [attempt] = budget.records
    assert attempt.outcome is AttemptOutcome.NO_RESPONSE
    assert type(exc).__name__ in attempt.error


class FailsOnSecondRecord(StubBudget):
    def record(self, attempt):
        super().record(attempt)
        if len(self.records) == 2:
            raise OSError("disk full")


def test_failed_record_after_an_earlier_recorded_failure_names_the_second_attempt():
    client = mock_client(timeout_error(), timeout_error(), message())
    budget = FailsOnSecondRecord()
    with pytest.raises(LedgerWriteError) as info:
        extractor(client, budget).extract(DOC, PROMPT)
    first, second = budget.records
    assert info.value.attempt is second
    assert client.messages.create.call_count == 2


def test_failed_record_of_a_paid_call_of_unknown_cost_says_cost_none():
    usage = {"input_tokens": -1, "output_tokens": 200}
    budget = FailingRecordBudget()
    with pytest.raises(LedgerWriteError) as info:
        extractor(mock_client(message(usage=usage)), budget).extract(DOC, PROMPT)
    [attempt] = budget.records
    assert attempt.outcome is AttemptOutcome.PRICING_ERROR
    assert attempt.cost_usd is None
    assert "cost_usd=None" in str(info.value)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupt_inside_record_on_the_pricing_error_path_stays_an_interrupt(interrupt):
    usage = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 50}

    class Interrupted(StubBudget):
        def record(self, attempt):
            super().record(attempt)
            raise interrupt

    budget = Interrupted()
    with pytest.raises(interrupt) as info:
        extractor(mock_client(message(usage=usage)), budget).extract(DOC, PROMPT)
    assert isinstance(info.value.__cause__, CacheWriteError)
    _assert_unrecorded(info.value, budget.records[0])


def test_backoff_is_capped_at_30_seconds():
    from llmeval.models.claude import _backoff

    assert [_backoff(n) for n in (1, 5, 6, 20)] == [1.0, 16.0, 30.0, 30.0]


def test_interrupt_during_count_tokens_records_nothing():
    client = mock_client(message())
    client.messages.count_tokens.side_effect = KeyboardInterrupt
    budget = StubBudget()
    with pytest.raises(KeyboardInterrupt):
        extractor(client, budget).extract(DOC, PROMPT)
    assert budget.events == []
    assert client.messages.create.call_count == 0


@pytest.mark.parametrize("error_type", ["uncertain", "lost"])
def test_call_error_is_noted_when_the_record_is_uncertain_or_lost(error_type):
    from llmeval.models.base import RecordUncertainError

    class Broken(StubBudget):
        def record(self, attempt):
            super().record(attempt)
            if error_type == "uncertain":
                raise RecordUncertainError("unknown whether")
            raise LedgerWriteError(attempt)

    exc = connection_error()
    with pytest.raises((RecordUncertainError, LedgerWriteError)) as info:
        extractor(mock_client(exc), Broken()).extract(DOC, PROMPT)
    assert any("call error: APIConnectionError" in n for n in info.value.__notes__)


def test_response_from_another_model_is_still_priced_as_the_requested_one():
    budget = StubBudget()
    result = extractor(mock_client(message(model=SONNET)), budget, model=HAIKU).extract(DOC, PROMPT)
    assert result.response_model == SONNET
    assert result.cost.model == HAIKU
    assert result.cost.total_usd == Decimal("0.0002")


def test_interrupt_of_uncertain_record_gets_no_unrecorded_note():
    class Uncertain(StubBudget):
        def record(self, attempt):
            super().record(attempt)
            exc = KeyboardInterrupt()
            exc.ledger_line_uncertain = True
            exc.attempt_id = "a" * 32
            exc.add_note("look for attempt_id " + "a" * 32)
            raise exc

    with pytest.raises(KeyboardInterrupt) as info:
        extractor(mock_client(message()), Uncertain()).extract(DOC, PROMPT)
    assert not any("unrecorded attempt" in n for n in info.value.__notes__)


@pytest.mark.parametrize("raised", [OSError("disk"), KeyboardInterrupt()])
def test_unrecorded_note_names_the_attempt_id_of_a_failed_append(raised):
    class AppendFailed(StubBudget):
        def record(self, attempt):
            super().record(attempt)
            raised.attempt_id = "b" * 32
            raise raised

    with pytest.raises((LedgerWriteError, KeyboardInterrupt)) as info:
        extractor(mock_client(message()), AppendFailed()).extract(DOC, PROMPT)
    text = "\n".join([str(info.value), *getattr(info.value, "__notes__", [])])
    assert "unrecorded attempt" in text and f"attempt_id={'b' * 32}" in text
