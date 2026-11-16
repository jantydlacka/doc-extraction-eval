"""Extraction through the Claude Messages API, with every attempt passed through a Budget.

Per ``extract``:

1. ``count_tokens`` with the same parameters as the real call (without ``max_tokens``);
2. ``pricing.check_prompt`` refuses a prompt priced on a higher rate card before paying;
3. per attempt: ``budget.check`` -> ``messages.create`` -> ``budget.record``.

Retries are ours, not the SDK's (the client must have ``max_retries=0``), so that each one
passes the budget. Retried: connection errors, timeouts, 408, 409, 429 and 5xx.

Contract for the runner: any exception out of ``extract`` stops the run. In particular:

- ``BudgetExceededError`` from ``budget.check``: nothing was sent;
- a ``PricingError`` after a response: recorded once with the cost it carries, because the
  price table no longer describes what is being billed;
- any other exception during the call, an interrupt included: recorded, then raised;
- ``LedgerWriteError``: ``budget.record`` failed, spend is missing from the ledger; the error
  carries the attempt (cost, request id) and the call's own exception. If ``record`` is
  interrupted instead, the interrupt itself is raised with that attempt in a note.

Never sent: ``thinking`` (Sonnet 5.5 rejects disabling it), ``temperature`` and other sampling
parameters (rejected), ``fallbacks`` (another model would answer and be priced as this one),
``inference_geo`` (surcharge) and ``cache_control`` (a cache write is treated as an error).
"""

import base64
import copy
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic

from llmeval.models.base import (
    Attempt,
    AttemptOutcome,
    Budget,
    CallEstimate,
    LedgerWriteError,
    RecordedStopError,
    RecordUncertainError,
    Result,
    Status,
    unrecorded,
)
from llmeval.pricing import PriceTable, PricingError, call_cost, check_prompt, load_prices
from llmeval.schema import JSON_SCHEMA

log = logging.getLogger(__name__)

DEFAULT_EFFORT = "medium"
# A rough guess that leaves room for thinking before the short JSON answer.
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MAX_ATTEMPTS = 3

_RETRYABLE_STATUS = {408, 409, 429}


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, anthropic.APIConnectionError):  # includes APITimeoutError
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS or exc.status_code >= 500
    return False


def _backoff(attempt: int) -> float:
    return min(2.0 ** (attempt - 1), 30.0)


class ClaudeExtractor:
    def __init__(
        self,
        model: str,
        budget: Budget,
        *,
        client: Any = None,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        prices: PriceTable | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not isinstance(budget, Budget):
            raise TypeError(f"ClaudeExtractor needs a Budget, got {budget!r}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.prices = prices or load_prices()
        self.prices.for_model(model)  # unknown model: fail before any call
        if client is None:
            client = anthropic.Anthropic(max_retries=0)
        elif client.max_retries != 0:
            raise ValueError(f"client must have max_retries=0, has {client.max_retries!r}")
        self.client = client
        self.model = model
        self.budget = budget
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self._sleep = sleep

    def _common(self, document: Path, prompt: str) -> dict[str, Any]:
        """Parameters shared by count_tokens and create; one place so they cannot drift."""
        data = base64.standard_b64encode(Path(document).read_bytes()).decode("ascii")
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": data,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "output_config": {
                "format": {"type": "json_schema", "schema": copy.deepcopy(JSON_SCHEMA)},
                "effort": self.effort,
            },
        }

    def request(self, document: Path, prompt: str) -> dict[str, Any]:
        """Keyword arguments of ``messages.create``."""
        return {**self._common(document, prompt), "max_tokens": self.max_tokens}

    def _count(self, params: dict[str, Any]) -> int:
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self.client.messages.count_tokens(**params).input_tokens
            except anthropic.APIError as exc:
                if not _retryable(exc) or attempt == self.max_attempts:
                    raise
                log.warning("count_tokens attempt %d failed (%s), retrying", attempt, exc)
                self._sleep(_backoff(attempt))
        raise AssertionError("unreachable")

    def estimate(self, document: Path, prompt: str) -> CallEstimate:
        """Bound of one ``extract`` attempt, counted with the parameters ``create`` gets.

        Refuses a prompt priced on a higher rate card. Free: neither checked nor recorded.
        """
        return self._estimate(self.request(document, prompt))

    def _estimate(self, params: dict[str, Any]) -> CallEstimate:
        counted = {k: v for k, v in params.items() if k != "max_tokens"}
        prompt_tokens = self._count(counted)
        check_prompt(self.model, prompt_tokens, self.prices)
        return CallEstimate(self.model, prompt_tokens, self.max_tokens)

    def extract(self, document: Path, prompt: str) -> Result:
        params = self.request(document, prompt)
        estimate = self._estimate(params)
        prompt_tokens = estimate.prompt_tokens

        last_error = ""
        for attempt in range(1, self.max_attempts + 1):
            self.budget.check(estimate)
            start = time.perf_counter()
            try:
                response = self.client.messages.create(**params)
            # BaseException: an interrupt while waiting may still leave a billed call behind.
            except BaseException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                failed = Attempt(
                    self.model,
                    AttemptOutcome.NO_RESPONSE,
                    None,
                    error=last_error,
                    latency_s=time.perf_counter() - start,
                    estimate=estimate,
                )
                self._record(failed, exc)
                if not _retryable(exc):
                    raise
                log.warning(
                    "%s attempt %d/%d failed: %s",
                    self.model,
                    attempt,
                    self.max_attempts,
                    last_error,
                )
                if attempt < self.max_attempts:
                    self._sleep(_backoff(attempt))
                continue
            latency = time.perf_counter() - start
            return self._result(response, latency, attempt, estimate)

        return Result(
            raw=None,
            status=Status.NO_RESPONSE,
            requested_model=self.model,
            effort=self.effort,
            reason=f"no response after {self.max_attempts} attempts; last: {last_error}",
            attempts=self.max_attempts,
            prompt_tokens_estimate=prompt_tokens,
        )

    def _record(self, attempt: Attempt, call_error: BaseException | None = None) -> None:
        try:
            self.budget.record(attempt)
        except RecordedStopError as stop:
            # Nothing is missing from the ledger, so an interrupt of the call stays the
            # primary fact (and the exit status of an interrupt).
            if call_error is not None and not isinstance(call_error, Exception):
                call_error.add_note(f"{type(stop).__name__}: {stop}")
                raise call_error from stop
            if call_error is not None:
                stop.add_note(f"call error: {type(call_error).__name__}: {call_error}")
            raise
        except (RecordUncertainError, LedgerWriteError) as error:
            # Already says what is (or may be) missing; wrapping would misstate it.
            if call_error is not None:
                error.add_note(f"call error: {type(call_error).__name__}: {call_error}")
            raise
        except Exception as record_exc:
            attempt_id = getattr(record_exc, "attempt_id", None)
            raise LedgerWriteError(attempt, call_error, attempt_id) from record_exc
        except BaseException as record_exc:
            # An interrupt stays an interrupt; the note keeps what the ledger is missing,
            # unless the budget says the line was written, or noted that it may be.
            if not (
                getattr(record_exc, "ledger_line_written", False)
                or getattr(record_exc, "ledger_line_uncertain", False)
            ):
                record_exc.add_note(unrecorded(attempt, getattr(record_exc, "attempt_id", None)))
            if call_error is not None:
                raise record_exc from call_error
            raise

    def _result(
        self, response: Any, latency: float, attempts: int, estimate: CallEstimate
    ) -> Result:
        request_id = getattr(response, "_request_id", None)
        try:
            cost = call_cost(self.model, response.usage, self.prices)
        except PricingError as exc:
            self._record(
                Attempt(
                    self.model,
                    AttemptOutcome.PRICING_ERROR,
                    getattr(exc, "cost", None),
                    error=str(exc),
                    response_model=response.model,
                    stop_reason=response.stop_reason,
                    latency_s=latency,
                    request_id=request_id,
                    estimate=estimate,
                ),
                exc,
            )
            raise
        self._record(
            Attempt(
                self.model,
                AttemptOutcome.RESPONSE,
                cost,
                response_model=response.model,
                stop_reason=response.stop_reason,
                latency_s=latency,
                request_id=request_id,
                estimate=estimate,
            )
        )
        if cost.tokens.thinking_tokens is not None:
            log.info("%s thinking_tokens=%d", self.model, cost.tokens.thinking_tokens)

        status, reason = self._status(response)
        return Result(
            raw="".join(block.text for block in response.content if block.type == "text"),
            status=status,
            requested_model=self.model,
            response_model=response.model,
            effort=self.effort,
            stop_reason=response.stop_reason,
            latency_s=latency,
            tokens=cost.tokens,
            cost=cost,
            reason=reason,
            attempts=attempts,
            prompt_tokens_estimate=estimate.prompt_tokens,
            request_id=request_id,
        )

    def _status(self, response: Any) -> tuple[Status, str]:
        stop = response.stop_reason
        if stop == "end_turn":
            return Status.OK, ""
        if stop == "max_tokens":
            return Status.MAX_TOKENS, f"output cut at max_tokens={self.max_tokens}"
        if stop == "refusal":
            details = response.stop_details
            if details is None:
                return Status.REFUSAL, "refused (no stop_details)"
            return Status.REFUSAL, f"refused: {details.category} ({details.explanation})"
        return Status.UNEXPECTED_STOP, f"unexpected stop_reason {stop!r}"
