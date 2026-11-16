"""What every extractor returns, and the spending interface the paid client needs.

``Budget`` is called by ``ClaudeExtractor`` around every ``messages.create`` attempt, in order:

1. ``check(CallEstimate)`` before the attempt; raising ``BudgetExceededError`` stops the
   extraction before anything is sent;
2. ``record(Attempt)`` exactly once after the attempt, whatever its outcome.

``count_tokens`` is free and is neither checked nor recorded.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from llmeval.pricing import CallCost, TokenCounts


class Status(StrEnum):
    OK = "ok"
    # The model hit max_tokens: an artefact of the setting, not a model mistake.
    MAX_TOKENS = "max_tokens"
    REFUSAL = "refusal"
    UNEXPECTED_STOP = "unexpected_stop"
    # Every attempt failed before a response arrived; raw is None.
    NO_RESPONSE = "no_response"


@dataclass(frozen=True)
class Result:
    # Model text as returned; None only for NO_RESPONSE. Never parsed or repaired here.
    raw: str | None
    status: Status
    requested_model: str
    response_model: str | None = None
    effort: str | None = None
    stop_reason: str | None = None
    latency_s: float | None = None
    tokens: TokenCounts | None = None
    cost: CallCost | None = None
    # Empty when status is OK.
    reason: str = ""
    attempts: int = 1
    prompt_tokens_estimate: int | None = None
    request_id: str | None = None


@runtime_checkable
class Extractor(Protocol):
    def extract(self, document: Path, prompt: str) -> Result: ...


class BudgetExceededError(Exception):
    """Raised by ``Budget.check`` when the next call could take spending over the cap."""


@dataclass(frozen=True)
class CallEstimate:
    """What the next call can cost at most: the prompt as counted plus ``max_tokens`` of output.

    Output tokens include thinking tokens, so ``max_tokens`` bounds both. ``prompt_tokens``
    comes from ``count_tokens``, which the API documents as an estimate.
    """

    model: str
    prompt_tokens: int
    max_tokens: int


class AttemptOutcome(StrEnum):
    # A response came back and was priced, whatever its stop_reason.
    RESPONSE = "response"
    # An exception before any response; recorded at cost 0 (whether it is billed is unverified).
    NO_RESPONSE = "no_response"
    # A response came back but pricing refused it; the error is raised after recording.
    PRICING_ERROR = "pricing_error"


@dataclass(frozen=True)
class Attempt:
    requested_model: str
    outcome: AttemptOutcome
    # None for NO_RESPONSE, and for PRICING_ERROR when the usage could not be priced at all.
    cost: CallCost | None
    error: str = ""
    response_model: str | None = None
    stop_reason: str | None = None
    latency_s: float | None = None
    request_id: str | None = None
    # The bound the budget was checked against; for NO_RESPONSE it is what the call may have
    # cost if it was billed after all.
    estimate: CallEstimate | None = None

    @property
    def cost_usd(self) -> Decimal | None:
        """USD to add to the ledger; None means a paid call of unknown cost."""
        if self.cost is not None:
            return self.cost.total_usd
        return Decimal(0) if self.outcome is AttemptOutcome.NO_RESPONSE else None


def unrecorded(attempt: Attempt, attempt_id: str | None = None) -> str:
    """One line with what a person needs to add the attempt to the ledger by hand."""
    text = (
        f"unrecorded attempt: model={attempt.requested_model} outcome={attempt.outcome} "
        f"cost_usd={attempt.cost_usd} request_id={attempt.request_id} "
        f"response_model={attempt.response_model} estimate={attempt.estimate}"
    )
    return text if attempt_id is None else f"{text} attempt_id={attempt_id}"


class LedgerWriteError(Exception):
    """``Budget.record`` failed: the attempt may have been paid but is not in the ledger."""

    def __init__(
        self,
        attempt: Attempt,
        call_error: BaseException | None = None,
        attempt_id: str | None = None,
    ):
        super().__init__(f"budget.record failed; {unrecorded(attempt, attempt_id)}")
        self.attempt = attempt
        # The exception of the call itself (API, pricing or interrupt), if there was one.
        self.call_error = call_error
        if call_error is not None:
            self.add_note(f"call error: {type(call_error).__name__}: {call_error}")


class RecordedStopError(Exception):
    """``Budget.record`` wrote the attempt to the ledger, then refused to let the run go on.

    Unlike ``LedgerWriteError`` nothing is missing from the ledger; it must not be added by hand.
    """


class RecordUncertainError(Exception):
    """``Budget.record`` appended the attempt but cannot tell whether exactly one copy of it is
    in the ledger now (it may be missing or duplicated); check the ledger before fixing it."""


@runtime_checkable
class Budget(Protocol):
    def check(self, estimate: CallEstimate) -> None: ...

    def record(self, attempt: Attempt) -> None: ...
