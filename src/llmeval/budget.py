"""Spending cap backed by an append-only JSONL ledger, fail-closed.

One line per API attempt, written (and fsynced) right after the attempt. ``cost_usd`` is a
decimal string; the ledger total is the exact sum of it over every line. A missing, unreadable
or malformed ledger stops a real run: a new one is only made by ``init_ledger``, which refuses
to overwrite, so losing the record of past spending takes a deliberate manual delete.

What counts toward the cap is not just the sum of ``cost_usd``: a line with ``"billed":
"unknown"`` (an attempt that got no response and may still have been billed) counts at its
``bound_usd``. After checking the real usage, a person resolves such a line by setting
``billed`` to ``"no"`` (with ``cost_usd`` 0) or to ``"yes"`` with the real ``cost_usd``; a
line whose billing cannot be counted stops the run like any other corrupt line.

Besides the cap on the whole ledger, a run may add at most ``run_limit`` (recorded spend plus
the bounds of its possibly billed attempts plus the bound of the next call).

``check`` re-reads the ledger before every attempt, so a deleted ledger or spending recorded
by another run is noticed before the next call. A total below what the ledger held at the start
plus what this run wrote means the ledger was truncated or edited, which also stops the run.

A ``LedgerBudget`` holds an exclusive ``flock`` on the ledger file itself until ``close``: two
runs checking the same ledger at once could each pass the cap on their own. The ledger rather
than a sibling lock file, because the ledger must exist anyway and a lock file could be deleted
on its own while the ledger it guards stays in use.

Lines are split on ``\n`` only and written ASCII-escaped: ``str.splitlines`` also splits on
U+2028, U+2029 and U+0085, which may appear in error text.
"""

import fcntl
import json
import math
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from llmeval.models.base import (
    Attempt,
    AttemptOutcome,
    BudgetExceededError,
    CallEstimate,
    LedgerWriteError,
    RecordedStopError,
    RecordUncertainError,
)
from llmeval.pricing import PriceTable

SPEND_CAP_USD = Decimal("5")
DEFAULT_RUN_LIMIT_USD = Decimal("1")
# count_tokens is documented as an estimate; the billed prompt may differ from it.
PROMPT_MARGIN = Decimal("0.10")
PROMPT_MARGIN_TOKENS = 100

_MTOK = Decimal(1_000_000)


class LedgerError(Exception):
    pass


class LedgerChangedError(LedgerError):
    """Someone else truncated, edited or replaced the ledger during the run."""


class LedgerChangedAfterRecordError(LedgerChangedError, RecordedStopError):
    pass


class LedgerUncertainError(LedgerChangedError, RecordUncertainError):
    pass


def default_ledger_path() -> Path:
    return Path.home() / ".local" / "share" / "llmeval" / "spend.jsonl"


def init_ledger(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8"):
            pass
    except FileExistsError:
        raise LedgerError(
            f"{path} exists; a ledger is never overwritten (delete it by hand to start over)"
        ) from None


def _decimal(entry: dict, key: str, where: str) -> Decimal:
    value = entry.get(key)
    if not isinstance(value, str):
        raise LedgerError(f"{where}: {key} must be a decimal string, got {value!r}")
    try:
        amount = Decimal(value)
    except InvalidOperation:
        raise LedgerError(f"{where}: {key} is not a number: {value!r}") from None
    if not amount.is_finite() or amount < 0:
        raise LedgerError(f"{where}: {key} must be finite and >= 0, got {value!r}")
    return amount


_RESOLVE = 'set billed to "no" (cost_usd "0") or to "yes" with the real cost_usd'


def _counted(entry: Any, where: str) -> Decimal:
    """What the line counts toward the cap; raises when that cannot be told."""
    if not isinstance(entry, dict):
        raise LedgerError(f"{where}: not a JSON object")
    cost = _decimal(entry, "cost_usd", where)
    billed = entry.get("billed", "yes")  # lines without the key: cost_usd as written
    if billed == "yes":
        if cost == 0 and entry.get("outcome") == "no_response":
            stated = 'billed "yes"' if "billed" in entry else "no billed key"
            raise LedgerError(
                f"{where}: {stated} and cost_usd 0 on an attempt without a response; {_RESOLVE}"
            )
        return cost
    if billed == "no":
        if cost != 0:
            raise LedgerError(f'{where}: billed "no" with cost_usd {cost}; {_RESOLVE}')
        return cost
    if billed == "unknown":
        try:
            bound = _decimal(entry, "bound_usd", where)
        except LedgerError as exc:
            raise LedgerError(
                f'{exc}; a line with billed "unknown" counts at its bound_usd, so without one '
                f"it cannot be counted: {_RESOLVE}"
            ) from None
        if bound == 0:
            # The tool never writes a zero bound; with one, "unknown" would count nothing.
            raise LedgerError(f'{where}: billed "unknown" with bound_usd 0; {_RESOLVE}')
        return max(cost, bound)
    raise LedgerError(f'{where}: billed must be "yes", "no" or "unknown", got {billed!r}')


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    keys = [key for key, _ in pairs]
    for key in keys:
        if keys.count(key) > 1:
            # json.loads would keep the last value; an edit could hide a cost that way.
            raise ValueError(f"duplicate key {key!r}")
    return dict(pairs)


def read_ledger(path: Path) -> list[dict]:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LedgerError(
            f"ledger {path} does not exist; create it with --init-ledger "
            "(only if no spending record should be kept)"
        ) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise LedgerError(f"ledger {path} cannot be read: {exc}") from exc
    if text and not text.endswith("\n"):
        raise LedgerError(f"ledger {path}: last line has no newline (interrupted write?)")
    entries = []
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        where = f"ledger {path} line {number}"
        try:
            entry = json.loads(line, object_pairs_hook=_unique_keys)
        except ValueError as exc:
            raise LedgerError(f"{where}: invalid JSON ({exc})") from None
        _counted(entry, where)
        entries.append(entry)
    return entries


@dataclass(frozen=True)
class LedgerTotals:
    # What the cap is checked against: cost_usd, and bound_usd for billed "unknown".
    counted: Decimal
    cost: Decimal
    unresolved: int


def ledger_totals(path: Path) -> LedgerTotals:
    entries = read_ledger(path)
    counted = sum((_counted(e, "") for e in entries), Decimal(0))
    cost = sum((Decimal(e["cost_usd"]) for e in entries), Decimal(0))
    unresolved = sum(e.get("billed") == "unknown" for e in entries)
    return LedgerTotals(counted, cost, unresolved)


def ledger_total(path: Path) -> Decimal:
    """The sum the cap is checked against."""
    return ledger_totals(path).counted


def prompt_with_margin(prompt_tokens: int) -> int:
    """Most prompt tokens a call counted at ``prompt_tokens`` is assumed to be billed for."""
    return math.ceil(prompt_tokens * (1 + PROMPT_MARGIN)) + PROMPT_MARGIN_TOKENS


def upper_bound(estimate: CallEstimate, prices: PriceTable) -> Decimal:
    """Most one call can cost: the counted prompt with a margin plus ``max_tokens`` of output.

    When the prompt with its margin could cross a model's prompt limit, the dearer rate card
    prices both input and output.
    """
    pricing = prices.for_model(estimate.model)
    prompt = prompt_with_margin(estimate.prompt_tokens)
    card = pricing.prices
    over = pricing.over_limit
    if pricing.prompt_limit is not None and over is not None and prompt > pricing.prompt_limit:
        card = over
    # Cache reads are cheaper than input, so pricing the whole prompt as input bounds them.
    return (prompt * card.input + estimate.max_tokens * card.output) / _MTOK


class LedgerBudget:
    def __init__(
        self,
        path: Path,
        prices: PriceTable,
        *,
        cap: Decimal = SPEND_CAP_USD,
        run_limit: Decimal = DEFAULT_RUN_LIMIT_USD,
        run_id: str = "",
    ):
        self.path = Path(path)
        self.prices = prices
        self.cap = cap
        self.run_limit = run_limit
        self.run_id = run_id
        # cost_usd this run has written to the ledger.
        self.recorded_usd = Decimal(0)
        # Bounds of attempts recorded at 0 that may have been billed anyway; they count toward
        # the cap and the run limit as well.
        self.possible_unrecorded_usd = Decimal(0)
        self._unknown_cost = False
        ledger_total(self.path)  # missing or corrupt: refuse before taking the lock
        try:
            self._lock_fd: int | None = os.open(self.path, os.O_RDONLY)
        except OSError as exc:
            raise LedgerError(f"ledger {self.path} cannot be opened: {exc}") from exc
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.close()
            raise LedgerError(f"ledger {self.path} is in use by another run") from None
        try:
            # Read again under the lock: another run may have written in between.
            self.start_total = ledger_total(self.path)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release the ledger lock; the budget must not be used afterwards."""
        if self._lock_fd is not None:
            os.close(self._lock_fd)  # closing the descriptor releases the flock
            self._lock_fd = None

    def __enter__(self) -> "LedgerBudget":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def run_counted_usd(self) -> Decimal:
        return self.recorded_usd + self.possible_unrecorded_usd

    def total(self) -> Decimal:
        """The ledger sum counted toward the cap, after checking the ledger was not changed."""
        return self.totals().counted

    def totals(self) -> LedgerTotals:
        if self._lock_fd is None:
            raise LedgerError(f"budget for {self.path} is closed")
        try:
            current = os.stat(self.path)
        except FileNotFoundError:
            read_ledger(self.path)  # raises with the --init-ledger hint
            raise
        except OSError as exc:
            raise LedgerError(f"ledger {self.path} cannot be read: {exc}") from exc
        locked = os.fstat(self._lock_fd)
        # A rename (sed -i, editors, cp+mv) leaves the lock on the old file, and another run
        # could then lock the new one.
        if (current.st_dev, current.st_ino) != (locked.st_dev, locked.st_ino):
            raise LedgerChangedError(f"ledger {self.path} was replaced during the run")
        totals = ledger_totals(self.path)
        expected = self.start_total + self.run_counted_usd
        if totals.counted < expected:
            raise LedgerChangedError(
                f"ledger {self.path} shrank during the run: counted {totals.counted} USD, "
                f"expected at least {expected} (start {self.start_total} + this run "
                f"{self.run_counted_usd})"
            )
        return totals

    def check(self, estimate: CallEstimate) -> None:
        if self._unknown_cost:
            raise BudgetExceededError(
                "an earlier call of unknown cost is in the ledger at its upper bound; "
                "no further calls in this run"
            )
        spent = self.total()
        bound = upper_bound(estimate, self.prices)
        call = (
            f"next call up to {bound} ({estimate.model}, {estimate.prompt_tokens} prompt tokens, "
            f"max_tokens {estimate.max_tokens})"
        )
        if spent + bound > self.cap:
            raise BudgetExceededError(
                f"spending cap {self.cap} USD: ledger {spent} (counted toward the cap) + {call}"
            )
        if self.run_counted_usd + bound > self.run_limit:
            raise BudgetExceededError(
                f"run limit {self.run_limit} USD: this run recorded {self.recorded_usd} + "
                f"possibly billed {self.possible_unrecorded_usd} + {call}"
            )

    def record(self, attempt: Attempt) -> None:
        bound = None if attempt.estimate is None else upper_bound(attempt.estimate, self.prices)
        cost = attempt.cost_usd
        cost_known = cost is not None
        if cost is None:
            if bound is None:
                raise LedgerError(f"paid call of unknown cost and no bound: {attempt}")
            cost = bound
        attempt_id = uuid.uuid4().hex
        entry = {
            # Lets record find its own line again after the ledger changed under it.
            "attempt_id": attempt_id,
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "model": attempt.requested_model,
            "outcome": str(attempt.outcome),
            "cost_usd": str(cost),
            "cost_known": cost_known,
            # Whether an attempt that got no response is billed is not known.
            "billed": "unknown" if attempt.outcome is AttemptOutcome.NO_RESPONSE else "yes",
            "bound_usd": None if bound is None else str(bound),
            "request_id": attempt.request_id,
            "response_model": attempt.response_model,
            "stop_reason": attempt.stop_reason,
            "latency_s": attempt.latency_s,
            "error": attempt.error,
        }
        if attempt.cost is not None:
            tokens = attempt.cost.tokens
            entry |= {
                "input_tokens": tokens.input_tokens,
                "cache_read_input_tokens": tokens.cache_read_input_tokens,
                "cache_creation_input_tokens": tokens.cache_creation_input_tokens,
                "output_tokens": tokens.output_tokens,
                "thinking_tokens": tokens.thinking_tokens,
            }
        line = json.dumps(entry)
        try:
            self._append(line + "\n")
        except BaseException as exc:
            exc.attempt_id = attempt_id  # type: ignore[attr-defined]
            if not isinstance(exc, Exception):
                # The write may have finished before the interrupt.
                exc.add_note(
                    f"interrupted while appending to the ledger {self.path}: the line may or may "
                    f"not be there; look for attempt_id {attempt_id} before adding it by hand: "
                    f"{line}"
                )
                exc.ledger_line_uncertain = True  # type: ignore[attr-defined]
            raise
        self.recorded_usd += cost
        if not cost_known:
            self._unknown_cost = True
        possible = Decimal(0)
        if attempt.outcome is AttemptOutcome.NO_RESPONSE and bound is not None:
            possible = bound
            self.possible_unrecorded_usd += bound
        try:
            self.total()
        except Exception as exc:
            try:
                self._raise_after_append(attempt, attempt_id, line, cost, possible, exc)
            except Exception:
                raise
            except BaseException as interrupt:
                interrupt.add_note(
                    f"the ledger {self.path} changed after the attempt was appended ({exc}) and "
                    f"the interrupt came while checking for its line; look for attempt_id "
                    f"{attempt_id} before adding it by hand: {line}"
                )
                interrupt.attempt_id = attempt_id  # type: ignore[attr-defined]
                interrupt.ledger_line_uncertain = True  # type: ignore[attr-defined]
                raise
        except BaseException as exc:
            exc.add_note(
                f"the attempt was appended to the ledger before the interrupt "
                f"({self.path}, attempt_id {attempt_id})"
            )
            exc.ledger_line_written = True  # type: ignore[attr-defined]
            exc.attempt_id = attempt_id  # type: ignore[attr-defined]
            raise

    def _raise_after_append(
        self,
        attempt: Attempt,
        attempt_id: str,
        line: str,
        cost: Decimal,
        possible: Decimal,
        exc: Exception,
    ) -> None:
        """The ledger changed around our append: say only what the ledger itself shows."""
        try:
            copies = sum(e.get("attempt_id") == attempt_id for e in read_ledger(self.path))
        except LedgerError as read_exc:
            copies = None
            exc.add_note(f"re-read failed: {read_exc}")
        if copies == 1:
            raise LedgerChangedAfterRecordError(
                f"attempt written to the ledger {self.path} (attempt_id {attempt_id}, request_id "
                f"{attempt.request_id}); do not add it by hand. The run stops because: {exc}"
            ) from exc
        if copies == 0:
            self.recorded_usd -= cost
            self.possible_unrecorded_usd -= possible
            error = LedgerWriteError(attempt)
            error.add_note(
                f"the ledger changed after the attempt was appended and its line is gone "
                f"({exc}); add this line to {self.path} by hand: {line}"
            )
            raise error from exc
        found = "could not be checked" if copies is None else f"is there {copies} times"
        raise LedgerUncertainError(
            f"unknown whether the attempt is in the ledger {self.path}: its line {found} after "
            f"the ledger changed ({exc}); look for attempt_id {attempt_id} before fixing it by "
            f"hand. The line: {line}"
        ) from exc

    def _append(self, line: str) -> None:
        # No O_CREAT: a ledger deleted mid-run must not be silently recreated from zero.
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        except OSError as exc:
            raise LedgerError(f"ledger {self.path} cannot be opened for append: {exc}") from exc
        try:
            data = line.encode("utf-8")
            if os.write(fd, data) != len(data):
                raise LedgerError(f"ledger {self.path}: short write")
            os.fsync(fd)
        finally:
            os.close(fd)
