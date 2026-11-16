"""Canned model outputs for a run without the API (``--fake``); touches no budget.

``data/fake/outputs.json`` maps a case id (the document's folder name) to the raw text and
``stop_reason`` a model could have returned. Most are correct; some carry typical mistakes so
that a fake run exercises every failure path of grading and reporting. The output does not
depend on the prompt.
"""

import json
import time
from pathlib import Path

from llmeval.models.base import Result, Status

DEFAULT_FIXTURES = Path(__file__).resolve().parents[3] / "data" / "fake" / "outputs.json"

_STATUS = {"end_turn": Status.OK, "max_tokens": Status.MAX_TOKENS, "refusal": Status.REFUSAL}


class FakeExtractor:
    name = "fake"

    def __init__(self, fixtures: Path = DEFAULT_FIXTURES):
        self._outputs = json.loads(Path(fixtures).read_text(encoding="utf-8"))
        if not isinstance(self._outputs, dict):
            raise ValueError(f"{fixtures}: expected an object keyed by case id")
        for case_id, entry in self._outputs.items():
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("raw"), str)
                or entry.get("stop_reason") not in _STATUS
            ):
                raise ValueError(
                    f"{fixtures}: fake output for {case_id!r} needs a string 'raw' and "
                    f"a stop_reason out of {sorted(_STATUS)}, got {entry!r}"
                )

    def extract(self, document: Path, prompt: str) -> Result:
        start = time.perf_counter()
        case_id = Path(document).parent.name
        try:
            entry = self._outputs[case_id]
        except KeyError:
            raise KeyError(f"no fake output for case {case_id!r}") from None
        stop_reason = entry["stop_reason"]
        status = _STATUS[stop_reason]
        return Result(
            raw=entry["raw"],
            status=status,
            requested_model=self.name,
            response_model=self.name,
            stop_reason=stop_reason,
            latency_s=time.perf_counter() - start,
            reason="" if status is Status.OK else f"fake {stop_reason}",
        )
