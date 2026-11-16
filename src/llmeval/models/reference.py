"""Baselines that bracket the score: Oracle is the ceiling, AllNull the floor."""

import json
import time
from pathlib import Path

from llmeval.models.base import Result, Status
from llmeval.schema import FIELDS


def _result(name: str, raw: str, start: float) -> Result:
    return Result(
        raw=raw,
        status=Status.OK,
        requested_model=name,
        response_model=name,
        stop_reason="end_turn",
        latency_s=time.perf_counter() - start,
    )


class OracleExtractor:
    """Returns the case's ``expected.json`` (next to the document) as the model output."""

    name = "oracle"

    def extract(self, document: Path, prompt: str) -> Result:
        start = time.perf_counter()
        expected = json.loads((Path(document).parent / "expected.json").read_text("utf-8"))
        return _result(self.name, json.dumps(expected, ensure_ascii=False), start)


class AllNullExtractor:
    """Says every document is readable and has none of the fields."""

    name = "all-null"

    def extract(self, document: Path, prompt: str) -> Result:
        start = time.perf_counter()
        output = {**dict.fromkeys(FIELDS), "citelne": True}
        return _result(self.name, json.dumps(output), start)
