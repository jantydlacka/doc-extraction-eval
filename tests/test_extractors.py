"""Contract shared by every extractor, and the extractors that need no API."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import anthropic
import httpx2
import pytest
from anthropic.types import Message, MessageTokensCount

from llmeval.grade import grade_text
from llmeval.models.base import Extractor, Result, Status
from llmeval.models.claude import ClaudeExtractor
from llmeval.models.fake import FakeExtractor
from llmeval.models.reference import AllNullExtractor, OracleExtractor
from llmeval.schema import FIELDS, validate

CASES_DIR = Path(__file__).resolve().parent.parent / "data" / "cases"
CASES = sorted(p.parent for p in CASES_DIR.glob("*/expected.json"))
PROMPT = "Vytěž údaje."


def _expected(case: Path) -> dict:
    return json.loads((case / "expected.json").read_text(encoding="utf-8"))


class _NoBudget:
    def check(self, estimate):
        pass

    def record(self, attempt):
        pass


def _claude() -> ClaudeExtractor:
    client = MagicMock()
    client.max_retries = 0
    client.messages.count_tokens.return_value = MessageTokensCount.model_validate(
        {"input_tokens": 1200}
    )
    client.messages.create.return_value = Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": '{"citelne": true}'}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 30},
        }
    )
    return ClaudeExtractor("claude-haiku-5-5", _NoBudget(), client=client, sleep=lambda s: None)


def _claude_no_response() -> ClaudeExtractor:
    extractor = _claude()
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    extractor.client.messages.create.side_effect = anthropic.APIConnectionError(request=request)
    extractor.max_attempts = 1
    return extractor


EXTRACTORS = {
    "claude": _claude,
    "claude_no_response": _claude_no_response,
    "fake": FakeExtractor,
    "oracle": OracleExtractor,
    "all_null": AllNullExtractor,
}


@pytest.mark.parametrize("make", EXTRACTORS.values(), ids=EXTRACTORS.keys())
@pytest.mark.parametrize("case", [CASES[0], CASES[-1]], ids=lambda p: p.name)
def test_result_shape(make, case):
    extractor = make()
    assert isinstance(extractor, Extractor)
    result = extractor.extract(case / "document.pdf", PROMPT)
    assert isinstance(result, Result)
    assert isinstance(result.status, Status)
    assert isinstance(result.reason, str)
    assert isinstance(result.raw, str) or (
        result.raw is None and result.status is Status.NO_RESPONSE
    )
    assert result.status is not Status.OK or result.reason == ""
    assert isinstance(result.requested_model, str) and result.requested_model
    assert result.response_model is None or isinstance(result.response_model, str)
    assert result.effort is None or isinstance(result.effort, str)
    assert result.stop_reason is None or isinstance(result.stop_reason, str)
    assert result.latency_s is None or result.latency_s >= 0
    assert (result.tokens is None) == (result.cost is None)
    assert result.attempts >= 1


def test_oracle_grades_every_case_as_passed():
    oracle = OracleExtractor()
    for case in CASES:
        result = oracle.extract(case / "document.pdf", PROMPT)
        grade = grade_text(_expected(case), result.raw)
        assert grade.passed, (case.name, grade.reason)
    assert len(CASES) == 20


def test_all_null_passes_the_schema_and_fails_most_cases():
    extractor = AllNullExtractor()
    passed = []
    for case in CASES:
        result = extractor.extract(case / "document.pdf", PROMPT)
        output = json.loads(result.raw)
        assert validate(output) == []
        assert output == {**dict.fromkeys(FIELDS), "citelne": True}
        if grade_text(_expected(case), result.raw).passed:
            passed.append(case.name)
    assert passed == []


# Which cases the committed fixtures fail, and on what; the --fake run relies on this mix.
FAKE_FAILURES = {
    "c06": "isin: format",
    "c07": "isin_platny: spatne",
    "c09": "castka: format",
    "c11": "castka: spatne",
    "c13": "invalid JSON",
    "c14": "cislo_smlouvy: navic_hodnota",
    "c18": "invalid JSON",
    "c20": "cislo_smlouvy: navic_hodnota",
}


def test_fake_gives_a_known_mix_of_passes_and_failures():
    fake = FakeExtractor()
    failures = {}
    statuses = {}
    for case in CASES:
        result = fake.extract(case / "document.pdf", PROMPT)
        statuses[case.name] = result.status
        grade = grade_text(_expected(case), result.raw)
        if not grade.passed:
            failures[case.name] = grade.reason
    assert failures.keys() == FAKE_FAILURES.keys()
    for case, reason in FAKE_FAILURES.items():
        assert reason in failures[case], (case, failures[case])
    assert statuses.pop("c13") is Status.MAX_TOKENS
    assert statuses.pop("c18") is Status.REFUSAL
    assert set(statuses.values()) == {Status.OK}


def test_fake_refuses_an_unknown_case(tmp_path):
    with pytest.raises(KeyError, match="c99"):
        FakeExtractor().extract(tmp_path / "c99" / "document.pdf", PROMPT)


def test_contract_includes_a_no_response_result():
    result = _claude_no_response().extract(CASES[0] / "document.pdf", PROMPT)
    assert result.status is Status.NO_RESPONSE


@pytest.mark.parametrize(
    "entry",
    [
        {"raw": "{}", "stop_reason": "pause_turn"},
        {"raw": "{}"},
        {"stop_reason": "end_turn"},
        "{}",
        ["{}", "end_turn"],
        None,
    ],
)
def test_fake_refuses_a_malformed_fixture(tmp_path, entry):
    fixtures = tmp_path / "outputs.json"
    fixtures.write_text(json.dumps({"c01": entry}), encoding="utf-8")
    with pytest.raises(ValueError, match="c01"):
        FakeExtractor(fixtures)


@pytest.mark.parametrize("top", [[], "x", None, 1])
def test_fake_refuses_fixtures_that_are_not_a_mapping(tmp_path, top):
    fixtures = tmp_path / "outputs.json"
    fixtures.write_text(json.dumps(top), encoding="utf-8")
    with pytest.raises(ValueError, match="outputs.json"):
        FakeExtractor(fixtures)


def test_fake_fixture_covers_every_case():
    from llmeval.models.fake import DEFAULT_FIXTURES

    outputs = json.loads(DEFAULT_FIXTURES.read_text(encoding="utf-8"))
    assert sorted(outputs) == [case.name for case in CASES]
