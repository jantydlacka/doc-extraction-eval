import json
from pathlib import Path

import pytest

from llmeval.grade import Grade, Status, grade, grade_text
from llmeval.schema import FIELDS

CASES_DIR = Path(__file__).resolve().parent.parent / "data" / "cases"
EXPECTED = {
    p.parent.name: json.loads(p.read_text()) for p in sorted(CASES_DIR.glob("*/expected.json"))
}

BASE = {
    "cislo_smlouvy": "SM-2026-738629",
    "isin": "CZ9902257167",
    "isin_platny": True,
    "castka": "12345.00",
    "mena": "CZK",
    "citelne": True,
}
UNREADABLE = dict.fromkeys(FIELDS[:-1]) | {"citelne": False}


def with_(**changes) -> dict:
    return BASE | changes


def only_failure(result: Grade) -> tuple[str, Status]:
    failed = [(name, r.status) for name, r in result.fields.items() if r.status is not Status.OK]
    assert len(failed) == 1, failed
    return failed[0]


def test_dataset_is_present():
    assert len(EXPECTED) == 20


@pytest.mark.parametrize("case_id", sorted(EXPECTED))
def test_oracle_passes_every_case(case_id):
    expected = EXPECTED[case_id]
    result = grade(expected, dict(expected))
    assert result.passed, result.reason
    assert result.reason == ""
    assert list(result.fields) == list(FIELDS)
    assert all(r.status is Status.OK for r in result.fields.values())
    assert grade_text(expected, json.dumps(expected)).passed


@pytest.mark.parametrize("case_id", sorted(EXPECTED))
def test_all_null_fails_exactly_where_expected_has_a_value(case_id):
    expected = EXPECTED[case_id]
    all_null = dict.fromkeys(FIELDS[:-1]) | {"citelne": True}
    result = grade(expected, all_null)
    failing = {name for name, r in result.fields.items() if r.status is not Status.OK}
    should_fail = {name for name in FIELDS[:-1] if expected[name] is not None}
    if not expected["citelne"]:
        should_fail.add("citelne")
    assert failing == should_fail
    assert all(result.fields[name].status is Status.SPATNE for name in failing)
    assert result.passed is (not should_fail)


# --- wrong values: the output is well formed, the content is not ---------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cislo_smlouvy", "SM-2026-736829"),  # two digits swapped
        ("castka", "1234.50"),  # decimal point shifted against 12345.00
        ("castka", "12354.00"),  # two digits swapped
        ("isin", "CZ9945150692"),  # another ISIN, check digit valid
        ("isin_platny", False),
        ("mena", "EUR"),
        ("citelne", False),
    ],
)
def test_wrong_value_is_spatne(field, value):
    result = grade(BASE, with_(**{field: value}))
    assert only_failure(result) == (field, Status.SPATNE)
    assert not result.passed
    assert field in result.reason


def test_invalid_isin_reported_as_valid_is_spatne():
    expected = with_(isin="CZ9985170590", isin_platny=False)
    result = grade(expected, expected | {"isin_platny": True})
    assert only_failure(result) == ("isin_platny", Status.SPATNE)


def test_shifted_decimal_point_the_other_way():
    result = grade(with_(castka="1234.50"), with_(castka="12345.00"))
    assert only_failure(result) == ("castka", Status.SPATNE)


def test_contract_number_is_compared_without_normalisation():
    result = grade(BASE, with_(cislo_smlouvy=" SM-2026-738629"))
    assert only_failure(result) == ("cislo_smlouvy", Status.SPATNE)


def test_null_where_a_value_is_expected_is_spatne():
    result = grade(BASE, with_(castka=None))
    assert only_failure(result) == ("castka", Status.SPATNE)
    assert "null" in result.fields["castka"].reason


# --- format: the value is not in the form the schema asks for ------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("castka", "1.234,50"),
        ("castka", "12345.0"),
        ("castka", "12 345.00"),
        ("castka", "12345.00 CZK"),
        ("castka", "012345.00"),
        ("castka", 12345.0),  # number instead of string
        ("castka", ""),
        ("isin", "CZ99 0225 7167"),  # spaces are not removed
        ("isin", "cz9902257167"),  # case is not changed
        ("isin", "CZ990225716"),
        ("isin", ""),
        ("mena", "Kč"),
        ("mena", "czk"),
        ("mena", ""),
        ("cislo_smlouvy", ""),
        ("cislo_smlouvy", 738629),
        ("isin_platny", "true"),  # bool as string
        ("isin_platny", 1),
        ("citelne", "true"),
        ("citelne", None),  # citelne is not nullable
    ],
)
def test_malformed_value_is_format(field, value):
    result = grade(BASE, with_(**{field: value}))
    assert only_failure(result) == (field, Status.FORMAT)
    assert not result.passed


def test_small_amounts_compare_exactly():
    assert grade(with_(castka="0.10"), with_(castka="0.10")).passed
    assert grade(with_(castka="0.10"), with_(castka="0.01")).fields["castka"].status is (
        Status.SPATNE
    )


# --- empty string, null and missing key are three different results -----------------------


@pytest.mark.parametrize("expected", [BASE, with_(cislo_smlouvy=None)])
def test_empty_null_and_missing_key_differ(expected):
    output_null = with_(cislo_smlouvy=None)
    output_empty = with_(cislo_smlouvy="")
    output_missing = {k: v for k, v in BASE.items() if k != "cislo_smlouvy"}
    statuses = {
        grade(expected, out).fields["cislo_smlouvy"].status
        for out in (output_null, output_empty, output_missing)
    }
    assert len(statuses) == 3
    assert grade(expected, output_missing).fields["cislo_smlouvy"].status is Status.CHYBI_KLIC
    assert grade(expected, output_empty).fields["cislo_smlouvy"].status is Status.FORMAT


def test_missing_key_is_chybi_klic():
    output = {k: v for k, v in BASE.items() if k != "citelne"}
    result = grade(BASE, output)
    assert only_failure(result) == ("citelne", Status.CHYBI_KLIC)
    assert result.fields["citelne"].actual is None


# --- values where expected is null --------------------------------------------------------


def test_value_where_null_is_expected_is_navic_hodnota():
    expected = with_(isin=None, isin_platny=None)
    result = grade(expected, with_(isin_platny=None))
    assert only_failure(result) == ("isin", Status.NAVIC_HODNOTA)


def test_unreadable_scan_with_invented_contract_number_fails():
    output = UNREADABLE | {"cislo_smlouvy": "SM-2026-000001"}
    result = grade(UNREADABLE, output)
    assert only_failure(result) == ("cislo_smlouvy", Status.NAVIC_HODNOTA)
    assert not result.passed
    assert "SM-2026-000001" in result.reason


def test_unreadable_scan_with_everything_null_passes():
    assert grade(UNREADABLE, dict(UNREADABLE)).passed


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("castka", "12 345 Kč"),
        ("castka", 12345),
        ("isin", "cz99 0225 7167"),
        ("isin_platny", "true"),
        ("mena", "Kč"),
    ],
)
def test_malformed_value_where_null_is_expected_is_navic_hodnota(field, value):
    result = grade(UNREADABLE, UNREADABLE | {field: value})
    assert only_failure(result) == (field, Status.NAVIC_HODNOTA)
    reason = result.fields[field].reason
    assert "expected null" in reason
    assert "instead of" in reason or "is not" in reason


def test_empty_string_where_null_is_expected_stays_format():
    result = grade(UNREADABLE, UNREADABLE | {"castka": ""})
    assert only_failure(result) == ("castka", Status.FORMAT)


def test_isin_platny_without_expected_isin_is_navic_hodnota():
    expected = with_(isin=None, isin_platny=None)
    result = grade(expected, with_(isin=None, isin_platny=True))
    assert only_failure(result) == ("isin_platny", Status.NAVIC_HODNOTA)


def test_currency_without_expected_amount_is_navic_hodnota():
    expected = with_(castka=None, mena=None)
    result = grade(expected, with_(castka=None, mena="CZK"))
    assert only_failure(result) == ("mena", Status.NAVIC_HODNOTA)


# --- whole-output problems ----------------------------------------------------------------


def test_extra_key_fails_the_case_but_keeps_field_results():
    result = grade(BASE, with_(poznamka="x"))
    assert all(r.status is Status.OK for r in result.fields.values())
    assert result.extra_keys == ("poznamka",)
    assert not result.passed
    assert "poznamka" in result.reason


def test_reason_has_no_raw_newline():
    result = grade(BASE, with_(castka="1\n2", **{"x\ny": 1}))
    assert not result.passed
    assert "\n" not in result.reason


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "{",
        "not json",
        '{"castka": "1.00",}',
        '{"castka": 1' + "0" * 5000 + "}",  # over the int digit limit: ValueError
        '{"a":' * 100000,  # too deep: RecursionError
    ],
    ids=["empty", "open-brace", "text", "trailing-comma", "huge-int", "deep-nesting"],
)
def test_invalid_json_fails_with_reason(raw):
    result = grade_text(BASE, raw)
    assert not result.passed
    assert result.fields == {}
    assert result.reason.startswith("invalid JSON")
    assert "\n" not in result.reason


@pytest.mark.parametrize("raw", ["[]", "null", '"text"', "42"])
def test_json_that_is_not_an_object_fails_with_reason(raw):
    result = grade_text(BASE, raw)
    assert not result.passed
    assert result.fields == {}
    assert "not an object" in result.reason


def test_grade_text_grades_fields():
    raw = json.dumps(with_(castka="1.234,50"))
    assert only_failure(grade_text(BASE, raw)) == ("castka", Status.FORMAT)


def test_reason_lists_every_failed_field():
    result = grade(BASE, with_(castka="1234.50", mena="Kč"))
    assert "castka" in result.reason
    assert "mena" in result.reason
    assert "isin" not in result.reason


def test_field_result_keeps_raw_and_expected_values():
    r = grade(BASE, with_(castka="1.234,50")).fields["castka"]
    assert (r.expected, r.actual) == ("12345.00", "1.234,50")
    assert r.reason


def test_invalid_expected_is_rejected():
    with pytest.raises(ValueError, match="castka"):
        grade(with_(castka="1 234,50"), BASE)


def test_status_values_are_plain_strings_for_csv():
    assert [s.value for s in Status] == ["ok", "spatne", "format", "chybi_klic", "navic_hodnota"]


def test_invalid_expected_is_rejected_even_with_invalid_json():
    with pytest.raises(ValueError, match="castka"):
        grade_text(with_(castka="1 234,50"), "{")


def test_contract_number_of_the_wrong_type_where_null_is_expected_notes_the_form():
    result = grade(with_(cislo_smlouvy=None), with_(cislo_smlouvy=738629))
    assert only_failure(result) == ("cislo_smlouvy", Status.NAVIC_HODNOTA)
    assert (
        result.fields["cislo_smlouvy"].reason == "738629, expected null, also int instead of string"
    )


@pytest.mark.parametrize("raw", [None, 42])
def test_raw_output_that_is_not_text_is_a_type_error(raw):
    # The runner never passes such a value: a result without text is graded without parsing.
    with pytest.raises(TypeError):
        grade_text(BASE, raw)


def test_duplicate_key_in_the_output_is_invalid_json():
    # json.loads would silently keep the last value, which may be the right one.
    raw = json.dumps(BASE)[:-1] + ', "isin": "CZ9902257167"}'
    result = grade_text(BASE, '{"isin": "XX0000000000", ' + raw[1:])
    assert not result.passed
    assert result.fields == {}
    assert result.reason == "invalid JSON: duplicate key 'isin'"


def test_duplicate_key_in_a_nested_object_is_invalid_json():
    raw = json.dumps(BASE | {"extra": {"a": 1}}).replace('{"a": 1}', '{"a": 1, "a": 2}')
    assert grade_text(BASE, raw).reason == "invalid JSON: duplicate key 'a'"


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_are_not_json(constant):
    raw = json.dumps(with_(castka=None)).replace("null", constant, 1)
    result = grade_text(BASE, raw)
    assert result.reason == f"invalid JSON: {constant} is not JSON"
