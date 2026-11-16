from decimal import Decimal

import pytest

from llmeval.schema import (
    FIELDS,
    JSON_SCHEMA,
    canonical_amount,
    is_canonical_amount,
    validate,
    validate_expected,
)

GOOD = {
    "cislo_smlouvy": "SM-2026-000001",
    "isin": "US0378331005",
    "isin_platny": True,
    "castka": "1234.50",
    "mena": "CZK",
    "citelne": True,
}

# Keywords that structured outputs does not support (the SDK strips them silently).
UNSUPPORTED = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
}


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def test_json_schema_shape():
    assert JSON_SCHEMA["type"] == "object"
    assert JSON_SCHEMA["additionalProperties"] is False
    assert tuple(JSON_SCHEMA["properties"]) == FIELDS
    assert JSON_SCHEMA["required"] == list(FIELDS)
    assert JSON_SCHEMA["properties"]["citelne"]["type"] == "boolean"
    for name in FIELDS[:-1]:
        variants = JSON_SCHEMA["properties"][name]["anyOf"]
        assert {"type": "null"} in variants
        assert len(variants) == 2


def test_json_schema_has_no_unsupported_keywords():
    for node in _walk(JSON_SCHEMA):
        assert not UNSUPPORTED & node.keys(), node


def test_every_property_is_described():
    for name in FIELDS:
        assert JSON_SCHEMA["properties"][name]["description"]


def test_valid_output_passes():
    assert validate(GOOD) == []
    assert validate_expected(GOOD) == []


def test_all_null_readable_passes_schema():
    assert validate({**dict.fromkeys(FIELDS), "citelne": True}) == []


@pytest.mark.parametrize(
    "change, fragment",
    [
        ({"castka": 1234.5}, "castka"),
        ({"isin_platny": "true"}, "isin_platny"),
        ({"citelne": None}, "citelne"),
        ({"cislo_smlouvy": 5}, "cislo_smlouvy"),
        ({"navic": "x"}, "navic"),
    ],
)
def test_type_errors_are_reported(change, fragment):
    errors = validate({**GOOD, **change})
    assert errors and fragment in " ".join(errors)


def test_missing_key_is_reported():
    output = dict(GOOD)
    del output["mena"]
    assert any("mena" in e for e in validate(output))


def test_bool_is_not_accepted_where_string_expected():
    assert validate({**GOOD, "castka": True})


def test_non_object_is_reported():
    assert validate(["x"])
    assert validate(None)


def test_empty_string_is_schema_valid_but_not_a_valid_expected_value():
    # "" and null stay distinguishable: the schema allows "" from a model, expected never uses it.
    output = {**GOOD, "cislo_smlouvy": ""}
    assert validate(output) == []
    assert validate_expected(output)


@pytest.mark.parametrize(
    "castka", ["1234.5", "1 234,50", "1234,50", "1.234,50", "01234.50", "-5.00"]
)
def test_expected_amount_must_be_canonical(castka):
    assert validate_expected({**GOOD, "castka": castka})


@pytest.mark.parametrize("mena", ["czk", "Kč", "CZKK", ""])
def test_expected_currency_must_be_iso_code(mena):
    assert validate_expected({**GOOD, "mena": mena})


def test_expected_isin_must_be_canonical():
    assert validate_expected({**GOOD, "isin": "us0378331005"})
    assert validate_expected({**GOOD, "isin": "US 0378 3310 05"})


def test_expected_isin_validity_flag_must_match_check_digit():
    assert validate_expected({**GOOD, "isin_platny": False})
    bad_digit = {**GOOD, "isin": "US0378331006", "isin_platny": False}
    assert validate_expected(bad_digit) == []
    assert validate_expected({**bad_digit, "isin_platny": True})


def test_expected_isin_validity_flag_is_null_without_isin():
    assert validate_expected({**GOOD, "isin": None, "isin_platny": None}) == []
    assert validate_expected({**GOOD, "isin": None, "isin_platny": False})


def test_expected_amount_and_currency_go_together():
    assert validate_expected({**GOOD, "mena": None})
    assert validate_expected({**GOOD, "castka": None})
    assert validate_expected({**GOOD, "castka": None, "mena": None}) == []


def test_unreadable_expected_has_all_fields_null():
    unreadable = {**dict.fromkeys(FIELDS), "citelne": False}
    assert validate_expected(unreadable) == []
    for name in FIELDS[:-1]:
        changed = {**unreadable, name: GOOD[name]}
        assert validate_expected(changed), name


def test_validate_expected_includes_schema_errors():
    assert validate_expected({**GOOD, "navic": 1})


@pytest.mark.parametrize(
    "value, text",
    [
        (Decimal("1234.5"), "1234.50"),
        (Decimal("35000"), "35000.00"),
        (Decimal("0.07"), "0.07"),
        (Decimal("1250000.50"), "1250000.50"),
    ],
)
def test_canonical_amount(value, text):
    assert canonical_amount(value) == text
    assert is_canonical_amount(text)


@pytest.mark.parametrize("value", [Decimal("1.234"), Decimal("-1.00")])
def test_canonical_amount_refuses_rounding_and_negatives(value):
    with pytest.raises(ValueError):
        canonical_amount(value)


# The schema goes out with every prompt version; rules for tricky cases belong to the prompt,
# otherwise the simple prompt gets them too and the prompt comparison measures nothing.
RULE_PHRASES = [
    "oprav",
    "konečný zůstatek",
    "poplatk",
    "ISO 6166",
    "Kč",
    "pokynu",
    "účtu",
    "ostatní pole",
    "chybnou",
    "Smlouva o investici",
    "Hlavní",
]


@pytest.mark.parametrize("name", FIELDS)
def test_descriptions_carry_no_extraction_rules(name):
    description = JSON_SCHEMA["properties"][name]["description"]
    for phrase in RULE_PHRASES:
        assert phrase.lower() not in description.lower(), (name, phrase)


def test_sdk_schema_transform_leaves_schema_unchanged():
    # The SDK rewrites unsupported keywords into descriptions when it builds a schema itself;
    # a schema without them must come out unchanged.
    from anthropic import transform_schema

    assert transform_schema(JSON_SCHEMA) == JSON_SCHEMA
