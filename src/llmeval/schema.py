"""Extraction output schema shared by the dataset, the grader and the model client.

Expected values (``expected.json``) obey stricter rules than a model output has to, see
:func:`validate_expected`: no ``""``, canonical amount/ISIN/currency, and consistent fields.
"""

import re
from decimal import Decimal

from llmeval.dataset.isin import is_valid_isin

FIELDS = ("cislo_smlouvy", "isin", "isin_platny", "castka", "mena", "citelne")

# Field meaning and output format only; rules for tricky documents belong in the prompt.
_DESCRIPTIONS = {
    "cislo_smlouvy": "Číslo smlouvy. null, pokud v dokumentu není.",
    "isin": "ISIN: 12 znaků, velká písmena, bez mezer. null, pokud v dokumentu není.",
    "isin_platny": "Zda je kontrolní číslice ISIN platná; null, když isin je null.",
    "castka": (
        "Částka v kanonickém tvaru: číslice, desetinná tečka, dvě desetinná místa, bez mezer "
        'a bez měny (např. "1234.50"). null, pokud v dokumentu není.'
    ),
    "mena": 'Kód měny ISO 4217 (např. "CZK"). null, když castka je null.',
    "citelne": "false, pokud dokument nejde přečíst.",
}

_TYPES = {
    "cislo_smlouvy": str,
    "isin": str,
    "isin_platny": bool,
    "castka": str,
    "mena": str,
}


def _nullable(json_type: str, name: str) -> dict:
    return {"anyOf": [{"type": json_type}, {"type": "null"}], "description": _DESCRIPTIONS[name]}


# No minimum/maxLength/pattern: structured outputs does not support them. A schema dict passed
# in output_config is sent as is; only SDK-built schemas get such keywords moved to descriptions.
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "cislo_smlouvy": _nullable("string", "cislo_smlouvy"),
        "isin": _nullable("string", "isin"),
        "isin_platny": _nullable("boolean", "isin_platny"),
        "castka": _nullable("string", "castka"),
        "mena": _nullable("string", "mena"),
        "citelne": {"type": "boolean", "description": _DESCRIPTIONS["citelne"]},
    },
    "required": list(FIELDS),
    "additionalProperties": False,
}

_AMOUNT = re.compile(r"(0|[1-9][0-9]*)\.[0-9]{2}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_ISIN_SHAPE = re.compile(r"[A-Z]{2}[A-Z0-9]{10}")


def is_canonical_amount(value: str) -> bool:
    return bool(_AMOUNT.fullmatch(value))


def is_currency_code(value: str) -> bool:
    return bool(_CURRENCY.fullmatch(value))


def is_isin_shape(value: str) -> bool:
    """12 uppercase letters/digits; says nothing about the check digit."""
    return bool(_ISIN_SHAPE.fullmatch(value))


def canonical_amount(value: Decimal) -> str:
    if value < 0 or value != value.quantize(Decimal("0.01")):
        raise ValueError(f"amount {value} is negative or needs rounding")
    return f"{value:.2f}"


def validate(output: object) -> list[str]:
    """Type-level check of a model output against JSON_SCHEMA."""
    if not isinstance(output, dict):
        return [f"output is {type(output).__name__}, not an object"]
    errors = [f"{name}: missing key" for name in FIELDS if name not in output]
    errors += [f"{name}: unexpected key" for name in output if name not in FIELDS]
    for name, expected_type in _TYPES.items():
        value = output.get(name)
        # bool is a subclass of int, not of str, so isinstance is exact here.
        if value is not None and not isinstance(value, expected_type):
            errors.append(f"{name}: {type(value).__name__} instead of {expected_type.__name__}")
    if "citelne" in output and not isinstance(output["citelne"], bool):
        errors.append(f"citelne: {type(output['citelne']).__name__} instead of bool")
    return errors


def validate_expected(expected: object) -> list[str]:
    """Schema check plus the canonical-form and consistency rules of expected values."""
    errors = validate(expected)
    if errors:
        return errors
    e = expected
    for name in FIELDS:
        if e[name] == "":
            errors.append(f"{name}: empty string, use null")
    if not e["citelne"]:
        errors += [f"{n}: must be null when unreadable" for n in FIELDS[:-1] if e[n] is not None]
        return errors
    if e["castka"] and not is_canonical_amount(e["castka"]):
        errors.append(f"castka: {e['castka']!r} is not canonical")
    if e["mena"] and not is_currency_code(e["mena"]):
        errors.append(f"mena: {e['mena']!r} is not an ISO 4217 code")
    if (e["castka"] is None) != (e["mena"] is None):
        errors.append("castka and mena must both be set or both be null")
    if e["isin"] is None:
        if e["isin_platny"] is not None:
            errors.append("isin_platny: must be null without isin")
    elif not is_isin_shape(e["isin"]):
        errors.append(f"isin: {e['isin']!r} is not 12 uppercase characters")
    elif e["isin_platny"] is not is_valid_isin(e["isin"]):
        errors.append("isin_platny: does not match the check digit")
    return errors
