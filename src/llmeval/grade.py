"""Field-by-field grading of a model output against ``expected.json``.

The model output is never normalised: a value in a form other than the schema asks for is a
``format`` error even if it means the right thing. Per field, in this order:

1. key missing                                -> ``chybi_klic``
2. null                                       -> ``ok`` if null is expected, else ``spatne``;
                                                 a null ``citelne`` (not nullable) -> ``format``
3. ``""``                                     -> ``format``, also where null is expected
4. any other value where null is expected     -> ``navic_hodnota``, a bad form noted in the reason
5. wrong type or non-canonical form           -> ``format``
6. a value equal to the expected one          -> ``ok``, otherwise ``spatne``

A case passes when every field is ``ok`` and the output has no keys outside the schema.
Output that is not strict JSON fails the case as invalid JSON, including a duplicate key (the
standard parser would keep the last value silently) and ``NaN``/``Infinity``.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

from llmeval.schema import (
    FIELDS,
    is_canonical_amount,
    is_currency_code,
    is_isin_shape,
    validate_expected,
)


class Status(StrEnum):
    OK = "ok"
    SPATNE = "spatne"
    FORMAT = "format"
    CHYBI_KLIC = "chybi_klic"
    NAVIC_HODNOTA = "navic_hodnota"


@dataclass(frozen=True)
class FieldResult:
    status: Status
    expected: object
    # Raw model value; None also when the key is missing (status tells the two apart).
    actual: object
    reason: str = ""


@dataclass(frozen=True)
class Grade:
    passed: bool
    # Keyed by FIELDS in schema order; empty when the output is not a JSON object.
    fields: dict[str, FieldResult] = field(default_factory=dict)
    extra_keys: tuple[str, ...] = ()
    # Empty when passed; otherwise every failure, "; "-separated.
    reason: str = ""


def _string(check: Callable[[str], bool] | None, rule: str) -> Callable[[object], str | None]:
    def problem(value: object) -> str | None:
        if not isinstance(value, str):
            return f"{type(value).__name__} instead of string"
        if value == "":
            return "empty string instead of null"
        if check is not None and not check(value):
            return f"{value!r} is not {rule}"
        return None

    return problem


def _boolean(value: object) -> str | None:
    if not isinstance(value, bool):
        return f"{type(value).__name__} instead of boolean"
    return None


_FORMAT = {
    "cislo_smlouvy": _string(None, ""),
    "isin": _string(is_isin_shape, "12 uppercase letters or digits without spaces"),
    "isin_platny": _boolean,
    "castka": _string(is_canonical_amount, 'in canonical form like "1234.50"'),
    "mena": _string(is_currency_code, "an ISO 4217 code"),
    "citelne": _boolean,
}


def _grade_field(name: str, expected: object, output: dict) -> FieldResult:
    if name not in output:
        return FieldResult(Status.CHYBI_KLIC, expected, None, "missing key")
    actual = output[name]
    if actual is None and name != "citelne":
        if expected is None:
            return FieldResult(Status.OK, expected, actual)
        return FieldResult(Status.SPATNE, expected, actual, f"null, expected {expected!r}")
    problem = _FORMAT[name](actual)
    if expected is None and actual != "":
        reason = f"{actual!r}, expected null" + (f", also {problem}" if problem else "")
        return FieldResult(Status.NAVIC_HODNOTA, expected, actual, reason)
    if problem:
        return FieldResult(Status.FORMAT, expected, actual, problem)
    # Same type on both sides here, so == cannot mix True with 1; a canonical amount has
    # exactly one spelling per value, so string equality is exact.
    if actual == expected:
        return FieldResult(Status.OK, expected, actual)
    return FieldResult(Status.SPATNE, expected, actual, f"{actual!r}, expected {expected!r}")


def _check_expected(expected: dict) -> None:
    errors = validate_expected(expected)
    if errors:
        raise ValueError("invalid expected: " + "; ".join(errors))


def grade(expected: dict, output: object) -> Grade:
    """Grade a parsed model output. Raises ValueError when ``expected`` itself is invalid."""
    _check_expected(expected)
    if not isinstance(output, dict):
        return Grade(False, reason=f"output is {type(output).__name__}, not an object")
    fields = {name: _grade_field(name, expected[name], output) for name in FIELDS}
    extra = tuple(key for key in output if key not in FIELDS)
    failures = [
        f"{name}: {r.status.value} ({r.reason})"
        for name, r in fields.items()
        if r.status is not Status.OK
    ]
    failures += [f"{key!r}: unexpected key" for key in extra]
    return Grade(not failures, fields, extra, "; ".join(failures))


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    output: dict = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate key {key!r}")
        output[key] = value
    return output


def _no_constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def grade_text(expected: dict, raw: str) -> Grade:
    """Parse the model's raw text as JSON and grade it; invalid JSON fails the case."""
    _check_expected(expected)
    # Besides JSONDecodeError: ValueError over the int digit limit, RecursionError when too deep.
    try:
        output = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_no_constant)
    except (ValueError, RecursionError) as exc:
        return Grade(False, reason=f"invalid JSON: {exc}")
    return grade(expected, output)
