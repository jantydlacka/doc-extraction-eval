"""ISIN check digit (ISO 6166): letters become numbers (A=10 … Z=35), then Luhn over the digits."""

import re

_BODY = re.compile(r"[A-Z]{2}[A-Z0-9]{9}")
_ISIN = re.compile(r"[A-Z]{2}[A-Z0-9]{9}[0-9]")


def check_digit(body: str) -> str:
    if not isinstance(body, str) or not _BODY.fullmatch(body):
        raise ValueError(f"not an 11-character ISIN body: {body!r}")
    digits = "".join(str(int(ch, 36)) for ch in body)
    total = 0
    # Doubling starts at the rightmost digit because the check digit will follow it.
    for i, ch in enumerate(reversed(digits)):
        n = int(ch) * (2 if i % 2 == 0 else 1)
        total += n - 9 if n > 9 else n
    return str((10 - total % 10) % 10)


def is_valid_isin(value: object) -> bool:
    if not isinstance(value, str) or not _ISIN.fullmatch(value):
        return False
    return check_digit(value[:11]) == value[11]


def make_isin(country: str, nsin: str) -> str:
    body = country + nsin
    return body + check_digit(body)
