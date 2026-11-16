import pytest

from llmeval.dataset.isin import check_digit, is_valid_isin, make_isin

# Well-known public ISINs: Apple, Bayer, Microsoft, BAE Systems and the Treasury Corporation
# of Victoria example from the ISIN article on Wikipedia.
KNOWN_VALID = ["US0378331005", "DE000BAY0017", "US5949181045", "GB0002634946", "AU0000XVGZA3"]


@pytest.mark.parametrize("isin", KNOWN_VALID)
def test_known_isins_are_valid(isin):
    assert is_valid_isin(isin)
    assert check_digit(isin[:11]) == isin[11]


@pytest.mark.parametrize("isin", KNOWN_VALID)
def test_every_other_check_digit_is_invalid(isin):
    for digit in "0123456789":
        if digit != isin[11]:
            assert not is_valid_isin(isin[:11] + digit)


@pytest.mark.parametrize(
    "value",
    ["us0378331005", "US037833100", "US03783310055", "US037833100X", "US-378331005", "", None],
)
def test_malformed_isins_are_invalid(value):
    assert not is_valid_isin(value)


def test_check_digit_rejects_malformed_body():
    with pytest.raises(ValueError):
        check_digit("US03783310")
    with pytest.raises(ValueError):
        check_digit("us037833100")


def test_make_isin_appends_check_digit():
    isin = make_isin("CZ", "990000001")
    assert isin[:11] == "CZ990000001"
    assert is_valid_isin(isin)
