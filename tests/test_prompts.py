import pytest

from llmeval.prompts import PROMPT_NAMES, load_prompt


def test_both_prompts_load():
    assert PROMPT_NAMES == ("v1", "v2")
    for name in PROMPT_NAMES:
        assert load_prompt(name).strip()


def test_v2_is_v1_plus_rules():
    v1, v2 = load_prompt("v1"), load_prompt("v2")
    assert v2.startswith(v1)
    rules = v2[len(v1) :]
    for phrase in ("ISO 6166", "platí opravená", "konečný zůstatek", "bez poplatků", '"Kč"'):
        assert phrase in rules
        assert phrase not in v1


@pytest.mark.parametrize("name", ["v3", "../pyproject", ""])
def test_unknown_prompt_is_refused(name):
    with pytest.raises(ValueError, match="unknown prompt"):
        load_prompt(name)
