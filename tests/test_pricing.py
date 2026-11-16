import decimal
import json
from dataclasses import fields
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from anthropic.types import CacheCreation, Usage

from llmeval.pricing import (
    DEFAULT_PRICES_PATH,
    CacheWriteError,
    CallCost,
    ModelPricing,
    PriceModifierError,
    PricingError,
    PromptTooLongError,
    UnknownModelError,
    call_cost,
    check_prompt,
    load_prices,
    read_usage,
)

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
D = Decimal


def usage(**fields_) -> dict:
    return {"input_tokens": 0, "output_tokens": 0} | fields_


# --- prices.json ---------------------------------------------------------------------------


def test_prices_json_is_the_repo_data_file():
    assert Path(__file__).resolve().parent.parent / "data" / "prices.json" == DEFAULT_PRICES_PATH


def test_prices_json_has_source_and_check_date():
    table = load_prices()
    assert table.source == "https://platform.claude.com/docs/en/about-claude/pricing"
    assert table.verified == date(2026, 11, 16)


def test_prices_json_stores_prices_as_strings():
    raw = json.loads(DEFAULT_PRICES_PATH.read_text())
    for model in raw["modely"].values():
        for tier in [
            model["ceny"],
            *([model["nad_prahem"]["ceny"]] if "nad_prahem" in model else []),
        ]:
            assert all(isinstance(v, str) for v in tier.values()), tier


def test_prices_match_the_checked_pricing_page():
    table = load_prices()
    haiku, sonnet = table.for_model(HAIKU), table.for_model(SONNET)

    assert haiku.prices.as_tuple() == (D("0.10"), D("0.125"), D("0.20"), D("0.01"), D("0.50"))
    assert haiku.prompt_limit == 100_000
    assert haiku.over_limit.as_tuple() == (D("0.50"), D("0.625"), D("1"), D("0.05"), D("2.50"))

    assert sonnet.prices.as_tuple() == (D("2"), D("2.50"), D("4"), D("0.10"), D("10"))
    assert sonnet.prompt_limit is None
    assert sonnet.over_limit is None

    assert set(table.models) == {HAIKU, SONNET}


def test_loaded_prices_are_decimal():
    for model in load_prices().models.values():
        for value in model.prices.as_tuple():
            assert type(value) is Decimal


def test_float_price_in_json_is_rejected(tmp_path):
    raw = json.loads(DEFAULT_PRICES_PATH.read_text())
    raw["modely"][SONNET]["ceny"]["vstup"] = 2.0
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(PricingError, match="vstup"):
        load_prices(path)


# --- hand-computed costs -------------------------------------------------------------------


def test_haiku_cost_by_hand():
    # input       2 000 x 0.10 / 1e6 = 0.0002
    # cache read  1 000 x 0.01 / 1e6 = 0.00001
    # output        300 x 0.50 / 1e6 = 0.00015
    # total                          = 0.00036
    cost = call_cost(
        HAIKU, usage(input_tokens=2000, cache_read_input_tokens=1000, output_tokens=300)
    )
    assert cost.input_usd == D("0.0002")
    assert cost.cache_read_usd == D("0.00001")
    assert cost.cache_write_usd == 0
    assert cost.output_usd == D("0.00015")
    assert cost.total_usd == D("0.00036")
    assert cost.model == HAIKU


def test_sonnet_cost_by_hand():
    # input       12 345 x 2    / 1e6 = 0.02469
    # cache read   5 000 x 0.10 / 1e6 = 0.0005
    # output       1 234 x 10   / 1e6 = 0.01234
    # total                           = 0.03753
    cost = call_cost(
        SONNET, usage(input_tokens=12_345, cache_read_input_tokens=5000, output_tokens=1234)
    )
    assert (cost.input_usd, cost.cache_read_usd, cost.output_usd) == (
        D("0.02469"),
        D("0.0005"),
        D("0.01234"),
    )
    assert cost.total_usd == D("0.03753")


def test_cost_is_exact_where_float_would_drift():
    # 3 x 0.10 / 1e6: as float 3 * 0.1 is 0.30000000000000004
    cost = call_cost(HAIKU, usage(input_tokens=3))
    assert cost.total_usd == D("0.0000003")


def test_every_amount_is_decimal():
    cost = call_cost(SONNET, usage(input_tokens=10, cache_read_input_tokens=10, output_tokens=10))
    amounts = [f.name for f in fields(CallCost) if f.name.endswith("_usd")]
    assert amounts == ["input_usd", "cache_write_usd", "cache_read_usd", "output_usd", "total_usd"]
    for name in amounts:
        assert type(getattr(cost, name)) is Decimal, name


def test_cost_carries_token_counts():
    cost = call_cost(HAIKU, usage(input_tokens=7, cache_read_input_tokens=5, output_tokens=3))
    assert cost.tokens.input_tokens == 7
    assert cost.tokens.cache_read_input_tokens == 5
    assert cost.tokens.cache_creation_input_tokens == 0
    assert cost.tokens.output_tokens == 3
    assert cost.tokens.prompt_tokens == 12


# --- Haiku prompt-length threshold ---------------------------------------------------------


def test_haiku_90k_input_plus_20k_cache_read_is_refused():
    with pytest.raises(PromptTooLongError) as exc:
        call_cost(HAIKU, usage(input_tokens=90_000, cache_read_input_tokens=20_000))
    assert exc.value.prompt_tokens == 110_000
    assert exc.value.limit == 100_000


def test_haiku_threshold_counts_all_three_input_fields():
    # 99 000 + 1 000 + 1 = 100 001 -> over, although each field alone is under
    with pytest.raises(PromptTooLongError):
        call_cost(
            HAIKU,
            usage(
                input_tokens=99_000,
                cache_read_input_tokens=1_000,
                cache_creation_input_tokens=1,
            ),
        )


def test_haiku_exactly_100k_is_allowed_at_the_lower_rate():
    # 90 000 x 0.10 + 10 000 x 0.01 = 9 000 + 100 = 9 100 -> / 1e6 = 0.0091
    cost = call_cost(HAIKU, usage(input_tokens=90_000, cache_read_input_tokens=10_000))
    assert cost.total_usd == D("0.0091")


def test_haiku_100_001_is_refused():
    with pytest.raises(PromptTooLongError):
        call_cost(HAIKU, usage(input_tokens=100_001))


def test_refused_call_still_carries_its_real_cost_at_the_higher_rate():
    # The call has already been billed, so the error must say how much:
    # input       100 001 x 0.50 / 1e6 = 0.0500005
    # output          100 x 2.50 / 1e6 = 0.00025
    # total                            = 0.0502505
    with pytest.raises(PromptTooLongError) as exc:
        call_cost(HAIKU, usage(input_tokens=100_001, output_tokens=100))
    assert exc.value.cost.total_usd == D("0.0502505")
    assert exc.value.cost.output_usd == D("0.00025")


def test_sonnet_has_no_threshold():
    # 500 000 x 2 / 1e6 = 1
    assert call_cost(SONNET, usage(input_tokens=500_000)).total_usd == D("1")


def test_check_prompt_before_the_call():
    check_prompt(HAIKU, 100_000)
    check_prompt(SONNET, 900_000)
    with pytest.raises(PromptTooLongError):
        check_prompt(HAIKU, 100_001)
    with pytest.raises(UnknownModelError):
        check_prompt("claude-haiku-4-5", 10)


# --- cache creation guard ------------------------------------------------------------------


def test_nonzero_cache_creation_is_an_error():
    with pytest.raises(CacheWriteError):
        call_cost(SONNET, usage(input_tokens=10, cache_creation_input_tokens=1))


def test_cache_write_error_carries_cost_with_ttl_breakdown():
    # 5m write  1 000 x 2.50 / 1e6 = 0.0025
    # 1h write    500 x 4    / 1e6 = 0.002
    # input       100 x 2    / 1e6 = 0.0002
    # total                        = 0.0047
    u = usage(
        input_tokens=100,
        cache_creation_input_tokens=1500,
        cache_creation={"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 500},
    )
    with pytest.raises(CacheWriteError) as exc:
        call_cost(SONNET, u)
    assert exc.value.cost.cache_write_usd == D("0.0045")
    assert exc.value.cost.total_usd == D("0.0047")


def test_cache_write_without_ttl_breakdown_is_priced_at_the_1h_rate():
    # Upper bound: 1 000 x 4 / 1e6 = 0.004
    with pytest.raises(CacheWriteError) as exc:
        call_cost(SONNET, usage(cache_creation_input_tokens=1000))
    assert exc.value.cost.cache_write_usd == D("0.004")


def test_prompt_length_is_checked_before_cache_write():
    # Both problems: the cost must come from the higher Haiku rate card.
    with pytest.raises(PromptTooLongError):
        call_cost(HAIKU, usage(input_tokens=100_000, cache_creation_input_tokens=1))


def test_zero_cache_creation_with_zero_breakdown_is_fine():
    u = usage(
        input_tokens=1,
        cache_creation_input_tokens=0,
        cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
    )
    assert call_cost(HAIKU, u).cache_write_usd == 0


# --- unknown model -------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-5", "", "CLAUDE-HAIKU-5-5"])
def test_unknown_model_raises_instead_of_costing_zero(model):
    with pytest.raises(UnknownModelError):
        call_cost(model, usage(input_tokens=1))


# --- usage input: SDK object and plain dict ------------------------------------------------


def test_sdk_usage_object_and_dict_give_the_same_result():
    raw = {
        "input_tokens": 1500,
        "output_tokens": 200,
        "cache_read_input_tokens": 100,
        "cache_creation_input_tokens": 0,
        "output_tokens_details": {"thinking_tokens": 50},
    }
    from_sdk = call_cost(HAIKU, Usage.model_validate(raw))
    from_dict = call_cost(HAIKU, raw)
    assert from_sdk == from_dict
    assert from_sdk.tokens.thinking_tokens == 50


def test_none_cache_fields_count_as_zero():
    sdk = Usage.model_validate(
        {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None,
            "cache_creation": None,
        }
    )
    tokens = read_usage(sdk)
    assert tokens.cache_read_input_tokens == 0
    assert tokens.cache_creation_input_tokens == 0
    assert read_usage(usage(input_tokens=10, cache_read_input_tokens=None)).prompt_tokens == 10


def test_missing_cache_fields_count_as_zero():
    tokens = read_usage({"input_tokens": 10, "output_tokens": 5})
    assert (tokens.cache_read_input_tokens, tokens.cache_creation_input_tokens) == (0, 0)


def test_thinking_tokens_absent_when_not_reported():
    assert read_usage(usage(output_tokens=5)).thinking_tokens is None
    assert read_usage(usage(output_tokens=5, output_tokens_details=None)).thinking_tokens is None
    sdk = Usage.model_validate({"input_tokens": 1, "output_tokens": 5})
    assert read_usage(sdk).thinking_tokens is None


def test_thinking_tokens_present_when_reported():
    tokens = read_usage(usage(output_tokens=40, output_tokens_details={"thinking_tokens": 0}))
    assert tokens.thinking_tokens == 0
    sdk = Usage.model_validate(
        {"input_tokens": 1, "output_tokens": 40, "output_tokens_details": {"thinking_tokens": 30}}
    )
    assert read_usage(sdk).thinking_tokens == 30


def test_thinking_tokens_are_not_billed_on_top_of_output():
    # output_tokens already includes thinking: 40 x 0.50 / 1e6 = 0.00002
    cost = call_cost(HAIKU, usage(output_tokens=40, output_tokens_details={"thinking_tokens": 30}))
    assert cost.output_usd == D("0.00002")


@pytest.mark.parametrize(
    "bad",
    [
        {"output_tokens": 1},
        {"input_tokens": 1},
        {"input_tokens": None, "output_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": 1.0, "output_tokens": 1},
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": "1", "output_tokens": 1},
        {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": -5},
    ],
)
def test_malformed_usage_is_rejected(bad):
    with pytest.raises(PricingError):
        read_usage(bad)


def test_cache_write_missing_from_ttl_breakdown_is_priced_at_the_1h_rate():
    # 5m write     200 x 2.50 / 1e6 = 0.0005
    # rest     (1 000 - 200) x 4 / 1e6 = 0.0032
    # total                        = 0.0037
    u = usage(
        cache_creation_input_tokens=1000,
        cache_creation={"ephemeral_5m_input_tokens": 200, "ephemeral_1h_input_tokens": 0},
    )
    with pytest.raises(CacheWriteError) as exc:
        call_cost(SONNET, u)
    assert exc.value.cost.cache_write_usd == D("0.0037")


def test_cache_write_only_in_ttl_breakdown_is_still_an_error():
    u = usage(
        cache_creation_input_tokens=0,
        cache_creation={"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 0},
    )
    with pytest.raises(CacheWriteError):
        call_cost(SONNET, u)


@pytest.mark.parametrize("limit", [0, -1, "100000", 100000.0, True])
def test_bad_prompt_limit_in_json_is_rejected(tmp_path, limit):
    raw = json.loads(DEFAULT_PRICES_PATH.read_text())
    raw["modely"][HAIKU]["nad_prahem"]["prah_promptu"] = limit
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(PricingError, match="prah_promptu"):
        load_prices(path)


def test_explicit_price_table_is_used(tmp_path):
    raw = json.loads(DEFAULT_PRICES_PATH.read_text())
    raw["modely"][SONNET]["ceny"]["vstup"] = "3"
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(raw))
    # 1 000 x 3 / 1e6 = 0.003
    assert call_cost(SONNET, usage(input_tokens=1000), load_prices(path)).total_usd == D("0.003")


# --- usage fields that change the price ----------------------------------------------------

BASE_USAGE = {"input_tokens": 1000, "output_tokens": 100}
# Standard-rate cost of BASE_USAGE on Sonnet: 1 000 x 2 / 1e6 + 100 x 10 / 1e6 = 0.003
BASE_SONNET_USD = D("0.003")

MODIFIERS = {
    "inference_geo": {"inference_geo": "us"},
    "speed": {"speed": "fast"},
    "service_tier": {"service_tier": "batch"},
    "priority": {"service_tier": "priority"},
    "web_search": {"server_tool_use": {"web_search_requests": 1, "web_fetch_requests": 0}},
    "web_fetch": {"server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 2}},
}
REPORTED = {
    "inference_geo": "inference_geo='us'",
    "speed": "speed='fast'",
    "service_tier": "service_tier='batch'",
    "priority": "service_tier='priority'",
    "web_search": "server_tool_use.web_search_requests=1",
    "web_fetch": "server_tool_use.web_fetch_requests=2",
}


@pytest.mark.parametrize("as_sdk", [False, True], ids=["dict", "sdk"])
@pytest.mark.parametrize("name", list(MODIFIERS))
def test_usage_field_that_changes_the_price_is_an_error(name, as_sdk):
    raw = BASE_USAGE | MODIFIERS[name]
    u = Usage.model_validate(raw) if as_sdk else raw
    with pytest.raises(PriceModifierError) as exc:
        call_cost(SONNET, u)
    assert exc.value.cost.total_usd == BASE_SONNET_USD
    assert exc.value.modifiers == (REPORTED[name],)


@pytest.mark.parametrize("as_sdk", [False, True], ids=["dict", "sdk"])
def test_standard_values_of_price_fields_are_fine(as_sdk):
    raw = BASE_USAGE | {
        "inference_geo": "global",
        "speed": "standard",
        "service_tier": "standard",
        "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
    }
    u = Usage.model_validate(raw) if as_sdk else raw
    assert call_cost(SONNET, u).total_usd == BASE_SONNET_USD


def test_null_price_fields_are_fine():
    raw = BASE_USAGE | dict.fromkeys(["inference_geo", "speed", "service_tier", "server_tool_use"])
    assert call_cost(SONNET, raw).total_usd == BASE_SONNET_USD


def test_unknown_server_tool_counter_in_dict_is_an_error():
    raw = BASE_USAGE | {"server_tool_use": {"code_execution_requests": 1}}
    with pytest.raises(PriceModifierError):
        call_cost(SONNET, raw)


def test_all_price_modifiers_are_listed():
    raw = BASE_USAGE | MODIFIERS["inference_geo"] | MODIFIERS["speed"] | MODIFIERS["web_search"]
    with pytest.raises(PriceModifierError) as exc:
        call_cost(SONNET, raw)
    assert len(exc.value.modifiers) == 3


def test_price_modifier_is_checked_before_cache_write():
    raw = BASE_USAGE | MODIFIERS["speed"] | {"cache_creation_input_tokens": 10}
    with pytest.raises(PriceModifierError):
        call_cost(SONNET, raw)


def test_long_haiku_prompt_with_price_modifier_keeps_the_higher_rate_cost():
    # input 100 001 x 0.50 / 1e6 = 0.0500005 (higher rate card)
    raw = usage(input_tokens=100_001, inference_geo="us")
    with pytest.raises(PromptTooLongError) as exc:
        call_cost(HAIKU, raw)
    assert exc.value.cost.total_usd == D("0.0500005")
    assert exc.value.modifiers == ("inference_geo='us'",)


def test_long_haiku_prompt_without_price_modifier_lists_none():
    with pytest.raises(PromptTooLongError) as exc:
        call_cost(HAIKU, usage(input_tokens=100_001))
    assert exc.value.modifiers == ()


# --- prices.json validation ----------------------------------------------------------------


def write_prices(tmp_path, change) -> Path:
    raw = json.loads(DEFAULT_PRICES_PATH.read_text())
    change(raw)
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(raw))
    return path


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "-0.01", "abc", "", "sNaN"])
def test_bad_price_string_is_rejected(tmp_path, value):
    path = write_prices(tmp_path, lambda r: r["modely"][SONNET]["ceny"].update(vystup=value))
    with pytest.raises(PricingError, match="vystup"):
        load_prices(path)


def test_bad_price_in_the_higher_rate_card_is_rejected(tmp_path):
    def change(r):
        r["modely"][HAIKU]["nad_prahem"]["ceny"]["vstup"] = "NaN"

    with pytest.raises(PricingError, match="vstup"):
        load_prices(write_prices(tmp_path, change))


@pytest.mark.parametrize("key", ["vstup", "vystup"])
@pytest.mark.parametrize("value", ["0", "0.000"])
@pytest.mark.parametrize("card", ["ceny", "nad_prahem"])
def test_zero_input_or_output_price_is_rejected(tmp_path, key, value, card):
    def change(r):
        prices = r["modely"][HAIKU]
        (prices["ceny"] if card == "ceny" else prices["nad_prahem"]["ceny"])[key] = value

    with pytest.raises(PricingError, match=key):
        load_prices(write_prices(tmp_path, change))


def test_zero_price_is_allowed(tmp_path):
    path = write_prices(tmp_path, lambda r: r["modely"][SONNET]["ceny"].update(cteni_cache="0"))
    assert load_prices(path).for_model(SONNET).prices.cache_read == 0


@pytest.mark.parametrize(
    "drop",
    [
        lambda r: r.pop("modely"),
        lambda r: r.pop("zdroj"),
        lambda r: r.pop("overeno"),
        lambda r: r["modely"][SONNET].pop("ceny"),
        lambda r: r["modely"][SONNET]["ceny"].pop("vystup"),
        lambda r: r["modely"][HAIKU]["nad_prahem"].pop("prah_promptu"),
        lambda r: r["modely"][HAIKU]["nad_prahem"].pop("ceny"),
    ],
    ids=["modely", "zdroj", "overeno", "ceny", "vystup", "prah_promptu", "nad_prahem.ceny"],
)
def test_missing_key_in_prices_json_is_a_pricing_error(tmp_path, drop):
    with pytest.raises(PricingError):
        load_prices(write_prices(tmp_path, drop))


@pytest.mark.parametrize("over", [None, {}, [], "100000"])
def test_present_but_empty_higher_rate_card_is_an_error(tmp_path, over):
    path = write_prices(tmp_path, lambda r: r["modely"][HAIKU].update(nad_prahem=over))
    with pytest.raises(PricingError, match="nad_prahem"):
        load_prices(path)


@pytest.mark.parametrize(
    "change", [lambda r: r.update(overeno="16. 11. 2026"), lambda r: r.update(zdroj="")]
)
def test_bad_source_or_date_is_rejected(tmp_path, change):
    with pytest.raises(PricingError):
        load_prices(write_prices(tmp_path, change))


def test_invalid_json_is_a_pricing_error(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text("{")
    with pytest.raises(PricingError):
        load_prices(path)


def test_limit_without_higher_rate_card_cannot_be_built():
    p = load_prices().for_model(SONNET).prices
    with pytest.raises(PricingError):
        ModelPricing(model="x", prices=p, prompt_limit=10)
    with pytest.raises(PricingError):
        ModelPricing(model="x", prices=p, over_limit=p)


# --- check_prompt arguments ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [-5, 100_000.5, True, None, "10"])
def test_check_prompt_rejects_a_bad_token_count(bad):
    with pytest.raises(PricingError):
        check_prompt(SONNET, bad)


def test_check_prompt_uses_an_explicit_table():
    sonnet = load_prices().for_model(SONNET)
    table = load_prices()
    table.models[SONNET] = ModelPricing(
        SONNET, sonnet.prices, prompt_limit=10, over_limit=sonnet.prices
    )
    check_prompt(SONNET, 10, table)
    with pytest.raises(PromptTooLongError):
        check_prompt(SONNET, 11, table)


# --- smaller gaps --------------------------------------------------------------------------


def test_cache_write_error_message_counts_writes_seen_only_in_the_breakdown():
    u = usage(
        cache_creation_input_tokens=0,
        cache_creation={"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 7},
    )
    with pytest.raises(CacheWriteError, match=r"\b17 tokens"):
        call_cost(SONNET, u)


def test_thinking_tokens_none_when_details_lack_them():
    assert read_usage(usage(output_tokens_details={})).thinking_tokens is None


def test_sdk_cache_creation_object_is_read():
    u = Usage(
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=300,
        cache_creation=CacheCreation(ephemeral_5m_input_tokens=100, ephemeral_1h_input_tokens=200),
    )
    tokens = read_usage(u)
    assert (tokens.cache_write_5m_tokens, tokens.cache_write_1h_tokens) == (100, 200)
    # 100 x 2.50 / 1e6 + 200 x 4 / 1e6 = 0.00025 + 0.0008 = 0.00105
    with pytest.raises(CacheWriteError) as exc:
        call_cost(SONNET, u)
    assert exc.value.cost.cache_write_usd == D("0.00105")


def test_haiku_cache_write_error_cost():
    # 1h write 1 000 x 0.20 / 1e6 = 0.0002
    # 5m write 2 000 x 0.125 / 1e6 = 0.00025
    # input    1 000 x 0.10 / 1e6 = 0.0001
    # total                      = 0.00055
    u = usage(
        input_tokens=1000,
        cache_creation_input_tokens=3000,
        cache_creation={"ephemeral_5m_input_tokens": 2000, "ephemeral_1h_input_tokens": 1000},
    )
    with pytest.raises(CacheWriteError) as exc:
        call_cost(HAIKU, u)
    assert exc.value.cost.cache_write_usd == D("0.00045")
    assert exc.value.cost.total_usd == D("0.00055")


@pytest.fixture
def low_precision():
    ctx = decimal.getcontext()
    saved = ctx.prec
    ctx.prec = 3
    yield
    ctx.prec = saved


def test_global_decimal_precision_does_not_round_costs(low_precision):
    # Same figures as test_sonnet_cost_by_hand; 0.02469 has 4 significant digits.
    cost = call_cost(
        SONNET, usage(input_tokens=12_345, cache_read_input_tokens=5000, output_tokens=1234)
    )
    assert cost.input_usd == D("0.02469")
    assert cost.total_usd == D("0.03753")


def test_1h_write_reported_only_in_the_breakdown_is_priced():
    # 50 x 4 / 1e6 = 0.0002
    u = usage(
        cache_creation_input_tokens=0,
        cache_creation={"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 50},
    )
    assert read_usage(u).cache_write_1h_tokens == 50
    with pytest.raises(CacheWriteError) as exc:
        call_cost(SONNET, u)
    assert exc.value.cost.cache_write_usd == D("0.0002")


@pytest.mark.parametrize(
    "change, error",
    [
        (lambda r: r.update(modely=[]), "AttributeError"),
        (lambda r: r["modely"][SONNET].update(ceny="2.00"), "TypeError"),
    ],
    ids=["modely-list", "ceny-string"],
)
def test_price_file_of_the_wrong_shape_is_a_pricing_error(tmp_path, change, error):
    with pytest.raises(PricingError, match=error):
        load_prices(write_prices(tmp_path, change))


def test_missing_price_file_is_not_mistaken_for_a_malformed_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / "missing.json")


@pytest.mark.parametrize("value", ["garbled", 5, ["web_search_requests"]])
def test_server_tool_use_of_an_unknown_shape_is_a_price_modifier_with_a_cost(value):
    with pytest.raises(PriceModifierError) as exc:
        call_cost(SONNET, BASE_USAGE | {"server_tool_use": value})
    assert exc.value.cost.total_usd == BASE_SONNET_USD
    assert exc.value.modifiers == (f"server_tool_use={value!r}",)


def test_price_file_without_models_is_an_error(tmp_path):
    with pytest.raises(PricingError, match="no models"):
        load_prices(write_prices(tmp_path, lambda r: r.update(modely={})))
