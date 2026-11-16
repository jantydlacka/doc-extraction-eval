"""Price of one API call from its ``usage``, in Decimal USD.

Rules:

- prompt length = ``input_tokens + cache_creation_input_tokens + cache_read_input_tokens``;
  a Haiku 5.5 prompt over its limit is billed on a higher rate card and the tool refuses it;
- the tool does not use prompt caching, so a non-zero cache write is an error;
  cache reads are priced;
- ``output_tokens`` already includes thinking tokens, which are only reported, not added;
- an unknown model is an error, never a zero cost;
- a usage field that makes the call dearer than the standard rate (``inference_geo`` other
  than global, fast ``speed``, a non-standard ``service_tier``, server tool requests) is an
  error, because those surcharges are not in the price table.

Errors raised by ``call_cost`` carry the call's ``cost``: the call was billed, so the caller can
still record it before stopping. Checks run in this order and the first one raises:

1. Haiku prompt over the limit -> ``PromptTooLongError``, cost on the higher rate card;
2. price modifier -> ``PriceModifierError``;
3. cache write -> ``CacheWriteError``, exact cost.

Whenever ``.modifiers`` of the error is non-empty, its ``cost`` is a lower bound.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

DEFAULT_PRICES_PATH = Path(__file__).resolve().parents[2] / "data" / "prices.json"

_MTOK = Decimal(1_000_000)
# Token counts and prices have few digits; 50 keeps every product and sum exact.
_PRECISION = 50
_PRICE_KEYS = ("vstup", "zapis_cache_5m", "zapis_cache_1h", "cteni_cache", "vystup")


class PricingError(Exception):
    pass


class UnknownModelError(PricingError):
    pass


class PromptTooLongError(PricingError):
    def __init__(
        self,
        model: str,
        prompt_tokens: int,
        limit: int,
        cost: "CallCost | None" = None,
        modifiers: tuple[str, ...] = (),
    ):
        super().__init__(
            f"{model}: prompt has {prompt_tokens} tokens, over the {limit} limit "
            "(higher rate card, refused)"
        )
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.limit = limit
        self.cost = cost
        self.modifiers = modifiers


class PriceModifierError(PricingError):
    def __init__(self, cost: "CallCost", modifiers: tuple[str, ...]):
        super().__init__(
            f"{cost.model}: usage reports {', '.join(modifiers)}, which the price table does not "
            "cover; the standard-rate cost is a lower bound"
        )
        self.cost = cost
        self.modifiers = modifiers


class CacheWriteError(PricingError):
    def __init__(self, cost: "CallCost"):
        tokens = cost.tokens
        written = max(
            tokens.cache_creation_input_tokens,
            (tokens.cache_write_5m_tokens or 0) + (tokens.cache_write_1h_tokens or 0),
        )
        super().__init__(
            f"{cost.model}: {written} tokens written to the prompt cache, "
            "which this tool never asks for"
        )
        self.cost = cost


@dataclass(frozen=True)
class Prices:
    """USD per million tokens."""

    input: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal
    cache_read: Decimal
    output: Decimal

    def as_tuple(self) -> tuple[Decimal, ...]:
        return (self.input, self.cache_write_5m, self.cache_write_1h, self.cache_read, self.output)


@dataclass(frozen=True)
class ModelPricing:
    model: str
    prices: Prices
    prompt_limit: int | None = None
    over_limit: Prices | None = None

    def __post_init__(self):
        if (self.prompt_limit is None) != (self.over_limit is None):
            raise PricingError(f"{self.model}: prompt limit and its rate card go together")


@dataclass(frozen=True)
class PriceTable:
    source: str
    verified: date
    models: dict[str, ModelPricing]

    def for_model(self, model: str) -> ModelPricing:
        try:
            return self.models[model]
        except KeyError:
            raise UnknownModelError(f"no price for model {model!r}") from None


@dataclass(frozen=True)
class TokenCounts:
    input_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    cache_write_5m_tokens: int | None = None
    cache_write_1h_tokens: int | None = None
    # None when the response does not report thinking tokens.
    thinking_tokens: int | None = None

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens


@dataclass(frozen=True)
class CallCost:
    model: str
    tokens: TokenCounts
    input_usd: Decimal
    cache_write_usd: Decimal
    cache_read_usd: Decimal
    output_usd: Decimal
    total_usd: Decimal


def _price(raw: Mapping[str, Any], key: str) -> Decimal:
    value = raw[key]
    if not isinstance(value, str):
        raise PricingError(f"price {key!r} must be a string, got {value!r}")
    try:
        price = Decimal(value)
    except InvalidOperation:
        raise PricingError(f"price {key!r} is not a number: {value!r}") from None
    if not price.is_finite() or price < 0:
        raise PricingError(f"price {key!r} must be finite and >= 0, got {value!r}")
    return price


def _prices(raw: Mapping[str, Any]) -> Prices:
    prices = Prices(*(_price(raw, key) for key in _PRICE_KEYS))
    # A zero input or output price would make the bound of a call without a response 0, and
    # the ledger refuses such a line.
    for key, price in (("vstup", prices.input), ("vystup", prices.output)):
        if price == 0:
            raise PricingError(f"price {key!r} must be above 0")
    return prices


def _model(model: str, entry: Mapping[str, Any]) -> ModelPricing:
    if "nad_prahem" not in entry:
        return ModelPricing(model=model, prices=_prices(entry["ceny"]))
    over = entry["nad_prahem"]
    if not isinstance(over, dict) or not {"prah_promptu", "ceny"} <= over.keys():
        raise PricingError(f"{model}: nad_prahem needs prah_promptu and ceny, got {over!r}")
    limit = over["prah_promptu"]
    if type(limit) is not int or limit <= 0:
        raise PricingError(f"{model}: prah_promptu must be a positive int, got {limit!r}")
    return ModelPricing(model, _prices(entry["ceny"]), limit, _prices(over["ceny"]))


def load_prices(path: Path = DEFAULT_PRICES_PATH) -> PriceTable:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        source = raw["zdroj"]
        if not isinstance(source, str) or not source:
            raise PricingError(f"zdroj must be a non-empty string, got {source!r}")
        verified = date.fromisoformat(raw["overeno"])
        models = {model: _model(model, entry) for model, entry in raw["modely"].items()}
        if not models:
            raise PricingError(f"{path}: no models")
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise PricingError(f"{path}: malformed price file ({type(e).__name__}: {e})") from e
    return PriceTable(source=source, verified=verified, models=models)


def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _checked_count(value: Any, name: str) -> int:
    # bool is an int subclass; a True token count is a bug upstream.
    if type(value) is not int or value < 0:
        raise PricingError(f"{name} must be a non-negative int, got {value!r}")
    return value


def _count(obj: Any, key: str, *, required: bool = False) -> int:
    value = _get(obj, key)
    if value is None and not required:
        return 0
    return _checked_count(value, f"usage.{key}")


_STANDARD = {"inference_geo": "global", "speed": "standard", "service_tier": "standard"}


def price_modifiers(usage: Any) -> tuple[str, ...]:
    """Usage fields that make the call dearer than the price table says, as ``key=value``."""
    found = [
        f"{key}={value!r}"
        for key, standard in _STANDARD.items()
        if (value := _get(usage, key)) not in (None, standard)
    ]
    server_tools = _get(usage, "server_tool_use")
    if isinstance(server_tools, Mapping):
        counters = server_tools
    elif hasattr(server_tools, "model_dump"):
        counters = server_tools.model_dump()
    else:
        counters = {}
        if server_tools is not None:
            # A shape this code does not know: assume it may carry a surcharge.
            found.append(f"server_tool_use={server_tools!r}")
    if counters:
        found += [
            f"server_tool_use.{key}={value!r}"
            for key, value in counters.items()
            if value not in (None, 0)
        ]
    return tuple(found)


def read_usage(usage: Any) -> TokenCounts:
    """Token counts from an SDK ``Usage`` object or a dict of the same shape.

    Optional fields that are missing or None count as 0.
    """
    breakdown = _get(usage, "cache_creation")
    details = _get(usage, "output_tokens_details")
    return TokenCounts(
        input_tokens=_count(usage, "input_tokens", required=True),
        cache_creation_input_tokens=_count(usage, "cache_creation_input_tokens"),
        cache_read_input_tokens=_count(usage, "cache_read_input_tokens"),
        output_tokens=_count(usage, "output_tokens", required=True),
        cache_write_5m_tokens=(
            None if breakdown is None else _count(breakdown, "ephemeral_5m_input_tokens")
        ),
        cache_write_1h_tokens=(
            None if breakdown is None else _count(breakdown, "ephemeral_1h_input_tokens")
        ),
        thinking_tokens=(
            None if _get(details, "thinking_tokens") is None else _count(details, "thinking_tokens")
        ),
    )


def _cost(model: str, tokens: TokenCounts, prices: Prices) -> CallCost:
    with localcontext(prec=_PRECISION):
        return _cost_exact(model, tokens, prices)


def _cost_exact(model: str, tokens: TokenCounts, prices: Prices) -> CallCost:
    def usd(n: int, per_mtok: Decimal) -> Decimal:
        return n * per_mtok / _MTOK

    write_5m = tokens.cache_write_5m_tokens or 0
    write_1h = tokens.cache_write_1h_tokens or 0
    # Cache writes the TTL breakdown does not account for are priced as the dearer 1h write.
    unattributed = max(tokens.cache_creation_input_tokens - write_5m - write_1h, 0)
    cache_write = usd(write_5m, prices.cache_write_5m) + usd(
        write_1h + unattributed, prices.cache_write_1h
    )
    parts = (
        usd(tokens.input_tokens, prices.input),
        cache_write,
        usd(tokens.cache_read_input_tokens, prices.cache_read),
        usd(tokens.output_tokens, prices.output),
    )
    return CallCost(model, tokens, *parts, total_usd=sum(parts, Decimal(0)))


def check_prompt(model: str, prompt_tokens: int, table: PriceTable | None = None) -> None:
    """Refuse, before the call, a prompt the tool will not pay the higher rate for."""
    _checked_count(prompt_tokens, "prompt_tokens")
    pricing = (table or load_prices()).for_model(model)
    if pricing.prompt_limit is not None and prompt_tokens > pricing.prompt_limit:
        raise PromptTooLongError(model, prompt_tokens, pricing.prompt_limit)


def call_cost(model: str, usage: Any, table: PriceTable | None = None) -> CallCost:
    """Cost of one finished call; ``usage`` is an SDK ``Usage`` or a dict of the same shape."""
    pricing = (table or load_prices()).for_model(model)
    tokens = read_usage(usage)
    modifiers = price_modifiers(usage)
    limit, over_limit = pricing.prompt_limit, pricing.over_limit
    if limit is not None and over_limit is not None and tokens.prompt_tokens > limit:
        cost = _cost(model, tokens, over_limit)
        raise PromptTooLongError(model, tokens.prompt_tokens, limit, cost, modifiers)
    cost = _cost(model, tokens, pricing.prices)
    if modifiers:
        raise PriceModifierError(cost, modifiers)
    if tokens.cache_creation_input_tokens or cost.cache_write_usd:
        raise CacheWriteError(cost)
    return cost
