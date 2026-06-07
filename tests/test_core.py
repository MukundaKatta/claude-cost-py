"""Tests for the Pricing + Usage + model-id normalization."""

from __future__ import annotations

import pytest

from claude_cost import (
    DEFAULT_PRICING_TABLE,
    Pricing,
    Usage,
    default_pricing,
    known_models,
    normalize_model_id,
)

# ---------------------------------------------------------------------------
# normalize_model_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_id,expected",
    [
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("claude-opus-4-7", "claude-opus-4-7"),
        ("claude-haiku-4-5", "claude-haiku-4-5"),
        # Bedrock versioned IDs
        ("anthropic.claude-sonnet-4-5-v1:0", "claude-sonnet-4-5"),
        ("anthropic.claude-haiku-4-5-v1:0", "claude-haiku-4-5"),
        ("anthropic.claude-opus-4-7-v1", "claude-opus-4-7"),
        # Inference-profile prefixes
        ("us.anthropic.claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("eu.anthropic.claude-haiku-4-5", "claude-haiku-4-5"),
        ("apac.anthropic.claude-sonnet-4-5-v1:0", "claude-sonnet-4-5"),
        ("global.anthropic.claude-sonnet-4-5", "claude-sonnet-4-5"),
        # Bedrock ARNs
        (
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-v1:0",
            "claude-sonnet-4-5",
        ),
        (
            "arn:aws:bedrock:eu-west-1:123456789012:foundation-model/"
            "anthropic.claude-opus-4-7-v1:0",
            "claude-opus-4-7",
        ),
        # ARN wrapping an inference-profile-prefixed id (region-routed)
        (
            "arn:aws:bedrock:us-east-1::foundation-model/us.anthropic.claude-sonnet-4-5-v1:0",
            "claude-sonnet-4-5",
        ),
        # Whitespace tolerance
        ("  claude-sonnet-4-5  ", "claude-sonnet-4-5"),
    ],
)
def test_normalize_model_id(input_id: str, expected: str):
    assert normalize_model_id(input_id) == expected


def test_normalize_unknown_model_passes_through():
    """An unknown model name should NOT be silently mangled — the function
    is a normalizer, not a validator."""
    out = normalize_model_id("future-model-xyz")
    assert out == "future-model-xyz"


# ---------------------------------------------------------------------------
# default_pricing
# ---------------------------------------------------------------------------


def test_default_pricing_known_model():
    p = default_pricing("claude-sonnet-4-5")
    assert p is not None
    assert p.input_per_mtok == 3.00
    assert p.output_per_mtok == 15.00
    assert p.cache_read_per_mtok == 0.30
    assert p.cache_write_per_mtok == 3.75


def test_default_pricing_haiku_4_5_rates():
    p = default_pricing("claude-haiku-4-5")
    assert p is not None
    assert p.input_per_mtok == 1.00
    assert p.output_per_mtok == 5.00
    assert p.cache_read_per_mtok == 0.10
    assert p.cache_write_per_mtok == 1.25


def test_default_pricing_resolves_bedrock_variants():
    """Same canonical model, four different IDs — all return the same row."""
    base = default_pricing("claude-sonnet-4-5")
    assert base is not None
    assert default_pricing("anthropic.claude-sonnet-4-5-v1:0") == base
    assert default_pricing("us.anthropic.claude-sonnet-4-5") == base
    assert (
        default_pricing(
            "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-v1:0"
        )
        == base
    )


def test_default_pricing_unknown_returns_none():
    assert default_pricing("claude-not-real-9000") is None
    assert default_pricing("gpt-5.4") is None


def test_known_models_lists_all_in_table():
    names = known_models()
    assert set(names) == set(DEFAULT_PRICING_TABLE.keys())
    assert "claude-sonnet-4-5" in names
    assert "claude-opus-4-7" in names


# ---------------------------------------------------------------------------
# Pricing.cost_for — the actual math
# ---------------------------------------------------------------------------


def test_cost_for_simple_call_no_cache():
    p = default_pricing("claude-sonnet-4-5")
    assert p is not None
    usage = Usage(input_tokens=1000, output_tokens=500)
    # 1000 * 3.00 + 500 * 15.00 = 3000 + 7500 = 10500 USD per 1M
    # / 1M = 0.0105 USD per call
    assert p.cost_for(usage) == pytest.approx(0.0105, abs=1e-9)


def test_cost_for_cache_aware_call():
    """Worked example from the README: Sonnet 4.5 with prompt caching."""
    p = default_pricing("claude-sonnet-4-5")
    assert p is not None
    usage = Usage(
        input_tokens=423,
        output_tokens=18,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=380,
    )
    # 423 * 3.00 + 18 * 15.00 + 380 * 0.30 = 1269 + 270 + 114 = 1653
    # / 1M = 0.001653 USD per call
    assert p.cost_for(usage) == pytest.approx(0.001653, abs=1e-9)


def test_cost_for_zero_usage_is_zero():
    p = default_pricing("claude-sonnet-4-5")
    assert p is not None
    assert p.cost_for(Usage()) == 0.0


def test_custom_pricing_round_trips_full_1m():
    """1M input tokens at $1.25/M should be exactly $1.25."""
    custom = Pricing(
        input_per_mtok=1.25,
        output_per_mtok=5.0,
        cache_read_per_mtok=0.125,
        cache_write_per_mtok=1.5625,
    )
    usage = Usage(input_tokens=1_000_000)
    assert custom.cost_for(usage) == pytest.approx(1.25, abs=1e-12)


def test_cost_scales_linearly_with_volume():
    """Sanity: 2x the tokens, 2x the dollar amount."""
    p = default_pricing("claude-haiku-4-5")
    assert p is not None
    a = p.cost_for(Usage(input_tokens=1000, output_tokens=500))
    b = p.cost_for(Usage(input_tokens=2000, output_tokens=1000))
    assert b == pytest.approx(2 * a, abs=1e-12)


def test_cache_write_costs_more_than_input():
    """Sanity: cache_write rate should always exceed plain input rate
    for every bundled model (writes carry the caching premium)."""
    for name, p in DEFAULT_PRICING_TABLE.items():
        assert p.cache_write_per_mtok > p.input_per_mtok, (
            f"{name}: cache_write should cost more than plain input"
        )


def test_cache_read_costs_less_than_input():
    """Sanity: cache_read should always be cheaper than plain input —
    that's the whole point of prompt caching."""
    for name, p in DEFAULT_PRICING_TABLE.items():
        assert p.cache_read_per_mtok < p.input_per_mtok, (
            f"{name}: cache_read should cost less than plain input"
        )
