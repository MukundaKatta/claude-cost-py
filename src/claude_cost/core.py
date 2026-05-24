"""Core types: Pricing + Usage + default_pricing lookup."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """USD-per-1M-tokens rates for one model.

    All fields are dollars per 1M tokens.  `cost_for(usage)` does the
    math: each token bucket is multiplied by its rate and divided by 1M.

    Cache pricing follows Anthropic's convention:
      * ``cache_read_per_mtok`` is what you pay when the model serves a
        cached prefix back (typically ~10% of input).
      * ``cache_write_per_mtok`` is the premium for writing a new prefix
        into the cache (typically 125% of input).
    """

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    cache_write_per_mtok: float

    def cost_for(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input_per_mtok
            + usage.output_tokens * self.output_per_mtok
            + usage.cache_read_input_tokens * self.cache_read_per_mtok
            + usage.cache_creation_input_tokens * self.cache_write_per_mtok
        ) / 1_000_000.0


@dataclass(frozen=True)
class Usage:
    """Mirrors the shape of Anthropic's ``usage`` block on a response.

    Defaults to zeros so callers can pass only the fields the API returned
    (e.g. cache buckets are absent on calls that don't use prompt caching).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# Pricing table — USD per 1M tokens, source: anthropic.com/pricing and
# aws.amazon.com/bedrock/pricing as of 2026-Q2.  **Verify before billing.**
DEFAULT_PRICING_TABLE: dict[str, Pricing] = {
    "claude-opus-4-7": Pricing(
        input_per_mtok=15.00,
        output_per_mtok=75.00,
        cache_read_per_mtok=1.50,
        cache_write_per_mtok=18.75,
    ),
    "claude-sonnet-4-5": Pricing(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_read_per_mtok=0.30,
        cache_write_per_mtok=3.75,
    ),
    "claude-haiku-4-5": Pricing(
        input_per_mtok=1.00,
        output_per_mtok=5.00,
        cache_read_per_mtok=0.10,
        cache_write_per_mtok=1.25,
    ),
    "claude-3-5-sonnet-20241022": Pricing(
        input_per_mtok=3.00,
        output_per_mtok=15.00,
        cache_read_per_mtok=0.30,
        cache_write_per_mtok=3.75,
    ),
    "claude-3-5-haiku-20241022": Pricing(
        input_per_mtok=0.80,
        output_per_mtok=4.00,
        cache_read_per_mtok=0.08,
        cache_write_per_mtok=1.00,
    ),
}


# Inference-profile prefix patterns that wrap a base model id.  Bedrock
# ships these for cross-region routing and they don't always carry the
# `-v1:0` suffix, so we strip them BEFORE the version-stripper.
_INFERENCE_PROFILE_PREFIX_RE = re.compile(r"^(us|eu|apac|global)\.")

# Bedrock model id is `anthropic.claude-xxxx-v1:0`. The `anthropic.` prefix
# and `-vN:M` suffix both need stripping to get the canonical name.
_ANTHROPIC_PREFIX = "anthropic."
_BEDROCK_VERSION_RE = re.compile(r"-v\d+(?::\d+)?$")

# Bedrock ARN — capture the foundation-model component then re-run.
_BEDROCK_ARN_RE = re.compile(
    r"^arn:aws:bedrock:[^:]*:[^:]*:foundation-model/(?P<id>.+)$"
)


def normalize_model_id(model_id: str) -> str:
    """Strip Bedrock prefixes/suffixes so the result matches a canonical
    Anthropic model name.  All of the following return "claude-sonnet-4-5":

      * "claude-sonnet-4-5"
      * "anthropic.claude-sonnet-4-5-v1:0"
      * "us.anthropic.claude-sonnet-4-5"
      * "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-5-v1:0"
    """
    name = model_id.strip()

    # ARN — extract the foundation-model component, then recurse.
    arn = _BEDROCK_ARN_RE.match(name)
    if arn:
        return normalize_model_id(arn.group("id"))

    # Inference-profile prefix (us. / eu. / apac. / global.)
    name = _INFERENCE_PROFILE_PREFIX_RE.sub("", name)

    # `anthropic.` prefix
    if name.startswith(_ANTHROPIC_PREFIX):
        name = name[len(_ANTHROPIC_PREFIX) :]

    # `-v1:0` / `-v2` style suffix
    name = _BEDROCK_VERSION_RE.sub("", name)

    return name


def default_pricing(model_id: str) -> Pricing | None:
    """Look up the bundled pricing row for ``model_id`` if known.

    Accepts any of the four shapes a caller is likely to pass — bare
    Anthropic name, ``anthropic.foo-v1:0``, ``us.anthropic.foo``, or a
    full Bedrock ARN.  Returns ``None`` for unknown models so callers
    can fall back to their own table without try/except.
    """
    canonical = normalize_model_id(model_id)
    return DEFAULT_PRICING_TABLE.get(canonical)


def known_models() -> list[str]:
    """Canonical names of every model in the bundled pricing table."""
    return sorted(DEFAULT_PRICING_TABLE.keys())
