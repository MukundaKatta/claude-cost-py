"""claude-cost — compute the USD cost of a Claude API call from its usage block.

Cache-aware (cache_creation_input_tokens, cache_read_input_tokens), works for
both the Anthropic first-party API and AWS Bedrock model IDs (versioned
`anthropic.claude-sonnet-4-5-v1:0`, inference profiles like
`us.anthropic.claude-haiku-4-5`, and full ARNs). Bring-your-own pricing
override. No SDK dependency.

Sibling to the Rust crate ``claude-cost`` (https://crates.io/crates/claude-cost).

Usage:

    from claude_cost import default_pricing, Usage

    pricing = default_pricing("claude-sonnet-4-5")
    usage = Usage(
        input_tokens=423,
        output_tokens=18,
        cache_read_input_tokens=380,
    )
    print(pricing.cost_for(usage))  # 0.001653 USD
"""

from .core import (
    DEFAULT_PRICING_TABLE,
    Pricing,
    Usage,
    default_pricing,
    known_models,
    normalize_model_id,
)

__all__ = [
    "DEFAULT_PRICING_TABLE",
    "Pricing",
    "Usage",
    "default_pricing",
    "known_models",
    "normalize_model_id",
]

__version__ = "0.1.0"
