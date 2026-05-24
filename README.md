# claude-cost-py

[![PyPI](https://img.shields.io/pypi/v/claude-cost-py.svg)](https://pypi.org/project/claude-cost-py/)
[![Python](https://img.shields.io/pypi/pyversions/claude-cost-py.svg)](https://pypi.org/project/claude-cost-py/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Compute the USD cost of a Claude API call from its `usage` block.**

Cache-aware (`cache_creation_input_tokens`, `cache_read_input_tokens`),
works for both the Anthropic first-party API and AWS Bedrock model IDs
(versioned `anthropic.claude-sonnet-4-5-v1:0`, inference profiles like
`us.anthropic.claude-haiku-4-5`, and full ARNs). Bring-your-own pricing
override. **Zero dependencies.**

Sibling to the Rust crate
[`claude-cost`](https://crates.io/crates/claude-cost).

## Install

```bash
pip install claude-cost-py
```

## Worked example

A Sonnet 4.5 call with prompt caching enabled:

```python
from claude_cost import default_pricing, Usage

pricing = default_pricing("claude-sonnet-4-5")
usage = Usage(
    input_tokens=423,            # fresh prompt tokens
    output_tokens=18,            # completion
    cache_read_input_tokens=380, # 380 tokens hit the cache
)

cost = pricing.cost_for(usage)
# 423 * 3.00 + 18 * 15.00 + 380 * 0.30
#   = 1269 + 270 + 114
#   = 1653 USD per 1M tokens
#   = 0.001653 USD per call
assert abs(cost - 0.001653) < 1e-9
```

## Bedrock-aware

Same model, four different IDs, all resolve to the same row:

```python
default_pricing("claude-sonnet-4-5")
default_pricing("anthropic.claude-sonnet-4-5-v1:0")
default_pricing("us.anthropic.claude-sonnet-4-5")
default_pricing(
    "arn:aws:bedrock:us-east-1::foundation-model/"
    "anthropic.claude-sonnet-4-5-v1:0"
)
```

`normalize_model_id` is also exposed if you want the canonical name for
your own bookkeeping:

```python
from claude_cost import normalize_model_id

normalize_model_id("us.anthropic.claude-sonnet-4-5")  # "claude-sonnet-4-5"
```

## Bring your own pricing

For a model that isn't in the bundled table, supply rates directly:

```python
from claude_cost import Pricing, Usage

custom = Pricing(
    input_per_mtok=1.25,
    output_per_mtok=5.0,
    cache_read_per_mtok=0.125,
    cache_write_per_mtok=1.5625,
)
usage = Usage(input_tokens=1_000_000)
assert custom.cost_for(usage) == 1.25
```

## Built-in models (0.1.x)

| Model                       | Input | Output | Cache read | Cache write |
|-----------------------------|-------|--------|------------|-------------|
| claude-opus-4-7             | 15.00 | 75.00  | 1.50       | 18.75       |
| claude-sonnet-4-5           | 3.00  | 15.00  | 0.30       | 3.75        |
| claude-haiku-4-5            | 1.00  | 5.00   | 0.10       | 1.25        |
| claude-3-5-sonnet-20241022  | 3.00  | 15.00  | 0.30       | 3.75        |
| claude-3-5-haiku-20241022   | 0.80  | 4.00   | 0.08       | 1.00        |

All USD per 1M tokens. Source: anthropic.com/pricing and
aws.amazon.com/bedrock/pricing as of 2026-Q2. **Verify before billing.**

## What it does NOT do

- **No HTTP client.** This library does not talk to Anthropic or Bedrock.
  Pass it the `usage` block your SDK already returned.
- **No usage tracking.** If you want a ledger (group by request id,
  daily aggregates, etc.), wrap it yourself or compose with
  [`agenttrace-rs`](https://crates.io/crates/agenttrace-rs).
- **No tokenizer.** Token counts come from the API response, not from
  re-tokenizing your prompt.
- **No prompt-caching logic.** Just the cost math given the usage
  numbers the API returns.

## Tests

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest tests/ -v
```

## License

MIT
