# NIAH Identifier Benchmark

This benchmark checks whether `utf-token`-encoded random identifiers hurt long-context
retrieval accuracy compared with raw `hex`, `base64`, and `uuid` strings.

It is inspired by Needle-In-A-Haystack (NIAH): a target identifier is embedded among many
distractor key/id records, then the model must return the identifier for one requested key as
structured JSON.

## Scenario

The initial benchmark intentionally keeps one fixed scenario and varies only model and
identifier encoding:

- Task: single needle, single query, one JSON string field named `identifier`.
- Baseline size: approximately 32,000 `o200k_base` tokens for hex records. The runner
  converts this to a hex character target, estimates the corresponding number of records, then
  keeps that record count fixed for every identifier format.
- Needle depth: 50% of the generated record list.
- Payload size: 16 bytes, so every sample has a UUID representation.
- Samples per cell: 20 by default.
- Total default calls: 4 encodings x 3 models x 20 samples = 240 OpenRouter calls.

Each distractor uses the same identifier encoding as the needle. The same number of records is
used for every encoding, so token savings show up in provider usage rather than as more
distractor rows. Prompt character count is still logged, but it is not the primary savings
metric for non-ASCII `utf-token` strings.

## Models

The runner uses OpenRouter's OpenAI-compatible chat completions endpoint:

- `openai/gpt-5.4-nano` with the `o200k` `utf-token` table.
- `google/gemma-4-26b-a4b-it` with the `gemma4` `utf-token` table.
- `anthropic/claude-haiku-4.5` with the `o200k` `utf-token` table as a cross-provider baseline.

Set the API key before running:

```shell
export OPENROUTER_API_KEY=...
```

## Run

Install all development and offline dependencies:

```shell
uv sync --all-groups
```

Dry-run prompt generation without network calls:

```shell
uv run scripts/benchmarks/run_niah_identifier_benchmark.py --dry-run --samples-per-cell 1
```

Run the default benchmark:

```shell
uv run scripts/benchmarks/run_niah_identifier_benchmark.py
```

Resume a partially completed run:

```shell
uv run scripts/benchmarks/run_niah_identifier_benchmark.py --resume
```

Outputs are written to:

- `docs/assets/benchmarks/niah_identifier_runs.jsonl`
- `docs/assets/benchmarks/niah_identifier_summary.csv`
- `docs/assets/benchmarks/niah_identifier_summary.md`

The JSONL file can be large and is ignored by git. Summary files are small enough to keep when
you want to publish a result.

## Metrics

The runner requests OpenRouter structured outputs with this JSON Schema:

```json
{
  "type": "object",
  "properties": {
    "identifier": {
      "type": "string"
    }
  },
  "required": ["identifier"],
  "additionalProperties": false
}
```

The per-call JSONL row includes:

- model slug, encoding, sample index, seed, payload hex, prompt hash.
- rendered needle value and model response.
- exact match, normalized match, and format-valid flags.
- fixed record count, prompt character count, latency, OpenRouter usage, and error field.
  Provider-reported prompt tokens are available in the OpenRouter usage payload when the model
  returns them.

The summary groups results by model and encoding. The main decision metric is normalized
accuracy: `utf_token` should not trail the raw formats for the same model by enough to offset
its token savings.

## Notes

- Record count is derived from a hex baseline for speed. At startup the runner tokenizes one
  hex calibration prompt with `tiktoken`'s `o200k_base`, computes average characters per token,
  uses that to estimate the number of hex records for the target size, then renders every
  encoding with that same record count. OpenRouter providers do not expose one shared tokenizer,
  and repeated full-prompt tokenization is too slow for this benchmark.
- The runner uses bounded retries for HTTP 429, network transport errors, and transient 5xx
  responses.
- Temperature is fixed at 0.0 and the completion budget defaults to 64 tokens.
- The benchmark is intentionally narrow. It does not sweep context lengths or needle depths, and
  it does not test multi-hop or tool-use agent workflows.

## Attribution

This benchmark adapts the NIAH evaluation idea, not source code, from:

- [gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
- [RULER paper, arXiv 2404.06654](https://arxiv.org/abs/2404.06654)

If future changes copy prompt templates or generation code from those repositories, preserve the
relevant MIT or Apache-2.0 license notices with the adapted files.
