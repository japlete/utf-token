# NIAH Identifier Benchmark

This benchmark checks whether `utf-token`-encoded random identifiers hurt long-context
retrieval accuracy. The default run compares full-input `utf-token` identifiers
(`keep_bytes="all"`) with the library's default `keep_bytes=3` encoding; raw `hex`,
`base64`, and `uuid` baselines are available with `--all-encodings`.

It is inspired by Needle-In-A-Haystack (NIAH): a target identifier is embedded among many
distractor key/id records, then the model must return the identifier for one requested key as
structured JSON.

## Scenario

The initial benchmark intentionally keeps one fixed scenario and varies only model and
identifier encoding:

- Task: single needle, single query, one JSON string field named `id`.
- Baseline size: approximately 32,000 `o200k_base` tokens for hex records. The runner
  converts this to a hex character target, estimates the corresponding number of records, then
  keeps that record count fixed for every identifier format.
- Needle depth: 50% of the generated record list.
- Payload size: 16 bytes, so every sample has a UUID representation.
- Samples per cell: 100 by default.
- Default encodings: `utf_token` (full-input) and `utf_token_keep_3` (the library default).
- Total default calls: 2 encodings x 3 models x 100 samples = 600 OpenRouter calls.
- With `--all-encodings`: 5 encodings x 3 models x 100 samples = 1,500 OpenRouter calls.

Each distractor uses the same identifier encoding as the needle. The same number of records is
used for every encoding, so token savings show up in provider usage rather than as more
distractor rows. Prompt character count is still logged, but it is not the primary savings
metric for `utf-token` strings.

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

Run the benchmark with raw `hex`, `base64`, and `uuid` baselines included:

```shell
uv run scripts/benchmarks/run_niah_identifier_benchmark.py --all-encodings
```

Resume a partially completed run:

```shell
uv run scripts/benchmarks/run_niah_identifier_benchmark.py --resume
```

Write one complete generated prompt and exit:

```shell
uv run scripts/benchmarks/run_niah_identifier_benchmark.py --write-prompt-example
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
    "id": {
      "type": "string",
      "pattern": "^[A-Za-z0-9_]+$",
      "minLength": 1,
      "maxLength": 90
    }
  },
  "required": ["id"],
  "additionalProperties": false
}
```

The `pattern` and `maxLength` are selected per encoding: raw hex, base64, UUID, and the two
`utf-token` encodings each use their own schema constraints.

The per-call JSONL row includes:

- run id, model slug, encoding, sample index, seed, payload hex, needle key, and prompt hash.
- rendered needle value, raw model response, and extracted response.
- exact match, normalized match, and format-valid flags.
- fixed record count, context targets, depth, prompt character count, latency, OpenRouter usage,
  and error field.
  Provider-reported prompt tokens are available in the OpenRouter usage payload when the model
  returns them.

The summary groups results by model and encoding. The main decision metric is normalized
accuracy. For the `utf-token` encodings, normalized matching decodes the model output with
`errors="fix"` and compares the recovered bytes with the original payload.

## Notes

- Record count is derived from a hex baseline for speed. At startup the runner tokenizes one
  hex calibration prompt with `tiktoken`'s `o200k_base`, computes average characters per token,
  uses that to estimate the number of hex records for the target size, then renders every
  encoding with that same record count. OpenRouter providers do not expose one shared tokenizer,
  and repeated full-prompt tokenization is too slow for this benchmark.
- The runner uses bounded retries for HTTP 429, network transport errors, and transient 5xx
  responses.
- Temperature is fixed at 0.0 and the answer completion budget defaults to 64 tokens. If a
  future Google Gemini Pro model is added to the model list, the runner also reserves a small
  excluded reasoning budget for that model family.
- The benchmark is intentionally narrow. It does not sweep context lengths or needle depths, and
  it does not test multi-hop or tool-use agent workflows.

## Attribution

This benchmark adapts the NIAH evaluation idea, not source code, from:

- [gkamradt/LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
- [NVIDIA/RULER](https://github.com/NVIDIA/RULER)
- [RULER paper, arXiv 2404.06654](https://arxiv.org/abs/2404.06654)

If future changes copy prompt templates or generation code from those repositories, preserve the
relevant MIT or Apache-2.0 license notices with the adapted files.
