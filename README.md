# utf-token

Convert random string identifiers to a LLM-friendly format to reduce token usage in certain retrieval and agentic tasks.

`utf-token` encodes only the first three bytes of each identifier by default, so any payload — UUIDs, hex hashes, base64 ids — typically becomes about two `o200k_base` tokens regardless of its original length:

![token savings vs hex, base64, and uuid](docs/assets/notebooks/token_savings.png)

The conversion stays fully reversible through `IdTokenBiMap`, which stores the original bytes behind every generated string.

## Install

```shell
uv add utf-token
```

## Reversible mapping with `IdTokenBiMap`

`IdTokenBiMap` is the main API. It encodes identifiers and stores the full original bytes so you can recover them later, even when only a prefix was actually encoded.

```python
from utf_token import IdTokenBiMap

bimap = IdTokenBiMap()

uuid_tokens = bimap.fromuuid("123e4567-e89b-12d3-a456-426614174000")
original_uuid = bimap.touuid(uuid_tokens)
```

Forward methods: `frombytes`, `fromhex`, `frombase64`, `fromuuid`.
Reverse methods: `tobytes`, `tohex`, `tobase64`, `touuid`.

Both forward and reverse methods accept either:

- a single value -> returns one encoded `str` (or recovered value)
- an iterable of values -> returns a lazy iterator

### Controlling how many bytes are encoded with `keep_bytes`

Each forward method takes a `keep_bytes` keyword:

- omitted (default): encode the first **3** bytes of the input
- a positive integer: encode that many leading bytes
- `None` or `"all"`: encode the full input

```python
short = bimap.frombytes(b"\x01\x02\x03\x04\x05\x06")            # encodes first 3 bytes
longer = bimap.frombytes(b"\x01\x02\x03\x04\x05\x06", keep_bytes=4)
full = bimap.frombytes(b"\x01\x02\x03\x04\x05\x06", keep_bytes="all")

bimap.tobytes(short) == b"\x01\x02\x03\x04\x05\x06"             # reverse returns the full input
```

Three bytes is enough entropy for retrieval workloads where you only need a handful of distinct identifiers visible to the model at once, and is also the minimum we recommend for the healing logic described below to stay reliable. Pass a larger value if you need more in-context disambiguation.

### Healing transcription errors on reverse lookup

LLMs occasionally make transcription errors when copying identifiers. Reverse methods accept an `errors` keyword to control what happens when the input is not an exact match in the reverse map:

- `errors="fix"` (default): return the closest previously encoded identifier by Levenshtein distance.
- `errors="raise"`: if the exact lookup misses, raise `KeyError`. Useful when you want to manage error handling yourself.

```python
bimap = IdTokenBiMap()
encoded = bimap.fromuuid("123e4567-e89b-12d3-a456-426614174000")

bimap.touuid(encoded)                                    # exact match
bimap.touuid(encoded[:-1] + "Z")                         # heals to nearest stored id

bimap.touuid("not_a_real_id", errors="raise")            # raises KeyError

if encoded in bimap:                                     # supports membership checks
    print("This will print")
```

### Persisting the reversible map

The internal map in `IdTokenBiMap` can be saved and restored:

- `to_dict` / `from_dict`
- `to_json` / `from_json`

`from_dict` accepts `null`, a positive integer, or the string `"all"` for each `keep_bytes` value. The canonical export always uses `null` for full-input encodings.

## Standalone forward-only helpers

`frombytes`, `fromhex`, `frombase64`, and `fromuuid` are also available as standalone module-level functions. They perform only the forward conversion, and they default to keeping the full input rather than truncating. They are useful when you want to plug `utf-token` into your own data flow or build your own reverse-lookup table:

```python
from utf_token import fromhex

my_hex = "215aada34d0987ebfb9de132d913e46b"
encoded_hex = fromhex(my_hex)                            # full input
short_hex = fromhex(my_hex, keep_bytes=3)                # only first 3 bytes
```

Both `keep_bytes=None` and `keep_bytes="all"` keep the full input.

## Supported token vocabularies

Pick the token vocabulary that matches the model you are using. Current options are:

- Default: `o200k` (OpenAI GPT-5+)
- `gemma4` (Google Gemma 4)

```python
bimap = IdTokenBiMap(vocab="gemma4")
```

For the standalone functions, pass the `vocab` parameter in the call.

## Included safe character set in tokens

Both `o200k` and `gemma4` lookup tables are restricted to ASCII (`A-Z`, `a-z`, `0-9`, `_`) to avoid LLM confusion.

Neither vocabulary emits quotes, slashes, brackets, commas, pipes, whitespace, or other delimiter characters, which makes the output easy to embed in JSON, Markdown, logs, tables, and prompts where the LLM or code needs to see clearly where an identifier begins and ends.

### Instructions to include in prompts/tools

To avoid confusion when your agent sees these IDs, you can adapt these instructions to your specific use case:

> Identifiers are random LLM token sequences containing only ASCII alphanumeric or `_` characters. They are delimited by `<insert your delimiters here>`. Some identifiers may contain words or part of words, it's just a coincidence due to the use of tokens. Do not translate or fix typos in the identifiers. Transcribe them **verbatim**.

#### Other recommendations for maximum reliability in identifier retrieval

1. Use consistent delimiters to clearly separate identifiers from other text in the prompt.
2. Keep the default `keep_bytes=3` (or higher) so the healing logic has enough signal to disambiguate identifiers.
3. Use structured outputs / JSON tools to request the identifiers. Provide a regex pattern such as `^[A-Za-z0-9_]+$` for the output strings in the JSON schema.
4. Use smart models. For OpenAI, use at least GPT-5.4-mini (not nano). For Gemini, use at least Gemma 4. For Anthropic, use at least Haiku 4.5.
5. Use low temperature if the model supports it.

## Retrieval benchmark

A NIAH-style benchmark is included to test small LLMs (GPT-5.4-mini, Gemma 4, Claude Haiku) on retrieval accuracy. With 100 samples for each model, and both full-input and `keep_bytes=3` identifiers, the success rate is 100%. The context length is 32k tokens (calibrated for hex identifiers, then re-encoded for each encoding), and identifiers have 16 bytes of entropy.

See [`docs/benchmarks_niah.md`](docs/benchmarks_niah.md).

The synthetic NIAH benchmark was adapted from [NVIDIA/RULER](https://github.com/NVIDIA/RULER).

## How it works

`utf-token` encodes the underlying bytes directly. Each vocabulary ships two pre-built lookup tables, generated offline by [`scripts/process_token_vocab.py`](scripts/process_token_vocab.py): a large pair table indexed by either 15 or 16 bits (depending on how many clean tokens the vocabulary can supply) and a small tail table indexed by 8 bits.

For 15-bit pair tables (both shipped vocabs) the encoder treats the input as an MSB-first bitstream, splits it into 15-bit chunks for the pair table, and uses the tail table for any 1–8 bit residual at the end. A 16-bit fast path is also implemented for any future vocabulary that can fill a 16-bit pair table under the curated `latin_16bit` recipe.

`IdTokenBiMap` keeps a forward map and a reverse map so the generated string can be resolved back to the original bytes later. The default `keep_bytes=3` means each identifier consumes about two tokens regardless of its original size; reverse lookups still return the full bytes that were passed in.

Collisions can happen when different inputs produce the same encoded string, especially when `keep_bytes` truncates them to a short prefix. When `IdTokenBiMap` sees that a new value would collide with an existing one, it deterministically moves to the next byte sequence until it finds an unused encoded string. The stored reverse map still points that generated string back to the original full input.

## Reproducing the savings plot

```shell
uv run --group offline scripts/token_savings_examples.py
```

This regenerates [`docs/assets/notebooks/token_savings.png`](docs/assets/notebooks/token_savings.png) from random payloads of 4 to 32 bytes.

## Project and release docs

For contributor workflows (testing, packaging, and release process), see `docs/releasing.md`.
