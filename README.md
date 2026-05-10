# utf-token

Convert random string identifiers to a LLM-friendly format to reduce token usage in certain retrieval and agentic tasks.

Average reduction over 16-byte identifiers (you can save a lot more with additional parameters, see below):

| Original format | Token savings average |
| --- | ---: |
| hex | 46% |
| base64 | 39% |
| uuid | 56% |

The conversion is fully reversible (see below).

## Python API

Install with `uv add utf-token`.

The standalone functions are the simplest API if you prefer to store a reverse map yourself.

Example with 16-byte decoded as hex:

```python
from utf_token import fromhex

my_hex = "215aada34d0987ebfb9de132d913e46b"
# 17 tokens: 215 a ada 34 d 098 7 eb fb 9 de 132 d 913 e 46 b

encoded_hex = fromhex(my_hex)
print(encoded_hex)
# Result: "udadreibenSnakeCheleighALLEhooamarqu"
# 9 tokens: udad reiben Snake Che leigh ALLE hoo amar qu
```

Supported formats: `frombytes`, `fromhex`, `frombase64`, `fromuuid`.

## Reversible mapping

The `IdTokenBiMap` class stores the reverse map for each string processed for easy inversion.

```python
from utf_token import IdTokenBiMap

bimap = IdTokenBiMap()

uuid_tokens = bimap.fromuuid("123e4567-e89b-12d3-a456-426614174000")
original_uuid = bimap.touuid(uuid_tokens)
```

Forward methods on the class: `frombytes`, `fromhex`, `frombase64`, `fromuuid`.
Reverse methods on the class: `tobytes`, `tohex`, `tobase64`, `touuid`.

Both forward and reverse methods accept either:

- a single value -> returns one encoded `str`
- an iterable of values -> returns a lazy iterator of encoded `str`

### Save more tokens with `truncate_bytes`

All forward methods also accept `truncate_bytes=<positive int>` to encode only the first N bytes of the input.

```python
token = bimap.frombytes(b"\x01\x02\x03\x04\x05\x06", truncate_bytes=4)
```

Reverse lookups still return the *full original identifier* pre-truncation, so you can reduce token usage without losing round-trip reversibility.

### Healing transcription errors on reverse lookup

LLMs occasionally make transcription errors when copying identifiers. Reverse methods accept an `errors` keyword to control what happens when the input is not an exact match in the reverse map:

- `errors="fix"` (default): the closest previously encoded identifier (by Levenshtein distance) is returned.
- `errors="raise"`: if the exact lookup misses, raise `KeyError`. Useful for if you want to manage error handling yourself.

```python
bimap = IdTokenBiMap()
encoded = bimap.fromuuid("123e4567-e89b-12d3-a456-426614174000")

bimap.touuid(encoded)                                    # exact match
bimap.touuid(encoded[:-1] + "Z")                         # heals to nearest stored id

bimap.touuid("not_a_real_id", errors="raise")            # raises KeyError

if encoded in bimap:                                     # Supports membership checks
    print("This will print")
```

The make the healing work with very high reliability, don't use a very low value of `truncate_bytes`. Keep a minimum of 3 bytes to avoid false matches.

### Persisting the reversible map

The internal map in `IdTokenBiMap` can be saved and restored:

- `to_dict` / `from_dict`
- `to_json` / `from_json`

## Supported token vocabularies

You should pick the token vocabulary that matches the model you are using. Current options are:

- Default: `o200k` (OpenAI GPT-5+)
- `gemma4` (Google Gemma 4)

You can specify the vocabulary when creating the `IdTokenBiMap` instance:

```python
bimap = IdTokenBiMap(vocab="gemma4")
```

For the standalone functions, you specify the `vocab` parameter in the function call.

## Included safe character set in tokens

The lookup tables only contain tokens made from alphanumeric characters and underscore (`_`).

Both `o200k` and `gemma4` are restricted to ASCII (`A-Z`, `a-z`, `0-9`, `_`) to avoid LLM confusion. The build pipeline first tries a curated Latin-script policy aimed at a 16-bit pair table, but neither shipped vocab supplies enough qualifying tokens, so both fall back to the 15-bit ASCII-only recipe. This trades a small efficiency loss (15-bit pair table) for higher transcription reliability on smaller LLMs that mistranscribe identifiers containing Devanagari, Arabic, Cuneiform, Mathematical or Fullwidth Latin variants.

Neither vocabulary emits quotes, slashes, brackets, commas, pipes, whitespace, or other delimiter characters, which makes the output easy to embed in JSON, Markdown, logs, tables, and prompts where the LLM or code needs to see clearly where an identifier begins and ends.

### Instructions to include in prompts/tools

To avoid confusion when your agent sees these IDs, you can adapt these instructions to your specific use case:

> Identifiers are random LLM token sequences containing only ASCII alphanumeric or `_` characters. They are delimited by `<insert your delimiters here>`. Some identifiers may contain words or part of words, it's just a coincidence due to the use of tokens. Do not translate or fix typos in the identifiers. Transcribe them **verbatim**.

#### Other recommendations for maximum reliability in identifier retrieval

1. Use low temperature if the model supports it.
2. Use structured outputs / JSON tools to request the identifiers. Provide a regex pattern such as `^[A-Za-z0-9_]+$` for the output strings in the JSON schema.
3. Use truncate_bytes to reduce identifier size and chances of error, while also reducing tokens and latency. But keep a minimum value of 3.
4. Use consistent delimiters to clearly separate identifiers from other text in the prompt.

## Retrieval benchmark

A NIAH-style benchmark is included to test small LLMs (GPT-5.4-nano, Gemma 4, Claude Haiku) on retrieval accuracy. With 100 samples for each model, and both full and truncated identifiers, the success rate is 100%. The context length is 32k tokens (calibrated for hex identifiers, then re-encoded for each encoding), and identifiers have 16 bytes of entropy.

See [`docs/benchmarks_niah.md`](docs/benchmarks_niah.md).

The synthetic NIAH benchmark was adapted from [NVIDIA/RULER](https://github.com/NVIDIA/RULER).

## How it works

`utf-token` encodes the underlying bytes directly. Each vocabulary ships two pre-built lookup tables, generated offline by [`scripts/process_token_vocab.py`](scripts/process_token_vocab.py): a large pair table indexed by either 15 or 16 bits (depending on how many clean tokens the vocabulary can supply) and a small tail table indexed by 8 bits.

For 15-bit pair tables (both shipped vocabs) the encoder treats the input as an MSB-first bitstream, splits it into 15-bit chunks for the pair table, and uses the tail table for any 1–8 bit residual at the end. A 16-bit fast path is also implemented for any future vocabulary that can fill a 16-bit pair table under the curated `latin_16bit` recipe.

That makes a 16-byte UUID typically become 9 tokens, instead of the much larger number of tokens needed by hex, base64, or canonical UUID text.

The standalone helpers only perform the forward conversion. `IdTokenBiMap` keeps both a forward map and a reverse map so the generated string can be resolved back to the original bytes later.

Collisions can happen when different inputs produce the same encoded string, especially when `truncate_bytes` is used. When `IdTokenBiMap` sees that a new value would collide with an existing value, it deterministically moves to the next byte sequence until it finds an unused encoded string. The stored reverse map still points that generated string back to the original full input.

## Project and release docs

For contributor workflows (testing, packaging, and release process), see `docs/releasing.md`.
