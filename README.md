# utf-token

Convert random string identifiers to a LLM-friendly format to reduce token usage in certain retrieval and agentic tasks.

Average reduction (can be further reduced with additional parameters, see below):

| Format | Token savings (avg) |
| --- | ---: |
| hex | 54% |
| base64 | 47% |
| uuid | 62% |

The conversion is fully reversible, provided it was originally done with the correct API and stored.

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
# Result: "mittөгөнcoinsientenándonosιαίençSamen"
# 8 tokens: mitt өгөн coins ienten ándonos ιαί enç Samen
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

Reverse lookups still return the full original bytes you encoded, so you can reduce prompt footprint without losing round-trip reversibility.

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

The lookup tables only contain tokens made from *alphanumeric characters and underscore* (`A-Z`, `a-z`, `0-9`, `_`). Encoded identifiers do not contain quotes, slashes, brackets, commas, pipes, whitespace, or other delimiter characters.

This makes the output easier to embed in JSON, Markdown, logs, tables, and prompts where the LLM or code needs to see clearly where an identifier begins and ends.

### Instructions to include in prompts/tools

To avoid confusion when your agent sees these IDs, you can include these instructions:

```text
Identifiers are encoded as compact LLM-friendly strings. They contain only alphanumeric characters from any alphabet plus underscores. Any other special character next to the identifier, such as quotes, slashes, brackets, commas, pipes, whitespace, or other delimiters, marks where the identifier starts or ends. Copy them exactly as written when filling tool arguments or table fields. Do not decode, normalize, translate, split, or reformat them. Some identifiers may resemble real text, it's just a coincidence due to the use of tokens.
```

## How it works

`utf-token` encodes the underlying bytes directly. It reads the byte sequence in 2-byte chunks, treats each chunk as an unsigned integer from 0 to 65,535, and uses that value as an index into a vocabulary-specific token table of size 2^16. If the input has an odd number of bytes, the final byte is encoded with a separate tail table of size 2^8.

This means a 16-byte UUID, hash prefix, or other identifier usually becomes 8 tokenizer tokens instead of the larger number of tokens needed by hex, base64, or UUID text.

The standalone helpers only perform the forward conversion. `IdTokenBiMap` keeps both a forward map and a reverse map so the generated string can be resolved back to the original bytes later.

Collisions can happen when different inputs produce the same encoded string, especially when `truncate_bytes` is used. When `IdTokenBiMap` sees that a new value would collide with an existing value, it deterministically moves to the next byte sequence until it finds an unused encoded string. The stored reverse map still points that generated string back to the original full input.

## Project and release docs

For contributor workflows (testing, packaging, and release process), see `docs/releasing.md`.
