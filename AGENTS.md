---
description: 
alwaysApply: true
---

# UTF-Token utility

This repo consists of a Python library (planned for Typescript) that converts random bytes to a compact LLM-friendly string representation. The goals is to reduce token usage for LLM calls that require the model to read/write large amounts of common random strings, such as hashes, UUIDs and other identifiers.

## Tech stack

- Python 3.13 (no TS env for now)

- Dev tools:
  - uv (package manager), ruff (linter), ty (type checker)
  - ipykernel
  - pandas
  - tiktoken, sentencepiece

## Rules

- Check Python code using ruff and ty when you have finished working, before execution/tests.
- Run tests with `uv run -m unittest discover -s tests`.
- Never use `pip install`, always use `uv add`.
- Never use plain `python file.py`, always use `uv run file.py`.
- Never use `hasattr` or `getattr`, always use static checks, such as `uv run ruff check` and `uv run ty check`.
- When changing which source or data files are packaged, locally verify the build with `uv build --no-build-isolation --sdist --wheel`, then install the built wheel into a fresh virtualenv and run `uv run --no-project --python <venv>/bin/python --script scripts/smoke_installed_package.py`.
- Avoid try-except blocks for offline scripts if possible.

## Repo structure

- `scripts/`: for offline work, such as building our special decoding tables.
- `src/utf_token/`: Python library code (importable as `utf_token`).
- `src/utf_token/data/`: lookup tables and metadata that ship with the package.
- `tests/`: unit tests for the library and offline scripts.
- `data/token_vocab/`: (gitignored) downloaded token vocabs to process in further scripts.
- `data/lookup_tables/`: processed token vocabs ready to be used as lookup tables (mirrored into `src/utf_token/data/` for packaging).

## Decoding tables

We work with the o200k OpenAI and Gemma4 token vocabs. For each vocab, [`scripts/process_token_vocab.py`](scripts/process_token_vocab.py) tries an ordered list of cleanup recipes and picks the first one that supplies enough tokens for both a pair table and a 256-entry tail table:

1. `latin_16bit`: UTF-8 printable, alphanumeric or `_`, at most 6 characters long, plus precomposed letters from the Latin-1 Supplement (`U+00C0`-`U+00FF`) and Latin Extended-A (`U+0100`-`U+017F`) blocks. Targets `2^16 + 2^8` tokens (16-bit pair index).
2. `ascii_15bit`: ASCII alphanumeric or `_` only, at most 6 characters long. Targets `2^15 + 2^8` tokens (15-bit pair index). Used as a fallback when a vocab does not have enough clean tokens for the 16-bit recipe.

Currently both `o200k` and `gemma4` land on `ascii_15bit`. The character and length policies are deliberate trade-offs: NIAH-style benchmarks showed that smaller LLMs mistranscribed identifiers that mixed unfamiliar scripts (Devanagari, Arabic, Cuneiform, Fullwidth or Mathematical Latin, etc.) or long, low-probability token fragments. The curated `latin_16bit` policy intentionally excludes those scripts, and the gemma4 vocab does not contain enough qualifying tokens to fill a 16-bit pair table, so it falls back to ASCII-only just like `o200k`. We sacrifice a small amount of efficiency for better visual reliability.

Each generated table is paired with a JSON metadata file that records the recipe, `pair_table_size`, `pair_index_bits`, `tail_table_size`, `tail_index_bits`, allowed character policy, maximum token length (`max_token_length`), and the inclusive non-ASCII Unicode codepoint ranges (`allowed_non_ascii_ranges`). Filenames are sized after the actual pair-table size so mixed sizes can coexist:

- `data/lookup_tables/o200k_base_32768_tokens.txt`, `o200k_base_32768_tail_256_tokens.txt`, `o200k_base_32768_metadata.json`.
- `data/lookup_tables/tokenizer_gemma4_32768_tokens.txt`, `tokenizer_gemma4_32768_tail_256_tokens.txt`, `tokenizer_gemma4_32768_metadata.json`.

Each table file has 1 row per token in a single column. The runtime (in [`src/utf_token/_tables.py`](src/utf_token/_tables.py) and [`src/utf_token/_api.py`](src/utf_token/_api.py)) reads `pair_index_bits` from metadata and dispatches to the appropriate path:

- For `pair_index_bits == 16`, each consecutive 2-byte chunk indexes the pair table and an odd trailing byte indexes the 256-entry tail table. (No shipped vocab currently uses this fast path; it is reserved for any future vocab that supplies enough qualifying tokens for the `latin_16bit` recipe.)
- For `pair_index_bits < 16` (e.g. `o200k` and `gemma4` at 15 bits), the input is consumed as an MSB-first bitstream in `pair_index_bits` chunks. A 1–8 bit residual indexes the tail table; a 9 to `pair_index_bits - 1` bit residual indexes the pair table (left-padded with zeros).

When changing recipes or filenames, also update [`src/utf_token/_tables.py`](src/utf_token/_tables.py) (`VOCAB_FILENAMES`), [`setup.py`](setup.py) (`PACKAGE_DATA_FILENAMES`), and [`scripts/smoke_installed_package.py`](scripts/smoke_installed_package.py) so the wheel ships the right data.

## Standalone functions vs IdTokenBiMap class

Both the standalone functions and the class implement:

- `frombytes`
- `fromhex`
- `frombase64`
- `fromuuid`

But the standalone functions are forward-only. From the resulting string of concatenated tokens, you can't map the original bytes. Since in most cases an LLM would write a transformed identifier back to the application, and this in turns needs to retrieve the original identifier, the class is necessary to achieve this. The class stored an intermal map with collision resolution and implements:

- `tobytes`
- `tohex`
- `tobase64`
- `touuid`

## Roadmap

Typescript npm package: implementation pending.

