# UTF-Token utility

This repo consists of a library (planned for Python and Typescript) that converts random bytes to a compact LLM-friendly string representation. The goals is to reduce token usage for LLM calls that require the model to read/write large amounts of common random strings, such as hashes, UUIDs and other identifiers.

## Tech stack

- Python 3.13 (no TS env for now)

- Dev tools:
  - uv (package manager), ruff (linter), ty (type checker)
  - ipykernel
  - pandas
  - tiktoken

## Rules

- Check Python code using ruff and ty when you have finished working, before execution/tests.
- Run tests with `uv run -m unittest discover -s tests`.
- Never use `pip install`, always use `uv add`.
- Never use plain `python file.py`, always use `uv run file.py`.
- Never use `hasattr` or `getattr`, always use static checks, such as `uv run ruff check` and `uv run ty check`.
- Avoid try-except blocks for offline scripts if possible.

## Repo structure

- `scripts/`: for offline work, such as building our special decoding tables.
- `src/utf-token-py/`: Python library code. Empty for now.
- `data/token_vocab/`: (gitignored) downloaded token vocabs to process in further scripts.
- `data/lookup_tables`: processed token vocabs ready to be used as lookup tables.

## Decoding tables

For now, we work only with the o200k OpenAI token vocabulary. We take a subset of size 2^16 and build a lookup table for each 2-byte combination. So, for every 2 bytes in the incoming sequence, we have 1 o200k token.

The 2^16 subset is alnum + '_' char, excluding the longest tokens to match the specified set size. The logic is to avoid special characters in JSON and Markdown, which are typical LLM I/O formats. Also, we want the LLM to clearly distinguish the resulting random string sequences from the surrounding context, so special characters used commonly to delimit table columns, sequences and strings are excluded.

The current produced lookup table is `data/lookup_tables/o200k_base_65536_tokens.txt`. It has 1 row per token in a single column. For a lookup, each 2-byte pair should be converted to an unsigned 16-bit integer, and use that to index the table.

## Roadmap

1. Simple library (importable functions, minimal deps)
2. Expand interface: include class to store executed mappings to perform exact encoding in reverse. The class could dump mappings to a json or expose a dictionary for the user to handle storage.
3. Typescript npm package.
