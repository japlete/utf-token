from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypeAlias


DATA_PACKAGE = "utf_token.data"
PROJECT_LOOKUP_TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "lookup_tables"

VocabName: TypeAlias = Literal["o200k", "gemma4"]
DEFAULT_VOCAB: VocabName = "o200k"

VOCAB_FILENAMES: dict[VocabName, tuple[str, str, str]] = {
    "o200k": (
        "o200k_base_65536_tokens.txt",
        "o200k_base_65536_tail_256_tokens.txt",
        "o200k_base_65536_metadata.json",
    ),
    "gemma4": (
        "tokenizer_gemma4_65536_tokens.txt",
        "tokenizer_gemma4_65536_tail_256_tokens.txt",
        "tokenizer_gemma4_65536_metadata.json",
    ),
}


def _read_text(filename: str) -> str:
    resource = files(DATA_PACKAGE).joinpath(filename)
    try:
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        fallback_path = PROJECT_LOOKUP_TABLE_DIR / filename
        if fallback_path.is_file():
            return fallback_path.read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"Could not find lookup table resource {filename!r} in package data or "
            f"{PROJECT_LOOKUP_TABLE_DIR}"
        )


def _read_lines(filename: str) -> tuple[str, ...]:
    return tuple(_read_text(filename).splitlines())


def supported_vocabs() -> tuple[VocabName, ...]:
    return tuple(VOCAB_FILENAMES.keys())


def _filenames_for_vocab(vocab: VocabName) -> tuple[str, str, str]:
    filenames = VOCAB_FILENAMES.get(vocab)
    if filenames is None:
        supported = ", ".join(supported_vocabs())
        raise ValueError(f"Unsupported vocab {vocab!r}. Expected one of: {supported}")
    return filenames


@lru_cache(maxsize=None)
def pair_table(vocab: VocabName = DEFAULT_VOCAB) -> tuple[str, ...]:
    table = _read_lines(_filenames_for_vocab(vocab)[0])
    if len(table) != 1 << 16:
        raise ValueError(f"Expected 65536 pair tokens, found {len(table)}")
    return table


@lru_cache(maxsize=None)
def tail_table(vocab: VocabName = DEFAULT_VOCAB) -> tuple[str, ...]:
    table = _read_lines(_filenames_for_vocab(vocab)[1])
    if len(table) != 1 << 8:
        raise ValueError(f"Expected 256 tail tokens, found {len(table)}")
    return table


@lru_cache(maxsize=None)
def metadata(vocab: VocabName = DEFAULT_VOCAB) -> dict[str, object]:
    return json.loads(_read_text(_filenames_for_vocab(vocab)[2]))
