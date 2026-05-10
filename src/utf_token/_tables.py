from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypeAlias


DATA_PACKAGE = "utf_token.data"
PROJECT_LOOKUP_TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "lookup_tables"

VocabName: TypeAlias = Literal["o200k", "gemma4"]
DEFAULT_VOCAB: VocabName = "o200k"


@dataclass(frozen=True, slots=True)
class _VocabFiles:
    """Resource filenames packaged for a given vocab."""

    pair_tokens: str
    tail_tokens: str
    metadata: str


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Compact runtime view of a vocab's lookup-table dimensions."""

    pair_table_size: int
    tail_table_size: int
    pair_index_bits: int
    tail_index_bits: int


VOCAB_FILENAMES: dict[VocabName, _VocabFiles] = {
    "o200k": _VocabFiles(
        pair_tokens="o200k_base_32768_tokens.txt",
        tail_tokens="o200k_base_32768_tail_256_tokens.txt",
        metadata="o200k_base_32768_metadata.json",
    ),
    "gemma4": _VocabFiles(
        pair_tokens="tokenizer_gemma4_32768_tokens.txt",
        tail_tokens="tokenizer_gemma4_32768_tail_256_tokens.txt",
        metadata="tokenizer_gemma4_32768_metadata.json",
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


def _files_for_vocab(vocab: VocabName) -> _VocabFiles:
    files_for_vocab = VOCAB_FILENAMES.get(vocab)
    if files_for_vocab is None:
        supported = ", ".join(supported_vocabs())
        raise ValueError(f"Unsupported vocab {vocab!r}. Expected one of: {supported}")
    return files_for_vocab


def _read_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Metadata field {key!r} must be an int, got {type(value).__name__}")
    return value


@lru_cache(maxsize=None)
def metadata(vocab: VocabName = DEFAULT_VOCAB) -> dict[str, object]:
    parsed = json.loads(_read_text(_files_for_vocab(vocab).metadata))
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Metadata for {vocab!r} did not decode to a JSON object"
        )
    return parsed


@lru_cache(maxsize=None)
def table_spec(vocab: VocabName = DEFAULT_VOCAB) -> TableSpec:
    payload = metadata(vocab)
    pair_table_size = _read_int(payload, "pair_table_size")
    tail_table_size = _read_int(payload, "tail_table_size")
    pair_index_bits = _read_int(payload, "pair_index_bits")
    tail_index_bits = _read_int(payload, "tail_index_bits")
    if pair_table_size != 1 << pair_index_bits:
        raise ValueError(
            f"Metadata for {vocab!r} declares pair_table_size={pair_table_size} but "
            f"pair_index_bits={pair_index_bits} (expected {1 << pair_index_bits})."
        )
    if tail_table_size != 1 << tail_index_bits:
        raise ValueError(
            f"Metadata for {vocab!r} declares tail_table_size={tail_table_size} but "
            f"tail_index_bits={tail_index_bits} (expected {1 << tail_index_bits})."
        )
    if pair_index_bits < tail_index_bits:
        raise ValueError(
            f"Metadata for {vocab!r} has pair_index_bits={pair_index_bits} smaller "
            f"than tail_index_bits={tail_index_bits}, which would leave residual bits "
            "unaddressable."
        )
    return TableSpec(
        pair_table_size=pair_table_size,
        tail_table_size=tail_table_size,
        pair_index_bits=pair_index_bits,
        tail_index_bits=tail_index_bits,
    )


@lru_cache(maxsize=None)
def pair_table(vocab: VocabName = DEFAULT_VOCAB) -> tuple[str, ...]:
    spec = table_spec(vocab)
    table = _read_lines(_files_for_vocab(vocab).pair_tokens)
    if len(table) != spec.pair_table_size:
        raise ValueError(
            f"Expected {spec.pair_table_size} pair tokens for vocab {vocab!r}, "
            f"found {len(table)}"
        )
    return table


@lru_cache(maxsize=None)
def tail_table(vocab: VocabName = DEFAULT_VOCAB) -> tuple[str, ...]:
    spec = table_spec(vocab)
    table = _read_lines(_files_for_vocab(vocab).tail_tokens)
    if len(table) != spec.tail_table_size:
        raise ValueError(
            f"Expected {spec.tail_table_size} tail tokens for vocab {vocab!r}, "
            f"found {len(table)}"
        )
    return table
