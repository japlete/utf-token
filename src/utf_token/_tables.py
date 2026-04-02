from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path


DATA_PACKAGE = "utf_token.data"
PROJECT_LOOKUP_TABLE_DIR = Path(__file__).resolve().parents[2] / "data" / "lookup_tables"
PAIR_TABLE_FILENAME = "o200k_base_65536_tokens.txt"
TAIL_TABLE_FILENAME = "o200k_base_65536_tail_256_tokens.txt"
METADATA_FILENAME = "o200k_base_65536_metadata.json"


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


@lru_cache(maxsize=1)
def pair_table() -> tuple[str, ...]:
    table = _read_lines(PAIR_TABLE_FILENAME)
    if len(table) != 1 << 16:
        raise ValueError(f"Expected 65536 pair tokens, found {len(table)}")
    return table


@lru_cache(maxsize=1)
def tail_table() -> tuple[str, ...]:
    table = _read_lines(TAIL_TABLE_FILENAME)
    if len(table) != 1 << 8:
        raise ValueError(f"Expected 256 tail tokens, found {len(table)}")
    return table


@lru_cache(maxsize=1)
def metadata() -> dict[str, object]:
    return json.loads(_read_text(METADATA_FILENAME))
