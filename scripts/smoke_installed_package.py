from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from uuid import UUID

import utf_token
from utf_token import frombase64, frombytes, fromhex, fromuuid


EXPECTED_DATA_FILES = {
    "o200k_base_32768_metadata.json",
    "o200k_base_32768_tail_256_tokens.txt",
    "o200k_base_32768_tokens.txt",
    "tokenizer_gemma4_32768_metadata.json",
    "tokenizer_gemma4_32768_tail_256_tokens.txt",
    "tokenizer_gemma4_32768_tokens.txt",
}
PROJECT_SRC_DIR = Path(__file__).resolve().parents[1] / "src" / "utf_token"


def main() -> None:
    module_path = Path(utf_token.__file__).resolve()
    if PROJECT_SRC_DIR in module_path.parents:
        raise RuntimeError(f"Expected an installed package, imported source tree module at {module_path}")

    data_files = {entry.name for entry in files("utf_token.data").iterdir()}
    missing_files = EXPECTED_DATA_FILES - data_files
    if missing_files:
        missing_list = ", ".join(sorted(missing_files))
        raise RuntimeError(f"Installed package is missing lookup table files: {missing_list}")

    sample_bytes = bytes.fromhex("0001ab")
    encoded = frombytes(sample_bytes)
    if fromhex("0001ab") != encoded:
        raise RuntimeError("fromhex produced a different result from frombytes")
    if frombase64("AAGr") != encoded:
        raise RuntimeError("frombase64 produced a different result from frombytes")
    gemma4_encoded = frombytes(sample_bytes, vocab="gemma4")
    if fromhex("0001ab", vocab="gemma4") != gemma4_encoded:
        raise RuntimeError("fromhex with gemma4 produced a different result from frombytes")
    if frombase64("AAGr", vocab="gemma4") != gemma4_encoded:
        raise RuntimeError("frombase64 with gemma4 produced a different result from frombytes")

    zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
    if fromuuid(zero_uuid) != frombytes(zero_uuid.bytes):
        raise RuntimeError("fromuuid(UUID) produced an unexpected result")
    if fromuuid(str(zero_uuid)) != frombytes(zero_uuid.bytes):
        raise RuntimeError("fromuuid(str) produced an unexpected result")
    if fromuuid(zero_uuid, vocab="gemma4") != frombytes(zero_uuid.bytes, vocab="gemma4"):
        raise RuntimeError("fromuuid(UUID, gemma4) produced an unexpected result")
    if fromuuid(str(zero_uuid), vocab="gemma4") != frombytes(zero_uuid.bytes, vocab="gemma4"):
        raise RuntimeError("fromuuid(str, gemma4) produced an unexpected result")

    print(f"Smoke test passed with installed package at {module_path}")


if __name__ == "__main__":
    main()
