from __future__ import annotations

import argparse
import base64
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


DEFAULT_INPUT_PATH = Path("data/token_vocab/o200k_base.tiktoken")
DEFAULT_OUTPUT_DIR = Path("data/lookup_tables")
DEFAULT_PAIR_TABLE_SIZE = 1 << 16
DEFAULT_TAIL_TABLE_SIZE = 1 << 8


@dataclass(frozen=True, slots=True)
class TokenEntry:
    rank: int
    token_bytes: bytes
    token_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply first-pass character exclusions to a downloaded tokenizer vocab "
            "(.tiktoken or SentencePiece .model) and preview a provisional 16-bit "
            "token subset."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the downloaded tokenizer vocab file.",
    )
    parser.add_argument(
        "--pair-table-size",
        type=int,
        default=DEFAULT_PAIR_TABLE_SIZE,
        help="Target subset size for 2-byte mappings. Defaults to 2^16.",
    )
    parser.add_argument(
        "--tail-table-size",
        type=int,
        default=DEFAULT_TAIL_TABLE_SIZE,
        help="Target subset size for odd trailing byte mappings. Defaults to 2^8.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the lookup table files will be written.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of random example 2-byte mappings to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for example mappings.",
    )
    return parser.parse_args()


def decode_tiktoken_vocab_lines(raw_contents: bytes) -> list[tuple[int, bytes]]:
    entries: list[tuple[int, bytes]] = []

    for line_number, raw_line in enumerate(raw_contents.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Malformed vocab line {line_number}: expected 2 fields, got {len(parts)}"
            )

        token_b64, rank_text = parts

        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError(
                f"Malformed rank on line {line_number}: {rank_text!r}"
            ) from exc

        try:
            token_bytes = base64.b64decode(token_b64, validate=True)
        except ValueError as exc:
            raise ValueError(
                f"Malformed base64 token on line {line_number}: {token_b64!r}"
            ) from exc

        entries.append((rank, token_bytes))

    return sorted(entries, key=lambda entry: entry[0])


def decode_sentencepiece_model(input_path: Path) -> list[tuple[int, bytes]]:
    processor = spm.SentencePieceProcessor()
    processor.Load(str(input_path))
    return [
        (rank, processor.IdToPiece(rank).encode("utf-8"))
        for rank in range(processor.GetPieceSize())
    ]


def load_vocab_entries(input_path: Path) -> list[tuple[int, bytes]]:
    if input_path.suffix == ".tiktoken":
        return decode_tiktoken_vocab_lines(input_path.read_bytes())
    if input_path.suffix == ".model":
        return decode_sentencepiece_model(input_path)
    raise ValueError(
        "Unsupported vocab format. Expected a .tiktoken or SentencePiece .model file, "
        f"got {input_path.name!r}"
    )


def classify_token(token_bytes: bytes) -> tuple[str, str | None]:
    try:
        token_text = token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return "invalid_utf8", None

    if not token_text:
        return "empty", None

    if any(character.isspace() for character in token_text):
        return "contains_whitespace", token_text

    if not token_text.isprintable():
        return "contains_non_printable", token_text

    if any(not (character.isalnum() or character == "_") for character in token_text):
        return "contains_disallowed_special_chars", token_text

    return "eligible", token_text


def collect_candidates(
    vocab_entries: list[tuple[int, bytes]],
) -> tuple[list[TokenEntry], Counter[str]]:
    counts: Counter[str] = Counter()
    candidates: list[TokenEntry] = []

    for rank, token_bytes in vocab_entries:
        classification, token_text = classify_token(token_bytes)
        counts[classification] += 1

        if classification != "eligible" or token_text is None:
            continue

        candidates.append(
            TokenEntry(rank=rank, token_bytes=token_bytes, token_text=token_text)
        )

    return candidates, counts


def format_example_mapping(index: int, entry: TokenEntry) -> str:
    source_bytes = index.to_bytes(2, byteorder="big", signed=False)
    return (
        f"{source_bytes!r} (0x{index:04x}) -> rank {entry.rank}: "
        f"{entry.token_text!r}"
    )


def select_subset(
    candidates: list[TokenEntry],
    *,
    pair_table_size: int,
    tail_table_size: int,
) -> tuple[list[TokenEntry], list[TokenEntry]]:
    ordered_candidates = sorted(
        candidates, key=lambda entry: (len(entry.token_text), entry.rank)
    )
    tail_entries = ordered_candidates[:tail_table_size]
    pair_entries = ordered_candidates[tail_table_size : tail_table_size + pair_table_size]
    return pair_entries, tail_entries


def output_file_stem(input_path: Path, pair_table_size: int) -> str:
    return f"{input_path.stem}_{pair_table_size}"


def write_lookup_table(
    *,
    input_path: Path,
    output_dir: Path,
    pair_table_size: int,
    tail_table_size: int,
    pair_entries: list[TokenEntry],
    tail_entries: list[TokenEntry],
) -> tuple[Path, Path, Path]:
    if len(pair_entries) != pair_table_size:
        raise ValueError(
            "Cannot write lookup table: provisional subset does not cover the "
            f"requested {pair_table_size:,} 2-byte indices."
        )

    if len(tail_entries) != tail_table_size:
        raise ValueError(
            "Cannot write lookup table: provisional subset does not cover the "
            f"requested {tail_table_size:,} trailing-byte indices."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    table_stem = output_file_stem(input_path, pair_table_size)
    pair_tokens_path = output_dir / f"{table_stem}_tokens.txt"
    tail_tokens_path = output_dir / f"{table_stem}_tail_{tail_table_size}_tokens.txt"
    metadata_path = output_dir / f"{table_stem}_metadata.json"

    pair_tokens_payload = "\n".join(entry.token_text for entry in pair_entries) + "\n"
    pair_tokens_path.write_text(pair_tokens_payload, encoding="utf-8")

    tail_tokens_payload = "\n".join(entry.token_text for entry in tail_entries) + "\n"
    tail_tokens_path.write_text(tail_tokens_payload, encoding="utf-8")

    metadata = {
        "source_vocab": input_path.as_posix(),
        "pair_table_size": pair_table_size,
        "pair_entry_count": len(pair_entries),
        "pair_index_bits": 16,
        "pair_index_byte_order": "big",
        "pair_lookup_rule": (
            "Zero-based line number equals the unsigned 16-bit value decoded from "
            "the source bytes."
        ),
        "tail_table_size": tail_table_size,
        "tail_entry_count": len(tail_entries),
        "tail_index_bits": 8,
        "tail_lookup_rule": (
            "Zero-based line number equals the unsigned 8-bit value decoded from "
            "the odd trailing source byte."
        ),
        "selection_order": "sorted by (len(token_text), rank)",
        "tail_selection_rule": "first tail_table_size entries from the selection order",
        "pair_selection_rule": (
            "entries immediately after the reserved tail slice, continuing for "
            "pair_table_size entries"
        ),
        "pair_tokens_file": pair_tokens_path.name,
        "tail_tokens_file": tail_tokens_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return pair_tokens_path, tail_tokens_path, metadata_path


def print_summary(
    *,
    input_path: Path,
    output_dir: Path,
    total_tokens: int,
    counts: Counter[str],
    candidates: list[TokenEntry],
    pair_table_size: int,
    tail_table_size: int,
    pair_entries: list[TokenEntry],
    tail_entries: list[TokenEntry],
    examples: int,
    seed: int,
    written_files: tuple[Path, Path, Path] | None = None,
) -> None:
    print(f"Input vocab: {input_path}")
    print(f"Total tokens loaded: {total_tokens}")
    print()
    print("First-pass character exclusions:")
    print("  - token must decode as UTF-8")
    print("  - token must not contain whitespace")
    print("  - token must be printable")
    print("  - token characters must be alphanumeric or `_`")
    print()
    print("Classification counts:")
    for name in (
        "eligible",
        "invalid_utf8",
        "empty",
        "contains_whitespace",
        "contains_non_printable",
        "contains_disallowed_special_chars",
    ):
        print(f"  - {name}: {counts.get(name, 0)}")
    print()

    candidate_count = len(candidates)
    print(f"Total candidate token subset: {candidate_count}")

    required_entries = pair_table_size + tail_table_size
    if candidate_count >= required_entries:
        overflow = candidate_count - required_entries
        print(
            "Feasible for paired and trailing tables: yes "
            f"({overflow:,} extra candidates)"
        )
    else:
        shortage = required_entries - candidate_count
        print(
            "Feasible for paired and trailing tables: no "
            f"({shortage:,} candidates short)"
        )
    print()

    if not pair_entries:
        print("No example mappings available because the candidate subset is empty.")
        return

    longest_entry = max(
        pair_entries + tail_entries, key=lambda entry: (len(entry.token_text), entry.rank)
    )
    print(
        "Longest token in the provisional subset: "
        f"rank {longest_entry.rank}, length {len(longest_entry.token_text)}: "
        f"{longest_entry.token_text!r}"
    )
    print()

    sample_count = min(examples, len(pair_entries))
    indices = random.Random(seed).sample(range(len(pair_entries)), k=sample_count)

    print(
        "Example 2-byte mappings using the provisional subset "
        "(after reserving the shortest eligible tail tokens first):"
    )
    for index in indices:
        print(f"  - {format_example_mapping(index, pair_entries[index])}")

    if tail_entries:
        print()
        print("Example trailing-byte mappings:")
        tail_indices = random.Random(seed).sample(range(len(tail_entries)), k=min(examples, len(tail_entries)))
        for index in tail_indices:
            print(f"  - {index:#04x} -> rank {tail_entries[index].rank}: {tail_entries[index].token_text!r}")

    if written_files is None:
        table_stem = output_file_stem(input_path, pair_table_size)
        print()
        print(
            "Lookup tables not written because the candidate subset does not cover "
            f"all {required_entries:,} required indices."
        )
        print(
            "If the subset becomes feasible, files will be written to "
            f"{output_dir / f'{table_stem}_tokens.txt'}, "
            f"{output_dir / f'{table_stem}_tail_{tail_table_size}_tokens.txt'}, and "
            f"{output_dir / f'{table_stem}_metadata.json'}"
        )
        return

    pair_tokens_path, tail_tokens_path, metadata_path = written_files
    print()
    print(f"Wrote ordered 2-byte token list: {pair_tokens_path}")
    print(f"Wrote ordered trailing-byte token list: {tail_tokens_path}")
    print(f"Wrote table metadata: {metadata_path}")


def main() -> int:
    args = parse_args()
    vocab_entries = load_vocab_entries(args.input_path)
    candidates, counts = collect_candidates(vocab_entries)
    pair_entries, tail_entries = select_subset(
        candidates,
        pair_table_size=args.pair_table_size,
        tail_table_size=args.tail_table_size,
    )
    written_files: tuple[Path, Path, Path] | None = None

    if len(pair_entries) == args.pair_table_size and len(tail_entries) == args.tail_table_size:
        written_files = write_lookup_table(
            input_path=args.input_path,
            output_dir=args.output_dir,
            pair_table_size=args.pair_table_size,
            tail_table_size=args.tail_table_size,
            pair_entries=pair_entries,
            tail_entries=tail_entries,
        )

    print_summary(
        input_path=args.input_path,
        output_dir=args.output_dir,
        total_tokens=len(vocab_entries),
        counts=counts,
        candidates=candidates,
        pair_table_size=args.pair_table_size,
        tail_table_size=args.tail_table_size,
        pair_entries=pair_entries,
        tail_entries=tail_entries,
        examples=args.examples,
        seed=args.seed,
        written_files=written_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
