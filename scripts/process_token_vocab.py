from __future__ import annotations

import argparse
import base64
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/token_vocab/o200k_base.tiktoken")
DEFAULT_OUTPUT_DIR = Path("data/lookup_tables")
DEFAULT_SUBSET_SIZE = 1 << 16


@dataclass(frozen=True, slots=True)
class TokenEntry:
    rank: int
    token_bytes: bytes
    token_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply first-pass character exclusions to a downloaded tokenizer vocab and "
            "preview a provisional 16-bit token subset."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the downloaded .tiktoken vocab file.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=DEFAULT_SUBSET_SIZE,
        help="Target subset size for 2-byte mappings. Defaults to 2^16.",
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


def decode_vocab_lines(raw_contents: bytes) -> list[tuple[int, bytes]]:
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


def select_subset(candidates: list[TokenEntry], subset_size: int) -> list[TokenEntry]:
    return sorted(candidates, key=lambda entry: (len(entry.token_text), entry.rank))[
        :subset_size
    ]


def output_file_stem(input_path: Path, subset_size: int) -> str:
    return f"{input_path.stem}_{subset_size}"


def write_lookup_table(
    *,
    input_path: Path,
    output_dir: Path,
    subset_size: int,
    provisional_subset: list[TokenEntry],
) -> tuple[Path, Path]:
    if len(provisional_subset) != subset_size:
        raise ValueError(
            "Cannot write lookup table: provisional subset does not cover the "
            f"requested {subset_size:,} indices."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    table_stem = output_file_stem(input_path, subset_size)
    tokens_path = output_dir / f"{table_stem}_tokens.txt"
    metadata_path = output_dir / f"{table_stem}_metadata.json"

    tokens_payload = "\n".join(entry.token_text for entry in provisional_subset) + "\n"
    tokens_path.write_text(tokens_payload, encoding="utf-8")

    metadata = {
        "source_vocab": input_path.as_posix(),
        "subset_size": subset_size,
        "entry_count": len(provisional_subset),
        "index_bits": 16,
        "index_byte_order": "big",
        "index_lookup_rule": (
            "Zero-based line number equals the unsigned 16-bit value decoded from "
            "the source bytes."
        ),
        "selection_order": "sorted by (len(token_text), rank)",
        "tokens_file": tokens_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    return tokens_path, metadata_path


def print_summary(
    *,
    input_path: Path,
    output_dir: Path,
    total_tokens: int,
    counts: Counter[str],
    candidates: list[TokenEntry],
    subset_size: int,
    provisional_subset: list[TokenEntry],
    examples: int,
    seed: int,
    written_files: tuple[Path, Path] | None = None,
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

    if candidate_count >= subset_size:
        overflow = candidate_count - subset_size
        print(
            f"Feasible for a {subset_size:,}-token 2-byte table: yes "
            f"({overflow:,} extra candidates)"
        )
    else:
        shortage = subset_size - candidate_count
        print(
            f"Feasible for a {subset_size:,}-token 2-byte table: no "
            f"({shortage:,} candidates short)"
        )
    print()

    if not provisional_subset:
        print("No example mappings available because the candidate subset is empty.")
        return

    longest_entry = max(
        provisional_subset, key=lambda entry: (len(entry.token_text), entry.rank)
    )
    print(
        "Longest token in the provisional subset: "
        f"rank {longest_entry.rank}, length {len(longest_entry.token_text)}: "
        f"{longest_entry.token_text!r}"
    )
    print()

    sample_count = min(examples, len(provisional_subset))
    indices = random.Random(seed).sample(range(len(provisional_subset)), k=sample_count)

    print(
        "Example mappings using the provisional subset "
        "(shortest eligible tokens first, rank tiebreaker):"
    )
    for index in indices:
        print(f"  - {format_example_mapping(index, provisional_subset[index])}")

    if written_files is None:
        table_stem = output_file_stem(input_path, subset_size)
        print()
        print(
            "Lookup table not written because the candidate subset does not cover "
            f"all {subset_size:,} indices."
        )
        print(
            "If the subset becomes feasible, files will be written to "
            f"{output_dir / f'{table_stem}_tokens.txt'} and "
            f"{output_dir / f'{table_stem}_metadata.json'}"
        )
        return

    tokens_path, metadata_path = written_files
    print()
    print(f"Wrote ordered token list: {tokens_path}")
    print(f"Wrote table metadata: {metadata_path}")


def main() -> int:
    args = parse_args()
    raw_contents = args.input_path.read_bytes()
    vocab_entries = decode_vocab_lines(raw_contents)
    candidates, counts = collect_candidates(vocab_entries)
    provisional_subset = select_subset(candidates, args.subset_size)
    written_files: tuple[Path, Path] | None = None

    if len(provisional_subset) == args.subset_size:
        written_files = write_lookup_table(
            input_path=args.input_path,
            output_dir=args.output_dir,
            subset_size=args.subset_size,
            provisional_subset=provisional_subset,
        )

    print_summary(
        input_path=args.input_path,
        output_dir=args.output_dir,
        total_tokens=len(vocab_entries),
        counts=counts,
        candidates=candidates,
        subset_size=args.subset_size,
        provisional_subset=provisional_subset,
        examples=args.examples,
        seed=args.seed,
        written_files=written_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
