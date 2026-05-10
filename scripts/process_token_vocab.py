from __future__ import annotations

import argparse
import ast
import base64
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm


DEFAULT_INPUT_PATH = Path("data/token_vocab/o200k_base.tiktoken")
DEFAULT_OUTPUT_DIR = Path("data/lookup_tables")
DEFAULT_TAIL_TABLE_SIZE = 1 << 8
DEFAULT_MAX_TOKEN_LENGTH = 10


# Per-block ranges for the curated Latin allowlist used by `latin_16bit`.
# Latin-1 Supplement covers common Western European diacritics (À-ÿ minus
# the maths symbols, which the baseline alnum filter rejects anyway).
# Latin Extended-A covers Central/Eastern European precomposed letters such as
# `ć`, `ř`, `ł`, `İ`, used by Polish, Czech, Croatian, Turkish, etc.
LATIN1_SUPPLEMENT_RANGE: tuple[int, int] = (0x00C0, 0x00FF)
LATIN_EXTENDED_A_RANGE: tuple[int, int] = (0x0100, 0x017F)


@dataclass(frozen=True, slots=True)
class TokenEntry:
    rank: int
    token_bytes: bytes
    token_text: str


@dataclass(frozen=True, slots=True)
class Recipe:
    """Declarative description of a vocab cleanup recipe.

    Recipes are tried in priority order until one yields enough candidates to
    fill its `pair_table_size + tail_table_size` slots. Each recipe records the
    table dimensions and the character constraints it enforces beyond the
    baseline (UTF-8 printable alnum-or-underscore).
    """

    name: str
    description: str
    pair_table_size: int
    tail_table_size: int
    pair_index_bits: int
    tail_index_bits: int
    require_ascii: bool
    # Inclusive (low, high) codepoint ranges that may appear alongside ASCII
    # alphanumerics and `_`. Empty means ASCII-only (which is also what
    # `require_ascii=True` enforces).
    allowed_non_ascii_ranges: tuple[tuple[int, int], ...]

    @property
    def required_total(self) -> int:
        return self.pair_table_size + self.tail_table_size

    @property
    def allowed_character_policy(self) -> str:
        if self.require_ascii or not self.allowed_non_ascii_ranges:
            return "ASCII alphanumeric or underscore"
        range_text = ", ".join(
            f"U+{low:04X}-U+{high:04X}" for low, high in self.allowed_non_ascii_ranges
        )
        return (
            "ASCII alphanumeric or underscore, plus characters in "
            f"{range_text}"
        )


# Cleanup recipes are ordered from most efficient (16-bit pair table) to least
# efficient (15-bit pair table with strict ASCII-only filtering). A vocab uses
# the first recipe whose candidate count meets the recipe's table dimensions.
RECIPES: tuple[Recipe, ...] = (
    Recipe(
        name="latin_16bit",
        description=(
            "16-bit pair table; ASCII alnum or underscore plus precomposed "
            "letters from the Latin-1 Supplement and Latin Extended-A blocks."
        ),
        pair_table_size=1 << 16,
        tail_table_size=1 << 8,
        pair_index_bits=16,
        tail_index_bits=8,
        require_ascii=False,
        allowed_non_ascii_ranges=(
            LATIN1_SUPPLEMENT_RANGE,
            LATIN_EXTENDED_A_RANGE,
        ),
    ),
    Recipe(
        name="ascii_15bit",
        description=(
            "15-bit pair table; ASCII alnum or underscore only, for vocabs that "
            "cannot supply enough clean tokens for a 16-bit table."
        ),
        pair_table_size=1 << 15,
        tail_table_size=1 << 8,
        pair_index_bits=15,
        tail_index_bits=8,
        require_ascii=True,
        allowed_non_ascii_ranges=(),
    ),
)


@dataclass(frozen=True, slots=True)
class RecipeAttempt:
    recipe: Recipe
    candidate_count: int
    feasible: bool


def parse_args() -> argparse.Namespace:
    recipe_names = tuple(recipe.name for recipe in RECIPES)
    parser = argparse.ArgumentParser(
        description=(
            "Apply ordered cleanup recipes to a downloaded tokenizer vocab "
            "(.tiktoken or SentencePiece .model) and write lookup tables for "
            "the first recipe that meets its candidate threshold."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the downloaded tokenizer vocab file.",
    )
    parser.add_argument(
        "--recipe",
        choices=recipe_names,
        default=None,
        help=(
            "Force a specific recipe instead of auto-selecting the first "
            "feasible one. Defaults to None, which tries recipes in order: "
            f"{', '.join(recipe_names)}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the lookup table files will be written.",
    )
    parser.add_argument(
        "--max-token-length",
        type=int,
        default=DEFAULT_MAX_TOKEN_LENGTH,
        help=(
            "Maximum decoded token length to keep in lookup-table candidates. "
            "Defaults to 10 to avoid long, low-copy-reliability tokens."
        ),
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of random example pair mappings to print.",
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


def decode_vocab_list_lines(raw_contents: str) -> list[tuple[int, bytes]]:
    entries: list[tuple[int, bytes]] = []

    for line_number, raw_line in enumerate(raw_contents.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split("\t", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"Malformed vocab-list line {line_number}: expected rank and repr fields"
            )

        rank_text, token_repr = parts
        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError(
                f"Malformed rank on line {line_number}: {rank_text!r}"
            ) from exc

        token_text = ast.literal_eval(token_repr)
        if not isinstance(token_text, str):
            raise ValueError(
                f"Malformed token repr on line {line_number}: expected str literal"
            )

        entries.append((rank, token_text.encode("utf-8")))

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
    if input_path.name.endswith("_vocab_list.txt"):
        return decode_vocab_list_lines(input_path.read_text(encoding="utf-8"))
    raise ValueError(
        "Unsupported vocab format. Expected a .tiktoken, SentencePiece .model, "
        "or *_vocab_list.txt file, "
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


def _is_ascii_alnum_or_underscore(text: str) -> bool:
    for character in text:
        if character == "_":
            continue
        if not (character.isascii() and character.isalnum()):
            return False
    return True


def _only_allowed_non_ascii_chars(
    text: str, allowed_ranges: tuple[tuple[int, int], ...]
) -> bool:
    """Return True when every non-ASCII character lies in an allowed range.

    ASCII characters always pass this check; the broader baseline filter in
    `classify_token` already restricts them to alphanumerics or `_`.
    """
    for character in text:
        if character.isascii():
            continue
        codepoint = ord(character)
        in_range = False
        for low, high in allowed_ranges:
            if low <= codepoint <= high:
                in_range = True
                break
        if not in_range:
            return False
    return True


def recipe_keeps(recipe: Recipe, text: str) -> bool:
    """Return True when `text` satisfies the recipe-specific constraints."""

    if recipe.require_ascii:
        return _is_ascii_alnum_or_underscore(text)
    return _only_allowed_non_ascii_chars(text, recipe.allowed_non_ascii_ranges)


def filter_candidates_for_recipe(
    candidates: list[TokenEntry],
    recipe: Recipe,
    *,
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
) -> list[TokenEntry]:
    return [
        entry
        for entry in candidates
        if recipe_keeps(recipe, entry.token_text)
        and len(entry.token_text) <= max_token_length
    ]


def format_example_mapping(index: int, entry: TokenEntry, *, index_bits: int) -> str:
    width = max(1, (index_bits + 3) // 4)
    return (
        f"index 0x{index:0{width}x} -> rank {entry.rank}: {entry.token_text!r}"
    )


def select_subset(
    candidates: list[TokenEntry],
    *,
    pair_table_size: int,
    tail_table_size: int,
) -> tuple[list[TokenEntry], list[TokenEntry]]:
    preferred_single_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    preferred_index = {character: index for index, character in enumerate(preferred_single_chars)}

    def selection_key(entry: TokenEntry) -> tuple[int, int]:
        preferred = (
            len(entry.token_text) == 1 and entry.token_text in preferred_index
        )
        if preferred:
            return (0, preferred_index[entry.token_text])
        return (1, entry.rank)

    ordered_candidates = sorted(candidates, key=selection_key)
    tail_entries = ordered_candidates[:tail_table_size]
    pair_entries = ordered_candidates[tail_table_size : tail_table_size + pair_table_size]
    return pair_entries, tail_entries


def output_file_stem(input_path: Path, pair_table_size: int) -> str:
    vocab_stem = input_path.stem
    if vocab_stem.endswith("_vocab_list"):
        vocab_stem = vocab_stem.removesuffix("_vocab_list")
    return f"{vocab_stem}_{pair_table_size}"


def write_lookup_table(
    *,
    input_path: Path,
    output_dir: Path,
    recipe: Recipe,
    pair_entries: list[TokenEntry],
    tail_entries: list[TokenEntry],
    max_token_length: int,
) -> tuple[Path, Path, Path]:
    if len(pair_entries) != recipe.pair_table_size:
        raise ValueError(
            "Cannot write lookup table: provisional subset does not cover the "
            f"requested {recipe.pair_table_size:,} large-table indices."
        )

    if len(tail_entries) != recipe.tail_table_size:
        raise ValueError(
            "Cannot write lookup table: provisional subset does not cover the "
            f"requested {recipe.tail_table_size:,} trailing-byte indices."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    table_stem = output_file_stem(input_path, recipe.pair_table_size)
    pair_tokens_path = output_dir / f"{table_stem}_tokens.txt"
    tail_tokens_path = output_dir / f"{table_stem}_tail_{recipe.tail_table_size}_tokens.txt"
    metadata_path = output_dir / f"{table_stem}_metadata.json"

    pair_tokens_payload = "\n".join(entry.token_text for entry in pair_entries) + "\n"
    pair_tokens_path.write_text(pair_tokens_payload, encoding="utf-8")

    tail_tokens_payload = "\n".join(entry.token_text for entry in tail_entries) + "\n"
    tail_tokens_path.write_text(tail_tokens_payload, encoding="utf-8")

    if recipe.pair_index_bits == 16:
        pair_lookup_rule = (
            "Zero-based line number equals the unsigned 16-bit value decoded "
            "from each consecutive 2-byte pair (big-endian)."
        )
    else:
        pair_lookup_rule = (
            f"Zero-based line number equals the unsigned {recipe.pair_index_bits}-bit "
            "value extracted from the input bitstream, MSB-first, in order. The "
            f"final residual of {recipe.pair_index_bits + 1}..{2 * recipe.pair_index_bits - 1} "
            f"bits also indexes this table; residuals of 1..{recipe.tail_index_bits} "
            "bits index the tail table instead."
        )

    allowed_ranges_payload = [
        {
            "start": f"U+{low:04X}",
            "end": f"U+{high:04X}",
            "start_codepoint": low,
            "end_codepoint": high,
        }
        for low, high in recipe.allowed_non_ascii_ranges
    ]

    metadata = {
        "source_vocab": input_path.as_posix(),
        "recipe_name": recipe.name,
        "recipe_description": recipe.description,
        "allowed_character_policy": recipe.allowed_character_policy,
        "allowed_non_ascii_ranges": allowed_ranges_payload,
        "max_token_length": max_token_length,
        "pair_table_size": recipe.pair_table_size,
        "pair_entry_count": len(pair_entries),
        "pair_index_bits": recipe.pair_index_bits,
        "pair_index_byte_order": "big",
        "pair_lookup_rule": pair_lookup_rule,
        "tail_table_size": recipe.tail_table_size,
        "tail_entry_count": len(tail_entries),
        "tail_index_bits": recipe.tail_index_bits,
        "tail_lookup_rule": (
            "Zero-based line number equals the unsigned 8-bit value decoded "
            "from the trailing residual bits of the input bitstream "
            "(left-padded with zero bits when fewer than 8)."
        ),
        "selection_order": (
            "single-char [0-9A-Za-z] first in fixed ASCII sequence, then rank order"
        ),
        "tail_selection_rule": "first tail_table_size entries from selection_order",
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


def evaluate_recipes(
    candidates: list[TokenEntry],
    *,
    selected_recipes: tuple[Recipe, ...],
    max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
) -> tuple[
    list[RecipeAttempt],
    tuple[Recipe, list[TokenEntry], list[TokenEntry]] | None,
]:
    """Try recipes in priority order and return the first feasible selection."""

    attempts: list[RecipeAttempt] = []
    chosen: tuple[Recipe, list[TokenEntry], list[TokenEntry]] | None = None

    for recipe in selected_recipes:
        filtered = filter_candidates_for_recipe(
            candidates,
            recipe,
            max_token_length=max_token_length,
        )
        pair_entries, tail_entries = select_subset(
            filtered,
            pair_table_size=recipe.pair_table_size,
            tail_table_size=recipe.tail_table_size,
        )
        feasible = (
            len(pair_entries) == recipe.pair_table_size
            and len(tail_entries) == recipe.tail_table_size
        )
        attempts.append(
            RecipeAttempt(recipe=recipe, candidate_count=len(filtered), feasible=feasible)
        )
        if feasible and chosen is None:
            chosen = (recipe, pair_entries, tail_entries)

    return attempts, chosen


def print_summary(
    *,
    input_path: Path,
    output_dir: Path,
    total_tokens: int,
    counts: Counter[str],
    candidates: list[TokenEntry],
    attempts: list[RecipeAttempt],
    chosen: tuple[Recipe, list[TokenEntry], list[TokenEntry]] | None,
    max_token_length: int,
    examples: int,
    seed: int,
    written_files: tuple[Path, Path, Path] | None = None,
) -> None:
    print(f"Input vocab: {input_path}")
    print(f"Total tokens loaded: {total_tokens}")
    print()
    print("Baseline character exclusions:")
    print("  - token must decode as UTF-8")
    print("  - token must not contain whitespace")
    print("  - token must be printable")
    print("  - token characters must be alphanumeric or `_`")
    print()
    print("Baseline classification counts:")
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
    print(f"Baseline candidate token subset: {len(candidates):,}")
    print(f"Maximum token length after recipe filtering: {max_token_length}")
    print()

    print("Recipe attempts (in priority order):")
    for attempt in attempts:
        recipe = attempt.recipe
        margin = attempt.candidate_count - recipe.required_total
        status = "FEASIBLE" if attempt.feasible else "short"
        print(
            f"  - {recipe.name} ({recipe.pair_index_bits}-bit pair, "
            f"{recipe.required_total:,} required): "
            f"{attempt.candidate_count:,} candidates ({margin:+,}) [{status}]"
        )
        if recipe.allowed_non_ascii_ranges:
            range_text = ", ".join(
                f"U+{low:04X}-U+{high:04X}"
                for low, high in recipe.allowed_non_ascii_ranges
            )
            print(f"      allows non-ASCII codepoints: {range_text}")
        print(f"      policy: {recipe.allowed_character_policy}")
    print()

    if chosen is None:
        print("No recipe yielded enough candidates; no lookup tables were written.")
        return

    recipe, pair_entries, tail_entries = chosen
    print(f"Selected recipe: {recipe.name}")
    print(f"  description: {recipe.description}")
    print(
        f"  table dimensions: pair {recipe.pair_table_size:,} entries "
        f"({recipe.pair_index_bits}-bit), tail {recipe.tail_table_size:,} entries "
        f"({recipe.tail_index_bits}-bit)"
    )
    print()

    longest_entry = max(
        pair_entries + tail_entries, key=lambda entry: (len(entry.token_text), entry.rank)
    )
    print(
        "Longest token in the selected subset: "
        f"rank {longest_entry.rank}, length {len(longest_entry.token_text)}: "
        f"{longest_entry.token_text!r}"
    )
    print()

    sample_count = min(examples, len(pair_entries))
    rng = random.Random(seed)
    indices = rng.sample(range(len(pair_entries)), k=sample_count)
    print("Example pair-table mappings:")
    for index in indices:
        print(
            "  - "
            + format_example_mapping(
                index, pair_entries[index], index_bits=recipe.pair_index_bits
            )
        )

    if tail_entries:
        print()
        print("Example tail-table mappings:")
        tail_rng = random.Random(seed)
        tail_indices = tail_rng.sample(
            range(len(tail_entries)), k=min(examples, len(tail_entries))
        )
        for index in tail_indices:
            print(
                f"  - 0x{index:02x} -> rank {tail_entries[index].rank}: "
                f"{tail_entries[index].token_text!r}"
            )

    if written_files is None:
        print()
        print("Lookup tables not written (see error above).")
        return

    pair_tokens_path, tail_tokens_path, metadata_path = written_files
    print()
    print(f"Wrote pair token list: {pair_tokens_path}")
    print(f"Wrote tail token list: {tail_tokens_path}")
    print(f"Wrote table metadata: {metadata_path}")


def _select_recipes(forced: str | None) -> tuple[Recipe, ...]:
    if forced is None:
        return RECIPES
    for recipe in RECIPES:
        if recipe.name == forced:
            return (recipe,)
    available = ", ".join(r.name for r in RECIPES)
    raise ValueError(f"Unknown recipe {forced!r}. Available: {available}")


def main() -> int:
    args = parse_args()
    selected_recipes = _select_recipes(args.recipe)
    vocab_entries = load_vocab_entries(args.input_path)
    candidates, counts = collect_candidates(vocab_entries)
    attempts, chosen = evaluate_recipes(
        candidates,
        selected_recipes=selected_recipes,
        max_token_length=args.max_token_length,
    )

    written_files: tuple[Path, Path, Path] | None = None
    if chosen is not None:
        recipe, pair_entries, tail_entries = chosen
        written_files = write_lookup_table(
            input_path=args.input_path,
            output_dir=args.output_dir,
            recipe=recipe,
            pair_entries=pair_entries,
            tail_entries=tail_entries,
            max_token_length=args.max_token_length,
        )

    print_summary(
        input_path=args.input_path,
        output_dir=args.output_dir,
        total_tokens=len(vocab_entries),
        counts=counts,
        candidates=candidates,
        attempts=attempts,
        chosen=chosen,
        max_token_length=args.max_token_length,
        examples=args.examples,
        seed=args.seed,
        written_files=written_files,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
