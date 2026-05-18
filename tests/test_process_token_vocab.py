from __future__ import annotations

import json
import runpy
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock


MODULE_GLOBALS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "process_token_vocab.py")
)
TokenEntry = cast(type[Any], MODULE_GLOBALS["TokenEntry"])
Recipe = cast(type[Any], MODULE_GLOBALS["Recipe"])
RECIPES: tuple[Any, ...] = cast(tuple[Any, ...], MODULE_GLOBALS["RECIPES"])
load_vocab_entries = MODULE_GLOBALS["load_vocab_entries"]
select_subset = MODULE_GLOBALS["select_subset"]
recipe_keeps = MODULE_GLOBALS["recipe_keeps"]
filter_candidates_for_recipe = MODULE_GLOBALS["filter_candidates_for_recipe"]
evaluate_recipes = MODULE_GLOBALS["evaluate_recipes"]
classify_token = MODULE_GLOBALS["classify_token"]
write_lookup_table = MODULE_GLOBALS["write_lookup_table"]


def _recipe_by_name(name: str) -> Any:
    for recipe in RECIPES:
        if recipe.name == name:
            return recipe
    raise AssertionError(f"Recipe {name!r} not found")


class SelectSubsetTests(unittest.TestCase):
    def test_select_subset_prioritizes_alnum_ascii_then_rank(self) -> None:
        candidates = [
            TokenEntry(rank=1, token_bytes=b"zz", token_text="zz"),
            TokenEntry(rank=2, token_bytes=b"a", token_text="a"),
            TokenEntry(rank=3, token_bytes=b"0", token_text="0"),
            TokenEntry(rank=4, token_bytes=b"A", token_text="A"),
            TokenEntry(rank=5, token_bytes=b"B", token_text="B"),
            TokenEntry(rank=6, token_bytes=b"_", token_text="_"),
        ]

        pair_entries, tail_entries = select_subset(
            candidates,
            pair_table_size=2,
            tail_table_size=2,
        )

        self.assertEqual([entry.token_text for entry in tail_entries], ["0", "A"])
        self.assertEqual([entry.token_text for entry in pair_entries], ["B", "a"])

    def test_select_subset_keeps_non_preferred_tokens_in_rank_order(self) -> None:
        candidates = [
            TokenEntry(rank=10, token_bytes=b"__", token_text="__"),
            TokenEntry(rank=20, token_bytes=b"_", token_text="_"),
            TokenEntry(rank=30, token_bytes=b"abc", token_text="abc"),
            TokenEntry(rank=40, token_bytes="é".encode("utf-8"), token_text="é"),
        ]

        pair_entries, tail_entries = select_subset(
            candidates,
            pair_table_size=2,
            tail_table_size=2,
        )

        self.assertEqual([entry.token_text for entry in tail_entries], ["__", "_"])
        self.assertEqual([entry.token_text for entry in pair_entries], ["abc", "é"])


class LoadVocabEntriesTests(unittest.TestCase):
    def test_load_vocab_entries_reads_sentencepiece_model(self) -> None:
        class FakeSentencePieceProcessor:
            def __init__(self) -> None:
                self.model_file: str | None = None

            def Load(self, model_file: str) -> None:
                self.model_file = model_file

            def GetPieceSize(self) -> int:
                return 3

            def IdToPiece(self, index: int) -> str:
                return ["foo", "Bar", "_baz"][index]

        with mock.patch.object(
            MODULE_GLOBALS["spm"], "SentencePieceProcessor", FakeSentencePieceProcessor
        ):
            entries = load_vocab_entries(Path("data/token_vocab/tokenizer_gemma4.model"))

        self.assertEqual(
            entries,
            [
                (0, b"foo"),
                (1, b"Bar"),
                (2, b"_baz"),
            ],
        )

    def test_load_vocab_entries_reads_vocab_list(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "tokenizer_gemma4_vocab_list.txt"
            input_path.write_text("0\t'<pad>'\n1\t'abc'\n2\t'café'\n", encoding="utf-8")

            entries = load_vocab_entries(input_path)

        self.assertEqual(
            entries,
            [
                (0, b"<pad>"),
                (1, b"abc"),
                (2, "café".encode("utf-8")),
            ],
        )

    def test_load_vocab_entries_rejects_unknown_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported vocab format"):
            load_vocab_entries(Path("data/token_vocab/unknown_vocab.txt"))


class RecipeFilterTests(unittest.TestCase):
    def test_latin_16bit_keeps_ascii_alnum(self) -> None:
        recipe = _recipe_by_name("latin_16bit")
        self.assertTrue(recipe_keeps(recipe, "abc"))
        self.assertTrue(recipe_keeps(recipe, "abc123"))
        self.assertTrue(recipe_keeps(recipe, "_underscore"))

    def test_latin_16bit_keeps_common_latin_diacritics(self) -> None:
        # The 16-bit recipe allows precomposed letters from Latin-1 Supplement
        # and Latin Extended-A so European words remain representable.
        recipe = _recipe_by_name("latin_16bit")
        for sample in ("café", "ñ", "zić", "řit", "Łódź", "İstanbul", "ÆØÅ"):
            self.assertTrue(recipe_keeps(recipe, sample), f"unexpected drop for {sample!r}")

    def test_latin_16bit_rejects_non_latin_scripts(self) -> None:
        recipe = _recipe_by_name("latin_16bit")
        # Mix of CJK, Hiragana, Katakana, Hangul, Bopomofo, Cyrillic, Greek,
        # Devanagari, Arabic, Cuneiform, mathematical Latin and Fullwidth Latin
        # samples drawn from observed NIAH transcription failures.
        for sample in (
            "\u4e2d",  # CJK
            "\u3042",  # Hiragana
            "\u30ab",  # Katakana
            "\uac00",  # Hangul
            "\u3105",  # Bopomofo
            "\u0410",  # Cyrillic
            "\u03b1",  # Greek
            "क",  # Devanagari KA
            "व",  # Devanagari VA
            "ي",  # Arabic YEH
            "𒐴",  # Cuneiform
            "𝙝",  # Mathematical sans-serif bold italic h
            "ｍ",  # Fullwidth Latin small m
        ):
            self.assertFalse(recipe_keeps(recipe, sample), f"unexpected keep for {sample!r}")

    def test_latin_16bit_rejects_extended_latin_blocks(self) -> None:
        recipe = _recipe_by_name("latin_16bit")
        # Latin Extended-B (e.g. ƀ U+0180) and IPA / phonetic extensions sit
        # outside the curated allowlist even though they share the LATIN script
        # name; this keeps the table free of obscure or visually confusable
        # variants.
        for sample in ("ƀ", "ǂ", "ɐ", "ʃ"):
            self.assertFalse(recipe_keeps(recipe, sample), f"unexpected keep for {sample!r}")

    def test_ascii_15bit_rejects_non_ascii(self) -> None:
        recipe = _recipe_by_name("ascii_15bit")
        self.assertTrue(recipe_keeps(recipe, "abc_123"))
        self.assertFalse(recipe_keeps(recipe, "café"))
        self.assertFalse(recipe_keeps(recipe, "\u4e2d"))


class RecipeOrderingTests(unittest.TestCase):
    def test_recipes_are_ordered_16bit_then_15bit(self) -> None:
        names = [recipe.name for recipe in RECIPES]
        self.assertEqual(names, ["latin_16bit", "ascii_15bit"])

    def test_first_recipe_targets_full_size(self) -> None:
        latin = _recipe_by_name("latin_16bit")
        self.assertEqual(latin.pair_table_size, 1 << 16)
        self.assertEqual(latin.tail_table_size, 1 << 8)
        self.assertEqual(latin.pair_index_bits, 16)

    def test_fallback_recipe_targets_15_bit(self) -> None:
        ascii_recipe = _recipe_by_name("ascii_15bit")
        self.assertEqual(ascii_recipe.pair_table_size, 1 << 15)
        self.assertEqual(ascii_recipe.tail_table_size, 1 << 8)
        self.assertEqual(ascii_recipe.pair_index_bits, 15)


class FilterCandidatesTests(unittest.TestCase):
    def test_filter_drops_non_latin_scripts_for_latin_16bit(self) -> None:
        recipe = _recipe_by_name("latin_16bit")
        candidates = [
            TokenEntry(rank=1, token_bytes=b"foo", token_text="foo"),
            TokenEntry(rank=2, token_bytes="\u4e2d".encode("utf-8"), token_text="\u4e2d"),
            TokenEntry(rank=3, token_bytes="café".encode("utf-8"), token_text="café"),
            TokenEntry(rank=4, token_bytes="कश".encode("utf-8"), token_text="कश"),
            TokenEntry(rank=5, token_bytes="ｍgtr".encode("utf-8"), token_text="ｍgtr"),
        ]
        kept = filter_candidates_for_recipe(candidates, recipe)
        self.assertEqual([entry.token_text for entry in kept], ["foo", "café"])

    def test_filter_keeps_only_ascii_for_ascii_15bit(self) -> None:
        recipe = _recipe_by_name("ascii_15bit")
        candidates = [
            TokenEntry(rank=1, token_bytes=b"foo", token_text="foo"),
            TokenEntry(rank=2, token_bytes="café".encode("utf-8"), token_text="café"),
            TokenEntry(rank=3, token_bytes=b"_a1", token_text="_a1"),
        ]
        kept = filter_candidates_for_recipe(candidates, recipe)
        self.assertEqual([entry.token_text for entry in kept], ["foo", "_a1"])

    def test_filter_drops_tokens_longer_than_default_limit(self) -> None:
        recipe = _recipe_by_name("ascii_15bit")
        candidates = [
            TokenEntry(rank=1, token_bytes=b"abcdef", token_text="abcdef"),
            TokenEntry(rank=2, token_bytes=b"abcdefg", token_text="abcdefg"),
        ]

        kept = filter_candidates_for_recipe(candidates, recipe)

        self.assertEqual([entry.token_text for entry in kept], ["abcdef"])


class EvaluateRecipesTests(unittest.TestCase):
    def _build_candidates(self, count: int, prefix: str) -> list[Any]:
        # Deterministic ASCII tokens that pass both recipe filters (max length 6).
        return [
            TokenEntry(rank=index, token_bytes=f"{prefix}{index}".encode("ascii"), token_text=f"{prefix}{index}")
            for index in range(count)
        ]

    def test_returns_first_feasible_recipe(self) -> None:
        candidates = self._build_candidates(70_000, "t")
        attempts, chosen = evaluate_recipes(candidates, selected_recipes=RECIPES)
        self.assertEqual([attempt.feasible for attempt in attempts], [True, True])
        self.assertIsNotNone(chosen)
        assert chosen is not None
        recipe, pair_entries, tail_entries = chosen
        self.assertEqual(recipe.name, "latin_16bit")
        self.assertEqual(len(pair_entries), recipe.pair_table_size)
        self.assertEqual(len(tail_entries), recipe.tail_table_size)

    def test_falls_back_when_first_recipe_short(self) -> None:
        candidates = self._build_candidates(40_000, "t")
        attempts, chosen = evaluate_recipes(candidates, selected_recipes=RECIPES)
        self.assertEqual([attempt.feasible for attempt in attempts], [False, True])
        self.assertIsNotNone(chosen)
        assert chosen is not None
        recipe, pair_entries, tail_entries = chosen
        self.assertEqual(recipe.name, "ascii_15bit")
        self.assertEqual(len(pair_entries), recipe.pair_table_size)
        self.assertEqual(len(tail_entries), recipe.tail_table_size)

    def test_returns_none_when_no_recipe_feasible(self) -> None:
        candidates = self._build_candidates(100, "tok")
        attempts, chosen = evaluate_recipes(candidates, selected_recipes=RECIPES)
        self.assertTrue(all(not attempt.feasible for attempt in attempts))
        self.assertIsNone(chosen)


class WriteLookupTableTests(unittest.TestCase):
    def test_metadata_records_recipe_and_dimensions(self) -> None:
        import tempfile

        recipe = _recipe_by_name("ascii_15bit")
        # Use a tiny stand-in recipe so the test stays fast: clone the ASCII
        # recipe with deliberately small table sizes.
        small_recipe = Recipe(
            name=recipe.name,
            description=recipe.description,
            pair_table_size=4,
            tail_table_size=2,
            pair_index_bits=2,
            tail_index_bits=1,
            require_ascii=True,
            allowed_non_ascii_ranges=(),
        )
        candidates = [
            TokenEntry(rank=index, token_bytes=f"a{index}".encode("ascii"), token_text=f"a{index}")
            for index in range(small_recipe.pair_table_size + small_recipe.tail_table_size)
        ]
        pair_entries, tail_entries = select_subset(
            candidates,
            pair_table_size=small_recipe.pair_table_size,
            tail_table_size=small_recipe.tail_table_size,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            input_path = Path("data/token_vocab/test_vocab.tiktoken")
            paths = write_lookup_table(
                input_path=input_path,
                output_dir=output_dir,
                recipe=small_recipe,
                pair_entries=pair_entries,
                tail_entries=tail_entries,
                max_token_length=10,
            )
            metadata_path = paths[2]
            payload = json.loads(metadata_path.read_text())

        self.assertEqual(payload["recipe_name"], "ascii_15bit")
        self.assertEqual(payload["pair_table_size"], 4)
        self.assertEqual(payload["tail_table_size"], 2)
        self.assertEqual(payload["pair_index_bits"], 2)
        self.assertEqual(payload["tail_index_bits"], 1)
        self.assertEqual(payload["allowed_character_policy"], small_recipe.allowed_character_policy)
        self.assertEqual(payload["allowed_non_ascii_ranges"], [])
        self.assertEqual(payload["max_token_length"], 10)

    def test_metadata_records_allowed_latin_ranges(self) -> None:
        import tempfile

        latin_recipe = _recipe_by_name("latin_16bit")
        small_recipe = Recipe(
            name=latin_recipe.name,
            description=latin_recipe.description,
            pair_table_size=4,
            tail_table_size=2,
            pair_index_bits=2,
            tail_index_bits=1,
            require_ascii=False,
            allowed_non_ascii_ranges=latin_recipe.allowed_non_ascii_ranges,
        )
        candidates = [
            TokenEntry(rank=index, token_bytes=f"a{index}".encode("ascii"), token_text=f"a{index}")
            for index in range(small_recipe.pair_table_size + small_recipe.tail_table_size)
        ]
        pair_entries, tail_entries = select_subset(
            candidates,
            pair_table_size=small_recipe.pair_table_size,
            tail_table_size=small_recipe.tail_table_size,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            input_path = Path("data/token_vocab/test_vocab.tiktoken")
            paths = write_lookup_table(
                input_path=input_path,
                output_dir=output_dir,
                recipe=small_recipe,
                pair_entries=pair_entries,
                tail_entries=tail_entries,
                max_token_length=10,
            )
            metadata_path = paths[2]
            payload = json.loads(metadata_path.read_text())

        self.assertEqual(payload["recipe_name"], "latin_16bit")
        self.assertEqual(
            payload["allowed_non_ascii_ranges"],
            [
                {
                    "start": "U+00C0",
                    "end": "U+00FF",
                    "start_codepoint": 0x00C0,
                    "end_codepoint": 0x00FF,
                },
                {
                    "start": "U+0100",
                    "end": "U+017F",
                    "start_codepoint": 0x0100,
                    "end_codepoint": 0x017F,
                },
            ],
        )
        self.assertIn("U+00C0-U+00FF", payload["allowed_character_policy"])
        self.assertIn("U+0100-U+017F", payload["allowed_character_policy"])


if __name__ == "__main__":
    unittest.main()
