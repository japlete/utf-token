from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock


MODULE_GLOBALS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "process_token_vocab.py")
)
TokenEntry = cast(type[Any], MODULE_GLOBALS["TokenEntry"])
load_vocab_entries = MODULE_GLOBALS["load_vocab_entries"]
select_subset = MODULE_GLOBALS["select_subset"]


class ProcessTokenVocabTests(unittest.TestCase):
    def test_select_subset_reserves_shortest_entries_for_tail_table(self) -> None:
        candidates = [
            TokenEntry(rank=30, token_bytes=b"bb", token_text="bb"),
            TokenEntry(rank=10, token_bytes=b"a", token_text="a"),
            TokenEntry(rank=20, token_bytes=b"c", token_text="c"),
            TokenEntry(rank=5, token_bytes=b"dd", token_text="dd"),
            TokenEntry(rank=40, token_bytes=b"eee", token_text="eee"),
        ]

        pair_entries, tail_entries = select_subset(
            candidates,
            pair_table_size=2,
            tail_table_size=2,
        )

        self.assertEqual([entry.token_text for entry in tail_entries], ["a", "c"])
        self.assertEqual([entry.token_text for entry in pair_entries], ["dd", "bb"])

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

    def test_load_vocab_entries_rejects_unknown_suffix(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported vocab format"):
            load_vocab_entries(Path("data/token_vocab/unknown_vocab.txt"))


if __name__ == "__main__":
    unittest.main()
