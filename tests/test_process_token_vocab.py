from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from typing import Any, cast


MODULE_GLOBALS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "process_token_vocab.py")
)
TokenEntry = cast(type[Any], MODULE_GLOBALS["TokenEntry"])
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


if __name__ == "__main__":
    unittest.main()
