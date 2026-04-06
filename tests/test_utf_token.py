from __future__ import annotations

import unittest
from collections.abc import Iterator
from typing import cast
from uuid import UUID

from utf_token import frombase64, frombytes, fromhex, fromuuid
from utf_token._tables import DEFAULT_VOCAB, VocabName, pair_table, supported_vocabs, tail_table


PAIR_TABLE = pair_table()
TAIL_TABLE = tail_table()
GEMMA4_PAIR_TABLE = pair_table("gemma4")
GEMMA4_TAIL_TABLE = tail_table("gemma4")


class UtfTokenTests(unittest.TestCase):
    def test_default_vocab_is_o200k(self) -> None:
        self.assertEqual(DEFAULT_VOCAB, "o200k")
        self.assertEqual(frombytes(b"\x12\x34"), frombytes(b"\x12\x34", vocab="o200k"))

    def test_frombytes_empty_returns_empty_string(self) -> None:
        self.assertEqual(frombytes(b""), "")

    def test_frombytes_known_pairs_use_big_endian_lookup(self) -> None:
        self.assertEqual(frombytes(b"\x00\x00"), PAIR_TABLE[0x0000])
        self.assertEqual(frombytes(b"\x00\x01"), PAIR_TABLE[0x0001])
        self.assertEqual(frombytes(b"\x00\x0a"), PAIR_TABLE[0x000A])
        self.assertEqual(frombytes(b"\x12\x34"), PAIR_TABLE[0x1234])

    def test_frombytes_odd_length_uses_tail_table(self) -> None:
        self.assertEqual(frombytes(b"\x00"), TAIL_TABLE[0x00])
        self.assertEqual(frombytes(b"\xff"), TAIL_TABLE[0xFF])
        self.assertEqual(frombytes(b"\x12\x34\xab"), PAIR_TABLE[0x1234] + TAIL_TABLE[0xAB])

    def test_frombytes_supports_gemma4_vocab(self) -> None:
        self.assertEqual(frombytes(b"\x00\x00", vocab="gemma4"), GEMMA4_PAIR_TABLE[0x0000])
        self.assertEqual(frombytes(b"\x12\x34\xab", vocab="gemma4"), GEMMA4_PAIR_TABLE[0x1234] + GEMMA4_TAIL_TABLE[0xAB])

    def test_frombytes_iterable_is_lazy_iterator(self) -> None:
        seen: list[str] = []

        def items() -> Iterator[bytes]:
            seen.append("first")
            yield b"\x00\x00"
            seen.append("second")
            yield b"\x00\x01"

        result = frombytes(items())
        self.assertIsInstance(result, Iterator)
        self.assertEqual(seen, [])
        self.assertEqual(next(result), PAIR_TABLE[0x0000])
        self.assertEqual(seen, ["first"])
        self.assertEqual(next(result), PAIR_TABLE[0x0001])
        self.assertEqual(seen, ["first", "second"])

    def test_frombytes_iterable_supports_gemma4_vocab(self) -> None:
        result = frombytes([b"\x00\x00", b"\x00\x01"], vocab="gemma4")
        self.assertEqual(list(result), [GEMMA4_PAIR_TABLE[0x0000], GEMMA4_PAIR_TABLE[0x0001]])

    def test_fromhex_matches_bytes_fromhex_behavior(self) -> None:
        self.assertEqual(fromhex("00 01 12 34 ab"), frombytes(bytes.fromhex("00 01 12 34 ab")))

    def test_fromhex_iterable_returns_expected_values(self) -> None:
        result = fromhex(["0000", "0001"])
        self.assertEqual(list(result), [PAIR_TABLE[0x0000], PAIR_TABLE[0x0001]])

    def test_fromhex_supports_gemma4_vocab(self) -> None:
        self.assertEqual(fromhex("1234ab", vocab="gemma4"), GEMMA4_PAIR_TABLE[0x1234] + GEMMA4_TAIL_TABLE[0xAB])

    def test_frombase64_supports_str_and_bytes(self) -> None:
        expected = frombytes(b"\x00\x01")
        self.assertEqual(frombase64("AAE="), expected)
        self.assertEqual(frombase64(b"AAE="), expected)

    def test_frombase64_supports_gemma4_vocab(self) -> None:
        expected = frombytes(b"\x00\x01", vocab="gemma4")
        self.assertEqual(frombase64("AAE=", vocab="gemma4"), expected)
        self.assertEqual(frombase64(b"AAE=", vocab="gemma4"), expected)

    def test_frombase64_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            frombase64("@@")

    def test_fromuuid_accepts_object_and_string(self) -> None:
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        expected = PAIR_TABLE[0x0000] * 8
        self.assertEqual(fromuuid(zero_uuid), expected)
        self.assertEqual(fromuuid(str(zero_uuid)), expected)

    def test_fromuuid_supports_gemma4_vocab(self) -> None:
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        expected = GEMMA4_PAIR_TABLE[0x0000] * 8
        self.assertEqual(fromuuid(zero_uuid, vocab="gemma4"), expected)
        self.assertEqual(fromuuid(str(zero_uuid), vocab="gemma4"), expected)

    def test_packaged_tables_have_expected_sizes(self) -> None:
        self.assertEqual(len(PAIR_TABLE), 1 << 16)
        self.assertEqual(len(TAIL_TABLE), 1 << 8)
        self.assertEqual(len(GEMMA4_PAIR_TABLE), 1 << 16)
        self.assertEqual(len(GEMMA4_TAIL_TABLE), 1 << 8)

    def test_tail_table_is_reserved_for_the_shortest_entries(self) -> None:
        self.assertLessEqual(max(map(len, TAIL_TABLE)), len(PAIR_TABLE[0]))
        self.assertLessEqual(max(map(len, GEMMA4_TAIL_TABLE)), len(GEMMA4_PAIR_TABLE[0]))

    def test_supported_vocabs_lists_public_options(self) -> None:
        self.assertEqual(supported_vocabs(), ("o200k", "gemma4"))

    def test_invalid_vocab_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            frombytes(b"\x00\x00", vocab=cast(VocabName, "invalid"))


if __name__ == "__main__":
    unittest.main()
