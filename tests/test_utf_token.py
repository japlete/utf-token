from __future__ import annotations

import unittest
from collections.abc import Iterator
from uuid import UUID

from utf_token import frombase64, frombytes, fromhex, fromuuid
from utf_token._tables import pair_table, tail_table


PAIR_TABLE = pair_table()
TAIL_TABLE = tail_table()


class UtfTokenTests(unittest.TestCase):
    def test_frombytes_empty_returns_empty_string(self) -> None:
        self.assertEqual(frombytes(b""), "")

    def test_frombytes_known_pairs_use_big_endian_lookup(self) -> None:
        self.assertEqual(frombytes(b"\x00\x00"), "0")
        self.assertEqual(frombytes(b"\x00\x01"), "1")
        self.assertEqual(frombytes(b"\x00\x0a"), "A")
        self.assertEqual(frombytes(b"\x12\x34"), PAIR_TABLE[0x1234])

    def test_frombytes_odd_length_uses_tail_table(self) -> None:
        self.assertEqual(frombytes(b"\x00"), TAIL_TABLE[0x00])
        self.assertEqual(frombytes(b"\xff"), TAIL_TABLE[0xFF])
        self.assertEqual(frombytes(b"\x12\x34\xab"), PAIR_TABLE[0x1234] + TAIL_TABLE[0xAB])

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
        self.assertEqual(next(result), "0")
        self.assertEqual(seen, ["first"])
        self.assertEqual(next(result), "1")
        self.assertEqual(seen, ["first", "second"])

    def test_fromhex_matches_bytes_fromhex_behavior(self) -> None:
        self.assertEqual(fromhex("00 01 12 34 ab"), frombytes(bytes.fromhex("00 01 12 34 ab")))

    def test_fromhex_iterable_returns_expected_values(self) -> None:
        result = fromhex(["0000", "0001"])
        self.assertEqual(list(result), ["0", "1"])

    def test_frombase64_supports_str_and_bytes(self) -> None:
        expected = frombytes(b"\x00\x01")
        self.assertEqual(frombase64("AAE="), expected)
        self.assertEqual(frombase64(b"AAE="), expected)

    def test_frombase64_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            frombase64("@@")

    def test_fromuuid_accepts_object_and_string(self) -> None:
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        expected = "00000000"
        self.assertEqual(fromuuid(zero_uuid), expected)
        self.assertEqual(fromuuid(str(zero_uuid)), expected)

    def test_packaged_tables_have_expected_sizes(self) -> None:
        self.assertEqual(len(PAIR_TABLE), 1 << 16)
        self.assertEqual(len(TAIL_TABLE), 1 << 8)


if __name__ == "__main__":
    unittest.main()
