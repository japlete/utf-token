from __future__ import annotations

import unittest
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from uuid import UUID

from utf_token import IdTokenBiMap, frombase64, frombytes, fromhex, fromuuid
from utf_token._reversible import FORMAT_VERSION, StoredMapping
from utf_token._tables import (
    DEFAULT_VOCAB,
    VocabName,
    metadata,
    pair_table,
    supported_vocabs,
    table_spec,
    tail_table,
)


O200K_PAIR_TABLE = pair_table("o200k")
O200K_TAIL_TABLE = tail_table("o200k")
GEMMA4_PAIR_TABLE = pair_table("gemma4")
GEMMA4_TAIL_TABLE = tail_table("gemma4")
O200K_SPEC = table_spec("o200k")
GEMMA4_SPEC = table_spec("gemma4")


def _bitstream_encode(
    data: bytes,
    *,
    pair_tokens: tuple[str, ...],
    tail_tokens: tuple[str, ...],
    pair_index_bits: int,
    tail_index_bits: int,
) -> str:
    """Reference encoder used by the 15-bit assertions, mirrors the production path."""

    if not data:
        return ""
    total_bits = 8 * len(data)
    full_chunks, residual_bits = divmod(total_bits, pair_index_bits)
    value = int.from_bytes(data, byteorder="big", signed=False)
    pair_mask = (1 << pair_index_bits) - 1
    parts: list[str] = []
    for chunk_index in range(full_chunks):
        shift = (full_chunks - 1 - chunk_index) * pair_index_bits + residual_bits
        parts.append(pair_tokens[(value >> shift) & pair_mask])
    if residual_bits == 0:
        return "".join(parts)
    residual_value = value & ((1 << residual_bits) - 1)
    if residual_bits <= tail_index_bits:
        parts.append(tail_tokens[residual_value])
    else:
        parts.append(pair_tokens[residual_value])
    return "".join(parts)


def _o200k_encode(data: bytes) -> str:
    return _bitstream_encode(
        data,
        pair_tokens=O200K_PAIR_TABLE,
        tail_tokens=O200K_TAIL_TABLE,
        pair_index_bits=O200K_SPEC.pair_index_bits,
        tail_index_bits=O200K_SPEC.tail_index_bits,
    )


def _gemma4_encode(data: bytes) -> str:
    return _bitstream_encode(
        data,
        pair_tokens=GEMMA4_PAIR_TABLE,
        tail_tokens=GEMMA4_TAIL_TABLE,
        pair_index_bits=GEMMA4_SPEC.pair_index_bits,
        tail_index_bits=GEMMA4_SPEC.tail_index_bits,
    )


class TableMetadataTests(unittest.TestCase):
    def test_default_vocab_is_o200k(self) -> None:
        self.assertEqual(DEFAULT_VOCAB, "o200k")
        self.assertEqual(frombytes(b"\x12\x34"), frombytes(b"\x12\x34", vocab="o200k"))

    def test_supported_vocabs_lists_public_options(self) -> None:
        self.assertEqual(supported_vocabs(), ("o200k", "gemma4"))

    def test_o200k_uses_15_bit_recipe(self) -> None:
        self.assertEqual(O200K_SPEC.pair_table_size, 1 << 15)
        self.assertEqual(O200K_SPEC.pair_index_bits, 15)
        self.assertEqual(O200K_SPEC.tail_table_size, 1 << 8)
        self.assertEqual(O200K_SPEC.tail_index_bits, 8)
        self.assertEqual(len(O200K_PAIR_TABLE), 1 << 15)
        self.assertEqual(len(O200K_TAIL_TABLE), 1 << 8)

    def test_gemma4_uses_15_bit_recipe(self) -> None:
        # The curated `latin_16bit` recipe is intentionally strict about which
        # non-Latin scripts it admits, and the gemma4 vocab does not supply
        # enough qualifying tokens. The pipeline therefore falls back to the
        # ASCII-only 15-bit recipe.
        self.assertEqual(GEMMA4_SPEC.pair_table_size, 1 << 15)
        self.assertEqual(GEMMA4_SPEC.pair_index_bits, 15)
        self.assertEqual(GEMMA4_SPEC.tail_table_size, 1 << 8)
        self.assertEqual(GEMMA4_SPEC.tail_index_bits, 8)
        self.assertEqual(len(GEMMA4_PAIR_TABLE), 1 << 15)
        self.assertEqual(len(GEMMA4_TAIL_TABLE), 1 << 8)

    def test_metadata_records_recipe(self) -> None:
        self.assertEqual(metadata("o200k")["recipe_name"], "ascii_15bit")
        self.assertEqual(metadata("gemma4")["recipe_name"], "ascii_15bit")

    def test_tail_tables_start_with_preferred_ascii_sequence(self) -> None:
        preferred = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        expected = tuple(preferred)
        self.assertEqual(O200K_TAIL_TABLE[: len(expected)], expected)
        self.assertEqual(GEMMA4_TAIL_TABLE[: len(expected)], expected)

    def test_pair_tables_are_pure_ascii(self) -> None:
        # Both shipped vocabs use the ascii_15bit recipe, so neither pair table
        # may contain non-ASCII characters that the LLM benchmarks have shown
        # are mistranscribed (Devanagari, Arabic, Cuneiform, Mathematical
        # styled Latin, Fullwidth Latin, etc.).
        for vocab_name, table in (("o200k", O200K_PAIR_TABLE), ("gemma4", GEMMA4_PAIR_TABLE)):
            for token in table:
                self.assertTrue(
                    all(ch == "_" or (ch.isascii() and ch.isalnum()) for ch in token),
                    f"non-ASCII token in {vocab_name} pair table: {token!r}",
                )

    def test_tail_tables_are_pure_ascii(self) -> None:
        for vocab_name, table in (("o200k", O200K_TAIL_TABLE), ("gemma4", GEMMA4_TAIL_TABLE)):
            for token in table:
                self.assertTrue(
                    all(ch == "_" or (ch.isascii() and ch.isalnum()) for ch in token),
                    f"non-ASCII token in {vocab_name} tail table: {token!r}",
                )


class FromBytesGemma4Tests(unittest.TestCase):
    """Cover the 15-bit bitstream path through gemma4."""

    def test_empty_returns_empty_string(self) -> None:
        self.assertEqual(frombytes(b"", vocab="gemma4"), "")

    def test_single_byte_uses_tail_table(self) -> None:
        for byte_value in (0x00, 0x12, 0xAB, 0xFF):
            data = bytes([byte_value])
            self.assertEqual(frombytes(data, vocab="gemma4"), GEMMA4_TAIL_TABLE[byte_value])

    def test_two_bytes_split_into_15_plus_1(self) -> None:
        for raw in (b"\x00\x00", b"\x00\x01", b"\x12\x34", b"\xff\xff"):
            value = int.from_bytes(raw, byteorder="big", signed=False)
            expected = GEMMA4_PAIR_TABLE[(value >> 1) & 0x7FFF] + GEMMA4_TAIL_TABLE[value & 1]
            self.assertEqual(frombytes(raw, vocab="gemma4"), expected)

    def test_three_bytes_split_into_15_plus_9(self) -> None:
        raw = b"\x12\x34\xab"
        value = int.from_bytes(raw, byteorder="big", signed=False)
        expected = (
            GEMMA4_PAIR_TABLE[(value >> 9) & 0x7FFF] + GEMMA4_PAIR_TABLE[value & 0x1FF]
        )
        self.assertEqual(frombytes(raw, vocab="gemma4"), expected)

    def test_truncates_before_encoding(self) -> None:
        raw = b"\x12\x34\xab"
        self.assertEqual(
            frombytes(raw, truncate_bytes=2, vocab="gemma4"),
            _gemma4_encode(b"\x12\x34"),
        )
        self.assertEqual(
            frombytes(raw, truncate_bytes=1, vocab="gemma4"),
            GEMMA4_TAIL_TABLE[0x12],
        )

    def test_iterable_supports_gemma4_vocab(self) -> None:
        result = frombytes([b"\x00\x00", b"\x00\x01"], vocab="gemma4")
        self.assertEqual(list(result), [_gemma4_encode(b"\x00\x00"), _gemma4_encode(b"\x00\x01")])

    def test_uuid_round_trip(self) -> None:
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        expected = (GEMMA4_PAIR_TABLE[0] * 8) + GEMMA4_TAIL_TABLE[0]
        self.assertEqual(fromuuid(zero_uuid, vocab="gemma4"), expected)
        self.assertEqual(fromuuid(str(zero_uuid), vocab="gemma4"), expected)
        self.assertEqual(frombytes(zero_uuid.bytes, vocab="gemma4"), expected)

    def test_bitstream_residual_widths(self) -> None:
        cases = [
            b"",
            b"\xab",
            b"\xab\xcd",
            b"\xab\xcd\xef",
            b"\xab\xcd\xef\x01",
            bytes(range(7)),
            bytes(range(15)),
            bytes(range(16)),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(frombytes(raw, vocab="gemma4"), _gemma4_encode(raw))


class FromBytesBitstreamTests(unittest.TestCase):
    """Cover the 15-bit bitstream path through o200k."""

    def test_empty_returns_empty_string(self) -> None:
        self.assertEqual(frombytes(b""), "")

    def test_single_byte_uses_tail_table(self) -> None:
        for byte_value in (0x00, 0x12, 0xAB, 0xFF):
            data = bytes([byte_value])
            self.assertEqual(frombytes(data), O200K_TAIL_TABLE[byte_value])

    def test_two_bytes_split_into_15_plus_1(self) -> None:
        # 16 bits -> 1 chunk of 15 bits (top) + 1 residual bit (tail).
        for raw in (b"\x00\x00", b"\x00\x01", b"\x12\x34", b"\xff\xff"):
            value = int.from_bytes(raw, byteorder="big", signed=False)
            expected = O200K_PAIR_TABLE[(value >> 1) & 0x7FFF] + O200K_TAIL_TABLE[value & 1]
            self.assertEqual(frombytes(raw), expected)

    def test_three_bytes_split_into_15_plus_9(self) -> None:
        # 24 bits -> 1 chunk of 15 bits + 9 residual bits (pair table, left-padded).
        raw = b"\x12\x34\xab"
        value = int.from_bytes(raw, byteorder="big", signed=False)
        expected = O200K_PAIR_TABLE[(value >> 9) & 0x7FFF] + O200K_PAIR_TABLE[value & 0x1FF]
        self.assertEqual(frombytes(raw), expected)

    def test_uuid_uses_8_pair_plus_tail(self) -> None:
        # 128 bits -> 8 chunks of 15 bits + 8 residual bits (tail table).
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        expected = (O200K_PAIR_TABLE[0] * 8) + O200K_TAIL_TABLE[0]
        self.assertEqual(fromuuid(zero_uuid), expected)
        self.assertEqual(fromuuid(str(zero_uuid)), expected)
        self.assertEqual(frombytes(zero_uuid.bytes), expected)

    def test_bitstream_residual_widths(self) -> None:
        # Spot-check encodings whose residual bit widths exercise both branches.
        cases = [
            b"",
            b"\xab",
            b"\xab\xcd",
            b"\xab\xcd\xef",
            b"\xab\xcd\xef\x01",
            bytes(range(7)),
            bytes(range(15)),
            bytes(range(16)),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(frombytes(raw), _o200k_encode(raw))

    def test_truncates_before_encoding(self) -> None:
        raw = b"\x12\x34\xab"
        self.assertEqual(frombytes(raw, truncate_bytes=2), _o200k_encode(b"\x12\x34"))
        self.assertEqual(frombytes(raw, truncate_bytes=1), O200K_TAIL_TABLE[0x12])

    def test_truncate_bytes_larger_than_input_is_noop(self) -> None:
        raw = b"\x12\x34\xab"
        self.assertEqual(frombytes(raw, truncate_bytes=10), frombytes(raw))
        self.assertEqual(fromhex("1234ab", truncate_bytes=10), frombytes(raw))


class FormatAdaptersTests(unittest.TestCase):
    def test_fromhex_matches_bytes_fromhex_behavior(self) -> None:
        self.assertEqual(fromhex("00 01 12 34 ab"), frombytes(bytes.fromhex("0001 1234 ab")))

    def test_fromhex_iterable_returns_expected_values(self) -> None:
        result = fromhex(["0000", "0001"])
        self.assertEqual(list(result), [_o200k_encode(b"\x00\x00"), _o200k_encode(b"\x00\x01")])

    def test_fromhex_supports_gemma4_vocab(self) -> None:
        self.assertEqual(fromhex("1234ab", vocab="gemma4"), _gemma4_encode(b"\x12\x34\xab"))

    def test_format_adapters_apply_truncation_after_decoding(self) -> None:
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        expected_two = _o200k_encode(b"\x12\x34")
        self.assertEqual(fromhex("1234ab", truncate_bytes=2), expected_two)
        self.assertEqual(frombase64("EjSr", truncate_bytes=2), expected_two)
        self.assertEqual(
            fromuuid(zero_uuid, truncate_bytes=2),
            _o200k_encode(b"\x00\x00"),
        )

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

    def test_invalid_vocab_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            frombytes(b"\x00\x00", vocab=cast(VocabName, "invalid"))

    def test_invalid_truncate_bytes_raises_value_error(self) -> None:
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        for truncate_bytes in (0, -1):
            with self.subTest(truncate_bytes=truncate_bytes):
                with self.assertRaises(ValueError):
                    frombytes(b"\x00\x01", truncate_bytes=truncate_bytes)
                with self.assertRaises(ValueError):
                    fromhex("0001", truncate_bytes=truncate_bytes)
                with self.assertRaises(ValueError):
                    frombase64("AAE=", truncate_bytes=truncate_bytes)
                with self.assertRaises(ValueError):
                    fromuuid(zero_uuid, truncate_bytes=truncate_bytes)


class FromBytesIterableTests(unittest.TestCase):
    def test_iterable_is_lazy_iterator(self) -> None:
        seen: list[str] = []

        def items() -> Iterator[bytes]:
            seen.append("first")
            yield b"\x00\x00"
            seen.append("second")
            yield b"\x00\x01"

        result = frombytes(items())
        self.assertIsInstance(result, Iterator)
        self.assertEqual(seen, [])
        self.assertEqual(next(result), _o200k_encode(b"\x00\x00"))
        self.assertEqual(seen, ["first"])
        self.assertEqual(next(result), _o200k_encode(b"\x00\x01"))
        self.assertEqual(seen, ["first", "second"])


class IdTokenBiMapTests(unittest.TestCase):
    def test_round_trips_scalar_bytes(self) -> None:
        codec = IdTokenBiMap()
        encoded = codec.frombytes(b"\x12\x34\xab")
        self.assertEqual(codec.tobytes(encoded), b"\x12\x34\xab")
        self.assertEqual(codec.tohex(encoded), "1234ab")
        self.assertEqual(codec.tobase64(encoded), "EjSr")

    def test_truncated_encoding_reverses_to_full_original(self) -> None:
        codec = IdTokenBiMap()
        encoded = codec.frombytes(b"\x12\x34\xab", truncate_bytes=2)
        self.assertEqual(encoded, _o200k_encode(b"\x12\x34"))
        self.assertEqual(codec.tobytes(encoded), b"\x12\x34\xab")
        self.assertEqual(codec.tohex(encoded), "1234ab")

    def test_round_trips_format_adapters(self) -> None:
        codec = IdTokenBiMap()
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")

        hex_encoded = codec.fromhex("00 01 ab")
        base64_encoded = codec.frombase64("AAE=")
        uuid_encoded = codec.fromuuid(zero_uuid)

        self.assertEqual(codec.tobytes(hex_encoded), bytes.fromhex("0001ab"))
        self.assertEqual(codec.tohex(hex_encoded), "0001ab")
        self.assertEqual(codec.tobase64(base64_encoded), "AAE=")
        self.assertEqual(codec.touuid(uuid_encoded), zero_uuid)

    def test_round_trips_under_gemma4_vocab(self) -> None:
        codec = IdTokenBiMap("gemma4")
        raw = b"\x12\x34\xab"
        encoded = codec.frombytes(raw)
        self.assertEqual(encoded, _gemma4_encode(raw))
        self.assertEqual(codec.tobytes(encoded), raw)

    def test_missing_reverse_lookup_returns_none_when_map_empty(self) -> None:
        codec = IdTokenBiMap()
        self.assertIsNone(codec.tobytes("missing"))
        self.assertIsNone(codec.tohex("missing"))
        self.assertIsNone(codec.tobase64("missing"))
        self.assertIsNone(codec.touuid("missing"))

    def test_missing_reverse_lookup_raises_when_errors_raise(self) -> None:
        codec = IdTokenBiMap()
        codec.frombytes(b"\x00\x01")
        with self.assertRaises(KeyError):
            codec.tobytes("definitely_not_stored", errors="raise")
        with self.assertRaises(KeyError):
            codec.tohex("definitely_not_stored", errors="raise")
        with self.assertRaises(KeyError):
            codec.tobase64("definitely_not_stored", errors="raise")
        with self.assertRaises(KeyError):
            codec.touuid("definitely_not_stored", errors="raise")

    def test_errors_raise_on_empty_map_still_raises(self) -> None:
        codec = IdTokenBiMap()
        with self.assertRaises(KeyError):
            codec.tobytes("missing", errors="raise")

    def test_invalid_errors_value_raises(self) -> None:
        from utf_token._reversible import _validate_error_mode

        with self.assertRaises(ValueError):
            _validate_error_mode("ignore")

    def test_exact_lookup_does_not_heal(self) -> None:
        codec = IdTokenBiMap()
        first = codec.frombytes(b"\x00\x01")
        second = codec.frombytes(b"\x00\x02")
        self.assertEqual(codec.tobytes(first), b"\x00\x01")
        self.assertEqual(codec.tobytes(second), b"\x00\x02")

    def test_heal_substitution(self) -> None:
        codec = IdTokenBiMap("gemma4")
        target = codec.frombytes(bytes.fromhex("3596e50a4202255eeeb0f5964097f49a"))
        if not target:
            self.skipTest("encoded target unexpectedly empty")
        garbled = target[:-1] + ("Z" if target[-1] != "Z" else "Y")
        self.assertNotIn(garbled, codec)
        self.assertEqual(codec.tobytes(garbled), codec.tobytes(target))

    def test_heal_deletion(self) -> None:
        codec = IdTokenBiMap("gemma4")
        target = codec.frombytes(bytes.fromhex("3596e50a4202255eeeb0f5964097f49a"))
        if len(target) < 2:
            self.skipTest("encoded target too short to delete a character")
        garbled = target[:-1]
        self.assertNotIn(garbled, codec)
        self.assertEqual(codec.tobytes(garbled), codec.tobytes(target))

    def test_heal_insertion_of_non_ascii(self) -> None:
        codec = IdTokenBiMap("gemma4")
        target = codec.frombytes(bytes.fromhex("3596e50a4202255eeeb0f5964097f49a"))
        if len(target) < 4:
            self.skipTest("encoded target too short for insertion test")
        midpoint = len(target) // 2
        garbled = target[:midpoint] + "\u4e2d" + target[midpoint:]
        self.assertNotIn(garbled, codec)
        self.assertEqual(codec.tobytes(garbled), codec.tobytes(target))

    def test_heal_whitespace_insertion(self) -> None:
        codec = IdTokenBiMap("gemma4")
        target = codec.frombytes(bytes.fromhex("3596e50a4202255eeeb0f5964097f49a"))
        if len(target) < 4:
            self.skipTest("encoded target too short for whitespace test")
        midpoint = len(target) // 2
        garbled = target[:midpoint] + " " + target[midpoint:]
        self.assertNotIn(garbled, codec)
        self.assertEqual(codec.tobytes(garbled), codec.tobytes(target))

    def test_heal_picks_nearest_among_distinct_identifiers(self) -> None:
        codec = IdTokenBiMap("gemma4")
        first = codec.frombytes(bytes.fromhex("3596e50a4202255eeeb0f5964097f49a"))
        second = codec.frombytes(bytes.fromhex("7a80ac94a5e07832ff53aa34473402bd"))
        garbled_first = first[:-1] + ("Z" if first[-1] != "Z" else "Y")
        self.assertNotIn(garbled_first, codec)
        self.assertEqual(codec.tobytes(garbled_first), codec.tobytes(first))
        self.assertNotEqual(codec.tobytes(garbled_first), codec.tobytes(second))

    def test_heal_tie_break_is_deterministic(self) -> None:
        codec = IdTokenBiMap()
        codec._reverse_map["bar"] = StoredMapping(
            original_bytes=b"\x02",
            encodings={None},
        )
        codec._reverse_map["baz"] = StoredMapping(
            original_bytes=b"\x03",
            encodings={None},
        )
        # Edit distance from "ba_" to "bar" is 1, same as to "baz".
        # The smaller string in lexicographic order ("bar") wins.
        self.assertEqual(codec.tobytes("ba_"), b"\x02")

    def test_heal_returns_none_with_empty_map(self) -> None:
        codec = IdTokenBiMap()
        self.assertIsNone(codec.tobytes("anything", errors="fix"))

    def test_contains_reports_exact_membership(self) -> None:
        codec = IdTokenBiMap()
        encoded = codec.frombytes(b"\x00\x01")
        self.assertIn(encoded, codec)
        self.assertNotIn("not_an_identifier", codec)
        self.assertNotIn(123, codec)

    def test_forward_iterable_is_lazy_iterator(self) -> None:
        codec = IdTokenBiMap()
        seen: list[str] = []

        def items() -> Iterator[bytes]:
            seen.append("first")
            yield b"\x00\x00"
            seen.append("second")
            yield b"\x00\x01"

        result = codec.frombytes(items())
        self.assertIsInstance(result, Iterator)
        self.assertEqual(seen, [])
        self.assertEqual(next(result), _o200k_encode(b"\x00\x00"))
        self.assertEqual(seen, ["first"])
        self.assertEqual(next(result), _o200k_encode(b"\x00\x01"))
        self.assertEqual(seen, ["first", "second"])

    def test_reverse_iterable_is_lazy_iterator(self) -> None:
        codec = IdTokenBiMap()
        first = codec.frombytes(b"\x00\x00")
        second = codec.frombytes(b"\x00\x01")
        seen: list[str] = []

        def items() -> Iterator[str]:
            seen.append("first")
            yield first
            seen.append("second")
            yield second

        result = codec.tobytes(items())
        self.assertIsInstance(result, Iterator)
        self.assertEqual(seen, [])
        self.assertEqual(next(result), b"\x00\x00")
        self.assertEqual(seen, ["first"])
        self.assertEqual(next(result), b"\x00\x01")
        self.assertEqual(seen, ["first", "second"])

    def test_reuses_same_output_for_repeated_bytes(self) -> None:
        codec = IdTokenBiMap()
        first = codec.frombytes(b"\x00\x01\x27")
        second = codec.frombytes(b"\x00\x01\x27")
        self.assertEqual(first, second)
        self.assertEqual(codec.tobytes(first), b"\x00\x01\x27")

    def test_reuses_same_output_for_repeated_truncation_setting(self) -> None:
        codec = IdTokenBiMap()
        first = codec.frombytes(b"\x12\x34\xab", truncate_bytes=2)
        second = codec.frombytes(b"\x12\x34\xab", truncate_bytes=2)
        self.assertEqual(first, second)

    def test_same_original_supports_multiple_truncation_lengths(self) -> None:
        codec = IdTokenBiMap()
        raw = b"\x12\x34\xab"

        full = codec.frombytes(raw)
        truncated = codec.frombytes(raw, truncate_bytes=2)
        oversized = codec.frombytes(raw, truncate_bytes=10)

        self.assertNotEqual(full, truncated)
        self.assertEqual(full, oversized)
        self.assertEqual(codec.tobytes(full), raw)
        self.assertEqual(codec.tobytes(truncated), raw)

    def test_remaps_known_collision(self) -> None:
        # The 15-bit bitstream encoder still admits natural collisions: two
        # distinct byte inputs whose forward encodings happen to be the same
        # string. The codec must hand the second input a different encoding so
        # the reverse map can disambiguate them.
        for vocab in ("o200k", "gemma4"):
            with self.subTest(vocab=vocab):
                first_input, second_input = self._find_natural_collision(vocab)

                stage1_first = frombytes(first_input, vocab=vocab)
                stage1_second = frombytes(second_input, vocab=vocab)
                self.assertEqual(stage1_first, stage1_second)

                codec = IdTokenBiMap(vocab)
                first = codec.frombytes(first_input)
                second = codec.frombytes(second_input)

                self.assertEqual(first, stage1_first)
                self.assertNotEqual(second, stage1_second)
                self.assertEqual(codec.tobytes(first), first_input)
                self.assertEqual(codec.tobytes(second), second_input)
                self.assertEqual(codec.frombytes(second_input), second)

    @staticmethod
    def _find_natural_collision(vocab: VocabName) -> tuple[bytes, bytes]:
        # Scan short byte strings until two distinct inputs encode to the same
        # string under the given vocab. The first natural collision is reached
        # well within ~100k samples for both shipped vocabs.
        seen: dict[str, bytes] = {}
        for length in (1, 2, 3, 4):
            for value in range(min(1 << (8 * length), 1 << 18)):
                data = value.to_bytes(length, byteorder="big", signed=False)
                encoded = frombytes(data, vocab=vocab)
                previous = seen.get(encoded)
                if previous is not None and previous != data:
                    return previous, data
                seen[encoded] = data
        raise AssertionError(f"No natural collision found in vocab {vocab!r}")

    def test_remaps_collision_created_by_truncation(self) -> None:
        # Two distinct 3-byte values collapse to the same 2-byte prefix under
        # truncation, so their forward encodings are identical until the codec
        # increments one. This invariant must hold regardless of the underlying
        # encoding scheme (15-bit o200k or 16-bit gemma4).
        vocabs: tuple[VocabName, ...] = ("o200k", "gemma4")
        for vocab in vocabs:
            with self.subTest(vocab=vocab):
                codec = IdTokenBiMap(vocab)
                first_raw = bytes.fromhex("123400")
                second_raw = bytes.fromhex("1234ff")

                first = codec.frombytes(first_raw, truncate_bytes=2)
                second = codec.frombytes(second_raw, truncate_bytes=2)

                self.assertEqual(first, frombytes(b"\x12\x34", vocab=vocab))
                self.assertNotEqual(first, second)
                self.assertEqual(codec.tobytes(first), first_raw)
                self.assertEqual(codec.tobytes(second), second_raw)

    def test_constructor_uses_specified_vocab(self) -> None:
        codec = IdTokenBiMap("gemma4")
        raw = b"\x00\x01"
        encoded = codec.frombytes(raw)
        self.assertEqual(encoded, _gemma4_encode(raw))
        self.assertEqual(codec.tobytes(encoded), raw)

    def test_dict_and_json_round_trip(self) -> None:
        codec = IdTokenBiMap()
        raw = b"\x00\x01"
        encoded = codec.frombytes(raw)
        oversized = codec.frombytes(raw, truncate_bytes=5)
        truncated = codec.frombytes(raw, truncate_bytes=1)

        self.assertEqual(encoded, oversized)

        payload = codec.to_dict()
        self.assertEqual(payload["format_version"], FORMAT_VERSION)
        self.assertEqual(payload["vocab"], "o200k")
        mappings = cast(dict[str, object], payload["mappings"])
        shared_entry = cast(dict[str, object], mappings[encoded])
        self.assertEqual(
            shared_entry["encodings"],
            [
                {"truncate_bytes": None},
                {"truncate_bytes": 5},
            ],
        )
        truncated_entry = cast(dict[str, object], mappings[truncated])
        self.assertEqual(
            truncated_entry["encodings"],
            [{"truncate_bytes": 1}],
        )

        clone = IdTokenBiMap.from_dict(payload)
        self.assertEqual(clone.tobytes(encoded), raw)
        self.assertEqual(clone.tobytes(truncated), raw)
        self.assertEqual(clone.frombytes(raw), encoded)
        self.assertEqual(clone.frombytes(raw, truncate_bytes=5), oversized)
        self.assertEqual(clone.frombytes(raw, truncate_bytes=1), truncated)

        json_clone = IdTokenBiMap.from_json(codec.to_json(indent=2))
        self.assertEqual(json_clone.tobytes(encoded), raw)
        self.assertEqual(json_clone.tobytes(truncated), raw)

    def test_import_rejects_conflicting_forward_mapping(self) -> None:
        with self.assertRaises(ValueError):
            IdTokenBiMap.from_dict(
                {
                    "format_version": FORMAT_VERSION,
                    "vocab": "o200k",
                    "mappings": {
                        "first": {
                            "original_hex": "0001",
                            "encodings": [{"truncate_bytes": None}],
                        },
                        "second": {
                            "original_hex": "0001",
                            "encodings": [{"truncate_bytes": None}],
                        },
                    },
                }
            )

    def test_import_rejects_unsupported_format_version(self) -> None:
        with self.assertRaises(ValueError):
            IdTokenBiMap.from_dict(
                {
                    "format_version": FORMAT_VERSION - 1,
                    "mappings": {},
                }
            )

    def test_touuid_rejects_non_uuid_lengths(self) -> None:
        codec = IdTokenBiMap()
        encoded = codec.frombytes(b"\x00\x01")
        with self.assertRaises(ValueError):
            codec.touuid(encoded)

    def test_invalid_truncate_bytes_raises_value_error(self) -> None:
        codec = IdTokenBiMap()
        with self.assertRaises(ValueError):
            codec.frombytes(b"\x00\x01", truncate_bytes=0)

    def test_thread_safe_per_instance(self) -> None:
        codec = IdTokenBiMap()
        codec.frombytes(bytes.fromhex("1f08"))

        def encode_collision() -> str:
            return codec.frombytes(bytes.fromhex("000127"))

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: encode_collision(), range(32)))

        self.assertEqual(len(set(results)), 1)
        self.assertEqual(codec.tobytes(results[0]), bytes.fromhex("000127"))


if __name__ == "__main__":
    unittest.main()
