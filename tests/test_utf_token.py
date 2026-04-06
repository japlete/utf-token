from __future__ import annotations

import unittest
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from uuid import UUID

from utf_token import IdTokenBiMap, frombase64, frombytes, fromhex, fromuuid
from utf_token._reversible import FORMAT_VERSION
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
        self.assertEqual(
            frombytes(b"\x12\x34\xab", vocab="gemma4"),
            GEMMA4_PAIR_TABLE[0x1234] + GEMMA4_TAIL_TABLE[0xAB],
        )

    def test_frombytes_truncates_before_encoding(self) -> None:
        self.assertEqual(frombytes(b"\x12\x34\xab", truncate_bytes=2), PAIR_TABLE[0x1234])
        self.assertEqual(frombytes(b"\x12\x34\xab", truncate_bytes=1), TAIL_TABLE[0x12])

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
        self.assertEqual(
            fromhex("1234ab", vocab="gemma4"),
            GEMMA4_PAIR_TABLE[0x1234] + GEMMA4_TAIL_TABLE[0xAB],
        )

    def test_format_adapters_apply_truncation_after_decoding(self) -> None:
        zero_uuid = UUID("00000000-0000-0000-0000-000000000000")
        self.assertEqual(fromhex("1234ab", truncate_bytes=2), PAIR_TABLE[0x1234])
        self.assertEqual(frombase64("EjSr", truncate_bytes=2), PAIR_TABLE[0x1234])
        self.assertEqual(fromuuid(zero_uuid, truncate_bytes=2), PAIR_TABLE[0x0000])

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

    def test_truncate_bytes_larger_than_input_is_noop(self) -> None:
        raw = b"\x12\x34\xab"
        self.assertEqual(frombytes(raw, truncate_bytes=10), frombytes(raw))
        self.assertEqual(fromhex("1234ab", truncate_bytes=10), frombytes(raw))

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
        self.assertEqual(encoded, PAIR_TABLE[0x1234])
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

    def test_missing_reverse_lookup_returns_none(self) -> None:
        codec = IdTokenBiMap()
        self.assertIsNone(codec.tobytes("missing"))
        self.assertIsNone(codec.tohex("missing"))
        self.assertIsNone(codec.tobase64("missing"))
        self.assertIsNone(codec.touuid("missing"))

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
        self.assertEqual(next(result), PAIR_TABLE[0x0000])
        self.assertEqual(seen, ["first"])
        self.assertEqual(next(result), PAIR_TABLE[0x0001])
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

    def test_reuses_same_output_for_repeated_bytes_and_vocab(self) -> None:
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
        codec = IdTokenBiMap()
        pair_value = bytes.fromhex("1f08")
        colliding_triplet = bytes.fromhex("000127")

        stage1_pair = frombytes(pair_value)
        stage1_triplet = frombytes(colliding_triplet)
        self.assertEqual(stage1_pair, stage1_triplet)

        first = codec.frombytes(pair_value)
        second = codec.frombytes(colliding_triplet)

        self.assertEqual(first, stage1_pair)
        self.assertNotEqual(second, stage1_triplet)
        self.assertEqual(codec.tobytes(first), pair_value)
        self.assertEqual(codec.tobytes(second), colliding_triplet)
        self.assertEqual(codec.frombytes(colliding_triplet), second)

    def test_remaps_collision_created_by_truncation(self) -> None:
        codec = IdTokenBiMap()
        first_raw = bytes.fromhex("123400")
        second_raw = bytes.fromhex("1234ff")

        first = codec.frombytes(first_raw, truncate_bytes=2)
        second = codec.frombytes(second_raw, truncate_bytes=2)

        self.assertEqual(first, PAIR_TABLE[0x1234])
        self.assertEqual(second, PAIR_TABLE[0x1235])
        self.assertEqual(codec.tobytes(first), first_raw)
        self.assertEqual(codec.tobytes(second), second_raw)

    def test_supports_mixed_vocabs(self) -> None:
        codec = IdTokenBiMap()
        raw = b"\x00\x01"

        o200k_encoded = codec.frombytes(raw, vocab="o200k")
        gemma4_encoded = codec.frombytes(raw, vocab="gemma4")

        self.assertEqual(codec.tobytes(o200k_encoded), raw)
        self.assertEqual(codec.tobytes(gemma4_encoded), raw)

        payload = codec.to_dict()
        mappings = cast(dict[str, object], payload["mappings"])
        self.assertIsInstance(mappings, dict)
        if o200k_encoded == gemma4_encoded:
            shared = cast(dict[str, object], mappings[o200k_encoded])
            self.assertIsInstance(shared, dict)
            self.assertEqual(
                shared["encodings"],
                [
                    {"vocab": "gemma4", "truncate_bytes": None},
                    {"vocab": "o200k", "truncate_bytes": None},
                ],
            )
        else:
            o200k_entry = cast(dict[str, object], mappings[o200k_encoded])
            gemma4_entry = cast(dict[str, object], mappings[gemma4_encoded])
            self.assertIsInstance(o200k_entry, dict)
            self.assertIsInstance(gemma4_entry, dict)
            self.assertEqual(
                o200k_entry["encodings"],
                [{"vocab": "o200k", "truncate_bytes": None}],
            )
            self.assertEqual(
                gemma4_entry["encodings"],
                [{"vocab": "gemma4", "truncate_bytes": None}],
            )

    def test_dict_and_json_round_trip(self) -> None:
        codec = IdTokenBiMap()
        raw = b"\x00\x01"
        o200k_encoded = codec.frombytes(raw, vocab="o200k")
        o200k_oversized = codec.frombytes(raw, vocab="o200k", truncate_bytes=5)
        o200k_truncated = codec.frombytes(raw, vocab="o200k", truncate_bytes=1)
        gemma4_encoded = codec.frombytes(raw, vocab="gemma4")

        self.assertEqual(o200k_encoded, o200k_oversized)

        payload = codec.to_dict()
        self.assertEqual(payload["format_version"], FORMAT_VERSION)
        mappings = cast(dict[str, object], payload["mappings"])
        shared_entry = cast(dict[str, object], mappings[o200k_encoded])
        self.assertEqual(
            shared_entry["encodings"],
            [
                {"vocab": "o200k", "truncate_bytes": None},
                {"vocab": "o200k", "truncate_bytes": 5},
            ],
        )
        truncated_entry = cast(dict[str, object], mappings[o200k_truncated])
        self.assertEqual(
            truncated_entry["encodings"],
            [{"vocab": "o200k", "truncate_bytes": 1}],
        )

        clone = IdTokenBiMap.from_dict(payload)
        self.assertEqual(clone.tobytes(o200k_encoded), raw)
        self.assertEqual(clone.tobytes(o200k_truncated), raw)
        self.assertEqual(clone.tobytes(gemma4_encoded), raw)
        self.assertEqual(clone.frombytes(raw, vocab="o200k"), o200k_encoded)
        self.assertEqual(clone.frombytes(raw, vocab="o200k", truncate_bytes=5), o200k_oversized)
        self.assertEqual(clone.frombytes(raw, vocab="o200k", truncate_bytes=1), o200k_truncated)
        self.assertEqual(clone.frombytes(raw, vocab="gemma4"), gemma4_encoded)

        json_clone = IdTokenBiMap.from_json(codec.to_json(indent=2))
        self.assertEqual(json_clone.tobytes(o200k_encoded), raw)
        self.assertEqual(json_clone.tobytes(o200k_truncated), raw)
        self.assertEqual(json_clone.tobytes(gemma4_encoded), raw)

    def test_import_rejects_conflicting_forward_mapping(self) -> None:
        with self.assertRaises(ValueError):
            IdTokenBiMap.from_dict(
                {
                    "format_version": FORMAT_VERSION,
                    "mappings": {
                        "first": {
                            "original_hex": "0001",
                            "encodings": [{"vocab": "o200k", "truncate_bytes": None}],
                        },
                        "second": {
                            "original_hex": "0001",
                            "encodings": [{"vocab": "o200k", "truncate_bytes": None}],
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
