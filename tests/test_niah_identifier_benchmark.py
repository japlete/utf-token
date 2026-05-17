from __future__ import annotations

import base64
import json
import re
import unittest
from typing import cast
from uuid import UUID

import httpx

from scripts.benchmarks.niah_dataset import (
    IDENTIFIER_FORMAT_INSTRUCTIONS,
    BenchmarkConfig,
    EncodingCondition,
    EncodingName,
    VocabName,
    codec_for_condition,
    estimate_hex_baseline_record_count,
    generate_sample,
    make_payload,
    render_identifier,
    score_response,
    summarize_rows,
)
from scripts.benchmarks.openrouter_client import OpenRouterClient, OpenRouterConfig
from scripts.benchmarks.run_niah_identifier_benchmark import (
    IDENTIFIER_MAX_LENGTHS,
    IDENTIFIER_PATTERNS,
    RUN_ID,
    _load_completed_keys,
    _max_tokens_for_model,
    _reasoning_config_for_model,
    identifier_response_format,
)
from utf_token import IdTokenBiMap


def context_lines(prompt: str) -> list[str]:
    records = prompt.split("Records:\n", maxsplit=1)[1].split("\n\nQuestion:", maxsplit=1)[0]
    return records.splitlines()


class NiahIdentifierBenchmarkTests(unittest.TestCase):
    def test_generate_sample_is_deterministic(self) -> None:
        config = BenchmarkConfig(context_length_target=80, base_seed=11)
        condition = EncodingCondition(encoding="raw_hex", vocab="o200k")

        first = generate_sample(
            config,
            condition,
            sample_index=1,
            record_count=12,
            context_character_target=500,
        )
        second = generate_sample(
            config,
            condition,
            sample_index=1,
            record_count=12,
            context_character_target=500,
        )

        self.assertEqual(first.payload_hex, second.payload_hex)
        self.assertEqual(first.needle_value_text, second.needle_value_text)
        self.assertEqual(first.prompt_hash, second.prompt_hash)

    def test_generate_sample_inserts_needle_at_requested_depth(self) -> None:
        config = BenchmarkConfig(
            context_length_target=100,
            depth_percent=50.0,
        )
        sample = generate_sample(
            config,
            EncodingCondition(encoding="raw_uuid", vocab="o200k"),
            sample_index=2,
            record_count=21,
            context_character_target=700,
        )
        lines = context_lines(sample.prompt)
        needle_index = lines.index(f"key={sample.needle_key}: id={sample.needle_value_text}")

        self.assertAlmostEqual(needle_index / (len(lines) - 1), 0.5, delta=0.15)

    def test_encoding_conditions_represent_same_payload_bytes(self) -> None:
        payload = make_payload(seed=3, payload_bytes=16)
        hex_text = render_identifier(
            payload,
            EncodingCondition(encoding="raw_hex", vocab="o200k"),
            IdTokenBiMap("o200k"),
        )
        base64_text = render_identifier(
            payload,
            EncodingCondition(encoding="raw_base64", vocab="o200k"),
            IdTokenBiMap("o200k"),
        )
        uuid_text = render_identifier(
            payload,
            EncodingCondition(encoding="raw_uuid", vocab="o200k"),
            IdTokenBiMap("o200k"),
        )
        utf_token_condition = EncodingCondition(encoding="utf_token", vocab="gemma4")
        keep_3_condition = EncodingCondition(encoding="utf_token_keep_3", vocab="gemma4")
        utf_token_codec = codec_for_condition(utf_token_condition)
        keep_3_codec = codec_for_condition(keep_3_condition)
        utf_token_text = render_identifier(payload, utf_token_condition, utf_token_codec)
        truncated_utf_token_text = render_identifier(
            payload, keep_3_condition, keep_3_codec
        )

        self.assertEqual(bytes.fromhex(hex_text), payload)
        self.assertEqual(base64.b64decode(base64_text, validate=True), payload)
        self.assertEqual(UUID(uuid_text).bytes, payload)
        self.assertEqual(utf_token_codec.tobytes(utf_token_text), payload)
        self.assertEqual(keep_3_codec.tobytes(truncated_utf_token_text), payload)
        self.assertNotEqual(truncated_utf_token_text, utf_token_text)

    def test_fixed_record_count_keeps_same_number_of_rows(self) -> None:
        config = BenchmarkConfig(context_length_target=100)
        context_character_target = 10_000
        record_count = estimate_hex_baseline_record_count(config, context_character_target)
        hex_sample = generate_sample(
            config,
            EncodingCondition(encoding="raw_hex", vocab="o200k"),
            sample_index=1,
            record_count=record_count,
            context_character_target=context_character_target,
        )
        utf_token_sample = generate_sample(
            config,
            EncodingCondition(encoding="utf_token", vocab="o200k"),
            sample_index=1,
            record_count=record_count,
            context_character_target=context_character_target,
        )

        self.assertEqual(len(context_lines(hex_sample.prompt)), record_count)
        self.assertEqual(len(context_lines(utf_token_sample.prompt)), record_count)

    def test_numeric_index_uses_unique_shuffled_integers(self) -> None:
        config = BenchmarkConfig(context_length_target=50)
        record_count = 12
        sample = generate_sample(
            config,
            EncodingCondition(encoding="numeric_index", vocab="o200k"),
            sample_index=1,
            record_count=record_count,
            context_character_target=500,
        )
        ids = [line.split("id=", maxsplit=1)[1] for line in context_lines(sample.prompt)]

        self.assertEqual(len(ids), record_count)
        self.assertEqual(len(set(ids)), record_count)
        for identifier in ids:
            self.assertRegex(identifier, r"^[0-9]+$")
            self.assertLessEqual(int(identifier), 2 * record_count - 1)
        self.assertRegex(sample.needle_value_text, r"^[0-9]+$")
        self.assertIn(sample.needle_value_text, ids)

    def test_numeric_index_prompt_describes_numbers_without_range(self) -> None:
        instructions = IDENTIFIER_FORMAT_INSTRUCTIONS["numeric_index"]
        self.assertIn("decimal numbers", instructions)
        self.assertNotRegex(instructions, r"\brange\b")
        self.assertNotRegex(instructions, r"2\s*\*")

    def test_numeric_index_scores_exact_string_answer(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="numeric_index", vocab="o200k"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )

        score = score_response(sample, json.dumps({"id": sample.needle_value_text}))

        self.assertTrue(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertTrue(score.format_valid)

    def test_truncated_utf_token_scores_by_reversible_mapping(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="utf_token_keep_3", vocab="o200k"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )

        score = score_response(sample, json.dumps({"id": sample.needle_value_text}))

        self.assertTrue(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertTrue(score.format_valid)

    def test_render_identifier_rejects_uuid_for_non_16_byte_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "UUID identifiers require 16-byte payloads"):
            render_identifier(
                b"short",
                EncodingCondition(encoding="raw_uuid", vocab="o200k"),
                IdTokenBiMap("o200k"),
            )

    def test_score_response_accepts_exact_answer(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="raw_hex", vocab="o200k"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )

        score = score_response(sample, json.dumps({"id": sample.needle_value_text}))

        self.assertTrue(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertTrue(score.format_valid)

    def test_score_response_normalizes_hex_case_and_wrappers(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="raw_hex", vocab="o200k"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )

        score = score_response(
            sample,
            json.dumps({"id": f"`{sample.needle_value_text.upper()}`"}),
        )

        self.assertFalse(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertTrue(score.format_valid)

    def test_score_response_validates_utf_token_by_mapping(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="utf_token", vocab="o200k"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )

        score = score_response(sample, json.dumps({"id": sample.needle_value_text}))

        self.assertTrue(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertTrue(score.format_valid)

    def test_score_response_heals_utf_token_substitution(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="utf_token", vocab="gemma4"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )
        target = sample.needle_value_text
        garbled = target[:-1] + ("Z" if target[-1] != "Z" else "Y")

        score = score_response(sample, json.dumps({"id": garbled}))

        self.assertFalse(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertFalse(score.format_valid)

    def test_score_response_heals_utf_token_non_ascii_insertion(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="utf_token", vocab="gemma4"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )
        target = sample.needle_value_text
        midpoint = max(1, len(target) // 2)
        garbled = target[:midpoint] + "\u4e2d" + target[midpoint:]

        score = score_response(sample, json.dumps({"id": garbled}))

        self.assertFalse(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertFalse(score.format_valid)

    def test_identifier_response_format_pattern_matches_real_outputs(self) -> None:
        config = BenchmarkConfig(context_length_target=50)
        cases: list[tuple[EncodingName, VocabName]] = [
            ("raw_hex", "o200k"),
            ("raw_base64", "o200k"),
            ("raw_uuid", "o200k"),
            ("utf_token", "gemma4"),
            ("utf_token_keep_3", "gemma4"),
            ("numeric_index", "o200k"),
        ]
        for encoding, vocab in cases:
            with self.subTest(encoding=encoding):
                sample = generate_sample(
                    config,
                    EncodingCondition(encoding=encoding, vocab=vocab),
                    sample_index=1,
                    record_count=10,
                    context_character_target=500,
                )
                compiled = re.compile(IDENTIFIER_PATTERNS[encoding])
                self.assertRegex(sample.needle_value_text, compiled)
                self.assertLessEqual(
                    len(sample.needle_value_text),
                    IDENTIFIER_MAX_LENGTHS[encoding],
                )

    def test_identifier_response_format_pattern_is_encoding_specific(self) -> None:
        uuid_pattern = re.compile(IDENTIFIER_PATTERNS["raw_uuid"])
        utf_pattern = re.compile(IDENTIFIER_PATTERNS["utf_token"])
        base64_pattern = re.compile(IDENTIFIER_PATTERNS["raw_base64"])

        sample_uuid = "123e4567-e89b-12d3-a456-426614174000"
        self.assertRegex(sample_uuid, uuid_pattern)
        self.assertNotRegex(sample_uuid, utf_pattern)

        self.assertRegex("AAEC", base64_pattern)
        self.assertNotRegex("not base64!", base64_pattern)
        self.assertNotRegex("with space", utf_pattern)

    def test_identifier_response_format_wraps_pattern_for_each_encoding(self) -> None:
        for encoding in IDENTIFIER_PATTERNS:
            with self.subTest(encoding=encoding):
                schema = identifier_response_format(encoding)
                json_schema = cast(dict[str, object], schema["json_schema"])
                inner = cast(dict[str, object], json_schema["schema"])
                properties = cast(dict[str, object], inner["properties"])
                id_field = cast(dict[str, object], properties["id"])
                self.assertEqual(id_field["pattern"], IDENTIFIER_PATTERNS[encoding])
                self.assertEqual(id_field["maxLength"], IDENTIFIER_MAX_LENGTHS[encoding])

    def test_score_response_rejects_non_schema_output(self) -> None:
        sample = generate_sample(
            BenchmarkConfig(context_length_target=50),
            EncodingCondition(encoding="raw_hex", vocab="o200k"),
            sample_index=1,
            record_count=10,
            context_character_target=500,
        )

        score = score_response(
            sample,
            json.dumps(
                {
                    "id": sample.needle_value_text,
                    "explanation": "extra fields are not part of the contract",
                }
            ),
        )

        self.assertIsNone(score.extracted)
        self.assertFalse(score.exact_match)
        self.assertFalse(score.normalized_match)
        self.assertFalse(score.format_valid)

    def test_summarize_rows_groups_by_model_and_encoding(self) -> None:
        summary = summarize_rows(
            [
                {
                    "model_slug": "model-a",
                    "encoding": "raw_hex",
                    "exact_match": True,
                    "normalized_match": True,
                    "latency_ms": 100,
                    "error": None,
                },
                {
                    "model_slug": "model-a",
                    "encoding": "raw_hex",
                    "exact_match": False,
                    "normalized_match": True,
                    "latency_ms": 300,
                    "error": "bad output",
                },
            ]
        )

        self.assertEqual(
            summary,
            [
                {
                    "model_slug": "model-a",
                    "encoding": "raw_hex",
                    "samples": 2,
                    "exact_accuracy": 0.5,
                    "normalized_accuracy": 1.0,
                    "mean_latency_ms": 200.0,
                    "error_count": 1,
                }
            ],
        )

    def test_load_completed_keys_matches_config_without_prompt_generation(self) -> None:
        config = BenchmarkConfig(context_length_target=32_000, depth_percent=50.0, base_seed=7)
        rows: list[dict[str, object]] = [
            {
                "model_slug": "openai/gpt-5.4-mini",
                "run_id": RUN_ID,
                "encoding": "raw_hex",
                "sample_index": 1,
                "seed": 7,
                "context_length_target": 32_000,
                "context_character_target": 77_929,
                "record_count": 1_350,
                "depth_percent": 50.0,
                "error": None,
            },
            {
                "model_slug": "openai/gpt-5.4-mini",
                "run_id": RUN_ID,
                "encoding": "raw_hex",
                "sample_index": 2,
                "seed": 8,
                "context_length_target": 32_000,
                "context_character_target": 77_929,
                "record_count": 1_350,
                "depth_percent": 50.0,
                "error": "transport failed",
            },
            {
                "model_slug": "openai/gpt-5.4-mini",
                "run_id": RUN_ID,
                "encoding": "raw_hex",
                "sample_index": 3,
                "seed": 9,
                "context_length_target": 16_000,
                "context_character_target": 77_929,
                "record_count": 1_350,
                "depth_percent": 50.0,
                "error": None,
            },
        ]

        self.assertEqual(
            _load_completed_keys(
                rows,
                config,
                context_character_target=77_929,
                record_count=1_350,
            ),
            {("openai/gpt-5.4-mini", "raw_hex", 1)},
        )

    def test_reasoning_config_uses_minimum_for_gemini_pro(self) -> None:
        self.assertEqual(
            _reasoning_config_for_model("google/gemini-2.5-pro"),
            {"max_tokens": 128, "exclude": True},
        )
        self.assertEqual(
            _reasoning_config_for_model("google/gemini-3-pro-preview"),
            {"max_tokens": 128, "exclude": True},
        )
        self.assertEqual(
            _reasoning_config_for_model("google/gemini-3-flash-preview"),
            {"enabled": False},
        )

    def test_max_tokens_adds_reasoning_budget_for_gemini_pro(self) -> None:
        self.assertEqual(_max_tokens_for_model("google/gemini-2.5-pro", 64), 192)
        self.assertEqual(_max_tokens_for_model("google/gemini-3-pro-preview", 64), 192)
        self.assertEqual(_max_tokens_for_model("google/gemini-3-flash-preview", 64), 64)


class OpenRouterClientTests(unittest.TestCase):
    def test_complete_sends_openrouter_request_and_retries_transient_status(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            if len(requests) == 1:
                return httpx.Response(429, json={"error": "rate limit"})
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "Answer: abc123"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            )

        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(transport=transport)
        client = OpenRouterClient(
            OpenRouterConfig(
                api_key="test-key",
                app_name="utf-token tests",
                http_referer="https://example.test",
                retry_backoff_seconds=0,
            ),
            client=http_client,
        )

        response_format = identifier_response_format("utf_token")
        result = client.complete(
            model_slug="model/test",
            prompt="Prompt",
            max_tokens=8,
            response_format=response_format,
            reasoning={"enabled": False},
        )

        self.assertEqual(result.content, "Answer: abc123")
        self.assertEqual(result.usage, {"prompt_tokens": 10, "completion_tokens": 4})
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1].headers["authorization"], "Bearer test-key")
        self.assertEqual(requests[1].headers["x-title"], "utf-token tests")
        payload = json.loads(requests[1].content)
        self.assertEqual(payload["model"], "model/test")
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(payload["response_format"], response_format)
        self.assertEqual(payload["reasoning"], {"enabled": False})


if __name__ == "__main__":
    unittest.main()
