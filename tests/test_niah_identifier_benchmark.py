from __future__ import annotations

import base64
import json
import unittest
from uuid import UUID

import httpx

from scripts.benchmarks.niah_dataset import (
    BenchmarkConfig,
    EncodingCondition,
    estimate_hex_baseline_record_count,
    generate_sample,
    make_payload,
    render_identifier,
    score_response,
    summarize_rows,
)
from scripts.benchmarks.openrouter_client import OpenRouterClient, OpenRouterConfig
from scripts.benchmarks.run_niah_identifier_benchmark import (
    IDENTIFIER_RESPONSE_FORMAT,
    RUN_ID,
    _load_completed_keys,
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
        codec = IdTokenBiMap("gemma4")
        utf_token_text = render_identifier(
            payload,
            EncodingCondition(encoding="utf_token", vocab="gemma4"),
            codec,
        )

        self.assertEqual(bytes.fromhex(hex_text), payload)
        self.assertEqual(base64.b64decode(base64_text, validate=True), payload)
        self.assertEqual(UUID(uuid_text).bytes, payload)
        self.assertEqual(codec.tobytes(utf_token_text), payload)

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

        score = score_response(sample, json.dumps({"identifier": sample.needle_value_text}))

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
            json.dumps({"identifier": f"`{sample.needle_value_text.upper()}`"}),
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

        score = score_response(sample, json.dumps({"identifier": sample.needle_value_text}))

        self.assertTrue(score.exact_match)
        self.assertTrue(score.normalized_match)
        self.assertTrue(score.format_valid)

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
                    "identifier": sample.needle_value_text,
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

        result = client.complete(
            model_slug="model/test",
            prompt="Prompt",
            max_tokens=8,
            response_format=IDENTIFIER_RESPONSE_FORMAT,
        )

        self.assertEqual(result.content, "Answer: abc123")
        self.assertEqual(result.usage, {"prompt_tokens": 10, "completion_tokens": 4})
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1].headers["authorization"], "Bearer test-key")
        self.assertEqual(requests[1].headers["x-title"], "utf-token tests")
        payload = json.loads(requests[1].content)
        self.assertEqual(payload["model"], "model/test")
        self.assertEqual(payload["max_tokens"], 8)
        self.assertEqual(payload["response_format"], IDENTIFIER_RESPONSE_FORMAT)


if __name__ == "__main__":
    unittest.main()
