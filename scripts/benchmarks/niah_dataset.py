from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from random import Random
from typing import Literal, TypeAlias
from uuid import UUID

from utf_token import IdTokenBiMap

EncodingName: TypeAlias = Literal[
    "raw_hex",
    "raw_base64",
    "raw_uuid",
    "utf_token",
    "utf_token_keep_30",
    "numeric_index",
]
VocabName: TypeAlias = Literal["o200k", "gemma4"]

ENCODINGS: tuple[EncodingName, ...] = (
    "raw_hex",
    "raw_base64",
    "raw_uuid",
    "utf_token",
    "utf_token_keep_30",
    "numeric_index",
)
IDENTIFIER_FIELD = "id"

IDENTIFIER_FORMAT_INSTRUCTIONS: dict[EncodingName, str] = {
    "raw_hex": (
        "The ids are lowercase hexadecimal that always start *after* the key= prefix, and end *before* a new line."
    ),
    "raw_base64": (
        "The ids are base64 and may contain +, /, and = padding. They always start *after* the key= prefix, and end *before* a new line."
    ),
    "raw_uuid": (
        "The ids are UUIDs with hyphens. They always start *after* the key= prefix, and end *before* a new line."
    ),
    "utf_token": (
        "The ids are random LLM token sequences containing only ASCII alphanumeric or `_` characters. "
        "They always start *after* the key= prefix, and end *before* a new line."
        "Some ids may contain words or part of words, it's just a coincidence due to the use of tokens. "
        "Do not translate or fix typos in the ids. Transcribe them **verbatim**."
    ),
    "utf_token_keep_30": (
        "The ids are random LLM token sequences containing only ASCII alphanumeric or `_` characters. "
        "They always start *after* the key= prefix, and end *before* a new line."
        "Some ids may contain words or part of words, it's just a coincidence due to the use of tokens. "
        "Do not translate or fix typos in the ids. Transcribe them **verbatim**."
    ),
    "numeric_index": (
        "The ids are decimal numbers. "
        "They always start *after* the key= prefix, and end *before* a new line."
        "Transcribe them **verbatim**, including any leading zeros."
    ),
}


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    context_length_target: int = 32_000
    depth_percent: float = 50.0
    samples_per_cell: int = 50
    payload_bytes: int = 16
    base_seed: int = 7


@dataclass(frozen=True, slots=True)
class EncodingCondition:
    encoding: EncodingName
    vocab: VocabName


@dataclass(frozen=True, slots=True)
class NiahSample:
    encoding: EncodingName
    vocab: VocabName
    sample_index: int
    seed: int
    payload_hex: str
    needle_key: str
    needle_value_text: str
    prompt: str
    prompt_hash: str
    prompt_character_count: int
    record_count: int
    context_length_target: int
    context_character_target: int
    depth_percent: float
    codec: IdTokenBiMap


@dataclass(frozen=True, slots=True)
class ScoreResult:
    extracted: str | None
    exact_match: bool
    normalized_match: bool
    format_valid: bool


def sample_seed(config: BenchmarkConfig, sample_index: int) -> int:
    if sample_index < 1:
        raise ValueError("sample_index must be >= 1")
    return config.base_seed + sample_index - 1


def make_payload(seed: int, payload_bytes: int) -> bytes:
    if payload_bytes < 1:
        raise ValueError("payload_bytes must be >= 1")
    return Random(seed).randbytes(payload_bytes)


def codec_for_condition(condition: EncodingCondition) -> IdTokenBiMap:
    if condition.encoding == "utf_token_keep_30":
        return IdTokenBiMap(condition.vocab)
    if condition.encoding == "utf_token":
        return IdTokenBiMap(condition.vocab, keep_bits=None)
    return IdTokenBiMap(condition.vocab)


def render_identifier(payload: bytes, condition: EncodingCondition, codec: IdTokenBiMap) -> str:
    if condition.encoding == "raw_hex":
        return payload.hex()
    if condition.encoding == "raw_base64":
        return base64.b64encode(payload).decode("ascii")
    if condition.encoding == "raw_uuid":
        if len(payload) != 16:
            raise ValueError("UUID identifiers require 16-byte payloads")
        return str(UUID(bytes=payload))
    if condition.encoding in ("utf_token", "utf_token_keep_30"):
        return codec.frombytes(payload)
    raise ValueError(f"Unsupported encoding {condition.encoding!r}")


def build_prompt(context: str, needle_key: str, encoding: EncodingName) -> str:
    return "\n".join(
        [
            "You are given many key/id records. Exactly one record contains the requested key/id.",
            IDENTIFIER_FORMAT_INSTRUCTIONS[encoding],
            "",
            "Records:",
            context,
            "",
            f"Question: What is the id associated with key {needle_key}?",
            'Return a JSON object with exactly one string field named "id".',
            "Do not include any extra fields, markdown, or explanation.",
        ]
    )


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def generate_sample(
    config: BenchmarkConfig,
    condition: EncodingCondition,
    sample_index: int,
    record_count: int,
    context_character_target: int,
) -> NiahSample:
    _validate_generation_config(config)
    if record_count < 1:
        raise ValueError("record_count must be >= 1")
    if context_character_target < 1:
        raise ValueError("context_character_target must be >= 1")

    seed = sample_seed(config, sample_index)
    payload = make_payload(seed, config.payload_bytes)
    rng = Random(seed + 1_000_000)
    codec = codec_for_condition(condition)
    needle_key = f"needle_{sample_index:04d}"
    if condition.encoding == "numeric_index":
        _, needle_value, distractor_values = _make_numeric_index_assignments(
            seed=seed,
            record_count=record_count,
            payload=payload,
        )
        distractor_lines = [
            _format_record(
                f"item_{sample_index:04d}_{distractor_index:05d}",
                distractor_values[distractor_index - 1],
            )
            for distractor_index in range(1, record_count)
        ]
    else:
        needle_value = render_identifier(payload, condition, codec)
        distractor_lines = [
            _make_distractor_line(
                rng=rng,
                sample_index=sample_index,
                distractor_index=distractor_index,
                condition=condition,
                codec=codec,
                payload_bytes=config.payload_bytes,
            )
            for distractor_index in range(1, record_count)
        ]
    needle_line = _format_record(needle_key, needle_value)
    context = _insert_needle(distractor_lines, needle_line, config.depth_percent)
    prompt = build_prompt(context, needle_key, condition.encoding)
    character_count = len(prompt)

    return NiahSample(
        encoding=condition.encoding,
        vocab=condition.vocab,
        sample_index=sample_index,
        seed=seed,
        payload_hex=payload.hex(),
        needle_key=needle_key,
        needle_value_text=needle_value,
        prompt=prompt,
        prompt_hash=prompt_hash(prompt),
        prompt_character_count=character_count,
        record_count=record_count,
        context_length_target=config.context_length_target,
        context_character_target=context_character_target,
        depth_percent=config.depth_percent,
        codec=codec,
    )


def estimate_hex_baseline_record_count(
    config: BenchmarkConfig,
    context_character_target: int,
) -> int:
    _validate_generation_config(config)
    if context_character_target < 1:
        raise ValueError("context_character_target must be >= 1")

    sample_index = 1
    seed = sample_seed(config, sample_index)
    payload = make_payload(seed, config.payload_bytes)
    rng = Random(seed + 1_000_000)
    condition = EncodingCondition(encoding="raw_hex", vocab="o200k")
    codec = codec_for_condition(condition)
    needle_key = f"needle_{sample_index:04d}"
    needle_value = render_identifier(payload, condition, codec)
    needle_line = _format_record(needle_key, needle_value)

    distractor_lines: list[str] = []
    while True:
        context = _insert_needle(distractor_lines, needle_line, config.depth_percent)
        prompt = build_prompt(context, needle_key, condition.encoding)
        if len(prompt) >= context_character_target:
            return len(distractor_lines) + 1
        distractor_lines.append(
            _make_distractor_line(
                rng=rng,
                sample_index=sample_index,
                distractor_index=len(distractor_lines) + 1,
                condition=condition,
                codec=codec,
                payload_bytes=config.payload_bytes,
            )
        )


def extract_answer(response_text: str) -> str | None:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict) or set(parsed) != {IDENTIFIER_FIELD}:
        return None

    answer = parsed[IDENTIFIER_FIELD]
    if not isinstance(answer, str):
        return None

    answer = answer.strip()
    if not answer:
        return None
    return answer


def score_response(sample: NiahSample, response_text: str) -> ScoreResult:
    extracted = extract_answer(response_text)
    if extracted is None:
        return ScoreResult(
            extracted=None,
            exact_match=False,
            normalized_match=False,
            format_valid=False,
        )

    exact_match = extracted == sample.needle_value_text
    normalized = _normalize_answer(sample.encoding, extracted)
    expected_normalized = _normalize_answer(sample.encoding, sample.needle_value_text)
    format_valid = _is_format_valid(sample.encoding, normalized, sample.codec)
    normalized_match = _matches_expected(sample, normalized, expected_normalized)

    return ScoreResult(
        extracted=extracted,
        exact_match=exact_match,
        normalized_match=normalized_match,
        format_valid=format_valid,
    )


def summarize_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        model_slug = row["model_slug"]
        encoding = row["encoding"]
        if not isinstance(model_slug, str) or not isinstance(encoding, str):
            raise TypeError("rows must include string model_slug and encoding values")
        groups.setdefault((model_slug, encoding), []).append(row)

    summary: list[dict[str, object]] = []
    for (model_slug, encoding), group_rows in sorted(groups.items()):
        total = len(group_rows)
        exact = sum(1 for row in group_rows if row["exact_match"] is True)
        normalized = sum(1 for row in group_rows if row["normalized_match"] is True)
        errors = sum(1 for row in group_rows if row["error"] is not None)
        latencies = [
            row["latency_ms"]
            for row in group_rows
            if isinstance(row["latency_ms"], int | float)
        ]
        mean_latency = sum(latencies) / len(latencies) if latencies else None
        summary.append(
            {
                "model_slug": model_slug,
                "encoding": encoding,
                "samples": total,
                "exact_accuracy": exact / total if total else 0.0,
                "normalized_accuracy": normalized / total if total else 0.0,
                "mean_latency_ms": mean_latency,
                "error_count": errors,
            }
        )
    return summary


def _make_numeric_index_assignments(
    *,
    seed: int,
    record_count: int,
    payload: bytes,
) -> tuple[list[str], str, list[str]]:
    if record_count < 1:
        raise ValueError("record_count must be >= 1")
    rng = Random(seed + 2_000_000)
    values = rng.sample(range(2 * record_count), record_count)
    rng.shuffle(values)
    pool = [str(value) for value in values]
    needle_slot = int.from_bytes(payload[:4], "big") % record_count
    needle_value = pool[needle_slot]
    distractor_values = [value for index, value in enumerate(pool) if index != needle_slot]
    return pool, needle_value, distractor_values


def _format_record(key: str, identifier: str) -> str:
    return f"key={key}: id={identifier}"


def _insert_needle(distractor_lines: list[str], needle_line: str, depth_percent: float) -> str:
    insert_index = round(len(distractor_lines) * (depth_percent / 100.0))
    lines = [*distractor_lines]
    lines.insert(insert_index, needle_line)
    return "\n".join(lines)


def _make_distractor_line(
    *,
    rng: Random,
    sample_index: int,
    distractor_index: int,
    condition: EncodingCondition,
    codec: IdTokenBiMap,
    payload_bytes: int,
) -> str:
    key = f"item_{sample_index:04d}_{distractor_index:05d}"
    identifier = render_identifier(rng.randbytes(payload_bytes), condition, codec)
    return _format_record(key, identifier)


def _validate_generation_config(config: BenchmarkConfig) -> None:
    if not 0 <= config.depth_percent <= 100:
        raise ValueError("depth_percent must be between 0 and 100")
    if config.payload_bytes != 16:
        raise ValueError("This benchmark keeps payload_bytes at 16 so UUID is defined")


def _normalize_answer(encoding: EncodingName, value: str) -> str:
    stripped = value.strip().strip("`'\"")
    if encoding in ("raw_hex", "raw_uuid"):
        return stripped.lower()
    return stripped


def _is_format_valid(encoding: EncodingName, value: str, codec: IdTokenBiMap) -> bool:
    if encoding == "raw_hex":
        return _is_hex(value)
    if encoding == "raw_base64":
        return _is_base64(value)
    if encoding == "raw_uuid":
        return _is_uuid(value)
    if encoding == "numeric_index":
        return value.isdigit()
    return value in codec


def _matches_expected(sample: NiahSample, normalized: str, expected_normalized: str) -> bool:
    if sample.encoding in ("utf_token", "utf_token_keep_30"):
        return sample.codec.tobytes(normalized, errors="fix") == bytes.fromhex(sample.payload_hex)
    return normalized == expected_normalized


def _is_hex(value: str) -> bool:
    if len(value) % 2 != 0:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _is_base64(value: str) -> bool:
    try:
        base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error):
        return False
    return True


def _is_uuid(value: str) -> bool:
    try:
        UUID(value)
    except ValueError:
        return False
    return True
