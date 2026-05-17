from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tiktoken

from scripts.benchmarks.niah_dataset import (
    ENCODINGS,
    BenchmarkConfig,
    EncodingName,
    EncodingCondition,
    NiahSample,
    VocabName,
    build_prompt,
    estimate_hex_baseline_record_count,
    generate_sample,
    make_payload,
    render_identifier,
    sample_seed,
    score_response,
    summarize_rows,
)
from scripts.benchmarks.openrouter_client import CompletionResult, OpenRouterClient, OpenRouterConfig
from utf_token import IdTokenBiMap

DEFAULT_OUTPUT_DIR = Path("docs/assets/benchmarks")
DEFAULT_RUNS_PATH = DEFAULT_OUTPUT_DIR / "niah_identifier_runs.jsonl"
DEFAULT_SUMMARY_CSV_PATH = DEFAULT_OUTPUT_DIR / "niah_identifier_summary.csv"
DEFAULT_SUMMARY_MD_PATH = DEFAULT_OUTPUT_DIR / "niah_identifier_summary.md"
DEFAULT_PROMPT_EXAMPLE_PATH = DEFAULT_OUTPUT_DIR / "niah_identifier_prompt_example.txt"
DEFAULT_SAMPLES_PER_CELL = 100
DEFAULT_CONTEXT_LENGTH_TARGET = 32_000
DEFAULT_DEPTH_PERCENT = 50.0
DEFAULT_BASE_SEED = 39
DEFAULT_MAX_OUTPUT_TOKENS = 64
DEFAULT_ENCODINGS: tuple[EncodingName, ...] = (
    "utf_token",
    "utf_token_truncate_3",
)
GEMINI_PRO_MIN_REASONING_TOKENS = 128
RUN_ID = "niah_seed39"
IDENTIFIER_PATTERNS: dict[EncodingName, str] = {
    "raw_hex": "^[0-9a-fA-F]+$",
    "raw_base64": "^[A-Za-z0-9+/]+={0,2}$",
    "raw_uuid": (
        "^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        "[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),
    "utf_token": "^[A-Za-z0-9_]+$",
    "utf_token_truncate_3": "^[A-Za-z0-9_]+$",
    "numeric_index": "^[0-9]+$",
}

IDENTIFIER_MAX_LENGTHS: dict[EncodingName, int] = {
    "raw_hex": 32,
    "raw_base64": 24,
    "raw_uuid": 36,
    "utf_token": 54,
    "utf_token_truncate_3": 12,
    "numeric_index": 10,
}


def identifier_response_format(encoding: EncodingName) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "id_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "pattern": IDENTIFIER_PATTERNS[encoding],
                        "minLength": 1,
                        "maxLength": IDENTIFIER_MAX_LENGTHS[encoding],
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class ModelSpec:
    slug: str
    vocab: VocabName


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(slug="openai/gpt-5.4-mini", vocab="o200k"),
    ModelSpec(slug="google/gemma-4-26b-a4b-it", vocab="gemma4"),
    ModelSpec(slug="anthropic/claude-haiku-4.5", vocab="o200k"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a fixed NIAH-style benchmark comparing utf-token identifiers "
            "against raw hex/base64/uuid strings through OpenRouter."
        )
    )
    parser.add_argument("--samples-per-cell", type=int, default=DEFAULT_SAMPLES_PER_CELL)
    parser.add_argument(
        "--context-length-target",
        type=int,
        default=DEFAULT_CONTEXT_LENGTH_TARGET,
    )
    parser.add_argument("--depth-percent", type=float, default=DEFAULT_DEPTH_PERCENT)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_RUNS_PATH)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV_PATH)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD_PATH)
    parser.add_argument(
        "--all-encodings",
        action="store_true",
        help="Run all identifier encodings. If omitted, only utf-token encodings are run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows already present in the JSONL output by model/encoding/sample/hash.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate prompts and print metadata without calling OpenRouter.",
    )
    parser.add_argument(
        "--write-prompt-example",
        type=Path,
        nargs="?",
        const=DEFAULT_PROMPT_EXAMPLE_PATH,
        help=(
            "Write one full generated prompt to the given path, or to the default "
            "docs asset path if no path is supplied, then exit."
        ),
    )
    parser.add_argument(
        "--prompt-example-encoding",
        choices=ENCODINGS,
        default="utf_token",
        help="Identifier encoding to use with --write-prompt-example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_cell < 1:
        raise ValueError("--samples-per-cell must be >= 1")

    config = BenchmarkConfig(
        context_length_target=args.context_length_target,
        depth_percent=args.depth_percent,
        samples_per_cell=args.samples_per_cell,
        base_seed=args.base_seed,
    )
    chars_per_token = _estimate_hex_chars_per_token()
    context_character_target = round(config.context_length_target * chars_per_token)
    record_count = estimate_hex_baseline_record_count(config, context_character_target)
    print(
        "Estimated hex baseline: "
        f"{context_character_target} chars from "
        f"{config.context_length_target} tokens at {chars_per_token:.3f} chars/token; "
        f"fixed record_count={record_count}."
    )
    if args.write_prompt_example is not None:
        _write_prompt_example(
            path=args.write_prompt_example,
            config=config,
            encoding=args.prompt_example_encoding,
            context_character_target=context_character_target,
            record_count=record_count,
        )
        return

    target_encodings = ENCODINGS if args.all_encodings else DEFAULT_ENCODINGS

    all_rows = _load_rows(args.output_jsonl) if args.output_jsonl.exists() else []
    rows = _compatible_rows(all_rows, config, context_character_target, record_count)
    completed_keys = (
        _load_completed_keys(all_rows, config, context_character_target, record_count)
        if args.resume
        else set()
    )
    if args.resume:
        print(f"Resume loaded {len(completed_keys)} completed calls from {args.output_jsonl}.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else OpenRouterClient(OpenRouterConfig.from_env())
    try:
        with args.output_jsonl.open("a", encoding="utf-8") as output_file:
            for model in MODEL_SPECS:
                for encoding in target_encodings:
                    condition = EncodingCondition(encoding=encoding, vocab=model.vocab)
                    for sample_index in range(1, config.samples_per_cell + 1):
                        work_key = _work_key(model.slug, encoding, sample_index)
                        if work_key in completed_keys:
                            continue
                        sample = generate_sample(
                            config=config,
                            condition=condition,
                            sample_index=sample_index,
                            record_count=record_count,
                            context_character_target=context_character_target,
                        )
                        if args.dry_run:
                            _print_dry_run(model, sample)
                            continue

                        if client is None:
                            raise RuntimeError("OpenRouter client was not initialized")
                        row = _run_one_call(
                            client=client,
                            model=model,
                            sample=sample,
                            max_output_tokens=args.max_output_tokens,
                        )
                        rows.append(row)
                        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        output_file.flush()
                        print(
                            f"{model.slug} {encoding} sample={sample_index} "
                            f"exact={row['exact_match']} normalized={row['normalized_match']}"
                        )
    finally:
        if client is not None:
            client.close()

    if args.dry_run:
        print("Dry run complete; no OpenRouter calls were made.")
        return

    summary = summarize_rows(rows)
    _write_summary_csv(args.summary_csv, summary)
    _write_summary_md(args.summary_md, summary)
    print(_format_markdown_table(summary))


def _run_one_call(
    *,
    client: OpenRouterClient,
    model: ModelSpec,
    sample: NiahSample,
    max_output_tokens: int,
) -> dict[str, object]:
    started = time.perf_counter()
    completion = client.complete(
        model_slug=model.slug,
        prompt=sample.prompt,
        max_tokens=_max_tokens_for_model(model.slug, max_output_tokens),
        temperature=0.0,
        response_format=identifier_response_format(sample.encoding),
        reasoning=_reasoning_config_for_model(model.slug),
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    return _build_row(
        model=model,
        sample=sample,
        completion=completion,
        latency_ms=latency_ms,
        error=None,
    )


def _build_row(
    *,
    model: ModelSpec,
    sample: NiahSample,
    completion: CompletionResult,
    latency_ms: int | None,
    error: str | None,
) -> dict[str, object]:
    score = score_response(sample, completion.content)
    return {
        "run_id": RUN_ID,
        "model_slug": model.slug,
        "encoding": sample.encoding,
        "sample_index": sample.sample_index,
        "seed": sample.seed,
        "payload_hex": sample.payload_hex,
        "needle_key": sample.needle_key,
        "needle_value_text": sample.needle_value_text,
        "prompt_hash": sample.prompt_hash,
        "prompt_character_count": sample.prompt_character_count,
        "record_count": sample.record_count,
        "context_length_target": sample.context_length_target,
        "context_character_target": sample.context_character_target,
        "depth_percent": sample.depth_percent,
        "model_response_raw": completion.content,
        "model_response_extracted": score.extracted,
        "exact_match": score.exact_match,
        "normalized_match": score.normalized_match,
        "format_valid": score.format_valid,
        "latency_ms": latency_ms,
        "usage": completion.usage,
        "error": error,
    }


def _max_tokens_for_model(model_slug: str, answer_token_budget: int) -> int:
    if _is_google_gemini_pro(model_slug):
        return answer_token_budget + GEMINI_PRO_MIN_REASONING_TOKENS
    return answer_token_budget


def _reasoning_config_for_model(model_slug: str) -> dict[str, object]:
    if _is_google_gemini_pro(model_slug):
        return {"max_tokens": GEMINI_PRO_MIN_REASONING_TOKENS, "exclude": True}
    return {"enabled": False}


def _is_google_gemini_pro(model_slug: str) -> bool:
    return model_slug.startswith("google/gemini-") and "-pro" in model_slug


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object row in {path}")
            rows.append(parsed)
    return rows


def _load_completed_keys(
    rows: list[dict[str, object]],
    config: BenchmarkConfig,
    context_character_target: int,
    record_count: int,
) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for row in rows:
        model_slug = row["model_slug"]
        encoding = row["encoding"]
        sample_index = row["sample_index"]
        seed = row["seed"]
        context_length_target = row["context_length_target"]
        row_character_target = row.get("context_character_target")
        row_record_count = row.get("record_count")
        depth_percent = row["depth_percent"]
        error = row["error"]
        if (
            isinstance(model_slug, str)
            and isinstance(encoding, str)
            and isinstance(sample_index, int)
            and row.get("run_id") == RUN_ID
            and isinstance(seed, int)
            and context_length_target == config.context_length_target
            and row_character_target == context_character_target
            and row_record_count == record_count
            and depth_percent == config.depth_percent
            and seed == sample_seed(config, sample_index)
            and error is None
        ):
            keys.add((model_slug, encoding, sample_index))
    return keys


def _compatible_rows(
    rows: list[dict[str, object]],
    config: BenchmarkConfig,
    context_character_target: int,
    record_count: int,
) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("run_id") == RUN_ID
        and row.get("context_length_target") == config.context_length_target
        and row.get("context_character_target") == context_character_target
        and row.get("record_count") == record_count
        and row.get("depth_percent") == config.depth_percent
    ]


def _work_key(model_slug: str, encoding: str, sample_index: int) -> tuple[str, str, int]:
    return (model_slug, encoding, sample_index)


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_slug",
        "encoding",
        "samples",
        "exact_accuracy",
        "normalized_accuracy",
        "mean_latency_ms",
        "error_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_format_markdown_table(rows) + "\n", encoding="utf-8")


def _write_prompt_example(
    *,
    path: Path,
    config: BenchmarkConfig,
    encoding: EncodingName,
    context_character_target: int,
    record_count: int,
) -> None:
    condition = EncodingCondition(encoding=encoding, vocab=MODEL_SPECS[0].vocab)
    sample = generate_sample(
        config=config,
        condition=condition,
        sample_index=1,
        record_count=record_count,
        context_character_target=context_character_target,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample.prompt + "\n", encoding="utf-8")
    print(
        f"Wrote {path} with encoding={sample.encoding}, "
        f"vocab={sample.vocab}, prompt_hash={sample.prompt_hash}."
    )


def _format_markdown_table(rows: list[dict[str, object]]) -> str:
    headers = [
        "model_slug",
        "encoding",
        "samples",
        "exact_accuracy",
        "normalized_accuracy",
        "mean_latency_ms",
        "error_count",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [_format_cell(row[header]) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return ""
    return str(value)


def _print_dry_run(model: ModelSpec, sample: NiahSample) -> None:
    print(
        "\n".join(
            [
                f"model_slug={model.slug}",
                f"encoding={sample.encoding}",
                f"vocab={sample.vocab}",
                f"sample_index={sample.sample_index}",
                f"record_count={sample.record_count}",
                f"prompt_character_count={sample.prompt_character_count}",
                f"prompt_hash={sample.prompt_hash}",
                "--- prompt preview ---",
                sample.prompt[:2_000],
                "--- end preview ---",
            ]
        )
    )


def _estimate_hex_chars_per_token() -> float:
    tokenizer = tiktoken.get_encoding("o200k_base")
    calibration_lines: list[str] = []
    condition = EncodingCondition(encoding="raw_hex", vocab="o200k")
    codec = _codec_for(condition)
    for sample_index in range(1, 101):
        payload = make_payload(seed=100_000 + sample_index, payload_bytes=16)
        identifier = render_identifier(payload, condition, codec)
        calibration_lines.append(
            f"key=calibration_raw_hex_{sample_index:04d}: id={identifier}"
        )
    calibration_prompt = build_prompt(
        "\n".join(calibration_lines),
        "calibration_0001",
        "raw_hex",
    )
    token_count = len(tokenizer.encode(calibration_prompt))
    if token_count == 0:
        raise ValueError("Calibration prompt produced zero tokens")
    return len(calibration_prompt) / token_count


def _codec_for(condition: EncodingCondition) -> IdTokenBiMap:
    return IdTokenBiMap(condition.vocab)


if __name__ == "__main__":
    main()
