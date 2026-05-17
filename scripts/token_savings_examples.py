"""Measure token savings for random byte payloads encoded with utf-token.

For each byte length in a configurable range, the script samples random
payloads, encodes them as hex, base64, UUID (when the length is 16), and the
default ``utf-token`` representation (the standalone encoder with
``keep_bytes=3``), then measures the resulting ``o200k_base`` token counts
and saves a percent-savings plot for the README.

Run with::

    uv run --group offline scripts/token_savings_examples.py
"""

from __future__ import annotations

import argparse
import base64
import os
import random
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "utf-token-matplotlib"),
)

import matplotlib.pyplot as plt
import tiktoken

from utf_token import frombytes

DEFAULT_NUM_SAMPLES = 200
DEFAULT_SEED = 39
DEFAULT_MIN_BYTES = 4
DEFAULT_MAX_BYTES = 32
DEFAULT_OUTPUT_DIR = Path("docs/assets/benchmarks")
DEFAULT_PLOT_NAME = "token_savings.png"
UUID_BYTE_LENGTH = 16


@dataclass(frozen=True, slots=True)
class FormatAverages:
    avg_input_tokens: float
    avg_utf_tokens: float
    avg_savings_pct: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample random byte payloads at multiple lengths and plot the "
            "percent token savings of utf-token (keep_bytes=3) compared with "
            "hex, base64, and UUID textual encodings."
        )
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--plot-name",
        type=str,
        default=DEFAULT_PLOT_NAME,
        help="Filename for the saved PNG inside --output-dir.",
    )
    return parser.parse_args()


def make_payloads(num_samples: int, sample_bytes: int, *, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    return [rng.randbytes(sample_bytes) for _ in range(num_samples)]


def average_tokens(texts: list[str], tokenizer: tiktoken.Encoding) -> float:
    if not texts:
        return 0.0
    total = sum(len(tokenizer.encode(text)) for text in texts)
    return total / len(texts)


def measure_format(
    payloads: list[bytes],
    *,
    encode_input: Callable[[bytes], str],
    tokenizer: tiktoken.Encoding,
    utf_avg: float,
) -> FormatAverages:
    input_texts = [encode_input(payload) for payload in payloads]
    avg_input = average_tokens(input_texts, tokenizer)
    savings = 100.0 * (avg_input - utf_avg) / avg_input if avg_input > 0 else 0.0
    return FormatAverages(
        avg_input_tokens=avg_input,
        avg_utf_tokens=utf_avg,
        avg_savings_pct=savings,
    )


def encode_hex(payload: bytes) -> str:
    return payload.hex()


def encode_base64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def encode_uuid(payload: bytes) -> str:
    return str(UUID(bytes=payload))


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.min_bytes < 1:
        raise ValueError("--min-bytes must be >= 1")
    if args.max_bytes < args.min_bytes:
        raise ValueError("--max-bytes must be >= --min-bytes")

    tokenizer = tiktoken.get_encoding("o200k_base")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    byte_lengths = list(range(args.min_bytes, args.max_bytes + 1))
    hex_savings: list[float] = []
    base64_savings: list[float] = []
    uuid_savings: float | None = None

    for sample_bytes in byte_lengths:
        payloads = make_payloads(args.num_samples, sample_bytes, seed=args.seed)
        utf_texts = [frombytes(payload, keep_bytes=3) for payload in payloads]
        utf_avg = average_tokens(utf_texts, tokenizer)

        hex_metrics = measure_format(
            payloads,
            encode_input=encode_hex,
            tokenizer=tokenizer,
            utf_avg=utf_avg,
        )
        base64_metrics = measure_format(
            payloads,
            encode_input=encode_base64,
            tokenizer=tokenizer,
            utf_avg=utf_avg,
        )
        hex_savings.append(hex_metrics.avg_savings_pct)
        base64_savings.append(base64_metrics.avg_savings_pct)

        print(
            f"bytes={sample_bytes:>2} "
            f"utf_avg={utf_avg:5.2f} "
            f"hex_avg={hex_metrics.avg_input_tokens:6.2f} "
            f"hex_savings={hex_metrics.avg_savings_pct:5.1f}% "
            f"b64_avg={base64_metrics.avg_input_tokens:6.2f} "
            f"b64_savings={base64_metrics.avg_savings_pct:5.1f}%"
        )

        if sample_bytes == UUID_BYTE_LENGTH:
            uuid_metrics = measure_format(
                payloads,
                encode_input=encode_uuid,
                tokenizer=tokenizer,
                utf_avg=utf_avg,
            )
            uuid_savings = uuid_metrics.avg_savings_pct
            print(
                f"          uuid_avg={uuid_metrics.avg_input_tokens:6.2f} "
                f"uuid_savings={uuid_metrics.avg_savings_pct:5.1f}%"
            )

    plot_path = args.output_dir / args.plot_name
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.0, 2.8), layout="constrained")
    ax.plot(
        byte_lengths,
        hex_savings,
        marker="o",
        linewidth=2.0,
        color="#C4682D",
        label="vs hex",
    )
    ax.plot(
        byte_lengths,
        base64_savings,
        marker="s",
        linewidth=2.0,
        color="#2B6CB0",
        label="vs base64",
    )
    if uuid_savings is not None:
        ax.scatter(
            [UUID_BYTE_LENGTH],
            [uuid_savings],
            s=90,
            color="#2F855A",
            edgecolors="white",
            linewidths=1.5,
            zorder=5,
            label="vs uuid (16 bytes)",
        )
        ax.annotate(
            f"{uuid_savings:.0f}%",
            xy=(UUID_BYTE_LENGTH, uuid_savings),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
            color="#2F855A",
            fontweight="bold",
        )

    ax.set_xlabel("Input length (bytes)")
    ax.set_ylabel("Token savings (%)")
    ax.set_title("utf-token savings vs hex, base64, and UUID")
    ax.set_xticks(byte_lengths[::2] if len(byte_lengths) > 16 else byte_lengths)
    ax.legend(loc="lower right")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
