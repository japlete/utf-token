from __future__ import annotations

import argparse
import base64
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ENCODING_URLS = {
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download OpenAI tokenizer vocab files and materialize a decoded local vocab list."
        )
    )
    parser.add_argument(
        "--encoding",
        action="append",
        choices=sorted(ENCODING_URLS),
        help="Encoding(s) to download. Defaults to o200k_base.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/token_vocab"),
        help="Directory to write the downloaded vocab files into.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Network timeout in seconds for each download.",
    )
    return parser.parse_args()


def download_bytes(url: str, timeout: float) -> bytes:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while downloading {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc.reason}") from exc


def decode_vocab_lines(raw_contents: bytes) -> list[tuple[int, bytes]]:
    entries: list[tuple[int, bytes]] = []

    for line_number, raw_line in enumerate(raw_contents.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Malformed vocab line {line_number}: expected 2 fields, got {len(parts)}"
            )

        token_b64, rank_text = parts
        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValueError(
                f"Malformed rank on line {line_number}: {rank_text!r}"
            ) from exc

        try:
            token_bytes = base64.b64decode(token_b64, validate=True)
        except ValueError as exc:
            raise ValueError(
                f"Malformed base64 token on line {line_number}: {token_b64!r}"
            ) from exc

        entries.append((rank, token_bytes))

    return sorted(entries, key=lambda entry: entry[0])


def render_token(token_bytes: bytes) -> str:
    try:
        return repr(token_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        return repr(token_bytes)


def write_outputs(encoding: str, raw_contents: bytes, output_dir: Path) -> tuple[Path, Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{encoding}.tiktoken"
    raw_path.write_bytes(raw_contents)

    decoded_entries = decode_vocab_lines(raw_contents)
    vocab_path = output_dir / f"{encoding}_vocab_list.txt"

    with vocab_path.open("w", encoding="utf-8") as handle:
        for rank, token_bytes in decoded_entries:
            handle.write(f"{rank}\t{render_token(token_bytes)}\n")

    return raw_path, vocab_path, len(decoded_entries)


def main() -> int:
    args = parse_args()
    encodings = args.encoding or ["o200k_base"]

    for encoding in encodings:
        url = ENCODING_URLS[encoding]
        raw_contents = download_bytes(url, timeout=args.timeout)
        raw_path, vocab_path, entry_count = write_outputs(
            encoding=encoding,
            raw_contents=raw_contents,
            output_dir=args.output_dir,
        )
        print(
            f"{encoding}: wrote {entry_count} entries to {vocab_path} and raw file to {raw_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
