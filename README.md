# utf-token

`utf-token` encodes random-looking binary identifiers into a compact token-friendly `str`
using a fixed subset of the `o200k_base` vocabulary.

## Supported Python

The published package targets Python 3.10 and newer.

## V1 API

The first library version is intentionally forward-only. It supports:

- `frombytes`
- `fromhex`
- `frombase64`
- `fromuuid`

Each function accepts either a single value or an iterable of values. Scalar inputs return
a single encoded `str`; iterable inputs return a lazy iterator of encoded `str` values.

Odd-length byte payloads are supported by using a separate 256-entry tail lookup for the
final byte. That tail table reserves the 256 shortest eligible tokens first, and the
65,536-entry pair table uses the next slice of the same ordering.

## Current limitation

V1 does not implement `tobytes` or other reverse helpers. Concatenated token strings are
not reliably self-delimiting under `o200k_base`, so exact reversal is deferred to the next
roadmap stage where executed mappings will be stored explicitly.

## Publishing

CI builds and smoke-tests both wheel and sdist artifacts before release. The release flow
publishes tagged versions to TestPyPI first, runs an install smoke test against TestPyPI,
and only then publishes the same artifacts to PyPI.

See `docs/releasing.md` for the local preflight commands and repository setup steps.

For local development install all deps with `uv sync --all-groups`
