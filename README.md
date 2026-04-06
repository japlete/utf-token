# utf-token

`utf-token` encodes random-looking binary identifiers into a compact token-friendly `str`
using a fixed subset of the `o200k_base` vocabulary.

## Supported Python

The published package targets Python 3.10 and newer.

## Python API

The library supports these forward helpers:

- `frombytes`
- `fromhex`
- `frombase64`
- `fromuuid`

Each function accepts either a single value or an iterable of values. Scalar inputs return
a single encoded `str`; iterable inputs return a lazy iterator of encoded `str` values.

Odd-length byte payloads are supported by using a separate 256-entry tail lookup for the
final byte. That tail table reserves the 256 shortest eligible tokens first, and the
65,536-entry pair table uses the next slice of the same ordering.

## Reversible Mappings

`IdTokenBiMap` is a stateful companion to the standalone forward helpers.
It records encoded strings produced so far so they can be reversed later with:

- `tobytes`
- `tohex`
- `tobase64`
- `touuid`

The reversible class also supports `to_dict` / `to_json` exports plus `from_dict` /
`from_json` imports for offline storage. When a newly encoded value would collide with an
existing stored string, the class deterministically remaps the output so reverse lookups
stay exact.

## Publishing

CI builds and smoke-tests both wheel and sdist artifacts before release. The release flow
publishes tagged versions to TestPyPI first, runs an install smoke test against TestPyPI,
and only then publishes the same artifacts to PyPI.

See `docs/releasing.md` for the local preflight commands and repository setup steps.

For local development install all deps with `uv sync --all-groups`
