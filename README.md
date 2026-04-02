# utf-token

`utf-token` encodes random-looking binary identifiers into a compact token-friendly `str`
using a fixed subset of the `o200k_base` vocabulary.

## V1 API

The first library version is intentionally forward-only. It supports:

- `frombytes`
- `fromhex`
- `frombase64`
- `fromuuid`

Each function accepts either a single value or an iterable of values. Scalar inputs return
a single encoded `str`; iterable inputs return a lazy iterator of encoded `str` values.

Odd-length byte payloads are supported by using a separate 256-entry tail lookup for the
final byte.

## Current limitation

V1 does not implement `tobytes` or other reverse helpers. Concatenated token strings are
not reliably self-delimiting under `o200k_base`, so exact reversal is deferred to the next
roadmap stage where executed mappings will be stored explicitly.
