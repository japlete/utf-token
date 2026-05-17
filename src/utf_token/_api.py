from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Iterator
from typing import TypeAlias, overload
from uuid import UUID

from ._tables import DEFAULT_VOCAB, TableSpec, VocabName, pair_table, table_spec, tail_table

Base64Value: TypeAlias = str | bytes
UUIDValue: TypeAlias = UUID | str


def _validate_keep_bytes(keep_bytes: int | None) -> int | None:
    if keep_bytes is None:
        return None
    if not isinstance(keep_bytes, int):
        raise TypeError("keep_bytes must be an int or None")
    if keep_bytes <= 0:
        raise ValueError("keep_bytes must be a positive integer")
    return keep_bytes


def _truncate_input(data: bytes, *, keep_bytes: int | None) -> bytes:
    limit = _validate_keep_bytes(keep_bytes)
    if limit is None or len(data) <= limit:
        return data
    return data[:limit]


def _encode_byte_aligned(
    data: bytes,
    *,
    pair_tokens: tuple[str, ...],
    trailing_tokens: tuple[str, ...],
) -> str:
    """Fast path used when `pair_index_bits == 16`: byte-aligned chunking."""

    encoded_parts: list[str] = []
    pair_limit = len(data) - (len(data) % 2)

    for start in range(0, pair_limit, 2):
        pair_index = int.from_bytes(data[start : start + 2], byteorder="big", signed=False)
        encoded_parts.append(pair_tokens[pair_index])

    if len(data) % 2 == 1:
        encoded_parts.append(trailing_tokens[data[-1]])

    return "".join(encoded_parts)


def _encode_bitstream(
    data: bytes,
    *,
    pair_tokens: tuple[str, ...],
    trailing_tokens: tuple[str, ...],
    spec: TableSpec,
) -> str:
    """Generic path that consumes the input as an MSB-first bitstream.

    The stream is split into `pair_index_bits` chunks. Trailing residuals of
    `1..tail_index_bits` bits index the tail table; residuals of
    `tail_index_bits + 1 .. pair_index_bits - 1` bits index the pair table
    (left-padded with zeros so the value fits the pair-table address space).
    """

    total_bits = 8 * len(data)
    pair_index_bits = spec.pair_index_bits
    tail_index_bits = spec.tail_index_bits
    full_chunks, residual_bits = divmod(total_bits, pair_index_bits)
    value = int.from_bytes(data, byteorder="big", signed=False)

    encoded_parts: list[str] = []
    pair_mask = (1 << pair_index_bits) - 1
    for chunk_index in range(full_chunks):
        shift = (full_chunks - 1 - chunk_index) * pair_index_bits + residual_bits
        index = (value >> shift) & pair_mask
        encoded_parts.append(pair_tokens[index])

    if residual_bits == 0:
        return "".join(encoded_parts)

    residual_value = value & ((1 << residual_bits) - 1)
    if residual_bits <= tail_index_bits:
        encoded_parts.append(trailing_tokens[residual_value])
    else:
        encoded_parts.append(pair_tokens[residual_value])

    return "".join(encoded_parts)


def _encode_bytes_single(
    data: bytes,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str:
    data = _truncate_input(data, keep_bytes=keep_bytes)
    if not data:
        return ""

    pair_tokens = pair_table(vocab)
    trailing_tokens = tail_table(vocab)
    spec = table_spec(vocab)

    if spec.pair_index_bits == 16 and spec.tail_index_bits == 8:
        return _encode_byte_aligned(
            data,
            pair_tokens=pair_tokens,
            trailing_tokens=trailing_tokens,
        )

    return _encode_bitstream(
        data,
        pair_tokens=pair_tokens,
        trailing_tokens=trailing_tokens,
        spec=spec,
    )


def _decode_hex_bytes(data: str) -> bytes:
    return bytes.fromhex(data)


def _decode_base64_bytes(data: str | bytes) -> bytes:
    payload = data.encode("ascii") if isinstance(data, str) else data
    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 input") from exc


def _decode_uuid_bytes(data: UUID | str) -> bytes:
    value = data if isinstance(data, UUID) else UUID(data)
    return value.bytes


def _fromhex_single(
    data: str,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str:
    return _encode_bytes_single(_decode_hex_bytes(data), vocab=vocab, keep_bytes=keep_bytes)


def _frombase64_single(
    data: str | bytes,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str:
    return _encode_bytes_single(
        _decode_base64_bytes(data),
        vocab=vocab,
        keep_bytes=keep_bytes,
    )


def _fromuuid_single(
    data: UUID | str,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str:
    return _encode_bytes_single(_decode_uuid_bytes(data), vocab=vocab, keep_bytes=keep_bytes)


@overload
def frombytes(
    data: bytes,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str: ...


@overload
def frombytes(
    data: Iterable[bytes],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> Iterator[str]: ...


def frombytes(
    data: bytes | Iterable[bytes],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode raw bytes into the UTF-token string representation.

    Pass a single ``bytes`` value to get one encoded ``str`` back. Pass an
    iterable of ``bytes`` values to get a lazy iterator that yields one encoded
    string per item.

    Args:
        data: Raw bytes to encode, or an iterable of byte strings.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bytes: Optional positive byte limit applied before encoding.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, bytes):
        return _encode_bytes_single(data, vocab=vocab, keep_bytes=keep_bytes)
    return (_encode_bytes_single(item, vocab=vocab, keep_bytes=keep_bytes) for item in data)


@overload
def fromhex(
    data: str,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str: ...


@overload
def fromhex(
    data: Iterable[str],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> Iterator[str]: ...


def fromhex(
    data: str | Iterable[str],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode hexadecimal input into the UTF-token string representation.

    This mirrors :meth:`bytes.fromhex`, so whitespace inside each hex string is
    allowed. Pass an iterable to process multiple values lazily.

    Args:
        data: A hex string such as ``"0012ab"`` or an iterable of hex strings.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bytes: Optional positive byte limit applied after hex decoding.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, str):
        return _fromhex_single(data, vocab=vocab, keep_bytes=keep_bytes)
    return (_fromhex_single(item, vocab=vocab, keep_bytes=keep_bytes) for item in data)


@overload
def frombase64(
    data: Base64Value,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str: ...


@overload
def frombase64(
    data: Iterable[Base64Value],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> Iterator[str]: ...


def frombase64(
    data: Base64Value | Iterable[Base64Value],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode base64-decoded bytes into the UTF-token string representation.

    Both ASCII ``str`` and ``bytes`` inputs are accepted for base64 payloads.
    Invalid base64 input raises ``ValueError``. Pass an iterable to process
    multiple values lazily.

    Args:
        data: A base64 string or bytes payload, or an iterable of them.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bytes: Optional positive byte limit applied after base64
            decoding.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (str, bytes)):
        return _frombase64_single(data, vocab=vocab, keep_bytes=keep_bytes)
    return (_frombase64_single(item, vocab=vocab, keep_bytes=keep_bytes) for item in data)


@overload
def fromuuid(
    data: UUIDValue,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str: ...


@overload
def fromuuid(
    data: Iterable[UUIDValue],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> Iterator[str]: ...


def fromuuid(
    data: UUIDValue | Iterable[UUIDValue],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode UUID values into the UTF-token string representation.

    UUID objects and canonical UUID strings are both accepted. Pass an iterable
    to process multiple UUID values lazily.

    Args:
        data: A UUID object, a UUID string, or an iterable of either form.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bytes: Optional positive byte limit applied after UUID
            conversion.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (UUID, str)):
        return _fromuuid_single(data, vocab=vocab, keep_bytes=keep_bytes)
    return (_fromuuid_single(item, vocab=vocab, keep_bytes=keep_bytes) for item in data)
