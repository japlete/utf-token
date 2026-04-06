from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Iterator
from typing import TypeAlias, overload
from uuid import UUID

from ._tables import DEFAULT_VOCAB, VocabName, pair_table, tail_table

Base64Value: TypeAlias = str | bytes
UUIDValue: TypeAlias = UUID | str


def _validate_truncate_bytes(truncate_bytes: int | None) -> int | None:
    if truncate_bytes is None:
        return None
    if not isinstance(truncate_bytes, int):
        raise TypeError("truncate_bytes must be an int or None")
    if truncate_bytes <= 0:
        raise ValueError("truncate_bytes must be a positive integer")
    return truncate_bytes


def _truncate_input(data: bytes, *, truncate_bytes: int | None) -> bytes:
    limit = _validate_truncate_bytes(truncate_bytes)
    if limit is None or len(data) <= limit:
        return data
    return data[:limit]


def _encode_bytes_single(
    data: bytes,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str:
    data = _truncate_input(data, truncate_bytes=truncate_bytes)
    pair_tokens = pair_table(vocab)
    trailing_tokens = tail_table(vocab)
    encoded_parts: list[str] = []
    pair_limit = len(data) - (len(data) % 2)

    for start in range(0, pair_limit, 2):
        pair_index = int.from_bytes(data[start : start + 2], byteorder="big", signed=False)
        encoded_parts.append(pair_tokens[pair_index])

    if len(data) % 2 == 1:
        encoded_parts.append(trailing_tokens[data[-1]])

    return "".join(encoded_parts)


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
    truncate_bytes: int | None = None,
) -> str:
    return _encode_bytes_single(_decode_hex_bytes(data), vocab=vocab, truncate_bytes=truncate_bytes)


def _frombase64_single(
    data: str | bytes,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str:
    return _encode_bytes_single(
        _decode_base64_bytes(data),
        vocab=vocab,
        truncate_bytes=truncate_bytes,
    )


def _fromuuid_single(
    data: UUID | str,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str:
    return _encode_bytes_single(_decode_uuid_bytes(data), vocab=vocab, truncate_bytes=truncate_bytes)


@overload
def frombytes(
    data: bytes,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str: ...


@overload
def frombytes(
    data: Iterable[bytes],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> Iterator[str]: ...


def frombytes(
    data: bytes | Iterable[bytes],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode raw bytes into the UTF-token string representation.

    Pass a single ``bytes`` value to get one encoded ``str`` back. Pass an
    iterable of ``bytes`` values to get a lazy iterator that yields one encoded
    string per item.

    Args:
        data: Raw bytes to encode, or an iterable of byte strings.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        truncate_bytes: Optional positive byte limit applied before encoding.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, bytes):
        return _encode_bytes_single(data, vocab=vocab, truncate_bytes=truncate_bytes)
    return (_encode_bytes_single(item, vocab=vocab, truncate_bytes=truncate_bytes) for item in data)


@overload
def fromhex(
    data: str,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str: ...


@overload
def fromhex(
    data: Iterable[str],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> Iterator[str]: ...


def fromhex(
    data: str | Iterable[str],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode hexadecimal input into the UTF-token string representation.

    This mirrors :meth:`bytes.fromhex`, so whitespace inside each hex string is
    allowed. Pass an iterable to process multiple values lazily.

    Args:
        data: A hex string such as ``"0012ab"`` or an iterable of hex strings.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        truncate_bytes: Optional positive byte limit applied after hex decoding.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, str):
        return _fromhex_single(data, vocab=vocab, truncate_bytes=truncate_bytes)
    return (_fromhex_single(item, vocab=vocab, truncate_bytes=truncate_bytes) for item in data)


@overload
def frombase64(
    data: Base64Value,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str: ...


@overload
def frombase64(
    data: Iterable[Base64Value],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> Iterator[str]: ...


def frombase64(
    data: Base64Value | Iterable[Base64Value],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode base64-decoded bytes into the UTF-token string representation.

    Both ASCII ``str`` and ``bytes`` inputs are accepted for base64 payloads.
    Invalid base64 input raises ``ValueError``. Pass an iterable to process
    multiple values lazily.

    Args:
        data: A base64 string or bytes payload, or an iterable of them.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        truncate_bytes: Optional positive byte limit applied after base64
            decoding.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (str, bytes)):
        return _frombase64_single(data, vocab=vocab, truncate_bytes=truncate_bytes)
    return (_frombase64_single(item, vocab=vocab, truncate_bytes=truncate_bytes) for item in data)


@overload
def fromuuid(
    data: UUIDValue,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str: ...


@overload
def fromuuid(
    data: Iterable[UUIDValue],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> Iterator[str]: ...


def fromuuid(
    data: UUIDValue | Iterable[UUIDValue],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    truncate_bytes: int | None = None,
) -> str | Iterator[str]:
    """Encode UUID values into the UTF-token string representation.

    UUID objects and canonical UUID strings are both accepted. Pass an iterable
    to process multiple UUID values lazily.

    Args:
        data: A UUID object, a UUID string, or an iterable of either form.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        truncate_bytes: Optional positive byte limit applied after UUID
            conversion.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (UUID, str)):
        return _fromuuid_single(data, vocab=vocab, truncate_bytes=truncate_bytes)
    return (_fromuuid_single(item, vocab=vocab, truncate_bytes=truncate_bytes) for item in data)
