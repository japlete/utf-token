from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Iterator
from typing import TypeAlias, overload
from uuid import UUID

from ._tables import pair_table, tail_table

Base64Value: TypeAlias = str | bytes
UUIDValue: TypeAlias = UUID | str


def _encode_bytes_single(data: bytes) -> str:
    pair_tokens = pair_table()
    trailing_tokens = tail_table()
    encoded_parts: list[str] = []
    pair_limit = len(data) - (len(data) % 2)

    for start in range(0, pair_limit, 2):
        pair_index = int.from_bytes(data[start : start + 2], byteorder="big", signed=False)
        encoded_parts.append(pair_tokens[pair_index])

    if len(data) % 2 == 1:
        encoded_parts.append(trailing_tokens[data[-1]])

    return "".join(encoded_parts)


def _fromhex_single(data: str) -> str:
    return _encode_bytes_single(bytes.fromhex(data))


def _frombase64_single(data: str | bytes) -> str:
    payload = data.encode("ascii") if isinstance(data, str) else data
    try:
        decoded = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("Invalid base64 input") from exc
    return _encode_bytes_single(decoded)


def _fromuuid_single(data: UUID | str) -> str:
    value = data if isinstance(data, UUID) else UUID(data)
    return _encode_bytes_single(value.bytes)


@overload
def frombytes(data: bytes, /) -> str: ...


@overload
def frombytes(data: Iterable[bytes], /) -> Iterator[str]: ...


def frombytes(data: bytes | Iterable[bytes], /) -> str | Iterator[str]:
    """Encode raw bytes into the UTF-token string representation.

    Pass a single ``bytes`` value to get one encoded ``str`` back. Pass an
    iterable of ``bytes`` values to get a lazy iterator that yields one encoded
    string per item.

    Args:
        data: Raw bytes to encode, or an iterable of byte strings.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, bytes):
        return _encode_bytes_single(data)
    return (_encode_bytes_single(item) for item in data)


@overload
def fromhex(data: str, /) -> str: ...


@overload
def fromhex(data: Iterable[str], /) -> Iterator[str]: ...


def fromhex(data: str | Iterable[str], /) -> str | Iterator[str]:
    """Encode hexadecimal input into the UTF-token string representation.

    This mirrors :meth:`bytes.fromhex`, so whitespace inside each hex string is
    allowed. Pass an iterable to process multiple values lazily.

    Args:
        data: A hex string such as ``"0012ab"`` or an iterable of hex strings.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, str):
        return _fromhex_single(data)
    return (_fromhex_single(item) for item in data)


@overload
def frombase64(data: Base64Value, /) -> str: ...


@overload
def frombase64(data: Iterable[Base64Value], /) -> Iterator[str]: ...


def frombase64(data: Base64Value | Iterable[Base64Value], /) -> str | Iterator[str]:
    """Encode base64-decoded bytes into the UTF-token string representation.

    Both ASCII ``str`` and ``bytes`` inputs are accepted for base64 payloads.
    Invalid base64 input raises ``ValueError``. Pass an iterable to process
    multiple values lazily.

    Args:
        data: A base64 string or bytes payload, or an iterable of them.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (str, bytes)):
        return _frombase64_single(data)
    return (_frombase64_single(item) for item in data)


@overload
def fromuuid(data: UUIDValue, /) -> str: ...


@overload
def fromuuid(data: Iterable[UUIDValue], /) -> Iterator[str]: ...


def fromuuid(data: UUIDValue | Iterable[UUIDValue], /) -> str | Iterator[str]:
    """Encode UUID values into the UTF-token string representation.

    UUID objects and canonical UUID strings are both accepted. Pass an iterable
    to process multiple UUID values lazily.

    Args:
        data: A UUID object, a UUID string, or an iterable of either form.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (UUID, str)):
        return _fromuuid_single(data)
    return (_fromuuid_single(item) for item in data)
