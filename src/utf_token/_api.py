from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Iterator
from typing import Literal, TypeAlias, overload
from uuid import UUID

from ._tables import DEFAULT_VOCAB, TableSpec, VocabName, pair_table, table_spec, tail_table

Base64Value: TypeAlias = str | bytes
UUIDValue: TypeAlias = UUID | str
KeepBitsValue: TypeAlias = int | Literal["all"] | None


def _validate_keep_bits(
    keep_bits: KeepBitsValue,
    *,
    pair_index_bits: int,
    vocab: VocabName,
) -> int | None:
    if keep_bits is None or keep_bits == "all":
        return None
    if isinstance(keep_bits, str):
        raise ValueError(f"keep_bits string must be 'all'; got {keep_bits!r}")
    if not isinstance(keep_bits, int):
        raise TypeError("keep_bits must be a positive int, None, or 'all'")
    if keep_bits <= 0:
        raise ValueError("keep_bits must be a positive integer")
    if keep_bits % pair_index_bits != 0:
        raise ValueError(
            f"keep_bits must be a multiple of {pair_index_bits} "
            f"(pair_index_bits for vocab {vocab!r}); got {keep_bits}"
        )
    return keep_bits


def _truncate_for_encode(
    data: bytes,
    *,
    keep_bits: int | None,
) -> tuple[bytes, int | None]:
    if keep_bits is None:
        return data, None
    total_bits = len(data) * 8
    if total_bits <= keep_bits:
        return data, None
    value = int.from_bytes(data, byteorder="big", signed=False)
    truncated_value = value >> (total_bits - keep_bits)
    out_len = (keep_bits + 7) // 8
    return truncated_value.to_bytes(out_len, byteorder="big"), keep_bits


def _increment_prefix(data: bytes, *, keep_bits: int | None) -> bytes:
    if keep_bits is None:
        next_value = int.from_bytes(data, byteorder="big", signed=False) + 1
        width = max(len(data), (next_value.bit_length() + 7) // 8, 1)
        return next_value.to_bytes(width, byteorder="big", signed=False)

    total_bits = len(data) * 8
    value = int.from_bytes(data, byteorder="big", signed=False)
    if total_bits > keep_bits:
        value >>= total_bits - keep_bits
    next_value = value + 1
    if next_value.bit_length() > keep_bits:
        out_len = (next_value.bit_length() + 7) // 8
        return next_value.to_bytes(out_len, byteorder="big", signed=False)
    out_len = (keep_bits + 7) // 8
    return next_value.to_bytes(out_len, byteorder="big", signed=False)


def _prefix_exceeds_keep_bits(data: bytes, *, keep_bits: int) -> bool:
    total_bits = len(data) * 8
    value = int.from_bytes(data, byteorder="big", signed=False)
    if total_bits > keep_bits:
        value >>= total_bits - keep_bits
    return value.bit_length() > keep_bits


def _encode_byte_aligned(
    data: bytes,
    *,
    pair_tokens: tuple[str, ...],
    trailing_tokens: tuple[str, ...],
    input_bit_length: int | None = None,
) -> str:
    """Fast path used when `pair_index_bits == 16`: byte-aligned chunking."""

    if input_bit_length is not None:
        total_bits = input_bit_length
        value = int.from_bytes(data, byteorder="big", signed=False)
        if len(data) * 8 > total_bits:
            value >>= len(data) * 8 - total_bits
        pair_byte_count = total_bits // 8
        data = value.to_bytes(pair_byte_count, byteorder="big", signed=False)
        if total_bits % 8 != 0:
            raise ValueError(
                "input_bit_length must be a multiple of 8 for byte-aligned encoding"
            )

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
    input_bit_length: int | None = None,
) -> str:
    """Generic path that consumes the input as an MSB-first bitstream.

    The stream is split into `pair_index_bits` chunks. Trailing residuals of
    `1..tail_index_bits` bits index the tail table; residuals of
    `tail_index_bits + 1 .. pair_index_bits - 1` bits index the pair table
    (left-padded with zeros so the value fits the pair-table address space).
    """

    total_bits = input_bit_length if input_bit_length is not None else 8 * len(data)
    pair_index_bits = spec.pair_index_bits
    tail_index_bits = spec.tail_index_bits
    full_chunks, residual_bits = divmod(total_bits, pair_index_bits)
    value = int.from_bytes(data, byteorder="big", signed=False)
    if input_bit_length is not None and len(data) * 8 > input_bit_length:
        value >>= len(data) * 8 - input_bit_length

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
    keep_bits: KeepBitsValue = None,
) -> str:
    spec = table_spec(vocab)
    validated_keep_bits = _validate_keep_bits(
        keep_bits,
        pair_index_bits=spec.pair_index_bits,
        vocab=vocab,
    )
    data, input_bit_length = _truncate_for_encode(data, keep_bits=validated_keep_bits)
    if not data:
        return ""

    pair_tokens = pair_table(vocab)
    trailing_tokens = tail_table(vocab)

    if spec.pair_index_bits == 16 and spec.tail_index_bits == 8:
        return _encode_byte_aligned(
            data,
            pair_tokens=pair_tokens,
            trailing_tokens=trailing_tokens,
            input_bit_length=input_bit_length,
        )

    return _encode_bitstream(
        data,
        pair_tokens=pair_tokens,
        trailing_tokens=trailing_tokens,
        spec=spec,
        input_bit_length=input_bit_length,
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
    keep_bits: KeepBitsValue = None,
) -> str:
    return _encode_bytes_single(_decode_hex_bytes(data), vocab=vocab, keep_bits=keep_bits)


def _frombase64_single(
    data: str | bytes,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str:
    return _encode_bytes_single(
        _decode_base64_bytes(data),
        vocab=vocab,
        keep_bits=keep_bits,
    )


def _fromuuid_single(
    data: UUID | str,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str:
    return _encode_bytes_single(_decode_uuid_bytes(data), vocab=vocab, keep_bits=keep_bits)


@overload
def frombytes(
    data: bytes,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str: ...


@overload
def frombytes(
    data: Iterable[bytes],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> Iterator[str]: ...


def frombytes(
    data: bytes | Iterable[bytes],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str | Iterator[str]:
    """Encode raw bytes into the UTF-token string representation.

    Pass a single ``bytes`` value to get one encoded ``str`` back. Pass an
    iterable of ``bytes`` values to get a lazy iterator that yields one encoded
    string per item.

    Args:
        data: Raw bytes to encode, or an iterable of byte strings.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bits: How many leading MSBs of each input to encode. Omitted or
            ``None`` keeps the full input; ``"all"`` is an explicit synonym for
            ``None``; a positive integer must be a multiple of the vocab's
            ``pair_index_bits`` (15 for shipped vocabs).

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, bytes):
        return _encode_bytes_single(data, vocab=vocab, keep_bits=keep_bits)
    return (_encode_bytes_single(item, vocab=vocab, keep_bits=keep_bits) for item in data)


@overload
def fromhex(
    data: str,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str: ...


@overload
def fromhex(
    data: Iterable[str],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> Iterator[str]: ...


def fromhex(
    data: str | Iterable[str],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str | Iterator[str]:
    """Encode hexadecimal input into the UTF-token string representation.

    This mirrors :meth:`bytes.fromhex`, so whitespace inside each hex string is
    allowed. Pass an iterable to process multiple values lazily.

    Args:
        data: A hex string such as ``"0012ab"`` or an iterable of hex strings.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bits: How many leading MSBs of the decoded payload to encode.
            Omitted or ``None`` keeps the full input; ``"all"`` is an explicit
            synonym for ``None``; a positive integer must be a multiple of the
            vocab's ``pair_index_bits``.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, str):
        return _fromhex_single(data, vocab=vocab, keep_bits=keep_bits)
    return (_fromhex_single(item, vocab=vocab, keep_bits=keep_bits) for item in data)


@overload
def frombase64(
    data: Base64Value,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str: ...


@overload
def frombase64(
    data: Iterable[Base64Value],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> Iterator[str]: ...


def frombase64(
    data: Base64Value | Iterable[Base64Value],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str | Iterator[str]:
    """Encode base64-decoded bytes into the UTF-token string representation.

    Both ASCII ``str`` and ``bytes`` inputs are accepted for base64 payloads.
    Invalid base64 input raises ``ValueError``. Pass an iterable to process
    multiple values lazily.

    Args:
        data: A base64 string or bytes payload, or an iterable of them.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bits: How many leading MSBs of the decoded payload to encode.
            Omitted or ``None`` keeps the full input; ``"all"`` is an explicit
            synonym for ``None``; a positive integer must be a multiple of the
            vocab's ``pair_index_bits``.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (str, bytes)):
        return _frombase64_single(data, vocab=vocab, keep_bits=keep_bits)
    return (_frombase64_single(item, vocab=vocab, keep_bits=keep_bits) for item in data)


@overload
def fromuuid(
    data: UUIDValue,
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str: ...


@overload
def fromuuid(
    data: Iterable[UUIDValue],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> Iterator[str]: ...


def fromuuid(
    data: UUIDValue | Iterable[UUIDValue],
    /,
    *,
    vocab: VocabName = DEFAULT_VOCAB,
    keep_bits: KeepBitsValue = None,
) -> str | Iterator[str]:
    """Encode UUID values into the UTF-token string representation.

    UUID objects and canonical UUID strings are both accepted. Pass an iterable
    to process multiple UUID values lazily.

    Args:
        data: A UUID object, a UUID string, or an iterable of either form.
        vocab: Lookup table vocabulary. Supported values are ``"o200k"`` and
            ``"gemma4"``.
        keep_bits: How many leading MSBs of the UUID payload to encode.
            Omitted or ``None`` keeps the full 16-byte UUID; ``"all"`` is an
            explicit synonym for ``None``; a positive integer must be a multiple
            of the vocab's ``pair_index_bits``.

    Returns:
        A single encoded string for scalar input, or a lazy iterator of encoded
        strings for iterable input.
    """
    if isinstance(data, (UUID, str)):
        return _fromuuid_single(data, vocab=vocab, keep_bits=keep_bits)
    return (_fromuuid_single(item, vocab=vocab, keep_bits=keep_bits) for item in data)
