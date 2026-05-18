from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator
from threading import RLock
from typing import Literal, TypeAlias, overload
from uuid import UUID

import Levenshtein

from ._api import (
    Base64Value,
    KeepBitsValue,
    UUIDValue,
    _decode_base64_bytes,
    _decode_hex_bytes,
    _decode_uuid_bytes,
    _encode_bytes_single,
    _increment_prefix,
    _prefix_exceeds_keep_bits,
    _truncate_for_encode,
    _validate_keep_bits,
)
from ._tables import DEFAULT_VOCAB, VocabName, table_spec

FORMAT_VERSION = 5
DEFAULT_KEEP_BITS = 30
ErrorMode: TypeAlias = Literal["fix", "raise"]
DEFAULT_ERROR_MODE: ErrorMode = "fix"


def _validate_error_mode(value: object) -> ErrorMode:
    if value == "fix":
        return "fix"
    if value == "raise":
        return "raise"
    raise ValueError(
        f"Unsupported errors mode {value!r}; expected one of: 'fix', 'raise'"
    )


def _validate_vocab_name(value: str) -> VocabName:
    if value == "o200k":
        return "o200k"
    if value == "gemma4":
        return "gemma4"
    raise ValueError(f"Unsupported vocab {value!r}")


def _parse_payload_keep_bits(value: object) -> KeepBitsValue:
    if value is None or value == "all":
        return None if value is None else "all"
    if isinstance(value, int):
        return value
    raise ValueError(
        "Mapping payload keep_bits must be null, a positive integer, or 'all'"
    )


class IdTokenBiMap:
    def __init__(
        self,
        vocab: VocabName = DEFAULT_VOCAB,
        *,
        keep_bits: KeepBitsValue = DEFAULT_KEEP_BITS,
    ) -> None:
        self._vocab = vocab
        spec = table_spec(vocab)
        self._keep_bits: int | None = _validate_keep_bits(
            keep_bits,
            pair_index_bits=spec.pair_index_bits,
            vocab=vocab,
        )
        self._reverse_map: dict[str, bytes] = {}
        self._forward_map: dict[bytes, str] = {}
        self._lock = RLock()

    @property
    def keep_bits(self) -> int | None:
        return self._keep_bits

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        with self._lock:
            return item in self._reverse_map

    def _store_mapping(self, encoded: str, original_bytes: bytes) -> str:
        self._reverse_map[encoded] = original_bytes
        self._forward_map[original_bytes] = encoded
        return encoded

    def _encode_candidate(self, candidate: bytes) -> str:
        if self._keep_bits is not None and _prefix_exceeds_keep_bits(
            candidate, keep_bits=self._keep_bits
        ):
            return _encode_bytes_single(candidate, vocab=self._vocab, keep_bits=None)
        return _encode_bytes_single(
            candidate, vocab=self._vocab, keep_bits=self._keep_bits
        )

    def _encode_and_store_single(self, data: bytes) -> str:
        with self._lock:
            cached = self._forward_map.get(data)
            if cached is not None:
                return cached

            encoded = _encode_bytes_single(
                data, vocab=self._vocab, keep_bits=self._keep_bits
            )
            existing = self._reverse_map.get(encoded)
            if existing is None or existing == data:
                return self._store_mapping(encoded, data)

            working_bytes, _ = _truncate_for_encode(data, keep_bits=self._keep_bits)
            candidate = working_bytes
            while True:
                candidate = _increment_prefix(candidate, keep_bits=self._keep_bits)
                encoded = self._encode_candidate(candidate)
                existing = self._reverse_map.get(encoded)
                if existing is None or existing == data:
                    return self._store_mapping(encoded, data)

    def _find_nearest_encoded(self, data: str) -> str | None:
        """Return the stored encoded identifier closest to `data`, or None if empty.

        Tie-breaking is deterministic: candidates are ordered by
        `(edit_distance, encoded_string)` and the first one wins.
        """
        if not self._reverse_map:
            return None
        best_distance: int | None = None
        best_encoded: str | None = None
        for encoded in self._reverse_map:
            distance = Levenshtein.distance(data, encoded)
            if best_distance is None or distance < best_distance or (
                distance == best_distance
                and best_encoded is not None
                and encoded < best_encoded
            ):
                best_distance = distance
                best_encoded = encoded
        return best_encoded

    def _lookup_bytes_single(self, data: str, errors: ErrorMode) -> bytes | None:
        with self._lock:
            original = self._reverse_map.get(data)
            if original is not None:
                return original
            if errors == "raise":
                raise KeyError(data)
            nearest = self._find_nearest_encoded(data)
            if nearest is None:
                return None
            return self._reverse_map[nearest]

    def _lookup_hex_single(self, data: str, errors: ErrorMode) -> str | None:
        value = self._lookup_bytes_single(data, errors)
        if value is None:
            return None
        return value.hex()

    def _lookup_base64_single(self, data: str, errors: ErrorMode) -> str | None:
        value = self._lookup_bytes_single(data, errors)
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")

    def _lookup_uuid_single(self, data: str, errors: ErrorMode) -> UUID | None:
        value = self._lookup_bytes_single(data, errors)
        if value is None:
            return None
        if len(value) != 16:
            raise ValueError(
                "Stored mapping is not a UUID-sized value: "
                f"expected 16 bytes, found {len(value)}"
            )
        return UUID(bytes=value)

    def to_dict(self) -> dict[str, object]:
        with self._lock:
            mappings: dict[str, str] = {}
            for encoded in sorted(self._reverse_map):
                mappings[encoded] = self._reverse_map[encoded].hex()
            return {
                "format_version": FORMAT_VERSION,
                "vocab": self._vocab,
                "keep_bits": self._keep_bits,
                "mappings": mappings,
            }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IdTokenBiMap:
        format_version = payload.get("format_version")
        if format_version != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported mapping format version {format_version!r}; "
                f"expected {FORMAT_VERSION}"
            )

        vocab_obj = payload.get("vocab")
        if not isinstance(vocab_obj, str):
            raise ValueError("Mapping payload must contain a 'vocab' string")
        vocab = _validate_vocab_name(vocab_obj)

        if "keep_bits" not in payload:
            raise ValueError("Mapping payload must contain 'keep_bits'")
        keep_bits_raw = _parse_payload_keep_bits(payload.get("keep_bits"))

        mappings_obj = payload.get("mappings")
        if not isinstance(mappings_obj, dict):
            raise ValueError("Mapping payload must contain a 'mappings' object")

        instance = cls(vocab, keep_bits=keep_bits_raw)

        for encoded, entry in mappings_obj.items():
            if not isinstance(encoded, str):
                raise ValueError("Mapping keys must be encoded strings")
            if not isinstance(entry, str):
                raise ValueError(
                    f"Mapping entry for {encoded!r} must be an original_hex string"
                )

            original_bytes = _decode_hex_bytes(entry)
            existing_reverse = instance._reverse_map.get(encoded)
            if existing_reverse is not None and existing_reverse != original_bytes:
                raise ValueError(
                    f"Conflicting mapping import for encoded {encoded!r}"
                )
            existing_forward = instance._forward_map.get(original_bytes)
            if existing_forward is not None and existing_forward != encoded:
                raise ValueError(
                    "Conflicting mapping import for "
                    f"{entry!r}"
                )

            instance._reverse_map[encoded] = original_bytes
            instance._forward_map[original_bytes] = encoded

        return instance

    @classmethod
    def from_json(cls, payload: str) -> IdTokenBiMap:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("JSON mapping payload must decode to an object")
        return cls.from_dict(parsed)

    @overload
    def frombytes(self, data: bytes, /) -> str: ...

    @overload
    def frombytes(self, data: Iterable[bytes], /) -> Iterator[str]: ...

    def frombytes(self, data: bytes | Iterable[bytes], /) -> str | Iterator[str]:
        """Encode raw bytes and store the original for later reverse lookup.

        Truncation is controlled by the instance ``keep_bits`` set at
        construction. Reverse lookups always return the full original bytes
        regardless of how many bits were encoded.
        """
        if isinstance(data, bytes):
            return self._encode_and_store_single(data)
        return (self._encode_and_store_single(item) for item in data)

    @overload
    def fromhex(self, data: str, /) -> str: ...

    @overload
    def fromhex(self, data: Iterable[str], /) -> Iterator[str]: ...

    def fromhex(self, data: str | Iterable[str], /) -> str | Iterator[str]:
        """Encode hex-decoded bytes and store the original for reverse lookup."""
        if isinstance(data, str):
            return self._encode_and_store_single(_decode_hex_bytes(data))
        return (
            self._encode_and_store_single(_decode_hex_bytes(item)) for item in data
        )

    @overload
    def frombase64(self, data: Base64Value, /) -> str: ...

    @overload
    def frombase64(self, data: Iterable[Base64Value], /) -> Iterator[str]: ...

    def frombase64(
        self, data: Base64Value | Iterable[Base64Value], /
    ) -> str | Iterator[str]:
        """Encode base64-decoded bytes and store the original for reverse lookup."""
        if isinstance(data, (str, bytes)):
            return self._encode_and_store_single(_decode_base64_bytes(data))
        return (
            self._encode_and_store_single(_decode_base64_bytes(item))
            for item in data
        )

    @overload
    def fromuuid(self, data: UUIDValue, /) -> str: ...

    @overload
    def fromuuid(self, data: Iterable[UUIDValue], /) -> Iterator[str]: ...

    def fromuuid(self, data: UUIDValue | Iterable[UUIDValue], /) -> str | Iterator[str]:
        """Encode UUID payloads and store the original for reverse lookup."""
        if isinstance(data, (UUID, str)):
            return self._encode_and_store_single(_decode_uuid_bytes(data))
        return (
            self._encode_and_store_single(_decode_uuid_bytes(item)) for item in data
        )

    @overload
    def tobytes(
        self,
        data: str,
        /,
        *,
        errors: ErrorMode = ...,
    ) -> bytes | None: ...

    @overload
    def tobytes(
        self,
        data: Iterable[str],
        /,
        *,
        errors: ErrorMode = ...,
    ) -> Iterator[bytes | None]: ...

    def tobytes(
        self,
        data: str | Iterable[str],
        /,
        *,
        errors: ErrorMode = DEFAULT_ERROR_MODE,
    ) -> bytes | None | Iterator[bytes | None]:
        mode = _validate_error_mode(errors)
        if isinstance(data, str):
            return self._lookup_bytes_single(data, mode)
        return (self._lookup_bytes_single(item, mode) for item in data)

    @overload
    def tohex(
        self,
        data: str,
        /,
        *,
        errors: ErrorMode = ...,
    ) -> str | None: ...

    @overload
    def tohex(
        self,
        data: Iterable[str],
        /,
        *,
        errors: ErrorMode = ...,
    ) -> Iterator[str | None]: ...

    def tohex(
        self,
        data: str | Iterable[str],
        /,
        *,
        errors: ErrorMode = DEFAULT_ERROR_MODE,
    ) -> str | None | Iterator[str | None]:
        mode = _validate_error_mode(errors)
        if isinstance(data, str):
            return self._lookup_hex_single(data, mode)
        return (self._lookup_hex_single(item, mode) for item in data)

    @overload
    def tobase64(
        self,
        data: str,
        /,
        *,
        errors: ErrorMode = ...,
    ) -> str | None: ...

    @overload
    def tobase64(
        self,
        data: Iterable[str],
        /,
        *,
        errors: ErrorMode = ...,
    ) -> Iterator[str | None]: ...

    def tobase64(
        self,
        data: str | Iterable[str],
        /,
        *,
        errors: ErrorMode = DEFAULT_ERROR_MODE,
    ) -> str | None | Iterator[str | None]:
        mode = _validate_error_mode(errors)
        if isinstance(data, str):
            return self._lookup_base64_single(data, mode)
        return (self._lookup_base64_single(item, mode) for item in data)

    @overload
    def touuid(
        self,
        data: str,
        /,
        *,
        errors: ErrorMode = ...,
    ) -> UUID | None: ...

    @overload
    def touuid(
        self,
        data: Iterable[str],
        /,
        *,
        errors: ErrorMode = ...,
    ) -> Iterator[UUID | None]: ...

    def touuid(
        self,
        data: str | Iterable[str],
        /,
        *,
        errors: ErrorMode = DEFAULT_ERROR_MODE,
    ) -> UUID | None | Iterator[UUID | None]:
        mode = _validate_error_mode(errors)
        if isinstance(data, str):
            return self._lookup_uuid_single(data, mode)
        return (self._lookup_uuid_single(item, mode) for item in data)
