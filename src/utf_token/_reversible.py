from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from threading import RLock
from typing import Literal, TypeAlias, cast, overload
from uuid import UUID

import Levenshtein

from ._api import (
    Base64Value,
    UUIDValue,
    _decode_base64_bytes,
    _decode_hex_bytes,
    _decode_uuid_bytes,
    _encode_bytes_single,
    _truncate_input,
    _validate_keep_bytes,
)
from ._tables import DEFAULT_VOCAB, VocabName

FORMAT_VERSION = 3
ForwardMapKey: TypeAlias = tuple[bytes, int | None]
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


def _increment_bytes(data: bytes) -> bytes:
    next_value = int.from_bytes(data, byteorder="big", signed=False) + 1
    width = max(len(data), (next_value.bit_length() + 7) // 8, 1)
    return next_value.to_bytes(width, byteorder="big", signed=False)


@dataclass(slots=True)
class StoredMapping:
    original_bytes: bytes
    encodings: set[int | None]


def _encoding_sort_key(value: int | None) -> int:
    return -1 if value is None else value


class IdTokenBiMap:
    def __init__(self, vocab: VocabName = DEFAULT_VOCAB) -> None:
        self._vocab = vocab
        self._reverse_map: dict[str, StoredMapping] = {}
        self._forward_map: dict[ForwardMapKey, str] = {}
        self._lock = RLock()

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, str):
            return False
        with self._lock:
            return item in self._reverse_map

    def _store_mapping(
        self,
        encoded: str,
        original_bytes: bytes,
        keep_bytes: int | None,
    ) -> str:
        mapping = self._reverse_map.get(encoded)
        if mapping is None:
            self._reverse_map[encoded] = StoredMapping(
                original_bytes=original_bytes,
                encodings={keep_bytes},
            )
        else:
            mapping.encodings.add(keep_bytes)
        self._forward_map[(original_bytes, keep_bytes)] = encoded
        return encoded

    def _encode_and_store_single(
        self,
        data: bytes,
        *,
        keep_bytes: int | None = None,
    ) -> str:
        keep_bytes = _validate_keep_bytes(keep_bytes)
        key = (data, keep_bytes)
        with self._lock:
            cached = self._forward_map.get(key)
            if cached is not None:
                return cached

            working_bytes = _truncate_input(data, keep_bytes=keep_bytes)
            encoded = _encode_bytes_single(working_bytes, vocab=self._vocab)
            mapping = self._reverse_map.get(encoded)
            if mapping is None or mapping.original_bytes == data:
                return self._store_mapping(encoded, data, keep_bytes)

            candidate = working_bytes
            while True:
                candidate = _increment_bytes(candidate)
                encoded = _encode_bytes_single(candidate, vocab=self._vocab)
                mapping = self._reverse_map.get(encoded)
                if mapping is None or mapping.original_bytes == data:
                    return self._store_mapping(encoded, data, keep_bytes)

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
            mapping = self._reverse_map.get(data)
            if mapping is not None:
                return mapping.original_bytes
            if errors == "raise":
                raise KeyError(data)
            nearest = self._find_nearest_encoded(data)
            if nearest is None:
                return None
            return self._reverse_map[nearest].original_bytes

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
            mappings: dict[str, object] = {}
            for encoded in sorted(self._reverse_map):
                mapping = self._reverse_map[encoded]
                mappings[encoded] = {
                    "original_hex": mapping.original_bytes.hex(),
                    "encodings": [
                        {"keep_bytes": keep_bytes}
                        for keep_bytes in sorted(
                            mapping.encodings,
                            key=_encoding_sort_key,
                        )
                    ],
                }
            return {
                "format_version": FORMAT_VERSION,
                "vocab": self._vocab,
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

        mappings_obj = payload.get("mappings")
        if not isinstance(mappings_obj, dict):
            raise ValueError("Mapping payload must contain a 'mappings' object")

        instance = cls(vocab)

        for encoded, entry in mappings_obj.items():
            if not isinstance(encoded, str):
                raise ValueError("Mapping keys must be encoded strings")
            if not isinstance(entry, dict):
                raise ValueError(f"Mapping entry for {encoded!r} must be an object")
            entry_dict = cast(dict[str, object], entry)

            original_hex = entry_dict.get("original_hex")
            if not isinstance(original_hex, str):
                raise ValueError(f"Mapping entry for {encoded!r} is missing 'original_hex'")

            encodings_obj = entry_dict.get("encodings")
            if not isinstance(encodings_obj, list):
                raise ValueError(f"Mapping entry for {encoded!r} is missing 'encodings'")

            original_bytes = _decode_hex_bytes(original_hex)
            encodings: set[int | None] = set()
            for item in encodings_obj:
                if not isinstance(item, dict):
                    raise ValueError(f"Mapping entry for {encoded!r} has a non-object encoding")
                encoding_dict = cast(dict[str, object], item)

                if "keep_bytes" not in encoding_dict:
                    raise ValueError(
                        f"Mapping entry for {encoded!r} is missing encoding keep_bytes"
                    )
                truncate_obj = encoding_dict.get("keep_bytes")
                if truncate_obj is not None and not isinstance(truncate_obj, int):
                    raise ValueError(
                        f"Mapping entry for {encoded!r} has a non-integer keep_bytes"
                    )
                encodings.add(_validate_keep_bytes(truncate_obj))

            if not encodings:
                raise ValueError(f"Mapping entry for {encoded!r} must list at least one encoding")

            instance._reverse_map[encoded] = StoredMapping(
                original_bytes=original_bytes,
                encodings=set(encodings),
            )
            for keep_bytes in encodings:
                key = (original_bytes, keep_bytes)
                existing = instance._forward_map.get(key)
                if existing is not None and existing != encoded:
                    raise ValueError(
                        "Conflicting mapping import for "
                        f"{original_hex!r} with keep_bytes={keep_bytes!r}"
                    )
                instance._forward_map[key] = encoded

        return instance

    @classmethod
    def from_json(cls, payload: str) -> IdTokenBiMap:
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("JSON mapping payload must decode to an object")
        return cls.from_dict(parsed)

    @overload
    def frombytes(
        self,
        data: bytes,
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str: ...

    @overload
    def frombytes(
        self,
        data: Iterable[bytes],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def frombytes(
        self,
        data: bytes | Iterable[bytes],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, bytes):
            return self._encode_and_store_single(data, keep_bytes=keep_bytes)
        return (
            self._encode_and_store_single(item, keep_bytes=keep_bytes)
            for item in data
        )

    @overload
    def fromhex(
        self,
        data: str,
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str: ...

    @overload
    def fromhex(
        self,
        data: Iterable[str],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def fromhex(
        self,
        data: str | Iterable[str],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, str):
            return self._encode_and_store_single(
                _decode_hex_bytes(data),
                keep_bytes=keep_bytes,
            )
        return (
            self._encode_and_store_single(
                _decode_hex_bytes(item),
                keep_bytes=keep_bytes,
            )
            for item in data
        )

    @overload
    def frombase64(
        self,
        data: Base64Value,
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str: ...

    @overload
    def frombase64(
        self,
        data: Iterable[Base64Value],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def frombase64(
        self,
        data: Base64Value | Iterable[Base64Value],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, (str, bytes)):
            return self._encode_and_store_single(
                _decode_base64_bytes(data),
                keep_bytes=keep_bytes,
            )
        return (
            self._encode_and_store_single(
                _decode_base64_bytes(item),
                keep_bytes=keep_bytes,
            )
            for item in data
        )

    @overload
    def fromuuid(
        self,
        data: UUIDValue,
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str: ...

    @overload
    def fromuuid(
        self,
        data: Iterable[UUIDValue],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def fromuuid(
        self,
        data: UUIDValue | Iterable[UUIDValue],
        /,
        *,
        keep_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, (UUID, str)):
            return self._encode_and_store_single(
                _decode_uuid_bytes(data),
                keep_bytes=keep_bytes,
            )
        return (
            self._encode_and_store_single(
                _decode_uuid_bytes(item),
                keep_bytes=keep_bytes,
            )
            for item in data
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
