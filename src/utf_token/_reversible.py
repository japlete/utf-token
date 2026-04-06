from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from threading import RLock
from typing import TypeAlias, cast, overload
from uuid import UUID

from ._api import (
    Base64Value,
    UUIDValue,
    _decode_base64_bytes,
    _decode_hex_bytes,
    _decode_uuid_bytes,
    _encode_bytes_single,
    _truncate_input,
    _validate_truncate_bytes,
)
from ._tables import DEFAULT_VOCAB, VocabName

FORMAT_VERSION = 2
EncodingConfig: TypeAlias = tuple[VocabName, int | None]
ForwardMapKey: TypeAlias = tuple[bytes, VocabName, int | None]


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
    encodings: set[EncodingConfig]


def _encoding_sort_key(value: EncodingConfig) -> tuple[str, int]:
    vocab, truncate_bytes = value
    truncate_sort = -1 if truncate_bytes is None else truncate_bytes
    return (vocab, truncate_sort)


class IdTokenBiMap:
    def __init__(self) -> None:
        self._reverse_map: dict[str, StoredMapping] = {}
        self._forward_map: dict[ForwardMapKey, str] = {}
        self._lock = RLock()

    def _store_mapping(
        self,
        encoded: str,
        original_bytes: bytes,
        vocab: VocabName,
        truncate_bytes: int | None,
    ) -> str:
        encoding = (vocab, truncate_bytes)
        mapping = self._reverse_map.get(encoded)
        if mapping is None:
            self._reverse_map[encoded] = StoredMapping(
                original_bytes=original_bytes,
                encodings={encoding},
            )
        else:
            mapping.encodings.add(encoding)
        self._forward_map[(original_bytes, vocab, truncate_bytes)] = encoded
        return encoded

    def _encode_and_store_single(
        self,
        data: bytes,
        *,
        vocab: VocabName,
        truncate_bytes: int | None = None,
    ) -> str:
        truncate_bytes = _validate_truncate_bytes(truncate_bytes)
        key = (data, vocab, truncate_bytes)
        with self._lock:
            cached = self._forward_map.get(key)
            if cached is not None:
                return cached

            working_bytes = _truncate_input(data, truncate_bytes=truncate_bytes)
            encoded = _encode_bytes_single(working_bytes, vocab=vocab)
            mapping = self._reverse_map.get(encoded)
            if mapping is None or mapping.original_bytes == data:
                return self._store_mapping(encoded, data, vocab, truncate_bytes)

            candidate = working_bytes
            while True:
                candidate = _increment_bytes(candidate)
                encoded = _encode_bytes_single(candidate, vocab=vocab)
                mapping = self._reverse_map.get(encoded)
                if mapping is None or mapping.original_bytes == data:
                    return self._store_mapping(encoded, data, vocab, truncate_bytes)

    def _lookup_bytes_single(self, data: str) -> bytes | None:
        with self._lock:
            mapping = self._reverse_map.get(data)
            if mapping is None:
                return None
            return mapping.original_bytes

    def _lookup_hex_single(self, data: str) -> str | None:
        value = self._lookup_bytes_single(data)
        if value is None:
            return None
        return value.hex()

    def _lookup_base64_single(self, data: str) -> str | None:
        value = self._lookup_bytes_single(data)
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")

    def _lookup_uuid_single(self, data: str) -> UUID | None:
        value = self._lookup_bytes_single(data)
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
                        {
                            "vocab": vocab,
                            "truncate_bytes": truncate_bytes,
                        }
                        for vocab, truncate_bytes in sorted(
                            mapping.encodings,
                            key=_encoding_sort_key,
                        )
                    ],
                }
            return {
                "format_version": FORMAT_VERSION,
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

        mappings_obj = payload.get("mappings")
        if not isinstance(mappings_obj, dict):
            raise ValueError("Mapping payload must contain a 'mappings' object")

        instance = cls()

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
            encodings: set[EncodingConfig] = set()
            for item in encodings_obj:
                if not isinstance(item, dict):
                    raise ValueError(f"Mapping entry for {encoded!r} has a non-object encoding")
                encoding_dict = cast(dict[str, object], item)

                vocab_obj = encoding_dict.get("vocab")
                if not isinstance(vocab_obj, str):
                    raise ValueError(f"Mapping entry for {encoded!r} is missing encoding vocab")

                if "truncate_bytes" not in encoding_dict:
                    raise ValueError(
                        f"Mapping entry for {encoded!r} is missing encoding truncate_bytes"
                    )
                truncate_obj = encoding_dict.get("truncate_bytes")
                if truncate_obj is not None and not isinstance(truncate_obj, int):
                    raise ValueError(
                        f"Mapping entry for {encoded!r} has a non-integer truncate_bytes"
                    )
                truncate_bytes = _validate_truncate_bytes(truncate_obj)
                encodings.add((_validate_vocab_name(vocab_obj), truncate_bytes))

            if not encodings:
                raise ValueError(f"Mapping entry for {encoded!r} must list at least one encoding")

            instance._reverse_map[encoded] = StoredMapping(
                original_bytes=original_bytes,
                encodings=set(encodings),
            )
            for vocab, truncate_bytes in encodings:
                key = (original_bytes, vocab, truncate_bytes)
                existing = instance._forward_map.get(key)
                if existing is not None and existing != encoded:
                    raise ValueError(
                        "Conflicting mapping import for "
                        f"{original_hex!r} under vocab {vocab!r} and "
                        f"truncate_bytes={truncate_bytes!r}"
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
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str: ...

    @overload
    def frombytes(
        self,
        data: Iterable[bytes],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def frombytes(
        self,
        data: bytes | Iterable[bytes],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, bytes):
            return self._encode_and_store_single(data, vocab=vocab, truncate_bytes=truncate_bytes)
        return (
            self._encode_and_store_single(item, vocab=vocab, truncate_bytes=truncate_bytes)
            for item in data
        )

    @overload
    def fromhex(
        self,
        data: str,
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str: ...

    @overload
    def fromhex(
        self,
        data: Iterable[str],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def fromhex(
        self,
        data: str | Iterable[str],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, str):
            return self._encode_and_store_single(
                _decode_hex_bytes(data),
                vocab=vocab,
                truncate_bytes=truncate_bytes,
            )
        return (
            self._encode_and_store_single(
                _decode_hex_bytes(item),
                vocab=vocab,
                truncate_bytes=truncate_bytes,
            )
            for item in data
        )

    @overload
    def frombase64(
        self,
        data: Base64Value,
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str: ...

    @overload
    def frombase64(
        self,
        data: Iterable[Base64Value],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def frombase64(
        self,
        data: Base64Value | Iterable[Base64Value],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, (str, bytes)):
            return self._encode_and_store_single(
                _decode_base64_bytes(data),
                vocab=vocab,
                truncate_bytes=truncate_bytes,
            )
        return (
            self._encode_and_store_single(
                _decode_base64_bytes(item),
                vocab=vocab,
                truncate_bytes=truncate_bytes,
            )
            for item in data
        )

    @overload
    def fromuuid(
        self,
        data: UUIDValue,
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str: ...

    @overload
    def fromuuid(
        self,
        data: Iterable[UUIDValue],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> Iterator[str]: ...

    def fromuuid(
        self,
        data: UUIDValue | Iterable[UUIDValue],
        /,
        *,
        vocab: VocabName = DEFAULT_VOCAB,
        truncate_bytes: int | None = None,
    ) -> str | Iterator[str]:
        if isinstance(data, (UUID, str)):
            return self._encode_and_store_single(
                _decode_uuid_bytes(data),
                vocab=vocab,
                truncate_bytes=truncate_bytes,
            )
        return (
            self._encode_and_store_single(
                _decode_uuid_bytes(item),
                vocab=vocab,
                truncate_bytes=truncate_bytes,
            )
            for item in data
        )

    @overload
    def tobytes(self, data: str, /) -> bytes | None: ...

    @overload
    def tobytes(self, data: Iterable[str], /) -> Iterator[bytes | None]: ...

    def tobytes(self, data: str | Iterable[str], /) -> bytes | None | Iterator[bytes | None]:
        if isinstance(data, str):
            return self._lookup_bytes_single(data)
        return (self._lookup_bytes_single(item) for item in data)

    @overload
    def tohex(self, data: str, /) -> str | None: ...

    @overload
    def tohex(self, data: Iterable[str], /) -> Iterator[str | None]: ...

    def tohex(self, data: str | Iterable[str], /) -> str | None | Iterator[str | None]:
        if isinstance(data, str):
            return self._lookup_hex_single(data)
        return (self._lookup_hex_single(item) for item in data)

    @overload
    def tobase64(self, data: str, /) -> str | None: ...

    @overload
    def tobase64(self, data: Iterable[str], /) -> Iterator[str | None]: ...

    def tobase64(self, data: str | Iterable[str], /) -> str | None | Iterator[str | None]:
        if isinstance(data, str):
            return self._lookup_base64_single(data)
        return (self._lookup_base64_single(item) for item in data)

    @overload
    def touuid(self, data: str, /) -> UUID | None: ...

    @overload
    def touuid(self, data: Iterable[str], /) -> Iterator[UUID | None]: ...

    def touuid(self, data: str | Iterable[str], /) -> UUID | None | Iterator[UUID | None]:
        if isinstance(data, str):
            return self._lookup_uuid_single(data)
        return (self._lookup_uuid_single(item) for item in data)
