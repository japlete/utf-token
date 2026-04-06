from __future__ import annotations

import runpy
import unittest
from pathlib import Path
from unittest import mock


MODULE_GLOBALS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "download_token_vocab.py")
)
VOCAB_SOURCES = MODULE_GLOBALS["VOCAB_SOURCES"]
decode_vocab_entries = MODULE_GLOBALS["decode_vocab_entries"]


class DownloadTokenVocabTests(unittest.TestCase):
    def test_gemma4_source_uses_sentencepiece_model(self) -> None:
        source = VOCAB_SOURCES["gemma4"]

        self.assertEqual(source.source_format, "sentencepiece")
        self.assertEqual(source.raw_filename, "tokenizer_gemma4.model")
        self.assertIn("tokenizer_gemma4.model", source.download_url)

    def test_decode_vocab_entries_reads_sentencepiece_proto(self) -> None:
        class FakeSentencePieceProcessor:
            def __init__(self) -> None:
                self.model_proto: bytes | None = None

            def LoadFromSerializedProto(self, model_proto: bytes) -> None:
                self.model_proto = model_proto

            def GetPieceSize(self) -> int:
                return 2

            def IdToPiece(self, index: int) -> str:
                return ["alpha", "_beta"][index]

        with mock.patch.object(
            MODULE_GLOBALS["spm"], "SentencePieceProcessor", FakeSentencePieceProcessor
        ):
            entries = decode_vocab_entries(b"fake-model", source_format="sentencepiece")

        self.assertEqual(entries, [(0, b"alpha"), (1, b"_beta")])


if __name__ == "__main__":
    unittest.main()
