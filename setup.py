from __future__ import annotations

from pathlib import Path
from shutil import copy2

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


PACKAGE_DATA_FILENAMES = (
    "o200k_base_65536_metadata.json",
    "o200k_base_65536_tail_256_tokens.txt",
    "o200k_base_65536_tokens.txt",
    "tokenizer_gemma4_65536_metadata.json",
    "tokenizer_gemma4_65536_tail_256_tokens.txt",
    "tokenizer_gemma4_65536_tokens.txt",
)


class build_py(_build_py):
    def run(self) -> None:
        super().run()

        project_root = Path(__file__).resolve().parent
        source_dir = project_root / "data" / "lookup_tables"
        target_dir = Path(self.build_lib) / "utf_token" / "data"
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename in PACKAGE_DATA_FILENAMES:
            copy2(source_dir / filename, target_dir / filename)


setup(cmdclass={"build_py": build_py})
