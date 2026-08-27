"""Canonical data and output paths for source and installed distributions."""

import sysconfig
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_DATA_DIR = _SOURCE_ROOT / "data"
_INSTALLED_DATA_DIR = Path(sysconfig.get_path("data")) / "share" / "quintropy" / "data"

if _SOURCE_DATA_DIR.is_dir():
    ROOT = _SOURCE_ROOT
    DATA_DIR = _SOURCE_DATA_DIR
else:
    ROOT = Path.cwd()
    DATA_DIR = _INSTALLED_DATA_DIR

WORDLISTS_DIR = DATA_DIR / "wordlists"
HISTORY_PATH = DATA_DIR / "history" / "played_5_2026-08-12.csv"
HISTORY_EXTENSION_PATH = DATA_DIR / "history" / "played_5_2026-08-13_2026-08-25.csv"
FREQUENCY_PATH = DATA_DIR / "frequency" / "frequency_list.txt"
RESULTS_DIR = ROOT / "results"
