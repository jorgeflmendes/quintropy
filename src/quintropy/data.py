"""Validated access to the local, versioned research inputs."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from wordfreq import zipf_frequency

from .paths import (
    FREQUENCY_PATH,
    HISTORY_EXTENSION_PATH,
    HISTORY_PATH,
    HISTORY_UPDATES_PATH,
    WORDLISTS_DIR,
)

WORD_RE = re.compile(r'"([a-z]{5})"')


def sha256(path: Path) -> str:
    """Return the content digest used in experiment provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_allowed_words() -> tuple[str, ...]:
    """Load the fixed Wordle action universe in a deterministic order."""
    files = (
        WORDLISTS_DIR / "solutions_classic_2315.txt",
        WORDLISTS_DIR / "nonsolutions_classic_10657.txt",
    )
    words = tuple(word for path in files for word in WORD_RE.findall(path.read_text()))
    if len(words) != 12_972 or len(set(words)) != len(words):
        raise ValueError(
            "The allowed-word universe is not the expected 12,972 unique words."
        )
    return words


@lru_cache(maxsize=1)
def load_answer_words() -> tuple[str, ...]:
    """Build the answer universe without consulting evaluation targets."""
    words = load_allowed_words()
    classic = set(words[:2_315])
    answers = tuple(
        word for word in words if word in classic or zipf_frequency(word, "en") > 0
    )
    if not answers or not classic <= set(answers):
        raise ValueError("The answer universe must contain every classic solution.")
    return answers


def answer_indices(words: tuple[str, ...]) -> np.ndarray:
    """Return action indices that belong to the static answer universe."""
    answer_set = set(load_answer_words())
    indices = np.fromiter(
        (index for index, word in enumerate(words) if word in answer_set),
        dtype=np.int32,
    )
    if not len(indices):
        raise ValueError("The action vocabulary has no words in the answer universe.")
    return indices


def answer_universe_metadata(words: tuple[str, ...]) -> dict[str, object]:
    """Describe and bind the ordered answer subset used by an experiment."""
    indices = answer_indices(words)
    selected = tuple(words[int(index)] for index in indices)
    digest = hashlib.sha256("\0".join(selected).encode("ascii")).hexdigest()
    return {
        "definition": "classic_2315_plus_wordfreq_3.1.1_positive",
        "size": len(selected),
        "sha256": digest,
    }


def load_frequency() -> dict[str, int]:
    """Load static lexical frequencies; non-five-letter entries are ignored."""
    frequency: dict[str, int] = {}
    for line in FREQUENCY_PATH.read_text(encoding="utf-8").splitlines():
        try:
            word, count = line.rsplit("\t", maxsplit=1)
        except ValueError:
            continue
        word = word.lower()
        if len(word) == 5 and word.isascii() and word.isalpha():
            frequency[word] = int(count)
    if not frequency:
        raise ValueError("Frequency source contained no usable five-letter words.")
    return frequency


def load_history() -> pd.DataFrame:
    """Load chronological answers and enforce the source's basic invariants."""
    history = pd.concat(
        (
            pd.read_csv(path, parse_dates=["date"])
            for path in (HISTORY_PATH, HISTORY_EXTENSION_PATH, HISTORY_UPDATES_PATH)
        ),
        ignore_index=True,
    )
    required = ["game", "date", "answer"]
    if list(history.columns) != required:
        raise ValueError(
            f"History columns must be {required}, got {list(history.columns)}."
        )
    if history.empty or history[required].isna().any().any():
        raise ValueError("History must contain complete game, date, and answer rows.")
    if (
        not history["date"].is_monotonic_increasing
        or history["date"].duplicated().any()
    ):
        raise ValueError(
            "History must contain one strictly chronological answer per date."
        )
    history["answer"] = history["answer"].str.lower()
    if (
        history["game"].duplicated().any()
        or not history["game"].is_monotonic_increasing
    ):
        raise ValueError("History games must be unique and chronological.")
    if not np.all(np.diff(history["game"].to_numpy(dtype=np.int64)) == 1):
        raise ValueError("History game numbers must be consecutive.")
    day_steps = history["date"].diff().dropna().dt.days.to_numpy(dtype=np.int64)
    if not np.all(day_steps == 1):
        raise ValueError("History dates must be consecutive daily observations.")
    if (
        not history["answer"]
        .map(
            lambda word: isinstance(word, str) and bool(re.fullmatch(r"[a-z]{5}", word))
        )
        .all()
    ):
        raise ValueError("History answers must be lowercase five-letter ASCII words.")
    allowed = set(load_allowed_words())
    unsupported = sorted(set(history["answer"]) - allowed)
    if unsupported:
        raise ValueError(
            f"History contains answers outside the action universe: {unsupported[:5]}."
        )
    answer_set = set(load_answer_words())
    unsupported_answers = sorted(set(history["answer"]) - answer_set)
    if unsupported_answers:
        raise ValueError(
            f"History contains answers outside the declared answer universe: {unsupported_answers[:5]}."
        )
    return history


@dataclass(frozen=True)
class DataProvenance:
    """Hashes that bind an experiment to its exact local inputs."""

    allowed_solutions_sha256: str
    allowed_nonsolutions_sha256: str
    frequency_sha256: str
    history_sha256: str
    history_extension_sha256: str
    history_updates_sha256: str


def data_provenance() -> DataProvenance:
    return DataProvenance(
        allowed_solutions_sha256=sha256(WORDLISTS_DIR / "solutions_classic_2315.txt"),
        allowed_nonsolutions_sha256=sha256(
            WORDLISTS_DIR / "nonsolutions_classic_10657.txt"
        ),
        frequency_sha256=sha256(FREQUENCY_PATH),
        history_sha256=sha256(HISTORY_PATH),
        history_extension_sha256=sha256(HISTORY_EXTENSION_PATH),
        history_updates_sha256=sha256(HISTORY_UPDATES_PATH),
    )


def implementation_sha256() -> str:
    """Digest the importable package sources used by a local experiment."""
    source_dir = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(source_dir.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def dependency_versions() -> dict[str, str]:
    """Return versions of libraries that materially affect generated priors."""
    return {
        distribution: version(distribution)
        for distribution in ("numpy", "pandas", "scikit-learn", "wordfreq", "cmudict")
    }
