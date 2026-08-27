"""Wordle feedback rules and optional full-universe feedback-table creation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - package dependency supplies numba
    njit = None


def encode(words: tuple[str, ...] | list[str]) -> np.ndarray:
    encoded = np.empty((len(words), 5), dtype=np.uint8)
    for index, word in enumerate(words):
        if len(word) != 5 or not word.isascii() or not word.isalpha():
            raise ValueError(f"Invalid five-letter word: {word!r}")
        encoded[index] = [ord(letter) - 97 for letter in word.lower()]
    return encoded


def feedback_code(guess, answer) -> int:
    """Encode Wordle feedback with correct duplicate-letter accounting."""
    if isinstance(guess, str):
        guess = [ord(char) - 97 for char in guess.lower()]
    if isinstance(answer, str):
        answer = [ord(char) - 97 for char in answer.lower()]
    result = [0] * 5
    counts = [0] * 26
    for position in range(5):
        if int(guess[position]) == int(answer[position]):
            result[position] = 2
        else:
            counts[int(answer[position])] += 1
    for position in range(5):
        if result[position] == 0:
            letter = int(guess[position])
            if counts[letter] > 0:
                result[position] = 1
                counts[letter] -= 1
    code, multiplier = 0, 1
    for value in result:
        code += value * multiplier
        multiplier *= 3
    return code


def _build_feedback_table_python(encoded: np.ndarray) -> np.ndarray:
    table = np.empty((len(encoded), len(encoded)), dtype=np.uint8)
    for guess_index, guess in enumerate(encoded):
        for answer_index, answer in enumerate(encoded):
            table[guess_index, answer_index] = feedback_code(guess, answer)
    return table


if njit is not None:

    @njit(cache=True)
    def _build_feedback_table_numba(encoded: np.ndarray) -> np.ndarray:
        size = len(encoded)
        table = np.empty((size, size), dtype=np.uint8)
        for guess_index in range(size):
            for answer_index in range(size):
                result = np.zeros(5, dtype=np.uint8)
                counts = np.zeros(26, dtype=np.uint8)
                for position in range(5):
                    if (
                        encoded[guess_index, position]
                        == encoded[answer_index, position]
                    ):
                        result[position] = 2
                    else:
                        counts[encoded[answer_index, position]] += 1
                for position in range(5):
                    if result[position] == 0:
                        letter = encoded[guess_index, position]
                        if counts[letter] > 0:
                            result[position] = 1
                            counts[letter] -= 1
                code = 0
                multiplier = 1
                for position in range(5):
                    code += result[position] * multiplier
                    multiplier *= 3
                table[guess_index, answer_index] = code
        return table


def build_feedback_table(
    words: tuple[str, ...] | list[str], cache_path=None
) -> np.ndarray:
    """Build an ``actions × answers`` table; intended for offline evaluation."""
    if cache_path is not None:
        from pathlib import Path

        cache_path = Path(cache_path)
        if cache_path.exists():
            table = np.load(cache_path, mmap_mode="r")
            expected = (len(words), len(words))
            if table.shape == expected and table.dtype == np.uint8:
                return table
            cache_path.unlink()
    encoded = encode(words)
    table = (
        _build_feedback_table_numba(encoded)
        if njit is not None
        else _build_feedback_table_python(encoded)
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, table)
    return table


def feedback_cache_path(cache_dir: Path, words: tuple[str, ...]) -> Path:
    """Return a cache path tied to the exact ordered action universe."""
    digest = hashlib.sha256("\0".join(words).encode("ascii")).hexdigest()[:16]
    return cache_dir / f"feedback-{digest}.npy"
