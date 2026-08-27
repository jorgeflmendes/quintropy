"""Deterministic spelling and pronunciation features."""

from __future__ import annotations

import cmudict
import numpy as np
from wordfreq import zipf_frequency

from .data import load_allowed_words, load_frequency


class EnglishOrthographyFeatures:
    """Dense, documented feature matrix indexed by the action vocabulary."""

    def __init__(self, words: tuple[str, ...]):
        self.words = words
        frequency = load_frequency()
        classic = set(load_allowed_words()[:2_315])
        letters = np.asarray(
            [[ord(character) - 97 for character in word] for word in words],
            dtype=np.int16,
        )
        n_words = len(words)

        # Position and bigram indicators encode local spelling structure.
        positional = np.zeros((n_words, 5 * 26), dtype=np.float32)
        positional[
            np.arange(n_words)[:, None], np.arange(5)[None, :] * 26 + letters
        ] = 1.0
        bigrams = np.zeros((n_words, 26 * 26), dtype=np.float32)
        bigram_ids = letters[:, :-1] * 26 + letters[:, 1:]
        bigrams[np.arange(n_words)[:, None], bigram_ids] = 1.0
        presence = np.zeros((n_words, 26), dtype=np.float32)
        presence[np.arange(n_words)[:, None], letters] = 1.0

        vowel_count = np.isin(letters, np.asarray([0, 4, 8, 14, 20])).sum(axis=1)
        local_frequency = np.log1p([frequency.get(word, 0) for word in words])
        external_frequency = np.asarray(
            [zipf_frequency(word, "en") for word in words], dtype=np.float32
        )
        scalars = np.column_stack(
            (
                local_frequency,
                external_frequency,
                vowel_count,
                5 - presence.sum(axis=1),
                np.isin(letters[:, 0], np.asarray([0, 4, 8, 14, 20])),
                np.isin(letters[:, -1], np.asarray([0, 4, 8, 14, 20])),
                np.fromiter(
                    (word in classic for word in words), dtype=np.float32, count=n_words
                ),
            )
        ).astype(np.float32)
        pronunciation_dictionary = cmudict.dict()
        primary = [pronunciation_dictionary.get(word, [[]])[0] for word in words]
        phone_vocabulary = sorted(
            {phone.rstrip("012") for sequence in primary for phone in sequence}
        )
        phone_index = {phone: index for index, phone in enumerate(phone_vocabulary)}
        first_phone = np.zeros((n_words, len(phone_vocabulary)), dtype=np.float32)
        last_phone = np.zeros_like(first_phone)
        phonetic_scalars = np.zeros((n_words, 6), dtype=np.float32)
        for index, sequence in enumerate(primary):
            if not sequence:
                continue
            first_phone[index, phone_index[sequence[0].rstrip("012")]] = 1.0
            last_phone[index, phone_index[sequence[-1].rstrip("012")]] = 1.0
            stresses = "".join(token[-1] for token in sequence if token[-1].isdigit())
            phonetic_scalars[index] = (
                1.0,
                len(sequence),
                len(stresses),
                stresses.count("1"),
                stresses.count("2"),
                float(sequence[0][-1].isdigit()),
            )
        self.matrix = np.concatenate(
            (
                positional,
                bigrams,
                presence,
                scalars,
                first_phone,
                last_phone,
                phonetic_scalars,
            ),
            axis=1,
        )

    @property
    def feature_count(self) -> int:
        return int(self.matrix.shape[1])
