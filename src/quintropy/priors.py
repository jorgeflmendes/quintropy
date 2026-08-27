"""Causal, deterministic prior construction and provenance-preserving storage."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    DataProvenance,
    answer_indices,
    answer_universe_metadata,
    data_provenance,
    dependency_versions,
    implementation_sha256,
    load_allowed_words,
    load_frequency,
)
from .linguistics import EnglishOrthographyFeatures


@dataclass(frozen=True)
class FrequencyPriorConfig:
    """Configuration for the static lexical prior."""

    floor_count: int = 1
    seen_word_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.seen_word_multiplier)
            or self.floor_count <= 0
            or not 0 < self.seen_word_multiplier <= 1
        ):
            raise ValueError(
                "floor_count must be positive and seen_word_multiplier in (0, 1]."
            )


@dataclass(frozen=True)
class LinguisticPriorConfig:
    """Configuration for the causal linguistic classifier."""

    regularization: float = 0.01
    negative_ratio: int = 8
    retrain_every_games: int = 28
    minimum_history: int = 120

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.regularization)
            or self.regularization <= 0
            or self.negative_ratio < 1
            or self.retrain_every_games < 1
            or self.minimum_history < 20
        ):
            raise ValueError("Linguistic-prior hyperparameters are invalid.")


@dataclass(frozen=True)
class HybridPriorConfig:
    """Configuration for the editorial and linguistic mixture."""

    editorial: "EditorialRegimeConfig" = field(
        default_factory=lambda: EditorialRegimeConfig()
    )
    linguistic: LinguisticPriorConfig = field(default_factory=LinguisticPriorConfig)
    linguistic_weight: float = 0.10

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.linguistic_weight)
            or not 0 <= self.linguistic_weight <= 1
        ):
            raise ValueError("linguistic_weight must be in [0, 1].")


@dataclass(frozen=True)
class PriorManifest:
    model: str
    config: dict[str, object]
    generated_for_dates: list[str]
    information_cutoff: str
    provenance: DataProvenance
    answer_universe: dict[str, object]
    generator_implementation_sha256: str
    dependencies: dict[str, str]


class FrequencyPrior:
    """Static frequency prior with an optional causal repeat penalty."""

    def __init__(
        self,
        words: tuple[str, ...],
        config: FrequencyPriorConfig = FrequencyPriorConfig(),
    ):
        self.words = words
        self.config = config
        counts = load_frequency()
        self._base = np.array(
            [max(counts.get(word, 0), config.floor_count) for word in words],
            dtype=float,
        )
        self._answer_indices = answer_indices(words)
        self._answer_mask = np.zeros(len(words), dtype=bool)
        self._answer_mask[self._answer_indices] = True

    def predict(self, history_before: pd.DataFrame) -> np.ndarray:
        """Return a distribution using only rows strictly before the target date."""
        weights = self._base.copy()
        weights[~self._answer_mask] = 0.0
        if self.config.seen_word_multiplier < 1:
            seen = set(history_before["answer"])
            weights[[i for i, word in enumerate(self.words) if word in seen]] *= (
                self.config.seen_word_multiplier
            )
        return weights / weights.sum()


class CausalLinguisticPrior:
    """Causal logistic density-ratio model over linguistic features."""

    def __init__(
        self,
        words: tuple[str, ...],
        config: LinguisticPriorConfig = LinguisticPriorConfig(),
    ):
        self.words = words
        self.config = config
        self._features = EnglishOrthographyFeatures(words)
        self._word_index = {word: position for position, word in enumerate(words)}
        self._answer_indices = answer_indices(words)
        self._model = None
        self._last_fit_size = -1
        self._last_fit_signature: tuple[tuple[int, int, str], ...] | None = None

    @staticmethod
    def _history_signature(
        history_before: pd.DataFrame,
    ) -> tuple[tuple[int, int, str], ...]:
        """Identify a history prefix by content rather than row count."""
        return tuple(
            (int(row.game), int(pd.Timestamp(row.date).value), str(row.answer))
            for row in history_before[["game", "date", "answer"]].itertuples(
                index=False
            )
        )

    def _negative_indices(self, positive_indices: np.ndarray) -> np.ndarray:
        """Sample deterministic references from the answer universe."""
        n_answers = len(self._answer_indices)
        ratio = self.config.negative_ratio
        rows = np.arange(len(positive_indices), dtype=np.int64)[:, None]
        offsets = np.arange(1, ratio + 1, dtype=np.int64)[None, :]
        positions = ((rows + 1) * 7_919 + offsets * 12_345) % n_answers
        candidates = self._answer_indices[positions]
        candidates = np.where(
            candidates == positive_indices[:, None],
            self._answer_indices[(positions + 1) % n_answers],
            candidates,
        )
        return candidates.astype(np.int32, copy=False).ravel()

    def _fit(self, history_before: pd.DataFrame) -> None:
        positives = np.asarray(
            [self._word_index[answer] for answer in history_before["answer"]],
            dtype=np.int32,
        )
        negatives = self._negative_indices(positives)
        sample_indices = np.concatenate((positives, negatives))
        labels = np.concatenate(
            (
                np.ones(len(positives), dtype=np.int8),
                np.zeros(len(negatives), dtype=np.int8),
            )
        )
        self._model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=self.config.regularization,
                solver="lbfgs",
                max_iter=400,
                class_weight="balanced",
            ),
        )
        self._model.fit(self._features.matrix[sample_indices], labels)
        self._last_fit_size = len(history_before)
        self._last_fit_signature = self._history_signature(history_before)

    def predict(self, history_before: pd.DataFrame, target_date=None) -> np.ndarray:
        if (
            target_date is not None
            and not history_before.empty
            and history_before["date"].max() >= target_date
        ):
            raise ValueError("Linguistic prior received a target or future row.")
        if len(history_before) < self.config.minimum_history:
            probability = np.zeros(len(self.words), dtype=float)
            probability[self._answer_indices] = 1.0 / len(self._answer_indices)
            return probability
        signature = self._history_signature(history_before)
        same_training_prefix = (
            self._last_fit_signature is not None
            and len(signature) >= self._last_fit_size
            and signature[: self._last_fit_size] == self._last_fit_signature
        )
        if (
            self._model is None
            or not same_training_prefix
            or len(history_before) - self._last_fit_size
            >= self.config.retrain_every_games
        ):
            self._fit(history_before)
        # The intercept cancels when density ratios are normalized.
        log_density_ratio = self._model.decision_function(
            self._features.matrix[self._answer_indices]
        )
        log_density_ratio -= float(np.max(log_density_ratio))
        probability = np.zeros(len(self.words), dtype=float)
        probability[self._answer_indices] = np.exp(log_density_ratio)
        return probability / probability.sum()


class HybridEditorialLinguisticPrior:
    """Arithmetic mixture of two independently causal, normalized priors."""

    def __init__(self, words: tuple[str, ...], config: HybridPriorConfig):
        self.config = config
        self._editorial = EditorialRegimePrior(words, config.editorial)
        self._linguistic = CausalLinguisticPrior(words, config.linguistic)

    def predict(self, history_before: pd.DataFrame, target_date) -> np.ndarray:
        editorial = self._editorial.predict(history_before, target_date)
        linguistic = self._linguistic.predict(history_before, target_date)
        return (
            1 - self.config.linguistic_weight
        ) * editorial + self.config.linguistic_weight * linguistic


@dataclass(frozen=True)
class EditorialRegimeConfig:
    """Configuration for the causal editorial-regime prior."""

    frequency_temperature: float = 4.0
    frequency_temperature_expanded: float | None = None
    regime_feature_weight: float = 0.5
    regime_frequency_profile_weight: float = 0.0
    frequency_profile_weight: float = 0.0
    positional_weight: float = 1.0
    bigram_weight: float = 0.0
    structural_weight: float = 0.5
    recent_overlap_weight: float = 0.0
    recent_overlap_window: int = 14
    positional_smoothing: float = 5.0
    bigram_smoothing: float = 2.0
    structural_smoothing: float = 2.0
    recency_half_life_days: float = 365.0

    def __post_init__(self) -> None:
        values = asdict(self)
        numeric = [
            value
            for value in values.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if not all(math.isfinite(value) for value in numeric if value is not None):
            raise ValueError("Editorial-regime hyperparameters must be finite.")
        if (
            self.frequency_temperature_expanded is not None
            and self.frequency_temperature_expanded <= 0
        ):
            raise ValueError(
                "frequency_temperature_expanded must be positive when provided."
            )
        if not 0 <= self.regime_feature_weight <= 1:
            raise ValueError("regime_feature_weight must be in [0, 1].")
        if self.regime_frequency_profile_weight < 0:
            raise ValueError("regime_frequency_profile_weight must be non-negative.")
        if (
            self.bigram_weight < 0
            or self.frequency_profile_weight < 0
            or self.recent_overlap_window < 1
            or any(
                value <= 0
                for key, value in values.items()
                if key
                not in {
                    "frequency_temperature_expanded",
                    "regime_feature_weight",
                    "regime_frequency_profile_weight",
                    "bigram_weight",
                    "frequency_profile_weight",
                    "recent_overlap_weight",
                    "recent_overlap_window",
                }
            )
        ):
            raise ValueError("Editorial-regime hyperparameters are invalid.")


class EditorialRegimePrior:
    """Causal prior over classic/expanded and unseen/repeated regimes."""

    def __init__(
        self,
        words: tuple[str, ...],
        config: EditorialRegimeConfig = EditorialRegimeConfig(),
    ):
        self.words = words
        self.config = config
        counts = load_frequency()
        raw_frequency = np.asarray([counts.get(word, 0) for word in words], dtype=float)
        self._frequency_score = np.log1p(raw_frequency)
        positive = self._frequency_score[self._frequency_score > 0]
        cutoffs = (
            np.quantile(positive, np.linspace(0, 1, 9))
            if len(positive)
            else np.array([0.0])
        )
        self._frequency_bin = np.where(
            self._frequency_score > 0,
            1 + np.digitize(self._frequency_score, cutoffs[1:-1]),
            0,
        ).astype(np.int8)
        self._frequency_bin_count = int(self._frequency_bin.max()) + 1
        self._base_frequency_mass = np.bincount(
            self._frequency_bin, minlength=self._frequency_bin_count
        ) / len(words)
        self._letters = np.asarray(
            [[ord(letter) - 97 for letter in word] for word in words], dtype=np.int16
        )
        self._bigrams = self._letters[:, :-1] * 26 + self._letters[:, 1:]
        self._letter_sets = np.zeros((len(words), 26), dtype=np.int8)
        for index, letters in enumerate(self._letters):
            self._letter_sets[index, np.unique(letters)] = 1
        self._vowel_counts = np.asarray(
            [sum(letter in "aeiou" for letter in word) for word in words], dtype=np.int8
        )
        self._repeat_counts = np.asarray(
            [5 - len(set(word)) for word in words], dtype=np.int8
        )
        self._word_index = {word: index for index, word in enumerate(words)}
        self._answer_indices = answer_indices(words)
        self._answer_mask = np.zeros(len(words), dtype=bool)
        self._answer_mask[self._answer_indices] = True
        classic_words = set(load_allowed_words()[:2_315])
        self._classic_mask = np.fromiter(
            (word in classic_words for word in words), dtype=bool, count=len(words)
        )

    def predict(self, history_before: pd.DataFrame, target_date=None) -> np.ndarray:
        """Return a normalized prior using only rows supplied as past history."""
        if (
            target_date is not None
            and not history_before.empty
            and history_before["date"].max() >= target_date
        ):
            raise ValueError("Editorial prior received a target or future row.")
        config = self.config
        expanded_temperature = (
            config.frequency_temperature_expanded or config.frequency_temperature
        )
        temperatures = np.where(
            self._classic_mask, config.frequency_temperature, expanded_temperature
        )
        score = self._frequency_score / temperatures
        position_counts = np.full((5, 26), config.positional_smoothing, dtype=float)
        bigram_counts = np.full((4, 26 * 26), config.bigram_smoothing, dtype=float)
        vowel_counts = np.full(6, config.structural_smoothing, dtype=float)
        repeat_counts = np.full(5, config.structural_smoothing, dtype=float)
        regime_position_counts = np.full(
            (2, 2, 5, 26), config.positional_smoothing, dtype=float
        )
        regime_bigram_counts = np.full(
            (2, 2, 4, 26 * 26), config.bigram_smoothing, dtype=float
        )
        regime_vowel_counts = np.full(
            (2, 2, 6), config.structural_smoothing, dtype=float
        )
        regime_repeat_counts = np.full(
            (2, 2, 5), config.structural_smoothing, dtype=float
        )
        regime_frequency_profile = np.full(
            (2, 2, self._frequency_bin_count), 0.5, dtype=float
        )
        frequency_profile = np.full(self._frequency_bin_count, 0.5, dtype=float)
        regime_counts = np.full((2, 2), 0.5, dtype=float)
        seen_before: set[str] = set()
        if not history_before.empty:
            if target_date is None:
                target_date = history_before["date"].max() + pd.Timedelta(days=1)
            age_days = (
                pd.Timestamp(target_date) - history_before["date"]
            ).dt.days.to_numpy(dtype=float)
            weights = np.exp2(-age_days / config.recency_half_life_days)
            for answer, weight in zip(history_before["answer"], weights):
                answer_index = self._word_index.get(answer)
                if answer_index is None:
                    raise ValueError(
                        f"Historical answer {answer!r} is outside the action universe."
                    )
                letters = [ord(letter) - 97 for letter in answer]
                position_counts[np.arange(5), letters] += weight
                bigrams = np.asarray(letters[:-1]) * 26 + np.asarray(letters[1:])
                bigram_counts[np.arange(4), bigrams] += weight
                vowel_counts[sum(letter in "aeiou" for letter in answer)] += weight
                repeat_counts[5 - len(set(answer))] += weight
                frequency_profile[self._frequency_bin[answer_index]] += weight
                regime = (
                    int(answer not in seen_before),
                    int(self._classic_mask[answer_index]),
                )
                regime_position_counts[regime[0], regime[1], np.arange(5), letters] += (
                    weight
                )
                regime_bigram_counts[regime[0], regime[1], np.arange(4), bigrams] += (
                    weight
                )
                regime_vowel_counts[
                    regime[0], regime[1], sum(letter in "aeiou" for letter in answer)
                ] += weight
                regime_repeat_counts[regime[0], regime[1], 5 - len(set(answer))] += (
                    weight
                )
                regime_frequency_profile[
                    regime[0], regime[1], self._frequency_bin[answer_index]
                ] += weight
                regime_counts[regime] += weight
                seen_before.add(answer)
        position_probability = position_counts / position_counts.sum(
            axis=1, keepdims=True
        )
        global_position_log = np.log(
            position_probability[np.arange(5)[:, None], self._letters.T]
        ).sum(axis=0)
        bigram_probability = bigram_counts / bigram_counts.sum(axis=1, keepdims=True)
        global_bigram_log = np.log(
            bigram_probability[np.arange(4)[:, None], self._bigrams.T]
        ).sum(axis=0)
        global_structural_log = np.log(
            vowel_counts[self._vowel_counts] / vowel_counts.sum()
        ) + np.log(repeat_counts[self._repeat_counts] / repeat_counts.sum())
        if config.frequency_profile_weight:
            profile_probability = frequency_profile / frequency_profile.sum()
            density_ratio = profile_probability / self._base_frequency_mass
            score += config.frequency_profile_weight * np.log(
                density_ratio[self._frequency_bin]
            )
        recent = history_before["answer"].iloc[-config.recent_overlap_window :]
        if config.recent_overlap_weight and len(recent):
            recent_indices = [self._word_index[answer] for answer in recent]
            overlap = (self._letter_sets @ self._letter_sets[recent_indices].T).max(
                axis=1
            )
            score += config.recent_overlap_weight * overlap
        seen = set(history_before["answer"])
        unseen_mask = np.fromiter(
            (word not in seen for word in self.words), dtype=bool, count=len(self.words)
        )
        regime_mass = regime_counts / regime_counts.sum()
        prior = np.zeros(len(self.words), dtype=float)
        for unseen in (False, True):
            for classic in (False, True):
                mask = (
                    self._answer_mask
                    & (unseen_mask if unseen else ~unseen_mask)
                    & (self._classic_mask if classic else ~self._classic_mask)
                )
                if mask.any():
                    regime = (int(unseen), int(classic))
                    regime_position_probability = regime_position_counts[
                        regime
                    ] / regime_position_counts[regime].sum(axis=1, keepdims=True)
                    regime_position_log = np.log(
                        regime_position_probability[
                            np.arange(5)[:, None], self._letters[mask].T
                        ]
                    ).sum(axis=0)
                    regime_bigram_probability = regime_bigram_counts[
                        regime
                    ] / regime_bigram_counts[regime].sum(axis=1, keepdims=True)
                    regime_bigram_log = np.log(
                        regime_bigram_probability[
                            np.arange(4)[:, None], self._bigrams[mask].T
                        ]
                    ).sum(axis=0)
                    regime_structural_log = np.log(
                        regime_vowel_counts[
                            regime[0], regime[1], self._vowel_counts[mask]
                        ]
                        / regime_vowel_counts[regime].sum()
                    ) + np.log(
                        regime_repeat_counts[
                            regime[0], regime[1], self._repeat_counts[mask]
                        ]
                        / regime_repeat_counts[regime].sum()
                    )
                    feature_weight = config.regime_feature_weight
                    score[mask] += config.positional_weight * (
                        (1 - feature_weight) * global_position_log[mask]
                        + feature_weight * regime_position_log
                    )
                    score[mask] += config.bigram_weight * (
                        (1 - feature_weight) * global_bigram_log[mask]
                        + feature_weight * regime_bigram_log
                    )
                    score[mask] += config.structural_weight * (
                        (1 - feature_weight) * global_structural_log[mask]
                        + feature_weight * regime_structural_log
                    )
                    if config.regime_frequency_profile_weight:
                        profile_probability = (
                            regime_frequency_profile[regime]
                            / regime_frequency_profile[regime].sum()
                        )
                        density_ratio = profile_probability / self._base_frequency_mass
                        score[mask] += config.regime_frequency_profile_weight * np.log(
                            density_ratio[self._frequency_bin[mask]]
                        )
        score -= score.max()
        unnormalized = np.exp(score)
        for unseen in (False, True):
            for classic in (False, True):
                mask = (
                    self._answer_mask
                    & (unseen_mask if unseen else ~unseen_mask)
                    & (self._classic_mask if classic else ~self._classic_mask)
                )
                if mask.any():
                    prior[mask] = (
                        regime_mass[int(unseen), int(classic)]
                        * unnormalized[mask]
                        / unnormalized[mask].sum()
                    )
        return prior / prior.sum()


def generate_priors(
    history: pd.DataFrame,
    words: tuple[str, ...],
    start: date,
    end: date,
    config: FrequencyPriorConfig
    | LinguisticPriorConfig
    | EditorialRegimeConfig
    | HybridPriorConfig = FrequencyPriorConfig(),
) -> tuple[pd.DataFrame, np.ndarray, PriorManifest]:
    """Generate daily priors using only earlier history rows."""
    targets = history[
        (history["date"].dt.date >= start) & (history["date"].dt.date <= end)
    ].copy()
    if targets.empty:
        raise ValueError(
            "The requested evaluation window has no games in the local history."
        )
    if isinstance(config, FrequencyPriorConfig):
        model = FrequencyPrior(words, config)
        model_name = "frequency_prior"
    elif isinstance(config, LinguisticPriorConfig):
        model = CausalLinguisticPrior(words, config)
        model_name = "causal_linguistic_prior"
    elif isinstance(config, HybridPriorConfig):
        model = HybridEditorialLinguisticPrior(words, config)
        model_name = "hybrid_editorial_linguistic_prior"
    elif isinstance(config, EditorialRegimeConfig):
        model = EditorialRegimePrior(words, config)
        model_name = "editorial_regime_prior"
    else:
        raise TypeError("Unsupported prior configuration.")
    priors = []
    for target in targets.itertuples(index=False):
        past = history[history["date"] < target.date]
        if not past.empty and past["date"].max() >= target.date:
            raise AssertionError("Future information reached prior construction.")
        priors.append(
            model.predict(past, target.date)
            if isinstance(
                model,
                (
                    EditorialRegimePrior,
                    CausalLinguisticPrior,
                    HybridEditorialLinguisticPrior,
                ),
            )
            else model.predict(past)
        )
    manifest = PriorManifest(
        model=model_name,
        config=asdict(config),
        generated_for_dates=[
            day.date.isoformat() for day in targets.itertuples(index=False)
        ],
        information_cutoff="strictly before each target date",
        provenance=data_provenance(),
        answer_universe=answer_universe_metadata(words),
        generator_implementation_sha256=implementation_sha256(),
        dependencies=dependency_versions(),
    )
    return targets.reset_index(drop=True), np.vstack(priors), manifest


def save_prior_artifact(
    path: Path,
    words: tuple[str, ...],
    targets: pd.DataFrame,
    priors: np.ndarray,
    manifest: PriorManifest,
) -> None:
    """Save arrays and immutable JSON metadata together in an NPZ artifact."""
    answers = answer_indices(words)
    outside = np.ones(len(words), dtype=bool)
    outside[answers] = False
    if np.any(np.asarray(priors)[:, outside] != 0):
        raise ValueError(
            "Priors must assign zero probability outside the answer universe."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        words=np.asarray(words),
        answer_indices=answers,
        games=targets["game"].to_numpy(dtype=np.int32),
        dates=np.asarray(targets["date"].dt.strftime("%Y-%m-%d"), dtype="U10"),
        answers=np.asarray(targets["answer"], dtype="U5"),
        priors=priors.astype(np.float64),
        manifest=json.dumps(asdict(manifest), sort_keys=True),
    )


def load_prior_artifact(
    path: Path,
) -> tuple[tuple[str, ...], np.ndarray, pd.DataFrame, np.ndarray, dict[str, object]]:
    """Load and validate a prior artifact written by :func:`save_prior_artifact`."""
    with np.load(path, allow_pickle=False) as artifact:
        schema_v1_required = {
            "words",
            "games",
            "dates",
            "answers",
            "priors",
            "manifest",
        }
        modern_required = schema_v1_required | {"answer_indices"}
        artifact_fields = set(artifact.files)
        if artifact_fields not in (schema_v1_required, modern_required):
            raise ValueError("Not a modern Quintropy prior artifact.")
        is_schema_two = artifact_fields == modern_required
        words = tuple(artifact["words"].astype(str))
        stored_answer_indices = (
            artifact["answer_indices"].astype(np.int32)
            if "answer_indices" in artifact.files
            else np.arange(len(words), dtype=np.int32)
        )
        priors = artifact["priors"].astype(float)
        targets = pd.DataFrame(
            {
                "game": artifact["games"].astype(int),
                "date": pd.to_datetime(artifact["dates"].astype(str)),
                "answer": artifact["answers"].astype(str),
            }
        )
        manifest = json.loads(str(artifact["manifest"].item()))
    if (
        priors.shape != (len(targets), len(words))
        or not np.isfinite(priors).all()
        or np.any(priors < 0)
    ):
        raise ValueError("Prior artifact has invalid dimensions or probabilities.")
    if not np.allclose(priors.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Prior rows must be normalized.")
    if (
        stored_answer_indices.ndim != 1
        or not len(stored_answer_indices)
        or np.any(stored_answer_indices < 0)
        or np.any(stored_answer_indices >= len(words))
        or len(np.unique(stored_answer_indices)) != len(stored_answer_indices)
    ):
        raise ValueError("Prior artifact has invalid answer indices.")
    outside = np.ones(len(words), dtype=bool)
    outside[stored_answer_indices] = False
    if np.any(priors[:, outside] != 0):
        raise ValueError(
            "Prior artifact assigns probability outside its answer universe."
        )
    word_index = {word: index for index, word in enumerate(words)}
    support = set(int(index) for index in stored_answer_indices)
    if any(word_index.get(answer) not in support for answer in targets["answer"]):
        raise ValueError(
            "Prior artifact contains a target outside its answer universe."
        )
    if is_schema_two:
        selected_words = tuple(words[int(index)] for index in stored_answer_indices)
        expected_digest = (
            __import__("hashlib")
            .sha256("\0".join(selected_words).encode("ascii"))
            .hexdigest()
        )
        metadata = manifest.get("answer_universe")
        if (
            not isinstance(metadata, dict)
            or metadata.get("size") != len(stored_answer_indices)
            or metadata.get("sha256") != expected_digest
        ):
            raise ValueError(
                "Prior artifact answer-universe metadata does not match its indices."
            )
        generator_hash = manifest.get("generator_implementation_sha256")
        if not isinstance(generator_hash, str) or len(generator_hash) != 64:
            raise ValueError(
                "Prior artifact lacks a valid generator implementation hash."
            )
        dependencies = manifest.get("dependencies")
        if not isinstance(dependencies, dict) or not dependencies:
            raise ValueError("Prior artifact lacks dependency provenance.")
    return words, stored_answer_indices, targets, priors, manifest
