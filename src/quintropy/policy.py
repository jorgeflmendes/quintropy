"""Decision policies with explicit units, configuration, and endgame search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from wordfreq import zipf_frequency

from .data import answer_indices as default_answer_indices
from .data import load_allowed_words

ALL_GREEN = 242


@dataclass(frozen=True)
class PolicyConfig:
    """Solver parameters; entropy-derived weights are measured in bits."""

    starter: str = "soare"
    adaptive_starter: bool = False
    direct_hit_weight: float = 3.0
    late_hit_weight: float = 0.0
    exploit_threshold: float = 0.4
    exact_endgame_limit: int = 3
    tail_wordfreq_weight: float = 0.0
    tail_wordfreq_gap: float = 0.1
    tail_wordfreq_start_turn: int = 3
    expanded_direct_hit_factor: float = 1.0
    expanded_language_override: bool = False
    expanded_language_min_probability: float = 0.2
    expanded_language_editorial_min: float = 0.15
    expanded_language_editorial_max: float = 0.2
    expanded_language_min_candidates: int = 3
    expanded_language_max_candidates: int = 20
    expanded_language_turn: int = 3

    def __post_init__(self) -> None:
        numeric_values = (
            self.direct_hit_weight,
            self.late_hit_weight,
            self.exploit_threshold,
            self.tail_wordfreq_weight,
            self.tail_wordfreq_gap,
            self.expanded_direct_hit_factor,
            self.expanded_language_min_probability,
            self.expanded_language_editorial_min,
            self.expanded_language_editorial_max,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Policy weights and thresholds must be finite.")
        if (
            self.direct_hit_weight < 0
            or self.late_hit_weight < 0
            or self.tail_wordfreq_weight < 0
            or self.tail_wordfreq_gap < 0
            or self.expanded_direct_hit_factor < 0
            or not 0 <= self.expanded_language_min_probability <= 1
            or not 0
            <= self.expanded_language_editorial_min
            <= self.expanded_language_editorial_max
            <= 1
            or not 0 <= self.exploit_threshold <= 1
        ):
            raise ValueError("Invalid policy weights or exploitation threshold.")
        if (
            self.exact_endgame_limit < 0
            or not 2 <= self.tail_wordfreq_start_turn <= 6
            or self.expanded_language_min_candidates < 2
            or self.expanded_language_max_candidates
            < self.expanded_language_min_candidates
            or not 2 <= self.expanded_language_turn <= 6
        ):
            raise ValueError(
                "Endgame, tail, and language-tail bounds must be valid values."
            )


def entropy_bits(patterns: np.ndarray, posterior: np.ndarray) -> float:
    mass = np.bincount(patterns, weights=posterior, minlength=243)
    nonzero = mass[mass > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def _expected_cost_for_action(
    table: np.ndarray,
    action: int,
    candidates: tuple[int, ...],
    weights: np.ndarray,
    value,
) -> float:
    candidate_array = np.asarray(candidates, dtype=np.int32)
    patterns = table[action, candidate_array]
    cost = 1.0
    for pattern in np.unique(patterns):
        group_mask = patterns == pattern
        mass = float(weights[group_mask].sum())
        if pattern != ALL_GREEN:
            group = tuple(candidate_array[group_mask].tolist())
            if group == candidates:
                return float("inf")
            cost += mass * value(group)
    return cost


class ExactEndgamePlanner:
    """Exact expected-cost search for states up to ``state_limit``."""

    def __init__(self, feedback_table: np.ndarray, prior: np.ndarray, state_limit: int):
        self.table = feedback_table
        self.prior = np.asarray(prior, dtype=float)
        self.state_limit = state_limit
        self._value = lru_cache(maxsize=None)(self._uncached_value)

    def _uncached_value(self, state: tuple[int, ...]) -> float:
        if len(state) == 1:
            return 1.0
        indices = np.asarray(state, dtype=np.int32)
        weights = self.prior[indices].astype(float)
        weights /= weights.sum()
        return min(
            _expected_cost_for_action(self.table, action, state, weights, self._value)
            for action in range(self.table.shape[0])
        )

    def best_action(self, candidates: np.ndarray) -> int | None:
        if self.state_limit < 2 or len(candidates) > self.state_limit:
            return None

        state = tuple(sorted(int(item) for item in candidates))
        indices = np.asarray(state, dtype=np.int32)
        weights = self.prior[indices].astype(float)
        weights /= weights.sum()
        costs = [
            _expected_cost_for_action(self.table, action, state, weights, self._value)
            for action in range(self.table.shape[0])
        ]
        return int(np.argmin(costs))


class EntropyPolicy:
    """Full-action Shannon entropy policy."""

    def __init__(
        self,
        words: tuple[str, ...],
        feedback_table: np.ndarray,
        config: PolicyConfig = PolicyConfig(),
        answer_indices: np.ndarray | None = None,
    ):
        self.words = words
        self.table = feedback_table
        self.config = config
        self.index = {word: i for i, word in enumerate(words)}
        if feedback_table.shape != (len(words), len(words)):
            raise ValueError(
                "Feedback table must be square over the action vocabulary."
            )
        selected = (
            default_answer_indices(words)
            if answer_indices is None
            else np.asarray(answer_indices, dtype=np.int32)
        )
        if (
            selected.ndim != 1
            or not len(selected)
            or np.any(selected < 0)
            or np.any(selected >= len(words))
        ):
            raise ValueError(
                "Answer indices must be a non-empty subset of the action vocabulary."
            )
        if len(np.unique(selected)) != len(selected):
            raise ValueError("Answer indices must be unique.")
        self.answer_indices = np.sort(selected)
        if config.starter not in self.index:
            raise ValueError(f"Starter {config.starter!r} is not an allowed word.")
        classic_words = set(load_allowed_words()[:2_315])
        self._expanded_mask = np.fromiter(
            (word not in classic_words for word in words), dtype=bool, count=len(words)
        )
        lexical = np.asarray(
            [zipf_frequency(word, "en") for word in words], dtype=float
        )
        lexical_std = float(lexical.std())
        self._tail_wordfreq_z = (
            (lexical - float(lexical.mean())) / lexical_std
            if lexical_std
            else np.zeros(len(words))
        )

    def _tail_adjusted_prior(
        self, candidates: np.ndarray, prior: np.ndarray, turn: int
    ) -> np.ndarray:
        """Apply lexical correction when the frequency gap clears its gate."""
        config = self.config
        if (
            config.tail_wordfreq_weight <= 0
            or turn < config.tail_wordfreq_start_turn
            or len(candidates) < 2
        ):
            return prior
        lexical = np.sort(self._tail_wordfreq_z[candidates])[::-1]
        if lexical[0] - lexical[1] < config.tail_wordfreq_gap:
            return prior
        adjusted = np.asarray(prior, dtype=float).copy()
        adjusted[candidates] *= np.exp(
            config.tail_wordfreq_weight * self._tail_wordfreq_z[candidates]
        )
        return adjusted / adjusted.sum()

    def _best_information_action(
        self,
        candidates: np.ndarray,
        posterior: np.ndarray,
        hit_weight: float,
        expanded_hit_factor: float = 1.0,
    ) -> int:
        candidate_probability = {
            int(index): float(probability)
            for index, probability in zip(candidates, posterior)
        }
        best_action = int(candidates[np.argmax(posterior)])
        best_score = -np.inf
        for action in range(len(self.words)):
            score = entropy_bits(self.table[action, candidates], posterior)
            hit_probability = candidate_probability.get(action, 0.0)
            if self._expanded_mask[action]:
                hit_probability *= expanded_hit_factor
            score += hit_weight * hit_probability
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _expanded_language_action(
        self,
        candidates: np.ndarray,
        prior: np.ndarray,
        auxiliary_prior: np.ndarray | None,
        turn: int,
    ) -> int | None:
        """Return the auxiliary action when every confidence gate passes."""
        config = self.config
        if (
            not config.expanded_language_override
            or auxiliary_prior is None
            or turn != config.expanded_language_turn
            or not config.expanded_language_min_candidates
            <= len(candidates)
            <= config.expanded_language_max_candidates
        ):
            return None
        auxiliary = np.asarray(auxiliary_prior, dtype=float)
        if (
            auxiliary.shape != prior.shape
            or not np.isfinite(auxiliary).all()
            or np.any(auxiliary < 0)
            or auxiliary.sum() <= 0
        ):
            raise ValueError(
                "Auxiliary linguistic prior must match the primary prior and be finite, non-negative, and positive."
            )
        editorial = prior[candidates].astype(float)
        editorial /= editorial.sum()
        linguistic = auxiliary[candidates].astype(float)
        linguistic /= linguistic.sum()
        top = int(np.argmax(linguistic))
        action = int(candidates[top])
        if not self._expanded_mask[action]:
            return None
        if linguistic[top] < config.expanded_language_min_probability:
            return None
        if (
            not config.expanded_language_editorial_min
            <= editorial[top]
            <= config.expanded_language_editorial_max
        ):
            return None
        return action

    def choose(
        self,
        candidates: np.ndarray,
        prior: np.ndarray,
        turn: int,
        exact_planner: ExactEndgamePlanner | None = None,
        auxiliary_prior: np.ndarray | None = None,
    ) -> int:
        if turn == 1:
            if self.config.adaptive_starter:
                candidates = self.answer_indices
                posterior = np.asarray(prior, dtype=float)[candidates]
                posterior /= posterior.sum()
                return self._best_information_action(
                    candidates, posterior, self.config.direct_hit_weight
                )
            return self.index[self.config.starter]
        adjusted_prior = self._tail_adjusted_prior(candidates, prior, turn)
        posterior = adjusted_prior[candidates].astype(float)
        posterior /= posterior.sum()
        order = np.argsort(posterior)[::-1]
        map_guess = int(candidates[order[0]])
        remaining_after_guess = 6 - turn
        if len(candidates) == 1:
            return map_guess
        planner = exact_planner
        if (
            adjusted_prior is not prior
            and exact_planner is not None
            and exact_planner.state_limit > 0
        ):
            planner = ExactEndgamePlanner(
                self.table, adjusted_prior, exact_planner.state_limit
            )
        exact = (
            planner
            or ExactEndgamePlanner(
                self.table, adjusted_prior, self.config.exact_endgame_limit
            )
        ).best_action(candidates)
        if exact is not None:
            return exact
        if len(candidates) <= remaining_after_guess + 1:
            return map_guess
        language_action = self._expanded_language_action(
            candidates, prior, auxiliary_prior, turn
        )
        if language_action is not None:
            return language_action
        if posterior[order[0]] >= self.config.exploit_threshold:
            return map_guess
        hit_weight = self.config.direct_hit_weight + self.config.late_hit_weight * max(
            turn - 2, 0
        )
        return self._best_information_action(
            candidates,
            posterior,
            hit_weight,
            self.config.expanded_direct_hit_factor,
        )
