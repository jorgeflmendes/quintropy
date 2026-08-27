"""Export the selected causal model as a deterministic browser snapshot."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import numpy as np
from wordfreq import zipf_frequency

from .data import (
    answer_indices,
    answer_universe_metadata,
    data_provenance,
    implementation_sha256,
    load_allowed_words,
    load_history,
)
from .priors import CausalLinguisticPrior, EditorialRegimePrior
from .selected import (
    SELECTED_EDITORIAL_CONFIG,
    SELECTED_LINGUISTIC_CONFIG,
    SELECTED_POLICY_CONFIG,
)

WEB_MODEL_SCHEMA_VERSION = 1


def _normalized(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a non-empty vector.")
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
        raise ValueError(f"{name} must be finite, non-negative, and positive.")
    return values / values.sum()


def build_web_snapshot() -> dict[str, object]:
    """Train on the complete frozen history and return the next-day snapshot."""
    words = load_allowed_words()
    history = load_history()
    prediction_date = history["date"].max() + timedelta(days=1)

    primary = EditorialRegimePrior(words, SELECTED_EDITORIAL_CONFIG).predict(
        history, prediction_date
    )
    auxiliary = CausalLinguisticPrior(words, SELECTED_LINGUISTIC_CONFIG).predict(
        history, prediction_date
    )
    answers = answer_indices(words)
    primary_answers = _normalized(primary[answers], "Primary answer prior")
    auxiliary_answers = _normalized(auxiliary[answers], "Auxiliary answer prior")

    lexical = np.asarray([zipf_frequency(word, "en") for word in words], dtype=float)
    lexical_std = float(lexical.std())
    lexical_z = (
        (lexical - float(lexical.mean())) / lexical_std
        if lexical_std
        else np.zeros(len(words), dtype=float)
    )

    return {
        "schemaVersion": WEB_MODEL_SCHEMA_VERSION,
        "model": "quintropy-selected-causal-model",
        "trainedThrough": history["date"].max().date().isoformat(),
        "predictionDate": prediction_date.date().isoformat(),
        "actionWords": list(words),
        "answerActionIndices": answers.tolist(),
        "primaryPrior": primary_answers.tolist(),
        "auxiliaryPrior": auxiliary_answers.tolist(),
        "answerLexicalZ": lexical_z[answers].tolist(),
        "classicSolutionCount": 2_315,
        "allGreenCode": 242,
        "policy": asdict(SELECTED_POLICY_CONFIG),
        "training": {
            "primary": asdict(SELECTED_EDITORIAL_CONFIG),
            "auxiliary": asdict(SELECTED_LINGUISTIC_CONFIG),
            "informationCutoff": "strictly before the prediction date",
        },
        "provenance": {
            **asdict(data_provenance()),
            "answerUniverse": answer_universe_metadata(words),
            "implementationSha256": implementation_sha256(),
        },
    }


def render_web_snapshot() -> bytes:
    """Serialize the browser snapshot without timestamps or platform-specific data."""
    payload = json.dumps(
        build_web_snapshot(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (payload + "\n").encode("utf-8")


def export_web_snapshot(output: Path, *, check: bool = False) -> None:
    """Write the snapshot, or verify that an existing snapshot is current."""
    output = Path(output)
    rendered = render_web_snapshot()
    if check:
        if not output.is_file() or output.read_bytes() != rendered:
            raise ValueError(
                f"Browser model snapshot is stale; regenerate {output.as_posix()}."
            )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rendered)
