"""Independent validation of modern Quintropy experiment artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import data_provenance, implementation_sha256, sha256
from .evaluation import bootstrap_mean_interval, evaluate_split, wilson_interval
from .feedback import build_feedback_table, feedback_cache_path
from .paths import ROOT
from .policy import EntropyPolicy, PolicyConfig
from .priors import load_prior_artifact


@dataclass(frozen=True)
class AuditResult:
    games: int
    schema_version: int
    implementation_matches: bool
    causal_status: str


def _resolve_artifact(experiment_dir: Path, recorded_path: str) -> Path:
    root = experiment_dir.resolve()
    normalized_record = Path(recorded_path.replace("\\", "/"))
    if ".." in normalized_record.parts:
        raise ValueError(
            "Prior artifact path may not traverse outside the experiment directory."
        )
    direct = (root / recorded_path).resolve()
    if direct.is_relative_to(root) and direct.is_file():
        return direct
    by_name = root / normalized_record.name
    if by_name.is_file():
        return by_name
    raise ValueError(f"Referenced prior artifact does not exist: {recorded_path}")


def _assert_close(actual: Any, expected: Any, path: str = "metrics") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path} fields do not match recomputed metrics.")
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} does not match recomputed metrics.")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_close(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, (float, np.floating)):
        if not np.isclose(
            float(actual), float(expected), rtol=1e-10, atol=1e-12, equal_nan=False
        ):
            raise ValueError(f"{path} differs from the recomputed value.")
        return
    if actual != expected:
        raise ValueError(f"{path} differs from the recomputed value.")


def _schema_one_report(
    report: dict[str, object], tries: np.ndarray, solved: np.ndarray
) -> dict[str, object]:
    schema_v1_report = dict(report)
    schema_v1_report.pop("mean_tries_moving_block_bootstrap_95")
    schema_v1_report.pop("accuracy_at_6_moving_block_bootstrap_95")
    schema_v1_report.pop("accuracy_at_6_wilson_95_iid_sensitivity")
    schema_v1_report.pop("uncertainty")
    schema_v1_report["top_k_coverage"].pop("243")
    schema_v1_report["mean_tries_bootstrap_95"] = list(bootstrap_mean_interval(tries))
    schema_v1_report["accuracy_at_6_wilson_95"] = list(
        wilson_interval(int(solved.sum()), len(solved))
    )
    return schema_v1_report


def _validate_manifest_dates(
    manifest: dict[str, object], targets: pd.DataFrame
) -> None:
    generated = manifest.get("generated_for_dates")
    expected = targets["date"].dt.strftime("%Y-%m-%d").tolist()
    if (
        not isinstance(generated, list)
        or [pd.Timestamp(value).strftime("%Y-%m-%d") for value in generated] != expected
    ):
        raise ValueError("Prior manifest dates do not align with its targets.")
    if manifest.get("information_cutoff") != "strictly before each target date":
        raise ValueError("Prior manifest does not declare the required causal cutoff.")
    recorded_provenance = manifest.get("provenance")
    local_provenance = data_provenance().__dict__
    if not isinstance(recorded_provenance, dict) or any(
        local_provenance.get(name) != digest
        for name, digest in recorded_provenance.items()
    ):
        raise ValueError("Prior input hashes do not match the local data sources.")


def audit_experiment(
    experiment_dir: Path, *, require_current_implementation: bool = False
) -> AuditResult:
    """Replay an experiment and reject any inconsistent result or provenance."""
    experiment_dir = Path(experiment_dir)
    games = pd.read_csv(experiment_dir / "games.csv")
    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (experiment_dir / "manifest.json").read_text(encoding="utf-8")
    )
    schema = int(manifest.get("schema_version", 1))
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, dict) or "source_prior_artifact" not in evaluation:
        raise ValueError(
            "Experiment manifest does not reference a source prior artifact."
        )
    prior_path = _resolve_artifact(
        experiment_dir, str(evaluation["source_prior_artifact"])
    )
    words, answer_indices, targets, priors, prior_manifest = load_prior_artifact(
        prior_path
    )
    _validate_manifest_dates(prior_manifest, targets)

    required = {
        "game",
        "date",
        "answer",
        "tries",
        "solved",
        "path",
        "true_probability",
        "true_rank",
    }
    if set(games) != required:
        raise ValueError("games.csv columns do not exactly match the audit schema.")
    expected_targets = pd.DataFrame(
        {
            "game": targets["game"].astype(int),
            "date": targets["date"].dt.strftime("%Y-%m-%d"),
            "answer": targets["answer"].astype(str),
        }
    )
    if (
        not games[["game", "date", "answer"]]
        .reset_index(drop=True)
        .equals(expected_targets.reset_index(drop=True))
    ):
        raise ValueError("games.csv targets do not match the source prior artifact.")
    if len(games) != metrics.get("n_games") or len(games) != evaluation.get("n_games"):
        raise ValueError("Game counts disagree across experiment artifacts.")
    if (
        "start" in evaluation
        and evaluation["start"] != expected_targets["date"].iloc[0]
    ):
        raise ValueError("Evaluation start does not match the source targets.")
    if "end" in evaluation and evaluation["end"] != expected_targets["date"].iloc[-1]:
        raise ValueError("Evaluation end does not match the source targets.")

    auxiliary_priors = None
    auxiliary_manifest = manifest.get("auxiliary_prior")
    auxiliary_name = manifest.get("auxiliary_prior_artifact")
    if auxiliary_name is not None:
        auxiliary_path = _resolve_artifact(experiment_dir, str(auxiliary_name))
        (
            auxiliary_words,
            auxiliary_indices,
            auxiliary_targets,
            auxiliary_priors,
            loaded_auxiliary_manifest,
        ) = load_prior_artifact(auxiliary_path)
        if (
            auxiliary_words != words
            or not np.array_equal(auxiliary_indices, answer_indices)
            or not auxiliary_targets.equals(targets)
        ):
            raise ValueError(
                "Auxiliary prior artifact does not align with the primary artifact."
            )
        if auxiliary_manifest != loaded_auxiliary_manifest:
            raise ValueError("Auxiliary prior manifest does not match its artifact.")
        _validate_manifest_dates(loaded_auxiliary_manifest, auxiliary_targets)
        if schema >= 2 and manifest.get("auxiliary_prior_sha256") != sha256(
            auxiliary_path
        ):
            raise ValueError(
                "Auxiliary prior artifact hash does not match the evaluation manifest."
            )
    elif auxiliary_manifest is not None:
        raise ValueError(
            "Auxiliary manifest exists without an auxiliary prior artifact."
        )

    if prior_manifest != manifest.get("prior"):
        raise ValueError("Primary prior manifest does not match its artifact.")
    if schema >= 2:
        if evaluation.get("source_prior_sha256") != sha256(prior_path):
            raise ValueError(
                "Primary prior artifact hash does not match the evaluation manifest."
            )
        required_prior_fields = {
            "answer_universe",
            "generator_implementation_sha256",
            "dependencies",
        }
        if not required_prior_fields <= set(prior_manifest):
            raise ValueError(
                "Schema-v2 prior lacks generator or answer-universe provenance."
            )

    table = build_feedback_table(
        words, feedback_cache_path(ROOT / ".quintropy-cache", words)
    )
    policy = EntropyPolicy(
        words, table, PolicyConfig(**manifest["policy"]), answer_indices
    )
    replayed, recomputed = evaluate_split(
        words, table, policy, targets, priors, auxiliary_priors
    )
    expected_games = pd.DataFrame(
        [
            {
                "game": result.game,
                "date": result.date,
                "answer": result.answer,
                "tries": result.tries,
                "solved": result.solved,
                "path": " ".join(result.path),
                "true_probability": result.true_probability,
                "true_rank": result.true_rank,
            }
            for result in replayed
        ]
    )
    for column in ("game", "date", "answer", "tries", "solved", "path", "true_rank"):
        if not games[column].equals(expected_games[column]):
            raise ValueError(f"games.csv column {column!r} differs from policy replay.")
    if not np.allclose(
        games["true_probability"],
        expected_games["true_probability"],
        rtol=1e-10,
        atol=1e-15,
    ):
        raise ValueError(
            "Reported true-answer probabilities differ from the source priors."
        )

    if schema == 1:
        tries = np.asarray([result.tries for result in replayed], dtype=float)
        solved = np.asarray([result.solved for result in replayed], dtype=bool)
        recomputed = _schema_one_report(recomputed, tries, solved)
    _assert_close(metrics, recomputed)

    recorded_implementation = manifest.get("implementation_sha256")
    if (
        not isinstance(recorded_implementation, str)
        or len(recorded_implementation) != 64
    ):
        raise ValueError("Experiment lacks a valid evaluator implementation hash.")
    matches = recorded_implementation == implementation_sha256()
    if require_current_implementation and not matches:
        raise ValueError(
            "Experiment was produced by different package sources than this checkout."
        )
    causal_status = "generator-bound" if schema >= 2 else "declared-only"
    return AuditResult(len(games), schema, matches, causal_status)
