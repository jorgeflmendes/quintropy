import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from quintropy.audit import audit_experiment
from quintropy.data import dependency_versions, implementation_sha256, sha256
from quintropy.evaluation import evaluate_split, save_evaluation
from quintropy.feedback import build_feedback_table
from quintropy.policy import EntropyPolicy, PolicyConfig
from quintropy.priors import FrequencyPriorConfig, generate_priors, save_prior_artifact


def create_experiment(root):
    words = ("cigar", "rebut", "sissy")
    history = pd.DataFrame(
        {
            "game": [0, 1],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "answer": ["cigar", "rebut"],
        }
    )
    targets, priors, prior_manifest = generate_priors(
        history,
        words,
        history.date.iloc[0].date(),
        history.date.iloc[-1].date(),
        FrequencyPriorConfig(),
    )
    prior_path = root / "priors.npz"
    save_prior_artifact(prior_path, words, targets, priors, prior_manifest)
    policy_config = PolicyConfig(starter="cigar", exploit_threshold=0.0)
    policy = EntropyPolicy(
        words, build_feedback_table(words), policy_config, np.arange(3, dtype=np.int32)
    )
    results, metrics = evaluate_split(words, policy.table, policy, targets, priors)
    manifest = {
        "schema_version": 2,
        "evaluation": {
            "start": "2026-01-01",
            "end": "2026-01-02",
            "n_games": 2,
            "source_prior_artifact": "priors.npz",
            "source_prior_sha256": sha256(prior_path),
        },
        "prior": asdict(prior_manifest),
        "auxiliary_prior": None,
        "auxiliary_prior_artifact": None,
        "auxiliary_prior_sha256": None,
        "policy": asdict(policy_config),
        "python": "test",
        "dependencies": dependency_versions(),
        "implementation_sha256": implementation_sha256(),
    }
    save_evaluation(root, results, metrics, manifest)


def test_audit_replays_complete_experiment(tmp_path):
    create_experiment(tmp_path)
    result = audit_experiment(tmp_path, require_current_implementation=True)
    assert result.games == 2
    assert result.causal_status == "generator-bound"


def test_audit_rejects_tampered_metrics(tmp_path):
    create_experiment(tmp_path)
    path = tmp_path / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["mean_tries"] = 1.0
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="mean_tries"):
        audit_experiment(tmp_path)


def test_audit_rejects_tampered_paths(tmp_path):
    create_experiment(tmp_path)
    path = tmp_path / "games.csv"
    games = pd.read_csv(path)
    games.loc[1, "path"] = "cigar sissy"
    games.to_csv(path, index=False)
    with pytest.raises(ValueError, match="path"):
        audit_experiment(tmp_path)
