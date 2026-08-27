import json

import numpy as np
import pytest

from quintropy.data import implementation_sha256
from quintropy.paths import ROOT
from quintropy.web_export import _normalized, export_web_snapshot


def test_normalized_rejects_invalid_probability_vectors():
    with pytest.raises(ValueError, match="non-empty"):
        _normalized(np.empty(0), "Prior")
    with pytest.raises(ValueError, match="finite"):
        _normalized(np.array([1.0, np.nan]), "Prior")


def test_export_check_detects_a_stale_snapshot(tmp_path, monkeypatch):
    output = tmp_path / "model.json"
    monkeypatch.setattr(
        "quintropy.web_export.build_web_snapshot",
        lambda: {"schemaVersion": 1, "model": "test"},
    )

    export_web_snapshot(output)
    assert json.loads(output.read_text(encoding="utf-8"))["model"] == "test"
    export_web_snapshot(output, check=True)

    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        export_web_snapshot(output, check=True)


def test_published_snapshot_is_bound_to_current_model_sources():
    snapshot = json.loads((ROOT / "web" / "model.json").read_text(encoding="utf-8"))
    assert snapshot["schemaVersion"] == 1
    assert len(snapshot["actionWords"]) == 12_972
    assert len(snapshot["answerActionIndices"]) == 8_926
    assert sum(snapshot["primaryPrior"]) == pytest.approx(1.0)
    assert snapshot["provenance"]["implementationSha256"] == implementation_sha256()
