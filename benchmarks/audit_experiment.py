"""Validate a modern Quintropy evaluation artifact and its input provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

from quintropy.audit import audit_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        type=Path,
        help="directory containing games.csv, metrics.json, and manifest.json",
    )
    parser.add_argument(
        "--require-current-implementation",
        action="store_true",
        help="fail when the artifact was not produced by this checkout",
    )
    args = parser.parse_args()
    try:
        result = audit_experiment(
            args.experiment,
            require_current_implementation=args.require_current_implementation,
        )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise SystemExit(f"Experiment audit failed: {exc}") from exc
    relation = (
        "matches current checkout"
        if result.implementation_matches
        else "replayed by current checkout"
    )
    causal = (
        "generator-bound causal provenance"
        if result.causal_status == "generator-bound"
        else "cutoff declared only"
    )
    print(
        f"Experiment audited: {result.games} games fully replayed; metrics and inputs verified; {relation}; {causal}."
    )


if __name__ == "__main__":
    main()
