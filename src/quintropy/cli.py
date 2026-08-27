"""Command-line entry points for reproducible Quintropy experiments."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

import numpy as np

from .data import (
    answer_indices,
    dependency_versions,
    implementation_sha256,
    load_allowed_words,
    load_history,
    sha256,
)
from .evaluation import evaluate_split, save_evaluation
from .feedback import build_feedback_table, feedback_cache_path
from .paths import RESULTS_DIR, ROOT
from .policy import EntropyPolicy, PolicyConfig
from .priors import (
    EditorialRegimeConfig,
    FrequencyPriorConfig,
    HybridPriorConfig,
    LinguisticPriorConfig,
    generate_priors,
    load_prior_artifact,
    save_prior_artifact,
)
from .selection import (
    select_editorial_config,
    select_hybrid_weight,
    select_linguistic_config,
)
from .web_export import export_web_snapshot


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Dates must use YYYY-MM-DD.") from exc


def add_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=parse_date, required=True)
    parser.add_argument("--end", type=parse_date, required=True)
    parser.add_argument("--seen-word-multiplier", type=float, default=1.0)


def add_prior_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prior-family",
        choices=("frequency", "editorial-regime", "linguistic", "hybrid"),
        default="frequency",
    )
    parser.add_argument("--frequency-temperature", type=float, default=4.0)
    parser.add_argument("--frequency-temperature-expanded", type=float, default=None)
    parser.add_argument("--regime-feature-weight", type=float, default=0.5)
    parser.add_argument("--regime-frequency-profile-weight", type=float, default=0.0)
    parser.add_argument("--positional-weight", type=float, default=1.0)
    parser.add_argument("--bigram-weight", type=float, default=0.0)
    parser.add_argument("--structural-weight", type=float, default=0.5)
    parser.add_argument("--recency-half-life-days", type=float, default=365.0)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--negative-ratio", type=int, default=8)
    parser.add_argument("--retrain-every-games", type=int, default=28)
    parser.add_argument("--linguistic-weight", type=float, default=0.10)


def prior_config_from_args(args: argparse.Namespace):
    if args.prior_family == "frequency":
        return FrequencyPriorConfig(seen_word_multiplier=args.seen_word_multiplier)
    if args.prior_family == "linguistic":
        return LinguisticPriorConfig(
            regularization=args.regularization,
            negative_ratio=args.negative_ratio,
            retrain_every_games=args.retrain_every_games,
        )
    editorial = EditorialRegimeConfig(
        frequency_temperature=args.frequency_temperature,
        frequency_temperature_expanded=args.frequency_temperature_expanded,
        regime_feature_weight=args.regime_feature_weight,
        regime_frequency_profile_weight=args.regime_frequency_profile_weight,
        positional_weight=args.positional_weight,
        bigram_weight=args.bigram_weight,
        structural_weight=args.structural_weight,
        recency_half_life_days=args.recency_half_life_days,
    )
    if args.prior_family == "hybrid":
        return HybridPriorConfig(
            editorial=editorial,
            linguistic=LinguisticPriorConfig(
                regularization=args.regularization,
                negative_ratio=args.negative_ratio,
                retrain_every_games=args.retrain_every_games,
            ),
            linguistic_weight=args.linguistic_weight,
        )
    return editorial


def command_generate_priors(args: argparse.Namespace) -> None:
    words = load_allowed_words()
    targets, priors, manifest = generate_priors(
        load_history(),
        words,
        args.start,
        args.end,
        prior_config_from_args(args),
    )
    save_prior_artifact(Path(args.output), words, targets, priors, manifest)
    print(f"Wrote {len(targets)} causal priors to {args.output}.")


def command_export_web(args: argparse.Namespace) -> None:
    export_web_snapshot(args.output, check=args.check)
    action = "Verified" if args.check else "Wrote"
    print(f"{action} browser model snapshot at {args.output}.")


def command_evaluate(args: argparse.Namespace) -> None:
    words = load_allowed_words()
    history = load_history()
    selection = None
    config = prior_config_from_args(args)
    if args.calibration_start is not None:
        if args.calibration_end is None or args.calibration_end >= args.start:
            raise ValueError(
                "The calibration window must end before the evaluation window starts."
            )
        if args.prior_family == "editorial-regime":
            config, leaderboard = select_editorial_config(
                history, words, args.calibration_start, args.calibration_end
            )
        elif args.prior_family == "linguistic":
            config, leaderboard = select_linguistic_config(
                history, words, args.calibration_start, args.calibration_end
            )
        elif args.prior_family == "hybrid":
            editorial, editorial_leaderboard = select_editorial_config(
                history, words, args.calibration_start, args.calibration_end
            )
            linguistic, linguistic_leaderboard = select_linguistic_config(
                history, words, args.calibration_start, args.calibration_end
            )
            config, leaderboard = select_hybrid_weight(
                history,
                words,
                args.calibration_start,
                args.calibration_end,
                editorial,
                linguistic,
            )
            leaderboard = {
                "editorial": editorial_leaderboard,
                "linguistic": linguistic_leaderboard,
                "hybrid_weight": leaderboard,
            }
        else:
            raise ValueError(
                "Automatic selection is available only for editorial-regime or linguistic priors."
            )
        selection = {
            "window": {
                "start": args.calibration_start.isoformat(),
                "end": args.calibration_end.isoformat(),
            },
            "criterion": "mean_log_loss_bits",
            "leaderboard": leaderboard,
        }
    targets, priors, prior_manifest = generate_priors(
        history, words, args.start, args.end, config
    )
    auxiliary_priors = None
    auxiliary_manifest = None
    if args.expanded_language_override:
        auxiliary_targets, auxiliary_priors, auxiliary_manifest_obj = generate_priors(
            history,
            words,
            args.start,
            args.end,
            LinguisticPriorConfig(
                regularization=args.regularization,
                negative_ratio=args.negative_ratio,
                retrain_every_games=args.retrain_every_games,
            ),
        )
        if not auxiliary_targets["game"].equals(targets["game"]):
            raise ValueError(
                "Auxiliary linguistic targets do not align with the primary prior targets."
            )
        auxiliary_manifest = asdict(auxiliary_manifest_obj)
    if args.max_games is not None:
        if args.max_games < 1:
            raise ValueError("--max-games must be a positive integer.")
        targets, priors = (
            targets.iloc[: args.max_games].copy(),
            priors[: args.max_games],
        )
        prior_manifest = replace(
            prior_manifest,
            generated_for_dates=[
                row.date.isoformat() for row in targets.itertuples(index=False)
            ],
        )
        if auxiliary_priors is not None:
            auxiliary_priors = auxiliary_priors[: args.max_games]
            auxiliary_manifest_obj = replace(
                auxiliary_manifest_obj,
                generated_for_dates=[
                    row.date.isoformat() for row in targets.itertuples(index=False)
                ],
            )
            auxiliary_manifest = asdict(auxiliary_manifest_obj)
    cache = feedback_cache_path(ROOT / ".quintropy-cache", words)
    table = build_feedback_table(words, cache)
    policy_config = PolicyConfig(
        starter=args.starter,
        adaptive_starter=args.adaptive_starter,
        direct_hit_weight=args.direct_hit_weight,
        late_hit_weight=args.late_hit_weight,
        exploit_threshold=args.exploit_threshold,
        exact_endgame_limit=args.exact_endgame_limit,
        tail_wordfreq_weight=args.tail_wordfreq_weight,
        tail_wordfreq_gap=args.tail_wordfreq_gap,
        tail_wordfreq_start_turn=args.tail_wordfreq_start_turn,
        expanded_direct_hit_factor=args.expanded_direct_hit_factor,
        expanded_language_override=args.expanded_language_override,
        expanded_language_min_probability=args.expanded_language_min_probability,
        expanded_language_editorial_min=args.expanded_language_editorial_min,
        expanded_language_editorial_max=args.expanded_language_editorial_max,
        expanded_language_min_candidates=args.expanded_language_min_candidates,
        expanded_language_max_candidates=args.expanded_language_max_candidates,
        expanded_language_turn=args.expanded_language_turn,
    )
    policy = EntropyPolicy(words, table, policy_config, answer_indices(words))
    results, report = evaluate_split(
        words, table, policy, targets, priors, auxiliary_priors
    )
    output = Path(args.output)
    prior_artifact = output / "priors.npz"
    save_prior_artifact(prior_artifact, words, targets, priors, prior_manifest)
    auxiliary_artifact = None
    if auxiliary_priors is not None:
        auxiliary_artifact = output / "linguistic_priors.npz"
        save_prior_artifact(
            auxiliary_artifact, words, targets, auxiliary_priors, auxiliary_manifest_obj
        )
    manifest = {
        "schema_version": 2,
        "evaluation": {
            "start": targets["date"].iloc[0].date().isoformat(),
            "end": targets["date"].iloc[-1].date().isoformat(),
            "n_games": len(targets),
            "source_prior_artifact": prior_artifact.name,
            "source_prior_sha256": sha256(prior_artifact),
        },
        "prior": asdict(prior_manifest),
        "auxiliary_prior": auxiliary_manifest,
        "auxiliary_prior_artifact": None
        if auxiliary_artifact is None
        else auxiliary_artifact.name,
        "auxiliary_prior_sha256": None
        if auxiliary_artifact is None
        else sha256(auxiliary_artifact),
        "selection": selection,
        "policy": asdict(policy_config),
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
        "implementation_sha256": implementation_sha256(),
        "warning": "This command is causal but does not make a prospective claim; do not tune on this reported window.",
    }
    save_evaluation(output, results, report, manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote inspectable evaluation artefacts to {output}.")


def command_evaluate_priors(args: argparse.Namespace) -> None:
    source_prior_path = Path(args.prior_artifact)
    words, prior_answer_indices, targets, priors, prior_manifest = load_prior_artifact(
        source_prior_path
    )
    auxiliary_priors = None
    auxiliary_manifest = None
    if args.auxiliary_prior_artifact is not None:
        (
            auxiliary_words,
            auxiliary_answer_indices,
            auxiliary_targets,
            auxiliary_priors,
            auxiliary_manifest,
        ) = load_prior_artifact(Path(args.auxiliary_prior_artifact))
        if (
            auxiliary_words != words
            or not np.array_equal(auxiliary_answer_indices, prior_answer_indices)
            or not auxiliary_targets.equals(targets)
        ):
            raise ValueError(
                "Auxiliary prior artifact does not align with the primary artifact."
            )
    cache = feedback_cache_path(ROOT / ".quintropy-cache", words)
    table = build_feedback_table(words, cache)
    policy_config = PolicyConfig(
        starter=args.starter,
        adaptive_starter=args.adaptive_starter,
        direct_hit_weight=args.direct_hit_weight,
        late_hit_weight=args.late_hit_weight,
        exploit_threshold=args.exploit_threshold,
        exact_endgame_limit=args.exact_endgame_limit,
        tail_wordfreq_weight=args.tail_wordfreq_weight,
        tail_wordfreq_gap=args.tail_wordfreq_gap,
        tail_wordfreq_start_turn=args.tail_wordfreq_start_turn,
        expanded_direct_hit_factor=args.expanded_direct_hit_factor,
        expanded_language_override=args.expanded_language_override,
        expanded_language_min_probability=args.expanded_language_min_probability,
        expanded_language_editorial_min=args.expanded_language_editorial_min,
        expanded_language_editorial_max=args.expanded_language_editorial_max,
        expanded_language_min_candidates=args.expanded_language_min_candidates,
        expanded_language_max_candidates=args.expanded_language_max_candidates,
        expanded_language_turn=args.expanded_language_turn,
    )
    results, report = evaluate_split(
        words,
        table,
        EntropyPolicy(words, table, policy_config, prior_answer_indices),
        targets,
        priors,
        auxiliary_priors,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bundled_prior_path = output / "priors.npz"
    if source_prior_path.resolve() != bundled_prior_path.resolve():
        shutil.copy2(source_prior_path, bundled_prior_path)
    bundled_auxiliary_path = None
    if args.auxiliary_prior_artifact is not None:
        source_auxiliary_path = Path(args.auxiliary_prior_artifact)
        bundled_auxiliary_path = output / "linguistic_priors.npz"
        if source_auxiliary_path.resolve() != bundled_auxiliary_path.resolve():
            shutil.copy2(source_auxiliary_path, bundled_auxiliary_path)
    manifest = {
        "schema_version": 2,
        "evaluation": {
            "start": targets["date"].iloc[0].date().isoformat(),
            "end": targets["date"].iloc[-1].date().isoformat(),
            "source_prior_artifact": bundled_prior_path.name,
            "source_prior_sha256": sha256(bundled_prior_path),
            "n_games": len(targets),
        },
        "prior": prior_manifest,
        "auxiliary_prior": auxiliary_manifest,
        "auxiliary_prior_artifact": None
        if bundled_auxiliary_path is None
        else bundled_auxiliary_path.name,
        "auxiliary_prior_sha256": None
        if bundled_auxiliary_path is None
        else sha256(bundled_auxiliary_path),
        "policy": asdict(policy_config),
        "python": platform.python_version(),
        "dependencies": dependency_versions(),
        "implementation_sha256": implementation_sha256(),
        "warning": "This reruns a saved prior artifact; its causal status is determined by that artifact's manifest.",
    }
    save_evaluation(output, results, report, manifest)
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quintropy",
        description="Reproducible five-letter puzzle research commands",
    )
    subparsers = parser.add_subparsers(dest="command")
    priors = subparsers.add_parser(
        "generate-priors", help="create a causal prior artifact"
    )
    add_window_arguments(priors)
    add_prior_arguments(priors)
    priors.add_argument("--output", type=Path, required=True)
    priors.set_defaults(handler=command_generate_priors)
    evaluate = subparsers.add_parser(
        "evaluate", help="run a causal, end-to-end evaluation"
    )
    add_window_arguments(evaluate)
    add_prior_arguments(evaluate)
    evaluate.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "experiments" / "local"
    )
    evaluate.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="smoke-test limit; omit for the full requested window",
    )
    evaluate.add_argument("--starter", default="soare")
    evaluate.add_argument("--adaptive-starter", action="store_true")
    evaluate.add_argument("--direct-hit-weight", type=float, default=3.0)
    evaluate.add_argument("--late-hit-weight", type=float, default=0.0)
    evaluate.add_argument("--exploit-threshold", type=float, default=0.5)
    evaluate.add_argument("--exact-endgame-limit", type=int, default=3)
    evaluate.add_argument("--tail-wordfreq-weight", type=float, default=1.0)
    evaluate.add_argument("--tail-wordfreq-gap", type=float, default=0.1)
    evaluate.add_argument("--tail-wordfreq-start-turn", type=int, default=3)
    evaluate.add_argument("--expanded-direct-hit-factor", type=float, default=1.5)
    evaluate.add_argument("--expanded-language-override", action="store_true")
    evaluate.add_argument(
        "--expanded-language-min-probability", type=float, default=0.2
    )
    evaluate.add_argument("--expanded-language-editorial-min", type=float, default=0.15)
    evaluate.add_argument("--expanded-language-editorial-max", type=float, default=0.2)
    evaluate.add_argument("--expanded-language-min-candidates", type=int, default=3)
    evaluate.add_argument("--expanded-language-max-candidates", type=int, default=20)
    evaluate.add_argument("--expanded-language-turn", type=int, default=3)
    evaluate.add_argument("--calibration-start", type=parse_date)
    evaluate.add_argument("--calibration-end", type=parse_date)
    evaluate.set_defaults(handler=command_evaluate)
    replay = subparsers.add_parser(
        "evaluate-priors", help="evaluate a saved modern prior artifact"
    )
    replay.add_argument("--prior-artifact", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--starter", default="soare")
    replay.add_argument("--adaptive-starter", action="store_true")
    replay.add_argument("--direct-hit-weight", type=float, default=3.0)
    replay.add_argument("--late-hit-weight", type=float, default=0.0)
    replay.add_argument("--exploit-threshold", type=float, default=0.5)
    replay.add_argument("--exact-endgame-limit", type=int, default=3)
    replay.add_argument("--tail-wordfreq-weight", type=float, default=1.0)
    replay.add_argument("--tail-wordfreq-gap", type=float, default=0.1)
    replay.add_argument("--tail-wordfreq-start-turn", type=int, default=3)
    replay.add_argument("--expanded-direct-hit-factor", type=float, default=1.5)
    replay.add_argument("--auxiliary-prior-artifact", type=Path)
    replay.add_argument("--expanded-language-override", action="store_true")
    replay.add_argument("--expanded-language-min-probability", type=float, default=0.2)
    replay.add_argument("--expanded-language-editorial-min", type=float, default=0.15)
    replay.add_argument("--expanded-language-editorial-max", type=float, default=0.2)
    replay.add_argument("--expanded-language-min-candidates", type=int, default=3)
    replay.add_argument("--expanded-language-max-candidates", type=int, default=20)
    replay.add_argument("--expanded-language-turn", type=int, default=3)
    replay.set_defaults(handler=command_evaluate_priors)
    web = subparsers.add_parser(
        "export-web", help="export the selected model for the static browser solver"
    )
    web.add_argument("--output", type=Path, default=ROOT / "web" / "model.json")
    web.add_argument(
        "--check",
        action="store_true",
        help="fail when the existing snapshot differs from a fresh export",
    )
    web.set_defaults(handler=command_export_web)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return
    args.handler(args)


if __name__ == "__main__":
    main()
