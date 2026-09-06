"""Command-line interface for the customer segmentation benchmark."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from customer_segmentation.external import (
    fit_external_segmentation,
    run_external_segmentation,
    score_external_segmentation,
)
from customer_segmentation.pipeline import run_pipeline


def _add_privacy_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-raw-identifiers",
        action="store_true",
        help="export raw customer IDs only when explicitly approved for that use",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce = subparsers.add_parser("reproduce", help="regenerate every artifact")
    reproduce.add_argument("--output-root", type=Path, default=Path.cwd())
    reproduce.add_argument("--config", type=Path)
    smoke = subparsers.add_parser("smoke", help="run the full pipeline in a temporary directory")
    smoke.add_argument("--config", type=Path)
    segment = subparsers.add_parser("segment", help="fit and assign an external feature CSV")
    segment.add_argument("--input", type=Path, required=True)
    segment.add_argument("--output-dir", type=Path, required=True)
    segment.add_argument("--seed", type=int, default=42)
    segment.add_argument("--config", type=Path)
    _add_privacy_arguments(segment)
    fit = subparsers.add_parser("fit", help="fit and persist a governed segment definition")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--model-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path)
    fit.add_argument("--config", type=Path)
    _add_privacy_arguments(fit)
    score = subparsers.add_parser("score", help="score a snapshot with a frozen definition")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--model-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--previous-assignments", type=Path)
    score.add_argument("--config", type=Path)
    score.add_argument(
        "--fail-on-review",
        action="store_true",
        help="exit with status 2 when monitoring requires review",
    )
    _add_privacy_arguments(score)
    args = parser.parse_args()
    policy = "raw" if getattr(args, "allow_raw_identifiers", False) else "pseudonymized"
    salt = os.environ.get("CUSTOMER_SEGMENTATION_ID_SALT")
    if args.command == "smoke":
        with tempfile.TemporaryDirectory(prefix="customer-segmentation-") as directory:
            metrics = run_pipeline(Path(directory), args.config)
    elif args.command == "segment":
        metrics = run_external_segmentation(
            args.input,
            args.output_dir,
            args.seed,
            args.config,
            identifier_policy=policy,
            identifier_salt=salt,
        )
    elif args.command == "fit":
        metrics = fit_external_segmentation(
            args.input,
            args.model_dir,
            args.output_dir,
            args.config,
            identifier_policy=policy,
            identifier_salt=salt,
        )
    elif args.command == "score":
        metrics = score_external_segmentation(
            args.input,
            args.model_dir,
            args.output_dir,
            args.previous_assignments,
            args.config,
            identifier_policy=policy,
            identifier_salt=salt,
        )
    else:
        metrics = run_pipeline(args.output_root, args.config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.command == "score" and args.fail_on_review:
        if not metrics["monitoring_gate_passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
