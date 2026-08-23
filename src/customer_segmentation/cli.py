"""Command-line interface for the customer segmentation benchmark."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from customer_segmentation.config import PROJECT_ROOT
from customer_segmentation.external import (
    fit_external_segmentation,
    run_external_segmentation,
    score_external_segmentation,
)
from customer_segmentation.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reproduce = subparsers.add_parser("reproduce", help="regenerate every artifact")
    reproduce.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    subparsers.add_parser("smoke", help="run the full pipeline in a temporary directory")
    segment = subparsers.add_parser("segment", help="fit and assign an external feature CSV")
    segment.add_argument("--input", type=Path, required=True)
    segment.add_argument("--output-dir", type=Path, required=True)
    segment.add_argument("--seed", type=int, default=42)
    fit = subparsers.add_parser("fit", help="fit and persist a governed segment definition")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--model-dir", type=Path, required=True)
    fit.add_argument("--output-dir", type=Path)
    score = subparsers.add_parser("score", help="score a snapshot with a frozen definition")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--model-dir", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--previous-assignments", type=Path)
    args = parser.parse_args()
    if args.command == "smoke":
        with tempfile.TemporaryDirectory(prefix="customer-segmentation-") as directory:
            metrics = run_pipeline(Path(directory))
    elif args.command == "segment":
        metrics = run_external_segmentation(args.input, args.output_dir, args.seed)
    elif args.command == "fit":
        metrics = fit_external_segmentation(args.input, args.model_dir, args.output_dir)
    elif args.command == "score":
        metrics = score_external_segmentation(
            args.input,
            args.model_dir,
            args.output_dir,
            args.previous_assignments,
        )
    else:
        metrics = run_pipeline(args.output_root)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
