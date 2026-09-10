#!/usr/bin/env python3
"""Generate IEEE LaTeX and Markdown comparison table: m4_roberta_focal vs m6_deberta_v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from src.paths import EXPORTS_DIR, ensure_artifact_dirs
from src.reporting.comparison_table import (
    _load_m4_rows,
    _load_m6_rows,
    format_latex,
    format_markdown,
    load_experiment_metrics,
    write_comparison_tables,
)


def _load_json(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate IEEE comparison table (RoBERTa-Focal vs DeBERTa-v3)"
    )
    parser.add_argument(
        "--m4-json",
        type=Path,
        default=None,
        help="Path to m4/E3 experiment JSON (default: artifacts/exports/experiment_E3.json)",
    )
    parser.add_argument(
        "--m6-json",
        type=Path,
        default=None,
        help="Path to m6/E2 JSON (default: experiment_E2_thresholds.json or experiment_E2.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPORTS_DIR,
        help="Directory for model_comparison_m4_m6.{tex,md}",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Re-run evaluation on unified multi-label test set (slow)",
    )
    parser.add_argument(
        "--skip-m4",
        action="store_true",
        help="Omit m4 rows when experiment JSON/checkpoint unavailable",
    )
    parser.add_argument(
        "--skip-m6",
        action="store_true",
        help="Omit m6 rows when experiment JSON/checkpoint unavailable",
    )
    parser.add_argument(
        "--stem",
        default="model_comparison_m4_m6",
        help="Output filename stem",
    )
    args = parser.parse_args()

    ensure_artifact_dirs()

    m6_payload = _load_json(args.m6_json)
    if m6_payload is None:
        m6_payload = load_experiment_metrics(EXPORTS_DIR, "E2")

    m4_payload = _load_json(args.m4_json)
    if m4_payload is None:
        m4_payload = load_experiment_metrics(EXPORTS_DIR, "E3")

    rows: list = []
    if not args.skip_m4:
        try:
            rows.extend(_load_m4_rows(m4_payload, args.evaluate))
        except (FileNotFoundError, ValueError) as exc:
            if m4_payload is None and not args.evaluate:
                print(f"Warning: skipping m4 rows ({exc}). Run E3 or pass --evaluate.")
            else:
                raise

    if not args.skip_m6:
        try:
            rows.extend(_load_m6_rows(m6_payload, args.evaluate))
        except FileNotFoundError as exc:
            if m6_payload is None and not args.evaluate:
                print(f"Warning: skipping m6 rows ({exc}). Run E2 or pass --evaluate.")
            else:
                raise

    if not rows:
        raise SystemExit("No comparison rows generated. Provide experiment JSON or use --evaluate.")

    tex_path, md_path = write_comparison_tables(rows, output_dir=args.output_dir, stem=args.stem)

    print("=== Markdown ===")
    print(format_markdown(rows))
    print("\n=== LaTeX (requires \\usepackage{booktabs}) ===")
    print(format_latex(rows))
    print(f"\nWrote {tex_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
