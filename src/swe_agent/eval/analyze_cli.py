"""Analyse a batch's result records into a success-rate + failure breakdown.

Examples:
    python -m swe_agent.eval.analyze_cli --results-dir runs/lite
    python -m swe_agent.eval.analyze_cli --results-dir runs/lite --out runs/lite/report.md

Reads the per-instance JSON written by the batch harness; no API key or docker
needed. With --out, also writes the Markdown report to a file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from swe_agent.eval.analysis import analyze, format_report, load_records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="swe-agent-analyze", description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runs/lite"),
        help="Directory of per-instance JSON records (default: runs/lite).",
    )
    parser.add_argument(
        "--out", type=Path, help="Also write the Markdown report to this path."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.results_dir.is_dir():
        print(f"error: {args.results_dir} is not a directory", file=sys.stderr)
        return 2

    records = load_records(args.results_dir)
    if not records:
        print(f"error: no result records found in {args.results_dir}", file=sys.stderr)
        return 2

    report = analyze(records)
    markdown = format_report(report)
    print(markdown)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown, encoding="utf-8")
        print(f"(written to {args.out})", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
