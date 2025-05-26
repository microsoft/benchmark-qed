# Copyright (c) 2025 Microsoft Corporation.
"""Main entry point for the benchmark_qed UI."""

import argparse
from pathlib import Path

from benchmark_qed.ui.compare_app import CompareApp

if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("results", type=str, help="The path to the results folder.")
    args: argparse.Namespace = parser.parse_args()
    app: CompareApp = CompareApp(Path(args.results))
    app.run()
