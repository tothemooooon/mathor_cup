#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    py = str(ROOT / ".venv" / "bin" / "python")

    run([py, "run_baseline.py", "--question", "Q1", "--seed-count", "2"])
    run([py, "run_baseline.py", "--question", "Q2", "--seed-count", "2"])
    run([
        py,
        "run_baseline.py",
        "--question",
        "Q3",
        "--customers",
        "20",
        "--cluster-size",
        "5",
        "--seed-count-per-cluster",
        "2",
    ])
    run([
        py,
        "run_baseline.py",
        "--question",
        "Q4",
        "--customers",
        "20",
        "--min-vehicles",
        "2",
        "--max-vehicles",
        "4",
        "--seed-count-per-vehicle",
        "2",
    ])

    print("smoke tests passed")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"smoke tests failed: {exc}", file=sys.stderr)
        raise
