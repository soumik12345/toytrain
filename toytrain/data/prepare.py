"""Run one or more dataset-local Nemotron preparation scripts.

For individual processing, run the ``prepare.py`` inside that dataset's own
directory. This launcher is only the convenience entry point for processing
multiple dataset directories in sequence.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def discover_prepare_scripts(data_dir: Path) -> dict[str, Path]:
    """Find dataset-local prepare.py files without maintaining a registry."""
    scripts = {}
    for directory, _, filenames in os.walk(data_dir):
        script = Path(directory) / "prepare.py"
        if "prepare.py" in filenames and script.resolve() != Path(__file__).resolve():
            scripts[script.parent.name] = script
    return dict(sorted(scripts.items()))


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    dataset_scripts = discover_prepare_scripts(data_dir)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run every dataset-local script.")
    parser.add_argument("--datasets", nargs="+", choices=dataset_scripts)
    parser.add_argument("--stop-on-error", action="store_true")
    args, forwarded_args = parser.parse_known_args()
    if not args.all and not args.datasets:
        parser.error("choose --all or --datasets ...")

    selected = list(dataset_scripts) if args.all else args.datasets
    failures = []
    for name in selected:
        script = dataset_scripts[name]
        print(f"\n=== Preparing {name} ===", flush=True)
        result = subprocess.run([sys.executable, str(script), *forwarded_args], check=False)
        if result.returncode:
            failures.append(name)
            print(f"{name} failed with exit code {result.returncode}")
            if args.stop_on_error:
                break

    if failures:
        raise SystemExit(f"Failed datasets: {', '.join(failures)}")


if __name__ == "__main__":
    main()
