"""Local verification runner for no-magic algorithm scripts.

Discovers scripts across all tier directories, runs each with a 600-second
timeout, and reports pass/fail/timeout. Exit code 0 means all passed.

Usage:
    python scripts/verify.py                           # full suite
    python scripts/verify.py --section 01-foundations  # one tier
    python scripts/verify.py microgpt.py               # specific script(s)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SECTIONS = ["01-foundations", "02-alignment", "03-systems", "04-agents"]
TIMEOUT_SECONDS = 600


def discover_scripts() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for section in SECTIONS:
        section_dir = REPO_ROOT / section
        if not section_dir.is_dir():
            continue
        scripts = sorted(p for p in section_dir.glob("*.py") if p.name != "__init__.py")
        if scripts:
            result[section] = scripts
    return result


def filter_by_section(all_scripts: dict[str, list[Path]], section: str) -> dict[str, list[Path]]:
    if section not in all_scripts:
        valid = ", ".join(all_scripts.keys())
        print(f"Error: unknown section '{section}'. Valid: {valid}", file=sys.stderr)
        sys.exit(1)
    return {section: all_scripts[section]}


def filter_by_names(all_scripts: dict[str, list[Path]], names: list[str]) -> dict[str, list[Path]]:
    lookup: dict[str, tuple[str, Path]] = {}
    for section, paths in all_scripts.items():
        for p in paths:
            lookup[p.name] = (section, p)

    unrecognized = [n for n in names if n not in lookup]
    if unrecognized:
        print(f"Error: unrecognized script(s): {', '.join(unrecognized)}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(lookup.keys()))}", file=sys.stderr)
        sys.exit(1)

    result: dict[str, list[Path]] = {}
    for name in names:
        section, path = lookup[name]
        result.setdefault(section, []).append(path)
    return result


def run_script(script_path: Path) -> tuple[str, float]:
    """Run script, return (status, elapsed_seconds). Status: 'pass', 'fail', 'timeout'."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=TIMEOUT_SECONDS,
        )
        elapsed = time.monotonic() - start
        status = "pass" if proc.returncode == 0 else "fail"
        if status == "fail":
            stderr_tail = proc.stderr.decode(errors="replace").splitlines()[-10:]
            for line in stderr_tail:
                print(f"    {line}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        status = "timeout"
    return status, round(elapsed, 1)


def format_duration(seconds: float) -> str:
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m {secs:02d}s"


def print_summary(results: list[tuple[str, str, float]]) -> None:
    """Print summary table. results: list of (label, status, elapsed)."""
    print()
    print("Verify results")
    print("\u2500" * 55)
    passed = failed = timed_out = 0
    for label, status, elapsed in results:
        if status == "pass":
            marker = "Pass"
            passed += 1
        elif status == "timeout":
            marker = "TIMEOUT"
            timed_out += 1
        else:
            marker = "FAIL"
            failed += 1
        dots = "." * max(1, 40 - len(label))
        print(f"  {label} {dots} {marker}  {format_duration(elapsed)}")

    total = len(results)
    print("\u2500" * 55)
    parts = [f"{passed}/{total} passed"]
    if failed:
        parts.append(f"{failed} failed")
    if timed_out:
        parts.append(f"{timed_out} timed out")
    print("  " + " | ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-magic scripts locally and report pass/fail/timeout."
    )
    parser.add_argument(
        "--section",
        choices=SECTIONS,
        help="Run only scripts in the given tier directory.",
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        help="Specific script filenames to verify (e.g. microgpt.py).",
    )
    args = parser.parse_args()

    if args.section and args.scripts:
        parser.error("--section and positional script names are mutually exclusive.")

    all_scripts = discover_scripts()
    if not all_scripts:
        print("Error: no algorithm scripts found.", file=sys.stderr)
        sys.exit(1)

    if args.section:
        targets = filter_by_section(all_scripts, args.section)
    elif args.scripts:
        targets = filter_by_names(all_scripts, args.scripts)
    else:
        targets = all_scripts

    results: list[tuple[str, str, float]] = []
    any_failed = False

    for section in SECTIONS:
        if section not in targets:
            continue
        for script_path in targets[section]:
            label = str(script_path.relative_to(REPO_ROOT))
            print(f"Running {label} ...", flush=True)
            status, elapsed = run_script(script_path)
            if status != "pass":
                any_failed = True
            results.append((label, status, elapsed))

    print_summary(results)
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
