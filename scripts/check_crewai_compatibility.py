"""Install one pinned CrewAI release in fresh, isolated preflight environments.

This opt-in setup check uses the network to download dependencies. It never
changes the Part 1 environment, requirements, model, knowledge base or database.
It performs no LLM request and no crew kickoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]


def run_logged(command: list[str], path: Path, env: dict[str, str]) -> int:
    """Save the complete subprocess output and exit code without dumping env."""
    with path.open("w", encoding="utf-8") as stream:
        stream.write("Command: " + subprocess.list2cmdline(command) + "\n\n")
        stream.flush()
        try:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=900, check=False,
            )
            code = result.returncode
        except subprocess.TimeoutExpired:
            stream.write("\nPreflight subprocess exceeded its time limit.\n")
            code = 124
        stream.write(f"\nExit code: {code}\n")
    print(f"{path.relative_to(ROOT)}: exit {code}", flush=True)
    return code


def environment_python(path: Path) -> Path:
    venv.EnvBuilder(with_pip=True).create(path)
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Exact official CrewAI release")
    parser.add_argument("--run-name", required=True, help="Unique ignored environment name")
    args = parser.parse_args()
    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        parser.error("Use an explicit stable release, for example major.minor.patch")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_name):
        parser.error("run-name must contain only letters, digits, underscores or hyphens")

    base = ROOT / ".runtime" / args.run_name
    if base.exists():
        parser.error("Environment directory already exists; choose a new run-name")
    base.mkdir(parents=True)
    transcripts = ROOT / "transcripts"
    transcripts.mkdir(exist_ok=True)
    requirements = ROOT / "requirements.txt"
    initial_hash = hashlib.sha256(requirements.read_bytes()).hexdigest()
    env = os.environ.copy()
    env.update({
        "CREWAI_DISABLE_TELEMETRY": "true",
        "OTEL_SDK_DISABLED": "true",
        "CREWAI_TRACING_ENABLED": "false",
        "CREWAI_STORAGE_DIR": str(base / "storage"),
        "PYTHONIOENCODING": "utf-8",
    })
    print(f"Python: {sys.version}", flush=True)
    print(f"Pinned CrewAI: {args.version}", flush=True)
    print(f"Fresh environments: {base.relative_to(ROOT)}", flush=True)
    combined_python = environment_python(base / "combined")

    def install(python: Path, suffix: str, include_part1: bool) -> int:
        command = [
            str(python), "-m", "pip", "--isolated", "--disable-pip-version-check",
            "--no-input", "--retries", "1", "--timeout", "20", "install",
            "--index-url", "https://pypi.org/simple", f"crewai=={args.version}",
        ]
        if include_part1:
            command.extend(["-r", str(requirements)])
        return run_logged(command, transcripts / f"crewai_preflight_{suffix}.txt", env)

    combined_code = install(combined_python, "combined_install", True)
    selected_python = combined_python
    standalone_code = None
    if combined_code != 0:
        selected_python = environment_python(base / "standalone")
        standalone_code = install(selected_python, "standalone_install", False)

    install_code = combined_code if standalone_code is None else standalone_code
    check_code = freeze_code = inspect_code = None
    if install_code == 0:
        check_code = run_logged(
            [str(selected_python), "-m", "pip", "check"],
            transcripts / "crewai_preflight_pip_check.txt", env,
        )
        freeze_code = run_logged(
            [str(selected_python), "-m", "pip", "freeze"],
            transcripts / "crewai_preflight_freeze.txt", env,
        )
        inspect_code = run_logged(
            [str(selected_python), "-c", (
                "import importlib.metadata as m, json, os, sys; "
                "print('Python:',sys.version); "
                "print('CrewAI:',m.version('crewai')); "
                "print('CrewAI requirements:',json.dumps(m.requires('crewai'),indent=2)); "
                "print('CREWAI_DISABLE_TELEMETRY:',os.environ['CREWAI_DISABLE_TELEMETRY']); "
                "print('OTEL_SDK_DISABLED:',os.environ['OTEL_SDK_DISABLED']); "
                "from crewai import Agent, Crew; "
                "from crewai.llms.base_llm import BaseLLM; "
                "print('CrewAI Agent, Crew and BaseLLM imports: PASS'); "
                "print('No LLM request or crew kickoff performed')"
            )], transcripts / "crewai_preflight_import.txt", env,
        )
    unchanged = hashlib.sha256(requirements.read_bytes()).hexdigest() == initial_hash
    summary = {
        "python": sys.version,
        "crewai_version": args.version,
        "release_source": "https://pypi.org/pypi/crewai/json",
        "environment_directory": str(base.relative_to(ROOT)),
        "combined_install_exit_code": combined_code,
        "standalone_install_exit_code": standalone_code,
        "pip_check_exit_code": check_code,
        "freeze_exit_code": freeze_code,
        "import_exit_code": inspect_code,
        "requirements_sha256": initial_hash,
        "requirements_unchanged": unchanged,
        "part2_implemented": False,
        "llm_requests_or_kickoff_performed": False,
        "combined_install_is_distinct_from_standalone_check": True,
    }
    path = transcripts / "crewai_preflight_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if install_code == check_code == freeze_code == inspect_code == 0 and unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
