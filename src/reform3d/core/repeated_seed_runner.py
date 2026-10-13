"""Subprocess helpers for REFORM-3D repeated-seed orchestration."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Sequence

from reform3d.core.seed_context import ACTIVE_SEED_ENV, normalize_seed


def option_present(argv: Sequence[str], option: str) -> bool:
    """Return whether a CLI option is present in an argv list."""
    return option in argv


def strip_options_with_values(argv: Sequence[str], options: set[str]) -> list[str]:
    """Remove options with variable-length values from an argv list.

    Values are consumed until the next ``--`` option.
    """
    stripped: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in options:
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        stripped.append(item)
        index += 1
    return stripped


def strip_options_with_one_value(argv: Sequence[str], options: set[str]) -> list[str]:
    """Remove options that take exactly one value."""
    stripped: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in options:
            index += 2
            continue
        stripped.append(item)
        index += 1
    return stripped


def strip_flags(argv: Sequence[str], flags: set[str]) -> list[str]:
    """Remove boolean flags from an argv list."""
    return [item for item in argv if item not in flags]


def repeated_seed_base_argv(argv: Sequence[str]) -> list[str]:
    """Return argv without repeated-seed orchestration controls."""
    base = strip_options_with_values(argv, {"--seeds"})
    base = strip_options_with_one_value(base, {"--seed"})
    base = strip_flags(base, {"--allow-incomplete-seeds"})
    return base


def command_line(command: Sequence[str], env_updates: dict[str, str] | None = None) -> str:
    """Return a shell-readable command line for logging."""
    env_parts = []
    if env_updates:
        env_parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(env_updates.items())]
    return " ".join([*env_parts, *[shlex.quote(part) for part in command]])


def run_seeded_module(
    *,
    module_name: str,
    base_argv: Sequence[str],
    seed: int | str,
    dry_run: bool,
    extra_args: Sequence[str] = (),
) -> None:
    """Run one module command with an active repeated-evaluation seed."""
    normalized_seed = normalize_seed(seed)
    env_updates = {ACTIVE_SEED_ENV: str(normalized_seed)}
    command = [
        sys.executable,
        "-u",
        "-m",
        module_name,
        *base_argv,
        "--seed",
        str(normalized_seed),
        *extra_args,
    ]
    print(f"[seed {normalized_seed}] {command_line(command, env_updates)}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env.update(env_updates)
    subprocess.run(command, check=True, env=env)


__all__ = [
    "command_line",
    "option_present",
    "repeated_seed_base_argv",
    "run_seeded_module",
    "strip_flags",
    "strip_options_with_one_value",
    "strip_options_with_values",
]
