# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Cargo build wrapper for benchmark binaries."""

import json
import os
import subprocess
from pathlib import Path
from typing import final

from rich.console import Console

from ..config import BuildConfig, Engine, get_workspace_root

console = Console()

_PROFILE_DIRS: dict[str, str] = {
    "dev": "debug",
    "test": "debug",
    "bench": "release",
}


@final
class BenchmarkBuilder:
    """Builds benchmark binaries with correct flags."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        config: BuildConfig | None = None,
        verbose: bool = False,
    ) -> None:
        self.workspace_root = workspace_root or get_workspace_root()
        self.config = config or BuildConfig()
        self.verbose = verbose
        self._target_dir_cache: tuple[tuple[str | None, str | None], Path] | None = None

    def _profile_dir(self) -> str:
        return _PROFILE_DIRS.get(self.config.profile, self.config.profile)

    def get_target_dir(self) -> Path:
        """Resolve the Cargo target directory for this workspace, honoring Cargo env and config."""
        env_key = (os.environ.get("CARGO_TARGET_DIR"), os.environ.get("CARGO_BUILD_TARGET_DIR"))
        if self._target_dir_cache is not None and self._target_dir_cache[0] == env_key:
            return self._target_dir_cache[1]

        target_dir = self.workspace_root / "target"
        try:
            completed = subprocess.run(
                ["cargo", "metadata", "--format-version", "1", "--no-deps"],
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=True,
            )
            metadata = json.loads(completed.stdout)
            resolved = metadata.get("target_directory")
            if isinstance(resolved, str) and resolved:
                target_dir = Path(resolved)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            pass

        self._target_dir_cache = (env_key, target_dir)
        return target_dir

    def get_binary_path(self, backend: Engine) -> Path:
        """Get path to built binary."""
        binary_name = backend.binary_name
        return self.get_target_dir() / self._profile_dir() / binary_name

    def get_data_generator_path(self) -> Path:
        """Get path to the built benchmark data generator binary."""
        return self.get_target_dir() / self._profile_dir() / "data-gen"

    def build(self, backends: list[Engine]) -> dict[Engine, Path]:
        """Build binaries for specified engines, return paths."""
        results: dict[Engine, Path] = {}

        env = os.environ.copy()
        env["RUSTFLAGS"] = self.config.rustflags

        for backend in backends:
            binary_name = backend.binary_name
            package_name = backend.package_name
            console.print(f"[blue]Building {binary_name}...[/blue]")

            cmd = [
                "cargo",
                "build",
                "--package",
                package_name,
                "--bin",
                binary_name,
                "--profile",
                self.config.profile,
            ]
            if self.config.features:
                cmd.extend(["--features", ",".join(self.config.features)])

            if self.verbose:
                console.print(f"[dim]$ RUSTFLAGS='{self.config.rustflags}' {' '.join(cmd)}[/dim]")

            try:
                subprocess.run(
                    cmd,
                    cwd=self.workspace_root,
                    env=env,
                    check=True,
                )
                results[backend] = self.get_binary_path(backend)
                console.print(f"[green]Built {binary_name}[/green]")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Failed to build {binary_name}: {e}[/red]")
                raise

        return results
