# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Cargo build wrapper for benchmark binaries."""

import os
import subprocess
from pathlib import Path
from typing import final

from rich.console import Console

from ..config import BuildConfig, Engine, get_workspace_root

console = Console()


@final
class BenchmarkBuilder:
    """Builds benchmark binaries with correct flags."""

    def __init__(
        self,
        workspace_root: Path | None = None,
        config: BuildConfig | None = None,
        verbose: bool = False,
    ):
        self.workspace_root = workspace_root or get_workspace_root()
        self.config = config or BuildConfig()
        self.verbose = verbose

    def get_binary_path(self, backend: Engine) -> Path:
        """Get path to built binary."""
        binary_name = backend.binary_name
        return self.workspace_root / "target" / self.config.profile / binary_name

    def get_data_generator_path(self) -> Path:
        """Get path to the built benchmark data generator binary."""
        return self.workspace_root / "target" / self.config.profile / "data-gen"

    def build(self, backends: list[Engine]) -> dict[Engine, Path]:
        """Build binaries for specified engines, return paths."""
        results: dict[Engine, Path] = {}

        env = os.environ.copy()
        env["RUSTFLAGS"] = self.config.rustflags

        for backend in backends:
            binary_name = backend.binary_name
            console.print(f"[blue]Building {binary_name}...[/blue]")

            cmd = [
                "cargo",
                "build",
                "-p",
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
