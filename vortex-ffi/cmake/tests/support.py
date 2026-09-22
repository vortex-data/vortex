# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Small helpers for isolated CMake projects and recording tools."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, Unpack, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
CMAKE_DIR = REPO_ROOT / "vortex-ffi/cmake"


class CommandOptions(TypedDict, total=False):
    env: Mapping[str, str] | None
    cwd: str | Path | None
    success: bool
    timeout: float


class CargoRecording(TypedDict):
    args: list[str]
    env: dict[str, str]


def rust_toolchain_environment() -> dict[str, str]:
    """Keep the repository's Rust selection when fixtures run outside its tree."""
    env = os.environ.copy()
    if not env.get("RUSTUP_TOOLCHAIN") and shutil.which("rustup"):
        result = subprocess.run(
            ["rustup", "show", "active-toolchain"],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        env["RUSTUP_TOOLCHAIN"] = result.stdout.split()[0]
    return env


class CMakeTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="vortex-cmake-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        self.repo = REPO_ROOT
        self.env = os.environ.copy()

    def write(self, path: str | Path, contents: str) -> Path:
        path = self.work / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(contents), encoding="utf-8")
        return path

    def executable(self, name: str, body: str) -> str:
        path = self.write(name, f"#!{sys.executable}\n" + textwrap.dedent(body))
        path.chmod(0o755)
        return str(path)

    def command(
        self,
        *args: str | Path,
        env: Mapping[str, str] | None = None,
        cwd: str | Path | None = None,
        success: bool = True,
        timeout: float = 120,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            list(map(str, args)),
            cwd=self.work if cwd is None else cwd,
            env=self.env if env is None else env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = result.stdout + result.stderr
        if success:
            self.assertEqual(result.returncode, 0, output)
        else:
            self.assertNotEqual(result.returncode, 0, output)
        return result

    def cmake_configure(
        self,
        source: str | Path,
        build: str | Path,
        *options: str,
        generator: str = "Ninja",
        **kwargs: Unpack[CommandOptions],
    ) -> subprocess.CompletedProcess[str]:
        return self.command("cmake", "-G", generator, "-S", source, "-B", build, *options, **kwargs)

    def cmake_build(
        self, build: str | Path, *options: str, **kwargs: Unpack[CommandOptions]
    ) -> subprocess.CompletedProcess[str]:
        return self.command("cmake", "--build", build, *options, **kwargs)

    def recording_cargo(self) -> str:
        """Record the handoff and emit an archive without compiling Vortex."""
        return self.executable(
            "cargo",
            """\
            import json, os, pathlib, sys
            args = sys.argv[1:]
            def option(name):
                return args[args.index(name) + 1]
            root = pathlib.Path(option('--target-dir'))
            profile = option('--profile')
            package = option('--package').replace('-', '_')
            archive = root / option('--target') / ('debug' if profile == 'dev' else profile) / f'lib{package}.a'
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(b'recorded archive')
            (root / 'environment.json').write_text(json.dumps({'args': args, 'env': dict(os.environ)}))
            """,
        )

    def fake_rustc(self, release: str = "1.95.0", host: str | None = None) -> str:
        arch = {"arm64": "aarch64", "AMD64": "x86_64"}.get(platform.machine(), platform.machine())
        host = host or f"{arch}-{'apple-darwin' if sys.platform == 'darwin' else 'unknown-linux-gnu'}"
        return self.executable(
            "rustc",
            f"""\
            import json, os, pathlib
            pathlib.Path(__file__).with_suffix('.json').write_text(json.dumps(
                [os.environ.get('RUSTUP_TOOLCHAIN'), os.getcwd()]))
            print('rustc {release}\\nhost: {host}\\nrelease: {release}')
            """,
        )

    def cargo_recording(self, target_dir: Path) -> CargoRecording:
        return cast(CargoRecording, json.loads((target_dir / "environment.json").read_text(encoding="utf-8")))
