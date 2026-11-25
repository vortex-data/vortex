# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import subprocess
from pathlib import Path

import pytest
from bench_orchestrator.config import BuildConfig, Engine
from bench_orchestrator.runner.builder import BenchmarkBuilder


@pytest.mark.parametrize(("profile", "profile_dir"), [("dev", "debug"), ("release_debug", "release_debug")])
def test_lookup_uses_cargo_metadata_target_directory_and_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    profile_dir: str,
) -> None:
    custom_target = tmp_path / "cargo-custom"
    switched_target = tmp_path / "cargo-switched"
    calls = 0

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        assert cmd == ["cargo", "metadata", "--format-version", "1", "--no-deps"]
        assert kwargs["cwd"] == tmp_path
        calls += 1
        target = custom_target if calls == 1 else switched_target
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"target_directory": str(target)}))

    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    monkeypatch.delenv("CARGO_BUILD_TARGET_DIR", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)

    builder = BenchmarkBuilder(workspace_root=tmp_path, config=BuildConfig(profile=profile))

    assert builder.get_binary_path(Engine.DATAFUSION) == custom_target / profile_dir / "datafusion-bench"
    assert builder.get_data_generator_path() == custom_target / profile_dir / "data-gen"
    assert calls == 1

    monkeypatch.setenv("CARGO_TARGET_DIR", str(switched_target))
    assert builder.get_binary_path(Engine.DATAFUSION) == switched_target / profile_dir / "datafusion-bench"
    assert calls == 2


def test_get_target_dir_falls_back_to_workspace_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_metadata(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("cargo not installed")

    monkeypatch.setattr(subprocess, "run", fail_metadata)

    assert BenchmarkBuilder(workspace_root=tmp_path).get_target_dir() == tmp_path / "target"
