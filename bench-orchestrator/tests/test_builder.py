# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from pathlib import Path

from bench_orchestrator.config import Engine
from bench_orchestrator.runner import builder as builder_module
from bench_orchestrator.runner.builder import BenchmarkBuilder


def test_datafusion_build_uses_package_and_binary_names(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, env, check):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["check"] = check

    monkeypatch.setattr(builder_module.subprocess, "run", fake_run)

    builder = BenchmarkBuilder(workspace_root=tmp_path)
    paths = builder.build([Engine.DATAFUSION])

    assert captured["cmd"] == [
        "cargo",
        "build",
        "--package",
        "datafusion-bench",
        "--bin",
        "df-bench",
        "--profile",
        "release_debug",
        "--features",
        "unstable_encodings",
    ]
    assert captured["cwd"] == tmp_path
    assert captured["check"] is True
    assert paths == {Engine.DATAFUSION: tmp_path / "target" / "release_debug" / "df-bench"}
