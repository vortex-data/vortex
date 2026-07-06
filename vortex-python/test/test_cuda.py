# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import tomllib
from pathlib import Path
from typing import cast

import vortex


def workspace_version() -> str:
    workspace_pyproject = tomllib.loads((Path(__file__).parents[2] / "Cargo.toml").read_text())
    return cast(str, workspace_pyproject["workspace"]["package"]["version"])


def test_cuda_extension_installed_returns_bool():
    assert isinstance(vortex.cuda_extension_installed(), bool)
    assert "cuda_extension_installed" in vortex.__all__


def test_cuda_extra_installs_exact_matching_extension():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert pyproject["project"]["optional-dependencies"]["cuda"] == [f"vortex-data-cuda=={workspace_version()}"]
