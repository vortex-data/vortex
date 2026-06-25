# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import tomllib
from pathlib import Path
from typing import cast

import vortex_cuda

import vortex


def workspace_version() -> str:
    workspace_pyproject = tomllib.loads((Path(__file__).parents[2] / "Cargo.toml").read_text())
    return cast(str, workspace_pyproject["workspace"]["package"]["version"])


def test_extension_is_detected_by_base():
    assert vortex.cuda_extension_installed() is True


def test_cuda_available_returns_bool():
    assert isinstance(vortex_cuda.cuda_available(), bool)


def test_extension_exact_pins_base_package():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert pyproject["project"]["dependencies"] == [f"vortex-data=={workspace_version()}"]
