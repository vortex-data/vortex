# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import hashlib
import math
import os
from typing import cast

import pyarrow as pa
import pytest

import vortex as vx


@pytest.fixture(
    scope="session",
    params=[{"x"}, {"x", "y"}, {"x", "z"}, {"x", "y", "z"}],
    ids=["int", "int_str", "int_float", "int_str_float"],
)
def vxf(
    tmpdir_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> vx.VortexFile:
    fname = tmpdir_factory.mktemp("data") / "foo.vortex"

    if not os.path.exists(fname):
        length = 100_000

        columns: dict[str, list[int] | list[float] | list[str]] = {}
        assert "x" in request.param  # pyright: ignore[reportAny]
        columns["x"] = list(range(length))

        if "y" in request.param:  # pyright: ignore[reportAny]
            columns["y"] = [hashlib.md5(x.to_bytes(length=4), usedforsecurity=False).hexdigest() for x in range(length)]
        if "z" in request.param:  # pyright: ignore[reportAny]
            columns["z"] = [math.sqrt(x) for x in range(length)]

        a = vx.array(pa.table(columns))  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]
        vx.io.write(a, str(fname))
    return vx.open(str(fname))


@pytest.fixture(scope="session", params=[10_000, 2_000_000], ids=["small", "large"])
def array_fixture(request: pytest.FixtureRequest) -> vx.Array:
    size = cast(int, request.param)
    return vx.array(pa.table({"x": list(range(size))}))
