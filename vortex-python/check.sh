#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

set -ex -o pipefail

ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )

# We do not use uv because: Ray's raylets do not initialize properly inside a Sphinx `doctest` that
# is further inside a `uv run`.
source $ROOT/.venv/bin/activate

pushd $ROOT/vortex-python
maturin develop
ruff format --check
ruff check
basedpyright
popd

pushd $ROOT/docs
make clean # Sphinx is bad at cache invalidation. Best not to rely on it.
make html
make doctest
popd

pushd $ROOT/vortex-python
pytest test
popd
