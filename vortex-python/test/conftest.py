# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from .hub_server import HubState, serve

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def local_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[HubState]:
    """A local Hugging Face Hub stand-in, installed as ``HF_ENDPOINT`` for the test."""
    root = tmp_path / "hub"
    root.mkdir()
    server, state = serve(root)
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    try:
        yield state
    finally:
        server.shutdown()
