# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from pathlib import Path

from vortex.store import LocalStore

import vortex as vx


def test_store_roundtrip(tmp_path: Path, session: vx.Session) -> None:
    # create a local store to write into
    local = LocalStore(prefix=tmp_path)

    records = vx.array([dict(name="Alice", salary=10), dict(name="Bob", salary=20), dict(name="Carol", salary=30)])

    assert len(records) == 3

    # write to the local store
    vx.io.write(records, "people.vortex", store=local, session=session)

    # verify file got written to correct location
    assert (tmp_path / "people.vortex").exists()

    # test vx.read for eager full-scan
    people = vx.io.read_url("people.vortex", store=local, session=session)

    assert people.to_pylist(session=session) == records.to_pylist(session=session)
