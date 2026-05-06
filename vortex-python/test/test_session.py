# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import vortex as vx


def test_session_exports_and_array_execution():
    session = vx.Session()

    array = vx.array([1, 2, 3])

    assert array.scalar_at(1, session=session).as_py() == 2
    assert array.to_arrow_array(session=session).to_pylist() == [1, 2, 3]
    assert vx.compress(array, session=session).to_arrow_array(session=session).to_pylist() == [
        1,
        2,
        3,
    ]


def test_file_dataset_and_scan_keep_session(tmp_path):
    session = vx.Session()
    path = tmp_path / "data.vortex"

    vx.io.write(vx.array([{"x": 1}, {"x": 2}]), str(path), session=session)

    vxf = vx.open(str(path), session=session)
    dataset = vxf.to_dataset()

    assert isinstance(vxf.session, vx.Session)
    assert isinstance(dataset.session, vx.Session)
    assert vxf.scan().read_all().to_arrow_table(session=session).to_pylist() == [
        {"x": 1},
        {"x": 2},
    ]
    assert dataset.to_table().to_pylist() == [{"x": 1}, {"x": 2}]
