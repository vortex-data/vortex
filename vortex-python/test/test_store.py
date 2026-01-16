from vortex.store import LocalStore

import vortex as vx


def test_store_roundtrip(tmpdir_factory):
    data_dir = tmpdir_factory.mktemp("data")

    # create a local store to write into
    local = LocalStore(prefix=str(data_dir))

    records = vx.array([dict(name="Alice", salary=10), dict(name="Bob", salary=20), dict(name="Carol", salary=30)])

    assert len(records) == 3

    # write to the local store
    vx.io.write(records, "people.vortex", store=local)

    # verify file got written to correct location
    assert (data_dir / "people.vortex").exists()

    # test vx.read for eager full-scan
    people = vx.io.read_url("people.vortex", store=local)

    print(people.to_pylist())
    print(records.to_pylist())
    assert people.to_pylist() == records.to_pylist()
