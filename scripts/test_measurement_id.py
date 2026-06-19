# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Cross-language golden-vector test for the `measurement_id` Python port.

`scripts/_measurement_id.py` ports the server-internal xxhash64 functions from
the `vortex-data/benchmarks-website` repo's `server/src/db.rs`. This test asserts
the port reproduces the golden vectors generated FROM that Rust source of truth
(`scripts/measurement_id_golden.json`, written by the
`measurement_id_golden_vectors` test in the `vortex-data/benchmarks-website`
repo's `server/tests/measurement_id_golden.rs`).

Because the Rust test computes the golden ids with the real `measurement_id_*`
functions and asserts the committed file matches them, and this test asserts the
Python port matches the same committed file, the two implementations are pinned
transitively: Rust == golden == Python. Either side drifting fails its half.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent
_GOLDEN_PATH = _SCRIPTS_DIR / "measurement_id_golden.json"


def _load_port():
    """Import `_measurement_id.py` by file path for symmetry with the sibling
    `test_migrate_schema.py` loader; a normal import also works since the module
    name is a valid identifier, but loading by path keeps the scripts/ tests
    uniform and independent of `sys.path` setup."""
    spec = importlib.util.spec_from_file_location("_measurement_id", _SCRIPTS_DIR / "_measurement_id.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pytest.importorskip("xxhash")
port = _load_port()


def _golden_vectors() -> list[dict]:
    document = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert document["seed"] == 0, "port assumes xxhash64 seed 0"
    vectors = document["vectors"]
    assert vectors, "golden file has no vectors; regenerate via the Rust test"
    return vectors


# Build the parametrize list once at import so each golden vector is its own test
# case: a single failure names the exact diverging vector (`NN-<table>`).
_VECTORS = _golden_vectors()
_PARAMS = [pytest.param(v, id=f"{i:02d}-{v['table']}") for i, v in enumerate(_VECTORS)]


@pytest.mark.parametrize("vector", _PARAMS)
def test_python_port_matches_golden(vector: dict) -> None:
    """The Python port reproduces the Rust-generated `measurement_id` for one
    golden vector."""
    table = vector["table"]
    fields = vector["fields"]
    expected = vector["measurement_id"]

    func = port.MEASUREMENT_ID_BY_TABLE.get(table)
    assert func is not None, f"no port function for table {table!r}"

    actual = func(**fields)

    assert actual == expected, (
        f"measurement_id mismatch for {table} vector {fields!r}: port produced {actual}, Rust golden is {expected}"
    )
    # The id is bit-cast to i64; confirm it stays in range (catches a port that
    # forgot the u64->i64 conversion and returned a bare u64).
    assert -(2**63) <= actual < 2**63, "measurement_id must be a signed 64-bit int"


def test_all_tables_covered() -> None:
    """Every fact table in the port's dispatch table appears in the golden
    vectors, so no family silently lacks cross-language coverage."""
    golden_tables = {v["table"] for v in _VECTORS}
    port_tables = set(port.MEASUREMENT_ID_BY_TABLE)
    assert port_tables == golden_tables, (
        f"port tables {sorted(port_tables)} != golden tables {sorted(golden_tables)}; "
        "a fact family is missing cross-language coverage"
    )


def test_multibyte_fixture_present() -> None:
    """Guard the regression that motivated byte-length prefixing: at least one
    vector must carry multibyte UTF-8, or the port could use char-length and
    still pass. If the golden set is ever regenerated without a multibyte
    fixture, fail loudly."""
    blob = json.dumps(_VECTORS, ensure_ascii=False)
    assert any(ord(ch) > 0x7F for ch in blob), (
        "no multibyte UTF-8 in golden vectors; the byte-vs-char length-prefix regression is unguarded"
    )
