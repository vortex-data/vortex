# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# pyright: reportAny=false, reportExplicitAny=false

import gc
import sys
import tomllib
import types
from pathlib import Path
from typing import Any, cast

import pytest
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


def test_to_cudf_is_exported():
    assert "to_cudf" in vortex_cuda.__all__


def test_import_installs_array_to_cudf(monkeypatch: pytest.MonkeyPatch):
    array = vortex.Array.from_range(range(0, 3))
    calls: list[tuple[object, str]] = []

    def fake_to_cudf(obj: object, *, fallback: str = "error") -> object:
        calls.append((obj, fallback))
        return "cudf-object"

    monkeypatch.setattr(vortex_cuda, "to_cudf", fake_to_cudf)

    assert cast(Any, array).to_cudf(fallback="error") == "cudf-object"
    assert calls == [(array, "error")]


def test_import_installs_array_arrow_c_device_array_on_cuda(monkeypatch: pytest.MonkeyPatch):
    array = vortex.Array.from_range(range(0, 3))

    if not vortex_cuda.cuda_available():
        assert not hasattr(array, "__arrow_c_device_array__")
        return

    calls: list[tuple[object, object | None, dict[str, object]]] = []

    def fake_export_device_array(
        exported_array: object,
        requested_schema: object | None = None,
        **kwargs: object,
    ) -> tuple[object, object]:
        calls.append((exported_array, requested_schema, kwargs))
        return "schema", "device_array"

    monkeypatch.setattr(vortex_cuda, "export_device_array", fake_export_device_array)

    assert cast(Any, array).__arrow_c_device_array__("requested", future=None) == ("schema", "device_array")
    assert calls == [(array, "requested", {"future": None})]


def test_to_cudf_rejects_unsupported_fallback():
    with pytest.raises(NotImplementedError, match="fallback='error'"):
        _ = vortex_cuda.to_cudf(vortex.Array.from_range(range(0, 3)), fallback="host")


def test_to_cudf_rejects_non_vortex_array():
    with pytest.raises(TypeError, match="vortex.Array"):
        _ = vortex_cuda.to_cudf(object())


def test_to_cudf_cuda_unavailable_rejects_without_importing_cudf(monkeypatch: pytest.MonkeyPatch):
    def fail_import_cudf_modules() -> tuple[object, object]:
        raise AssertionError("unexpected import of cuDF modules")

    monkeypatch.setattr(vortex_cuda, "cuda_available", lambda: False)
    monkeypatch.setattr("vortex_cuda._import_cudf_modules", fail_import_cudf_modules)

    with pytest.raises(RuntimeError, match="CUDA"):
        _ = vortex_cuda.to_cudf(vortex.Array.from_range(range(0, 3)))


def test_to_cudf_non_struct_uses_pylibcudf_column_from_arrow(monkeypatch: pytest.MonkeyPatch):
    array = vortex.Array.from_range(range(0, 3))
    fake_column = object()

    class FakeSeries:
        pass

    fake_series = FakeSeries()

    class FakePylibcudfColumn:
        @staticmethod
        def from_arrow(obj: object) -> object:
            assert obj is array
            return fake_column

    class FakePylibcudfTable:
        @staticmethod
        def from_arrow(_obj: object) -> object:
            raise AssertionError("non-struct array should not import through pylibcudf.Table")

    class FakeCudfSeries:
        @staticmethod
        def from_pylibcudf(column: object) -> object:
            assert column is fake_column
            return fake_series

    monkeypatch.setattr(vortex_cuda, "cuda_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "pylibcudf",
        types.SimpleNamespace(Column=FakePylibcudfColumn, Table=FakePylibcudfTable),
    )
    monkeypatch.setitem(sys.modules, "cudf", types.SimpleNamespace(Series=FakeCudfSeries))

    assert vortex_cuda.to_cudf(array) is fake_series


def test_to_cudf_struct_uses_pylibcudf_table_from_arrow(monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa

    array = vortex.Array.from_arrow(pa.table({"x": [1, None, 3], "y": [4.0, 5.0, 6.0]}))
    fake_table = object()

    class FakeDataFrame:
        columns: object = None

    fake_dataframe = FakeDataFrame()

    class FakePylibcudfColumn:
        @staticmethod
        def from_arrow(_obj: object) -> object:
            raise AssertionError("struct array should not import through pylibcudf.Column")

    class FakePylibcudfTable:
        @staticmethod
        def from_arrow(obj: object) -> object:
            assert obj is array
            return fake_table

    class FakeCudfDataFrame:
        @staticmethod
        def from_pylibcudf(table: object) -> object:
            assert table is fake_table
            return fake_dataframe

    monkeypatch.setattr(vortex_cuda, "cuda_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "pylibcudf",
        types.SimpleNamespace(Column=FakePylibcudfColumn, Table=FakePylibcudfTable),
    )
    monkeypatch.setitem(sys.modules, "cudf", types.SimpleNamespace(DataFrame=FakeCudfDataFrame))

    assert vortex_cuda.to_cudf(array) is fake_dataframe
    assert fake_dataframe.columns == ["x", "y"]


def test_to_cudf_nullable_struct_rejects_without_importing_cudf(monkeypatch: pytest.MonkeyPatch):
    import pyarrow as pa

    array = vortex.Array.from_arrow(pa.array([{"x": 1}, None], type=pa.struct([("x", pa.int64())])))

    def fail_import_cudf_modules() -> tuple[object, object]:
        raise AssertionError("unexpected import of cuDF modules")

    monkeypatch.setattr(vortex_cuda, "cuda_available", lambda: True)
    monkeypatch.setattr("vortex_cuda._import_cudf_modules", fail_import_cudf_modules)

    with pytest.raises(NotImplementedError, match="top-level nulls"):
        _ = vortex_cuda.to_cudf(array)


def test_to_cudf_real_cudf_smoke():
    cudf_module = cast(object, pytest.importorskip("cudf"))
    pylibcudf_module = cast(object, pytest.importorskip("pylibcudf"))
    assert cudf_module is not None
    assert pylibcudf_module is not None

    if not vortex_cuda.cuda_available():
        pytest.skip("CUDA device is not available")

    import pyarrow as pa

    int_series = cast(Any, vortex.array([1, 2, 3])).to_cudf()
    assert type(int_series).__name__ == "Series"
    assert int_series.to_arrow().to_pylist() == [1, 2, 3]

    nullable_int_series = cast(Any, vortex.array([1, None, 3])).to_cudf()
    assert type(nullable_int_series).__name__ == "Series"
    assert nullable_int_series.to_arrow().to_pylist() == [1, None, 3]
    assert "<NA>" in repr(nullable_int_series)

    nullable_bool_series = cast(Any, vortex.array([True, None, False])).to_cudf()
    assert type(nullable_bool_series).__name__ == "Series"
    assert nullable_bool_series.to_arrow().to_pylist() == [True, None, False]
    assert "<NA>" in repr(nullable_bool_series)

    string_series = cast(Any, vortex.array(["alpha", "beta", "gamma"])).to_cudf()
    assert type(string_series).__name__ == "Series"
    assert string_series.to_arrow().to_pylist() == ["alpha", "beta", "gamma"]

    struct_array = vortex.Array.from_arrow(pa.table({"x": [1, None, 3], "y": [4.0, 5.0, 6.0]}))
    dataframe = cast(Any, struct_array).to_cudf()
    assert type(dataframe).__name__ == "DataFrame"
    assert list(dataframe.columns) == ["x", "y"]
    assert len(dataframe) == 3
    assert "<NA>" in repr(dataframe)
    assert {name: str(dtype) for name, dtype in dataframe.dtypes.items()} == {
        "x": "int64",
        "y": "float64",
    }

    x_series = dataframe["x"]
    del dataframe
    _ = gc.collect()
    assert x_series.to_arrow().to_pylist() == [1, None, 3]
    assert "<NA>" in repr(x_series)
