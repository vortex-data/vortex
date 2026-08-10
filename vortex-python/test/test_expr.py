# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Coverage for the expression builders exposed by :mod:`vortex.expr`."""

import pickle
from collections.abc import Callable
from typing import cast

import pyarrow as pa
import pytest

import vortex as vx
import vortex.expr as ve
from vortex.expr import Expr


@pytest.fixture(scope="module")
def people(tmp_path_factory: pytest.TempPathFactory) -> vx.VortexFile:
    path = tmp_path_factory.mktemp("expr") / "people.vortex"
    array = vx.array(
        pa.array(
            [
                {"name": "Alice", "age": 30, "scores": [1, 2, 3], "nested": {"city": "Paris"}},
                {"name": "Bob", "age": 25, "scores": [4], "nested": {"city": "Berlin"}},
                {"name": "alfred", "age": None, "scores": [], "nested": {"city": "Paris"}},
                {"name": "Charlie", "age": 57, "scores": [5, 6], "nested": {"city": "Lima"}},
            ]
        )
    )
    vx.io.write(array, str(path))
    return vx.open(str(path))


def names(vxf: vx.VortexFile, expr: Expr) -> list[str | None]:
    rows = cast(list[dict[str, object]], vxf.scan(["name"], expr=expr).read_all().to_arrow_table().to_pylist())
    return [cast("str | None", row["name"]) for row in rows]


def column_values(vxf: vx.VortexFile, projection: Expr) -> list[object]:
    table = vxf.scan(ve.pack({"value": projection})).read_all().to_arrow_table()
    return cast(list[object], table.column("value").to_pylist())


# --------------------------------------------------------------------------------------
# Every builder is reachable and produces an expression
# --------------------------------------------------------------------------------------

BUILDERS: dict[str, Callable[[], Expr]] = {
    "root": ve.root,
    "column": lambda: ve.column("age"),
    "literal": lambda: ve.literal(vx.int_(), 42),
    "get_item": lambda: ve.get_item("city", ve.column("nested")),
    "get_item_default_child": lambda: ve.get_item("age"),
    "not_": lambda: ve.not_(ve.is_null(ve.column("age"))),
    "and_": lambda: ve.and_(ve.is_null(ve.column("age")), ve.is_not_null(ve.column("name"))),
    "or_": lambda: ve.or_(ve.is_null(ve.column("age")), ve.is_not_null(ve.column("name"))),
    "eq": lambda: ve.eq(ve.column("age"), 30),
    "not_eq": lambda: ve.not_eq(ve.column("age"), 30),
    "gt": lambda: ve.gt(ve.column("age"), 30),
    "gt_eq": lambda: ve.gt_eq(ve.column("age"), 30),
    "lt": lambda: ve.lt(ve.column("age"), 30),
    "lt_eq": lambda: ve.lt_eq(ve.column("age"), 30),
    "add": lambda: ve.add(ve.column("age"), 1),
    "sub": lambda: ve.sub(ve.column("age"), 1),
    "mul": lambda: ve.mul(ve.column("age"), 2),
    "div": lambda: ve.div(ve.column("age"), 2),
    "between": lambda: ve.between(ve.column("age"), 26, 40),
    "between_strict": lambda: ve.between(ve.column("age"), 26, 40, lower_strict=True, upper_strict=True),
    "is_null": lambda: ve.is_null(ve.column("age")),
    "is_not_null": lambda: ve.is_not_null(ve.column("age")),
    "fill_null": lambda: ve.fill_null(ve.column("age"), 0),
    "like": lambda: ve.like(ve.column("name"), "A%"),
    "ilike": lambda: ve.ilike(ve.column("name"), "a%"),
    "not_like": lambda: ve.not_like(ve.column("name"), "A%"),
    "not_ilike": lambda: ve.not_ilike(ve.column("name"), "a%"),
    "byte_length": lambda: ve.byte_length(ve.column("name")),
    "select": lambda: ve.select(["name", "age"]),
    "select_str": lambda: ve.select("name"),
    "select_exclude": lambda: ve.select_exclude(["scores"]),
    "pack_dict": lambda: ve.pack({"n": ve.column("name"), "constant": 7}),
    "pack_pairs": lambda: ve.pack([("n", ve.column("name"))]),
    "pack_nullable": lambda: ve.pack({"n": ve.column("name")}, nullable=True),
    "merge": lambda: ve.merge([ve.select(["name"]), ve.select(["age"])]),
    "merge_rightmost": lambda: ve.merge([ve.select(["name"]), ve.select(["name"])], duplicate_handling="rightmost"),
    "list_contains": lambda: ve.list_contains(ve.column("scores"), 5),
    "list_length": lambda: ve.list_length(ve.column("scores")),
    "list_sum": lambda: ve.list_sum(ve.column("scores")),
    "list_sum_nans": lambda: ve.list_sum(ve.column("scores"), skip_nans=False),
    "case_when": lambda: ve.case_when([(ve.gt(ve.column("age"), 30), "old")], else_value="young"),
    "case_when_no_else": lambda: ve.case_when([(ve.gt(ve.column("age"), 30), "old")]),
    "zip_": lambda: ve.zip_(ve.is_null(ve.column("age")), 0, ve.column("age")),
    "mask": lambda: ve.mask(ve.column("age"), ve.is_null(ve.column("age"))),
    "cast": lambda: ve.cast(ve.column("age"), vx.int_(32, nullable=True)),
    "and_collect": lambda: cast(Expr, ve.and_collect([ve.is_null(ve.column("age")), ve.gt(ve.column("age"), 1)])),
    "or_collect": lambda: cast(Expr, ve.or_collect([ve.is_null(ve.column("age")), ve.gt(ve.column("age"), 1)])),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_builder_returns_expr(name: str) -> None:
    assert isinstance(BUILDERS[name](), Expr)


# `CaseWhen::serialize` is deliberately disabled in vortex-array pending a stable wire format, so
# `case_when` expressions cannot cross a process boundary yet.
NOT_SERIALIZABLE = {"case_when", "case_when_no_else"}


@pytest.mark.parametrize("name", sorted(set(BUILDERS) - NOT_SERIALIZABLE))
def test_builder_round_trips_through_proto(name: str) -> None:
    expr = BUILDERS[name]()
    assert str(ve.deserialize(expr.serialize())) == str(expr)


@pytest.mark.parametrize("name", sorted(NOT_SERIALIZABLE))
def test_unserializable_builder_reports_clearly(name: str) -> None:
    with pytest.raises(RuntimeError, match="serial"):
        _ = BUILDERS[name]().serialize()


def test_variant_get_paths() -> None:
    # Variant columns are not exercised here; this only checks the path coercions are accepted.
    assert isinstance(ve.variant_get(ve.column("payload"), "user"), Expr)
    assert isinstance(ve.variant_get(ve.column("payload"), 0), Expr)
    assert isinstance(ve.variant_get(ve.column("payload"), ["user", 1, "id"]), Expr)
    assert isinstance(ve.variant_get(ve.column("payload"), "user", vx.int_(64, nullable=True)), Expr)


def test_variant_get_rejects_bad_path() -> None:
    with pytest.raises(TypeError):
        _ = ve.variant_get(ve.column("payload"), [1.5])  # pyright: ignore[reportArgumentType]


def test_ext_storage_returns_expr() -> None:
    assert isinstance(ve.ext_storage(ve.column("ts")), Expr)


def test_merge_rejects_unknown_duplicate_handling() -> None:
    with pytest.raises(ValueError):
        _ = ve.merge([ve.select(["name"])], duplicate_handling="nonsense")  # pyright: ignore[reportArgumentType]


def test_case_when_requires_a_pair() -> None:
    with pytest.raises(ValueError):
        _ = ve.case_when([])


def test_collect_returns_none_when_empty() -> None:
    assert ve.and_collect([]) is None
    assert ve.or_collect([]) is None


# --------------------------------------------------------------------------------------
# Semantics
# --------------------------------------------------------------------------------------


def test_like_and_ilike(people: vx.VortexFile) -> None:
    assert names(people, ve.like(ve.column("name"), "A%")) == ["Alice"]
    assert names(people, ve.ilike(ve.column("name"), "a%")) == ["Alice", "alfred"]
    assert names(people, ve.not_like(ve.column("name"), "A%")) == ["Bob", "alfred", "Charlie"]
    assert names(people, ve.not_ilike(ve.column("name"), "a%")) == ["Bob", "Charlie"]


def test_between(people: vx.VortexFile) -> None:
    assert names(people, ve.between(ve.column("age"), 25, 30)) == ["Alice", "Bob"]
    assert names(people, ve.between(ve.column("age"), 25, 30, lower_strict=True)) == ["Alice"]
    assert names(people, ve.between(ve.column("age"), 25, 30, upper_strict=True)) == ["Bob"]


def test_or_and_is_null(people: vx.VortexFile) -> None:
    assert names(people, ve.is_null(ve.column("age"))) == ["alfred"]
    assert names(people, ve.or_(ve.eq(ve.column("age"), 25), ve.eq(ve.column("age"), 57))) == ["Bob", "Charlie"]


def test_arithmetic_and_list_functions(people: vx.VortexFile) -> None:
    assert column_values(people, ve.add(ve.column("age"), 1)) == [31, 26, None, 58]
    assert column_values(people, ve.list_length(ve.column("scores"))) == [3, 1, 0, 2]
    assert column_values(people, ve.list_sum(ve.column("scores"))) == [6, 4, None, 11]
    assert column_values(people, ve.list_contains(ve.column("scores"), 5)) == [False, False, False, True]
    assert column_values(people, ve.byte_length(ve.column("name"))) == [5, 3, 6, 7]
    assert column_values(people, ve.fill_null(ve.column("age"), 0)) == [30, 25, 0, 57]
    assert column_values(people, ve.get_item("city", ve.column("nested"))) == ["Paris", "Berlin", "Paris", "Lima"]


def test_case_when_semantics(people: vx.VortexFile) -> None:
    expr = ve.case_when([(ve.gt(ve.column("age"), 40), "senior"), (ve.gt(ve.column("age"), 26), "mid")], "junior")
    assert column_values(people, expr) == ["mid", "junior", "junior", "senior"]


def test_select_exclude(people: vx.VortexFile) -> None:
    table = people.scan(ve.select_exclude(["scores", "nested"])).read_all().to_arrow_table()
    assert table.column_names == ["name", "age"]


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------


def test_invert_operator(people: vx.VortexFile) -> None:
    assert names(people, ~ve.is_null(ve.column("age"))) == ["Alice", "Bob", "Charlie"]


def test_reflected_operators(people: vx.VortexFile) -> None:
    # `5 < expr` falls back to `expr.__gt__(5)`; `1 + expr` needs `__radd__`.
    assert names(people, 26 < ve.column("age")) == ["Alice", "Charlie"]
    assert column_values(people, 1 + ve.column("age")) == [31, 26, None, 58]
    assert column_values(people, 100 - ve.column("age")) == [70, 75, None, 43]
    assert column_values(people, 2 * ve.column("age")) == [60, 50, None, 114]


def test_bool_literals_are_boolean(people: vx.VortexFile) -> None:
    # A Python bool is a subclass of int, so it must be checked before the int coercion.
    assert names(people, ve.is_not_null(ve.column("age")) & True) == ["Alice", "Bob", "Charlie"]


# --------------------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------------------


def test_pickle_round_trip_preserves_filter(people: vx.VortexFile) -> None:
    expr = (ve.column("age") > 26) & ve.ilike(ve.column("name"), "a%")
    restored = cast(Expr, pickle.loads(pickle.dumps(expr)))
    assert names(people, restored) == names(people, expr) == ["Alice"]


def test_deserialize_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        _ = ve.deserialize(b"\xff\xff\xff\xff\xff\xff")


def test_serialize_is_stable() -> None:
    expr = ve.column("age") > 21
    assert expr.serialize() == expr.serialize()
