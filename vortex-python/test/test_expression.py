# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

# Tests the _schema_for_substrait workaround in vortex/arrow/expression.py

from typing import TYPE_CHECKING, Literal, TypeAlias

import pyarrow as pa
import pyarrow.compute as pc
import pytest
from vortex.arrow.expression import _schema_for_substrait, arrow_to_vortex  # pyright: ignore[reportPrivateUsage]

import vortex as vx
import vortex.expr as ve
from vortex import substrait as vx_substrait

if TYPE_CHECKING:
    from substrait.algebra_pb2 import Expression
    from substrait.type_pb2 import Type
else:
    try:
        from substrait.algebra_pb2 import Expression
        from substrait.type_pb2 import Type
    except ImportError:
        from substrait.gen.proto.algebra_pb2 import Expression
        from substrait.gen.proto.type_pb2 import Type

UIntWidth: TypeAlias = Literal[8, 16, 32, 64]
UnsignedNullCase: TypeAlias = tuple[pa.DataType, UIntWidth]
UNSIGNED_NULL_CASES: list[UnsignedNullCase] = [
    (pa.uint8(), 8),
    (pa.uint16(), 16),
    (pa.uint32(), 32),
    (pa.uint64(), 64),
]


class TestSchemaForSubstrait:
    """Verifies mapping: string_view=>string, binary_view=>binary, else unchanged"""

    def test_string_view_mapped_to_string(self):
        schema = pa.schema([("col", pa.string_view())])
        result = _schema_for_substrait(schema)
        assert result.field("col").type == pa.string()  # pyright: ignore[reportUnknownMemberType]

    def test_binary_view_mapped_to_binary(self):
        schema = pa.schema([("col", pa.binary_view())])
        result = _schema_for_substrait(schema)
        assert result.field("col").type == pa.binary()  # pyright: ignore[reportUnknownMemberType]

    def test_other_types_unchanged(self):
        schema = pa.schema(
            [
                ("int_col", pa.int64()),
                ("str_col", pa.string()),
                ("bin_col", pa.binary()),
                ("float_col", pa.float64()),
            ]
        )
        result = _schema_for_substrait(schema)
        assert result == schema

    def test_mixed_schema(self):
        schema = pa.schema(
            [
                ("sv", pa.string_view()),
                ("bv", pa.binary_view()),
                ("s", pa.string()),
                ("i", pa.int64()),
            ]
        )
        result = _schema_for_substrait(schema)
        expected = pa.schema(
            [
                ("sv", pa.string()),
                ("bv", pa.binary()),
                ("s", pa.string()),
                ("i", pa.int64()),
            ]
        )
        assert result == expected


class TestArrowToVortexWithViews:
    """Tests comparisons over string_views and binary_views"""

    def test_string_view_equality_expression(self):
        schema = pa.schema([("name", pa.string_view())])
        expr = pc.field("name") == "alice"
        vortex_expr = arrow_to_vortex(expr, schema)
        assert vortex_expr is not None

    def test_binary_view_equality_expression(self):
        schema = pa.schema([("data", pa.binary_view())])
        expr = pc.field("data") == b"hello"
        vortex_expr = arrow_to_vortex(expr, schema)
        assert vortex_expr is not None

    def test_string_view_comparison_expression(self):
        schema = pa.schema([("name", pa.string_view())])
        expr = pc.field("name") > "bob"
        vortex_expr = arrow_to_vortex(expr, schema)
        assert vortex_expr is not None

    def test_mixed_view_and_regular_types(self):
        schema = pa.schema(
            [
                ("id", pa.int64()),
                ("name", pa.string_view()),
                ("data", pa.binary_view()),
            ]
        )
        expr = (pc.field("id") > 10) & (pc.field("name") == "test")
        vortex_expr = arrow_to_vortex(expr, schema)
        assert vortex_expr is not None

    @pytest.mark.parametrize(
        "view_type,value",
        [
            (pa.string_view(), "test"),
            (pa.binary_view(), b"test"),
        ],
    )
    def test_view_types_parametrized(self, view_type, value):  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        schema = pa.schema([("col", view_type)])  # pyright: ignore[reportUnknownArgumentType]
        expr = pc.field("col") == value  # pyright: ignore[reportUnknownVariableType]
        vortex_expr = arrow_to_vortex(expr, schema)  # pyright: ignore[reportUnknownArgumentType]
        assert vortex_expr is not None

    def test_null_literal_expression(self):
        schema = pa.schema([("id", pa.int64())])
        expr = pc.field("id") == pa.scalar(None, type=pa.int64())
        vortex_expr = arrow_to_vortex(expr, schema)
        assert vortex_expr is not None

    @pytest.mark.parametrize(("arrow_type", "width"), UNSIGNED_NULL_CASES)
    def test_unsigned_null_literal_expression(self, arrow_type: pa.DataType, width: UIntWidth):
        schema = pa.schema([("u", arrow_type)])
        expr = pc.field("u") == pa.scalar(None, type=arrow_type)

        actual = arrow_to_vortex(expr, schema)
        expected = ve.column("u") == ve.literal(vx.uint(width, nullable=True), None)

        assert str(actual) == str(expected)


def test_substrait_typed_null_literal():
    literal: Expression.Literal = Expression.Literal()
    literal.null.i64.nullability = Type.NULLABILITY_NULLABLE

    actual = vx_substrait.literal(literal)
    expected = ve.literal(vx.int_(64, nullable=True), None)

    assert str(actual) == str(expected)
