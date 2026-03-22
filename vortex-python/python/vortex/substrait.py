# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from google.protobuf.internal.containers import RepeatedCompositeFieldContainer

if TYPE_CHECKING:
    from substrait.algebra_pb2 import Expression, FunctionArgument
    from substrait.extended_expression_pb2 import ExpressionReference, ExtendedExpression
    from substrait.extensions.extensions_pb2 import (
        SimpleExtensionDeclaration,
        SimpleExtensionURI,  # pyright: ignore[reportDeprecated]
    )
    from substrait.type_pb2 import NamedStruct, Type
else:
    try:
        # substrait >= 0.27
        from substrait.algebra_pb2 import Expression, FunctionArgument
        from substrait.extended_expression_pb2 import ExpressionReference, ExtendedExpression
        from substrait.extensions.extensions_pb2 import SimpleExtensionDeclaration, SimpleExtensionURI
        from substrait.type_pb2 import NamedStruct, Type
    except ImportError:
        # substrait < 0.27
        from substrait.gen.proto.algebra_pb2 import Expression, FunctionArgument
        from substrait.gen.proto.extended_expression_pb2 import ExpressionReference, ExtendedExpression
        from substrait.gen.proto.extensions.extensions_pb2 import SimpleExtensionDeclaration, SimpleExtensionURI
        from substrait.gen.proto.type_pb2 import NamedStruct, Type

from ._lib import dtype as _dtype  # pyright: ignore[reportMissingModuleSource]
from ._lib import expr as _expr  # pyright: ignore[reportMissingModuleSource]

ExtensionTypes = dict[int, str]
TypeVariations = dict[int, str]
IntWidth = Literal[8, 16, 32, 64]


def literal(substrait_object: Expression.Literal) -> _expr.Expr:
    return _literal(substrait_object, extension_types={}, type_variations={})


def _literal(
    substrait_object: Expression.Literal,
    *,
    extension_types: ExtensionTypes,
    type_variations: TypeVariations,
) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L890
    match substrait_object.WhichOneof("literal_type"):
        case "boolean":
            return _expr.literal(_dtype.bool_(nullable=False), substrait_object.boolean)
        case "i8":
            return _expr.literal(_dtype.int_(8, nullable=False), substrait_object.i8)
        case "i16":
            return _expr.literal(_dtype.int_(16, nullable=False), substrait_object.i16)
        case "i32":
            return _expr.literal(_dtype.int_(32, nullable=False), substrait_object.i32)
        case "i64":
            return _expr.literal(_dtype.int_(64, nullable=False), substrait_object.i64)
        case "fp32":
            return _expr.literal(_dtype.float_(32, nullable=False), substrait_object.fp32)
        case "fp64":
            return _expr.literal(_dtype.float_(64, nullable=False), substrait_object.fp64)
        case "decimal":
            substrait_decimal = substrait_object.decimal
            return _expr.literal(
                _dtype.decimal(precision=substrait_decimal.precision, scale=substrait_decimal.scale, nullable=False),
                int.from_bytes(substrait_decimal.value, byteorder="little", signed=True),
            )
        case "string":
            return _expr.literal(_dtype.utf8(nullable=False), substrait_object.string)
        case "binary":
            return _expr.literal(_dtype.binary(nullable=False), substrait_object.binary)
        case "timestamp":
            # The unit here is from the substrait definition
            return _expr.literal(_dtype.timestamp(unit="us", nullable=False), substrait_object.timestamp)
        case "precision_timestamp":
            unit = _precision_to_unit("precision_timestamp", substrait_object.precision_timestamp.precision)

            return _expr.literal(
                _dtype.timestamp(unit=unit, nullable=False), substrait_object.precision_timestamp.value
            )
        case "date":
            # The unit here is from the substrait definition
            return _expr.literal(_dtype.date(unit="days", nullable=False), substrait_object.date)
        case "time":
            # The unit here is from the substrait definition
            return _expr.literal(_dtype.time(unit="us", nullable=False), substrait_object.time)
        case "precision_time":
            unit = _precision_to_unit("precision_time", substrait_object.precision_time.precision)
            return _expr.literal(_dtype.time(unit=unit, nullable=False), substrait_object.precision_time.value)
        case "null":
            return _expr.literal(
                _type_to_dtype(
                    substrait_object.null,
                    force_nullable=True,
                    extension_types=extension_types,
                    type_variations=type_variations,
                ),
                None,
            )
        case "interval_year_to_month":
            raise NotImplementedError
        case "interval_day_to_second":
            raise NotImplementedError
        case "interval_compound":
            raise NotImplementedError
        case "fixed_char":
            raise NotImplementedError
        case "var_char":
            raise NotImplementedError
        case "fixed_binary":
            raise NotImplementedError
        case "precision_timestamp_tz":
            raise NotImplementedError
        case "struct":
            raise NotImplementedError
        case "map":
            raise NotImplementedError
        case "timestamp_tz":
            raise NotImplementedError
        case "uuid":
            raise NotImplementedError
        case "list":
            raise NotImplementedError
        case "empty_list":
            # substrait_object.empty_list is a Type which needs to be converted
            raise NotImplementedError
        case "empty_map":
            # substrait_object.empty_map is a Type which needs to be converted
            raise NotImplementedError
        case "user_defined":
            raise NotImplementedError
        case literal_type:
            raise ValueError(f"unknown literal_type {literal_type}")


def _type_to_dtype(
    substrait_type: Type,
    *,
    force_nullable: bool = False,
    extension_types: ExtensionTypes | None = None,
    type_variations: TypeVariations | None = None,
) -> _dtype.DType:
    extension_types = extension_types or {}
    type_variations = type_variations or {}

    match substrait_type.WhichOneof("kind"):
        case "bool":
            return _dtype.bool_(nullable=_nullability(substrait_type.bool.nullability, force_nullable=force_nullable))
        case "i8":
            return _integer_dtype(
                8,
                substrait_type.i8.type_variation_reference,
                substrait_type.i8.nullability,
                force_nullable=force_nullable,
                type_variations=type_variations,
            )
        case "i16":
            return _integer_dtype(
                16,
                substrait_type.i16.type_variation_reference,
                substrait_type.i16.nullability,
                force_nullable=force_nullable,
                type_variations=type_variations,
            )
        case "i32":
            return _integer_dtype(
                32,
                substrait_type.i32.type_variation_reference,
                substrait_type.i32.nullability,
                force_nullable=force_nullable,
                type_variations=type_variations,
            )
        case "i64":
            return _integer_dtype(
                64,
                substrait_type.i64.type_variation_reference,
                substrait_type.i64.nullability,
                force_nullable=force_nullable,
                type_variations=type_variations,
            )
        case "fp32":
            return _dtype.float_(
                32,
                nullable=_nullability(substrait_type.fp32.nullability, force_nullable=force_nullable),
            )
        case "fp64":
            return _dtype.float_(
                64,
                nullable=_nullability(substrait_type.fp64.nullability, force_nullable=force_nullable),
            )
        case "string":
            return _dtype.utf8(nullable=_nullability(substrait_type.string.nullability, force_nullable=force_nullable))
        case "binary":
            return _dtype.binary(
                nullable=_nullability(substrait_type.binary.nullability, force_nullable=force_nullable)
            )
        case "decimal":
            return _dtype.decimal(
                precision=substrait_type.decimal.precision,
                scale=substrait_type.decimal.scale,
                nullable=_nullability(substrait_type.decimal.nullability, force_nullable=force_nullable),
            )
        case "date":
            return _dtype.date(
                unit="days",
                nullable=_nullability(substrait_type.date.nullability, force_nullable=force_nullable),
            )
        case "time":
            return _dtype.time(
                unit="us",
                nullable=_nullability(substrait_type.time.nullability, force_nullable=force_nullable),
            )
        case "precision_time":
            return _dtype.time(
                unit=_precision_to_unit("precision_time", substrait_type.precision_time.precision),
                nullable=_nullability(
                    substrait_type.precision_time.nullability,
                    force_nullable=force_nullable,
                ),
            )
        case "timestamp":
            return _dtype.timestamp(
                unit="us",
                nullable=_nullability(substrait_type.timestamp.nullability, force_nullable=force_nullable),
            )
        case "precision_timestamp":
            return _dtype.timestamp(
                unit=_precision_to_unit("precision_timestamp", substrait_type.precision_timestamp.precision),
                nullable=_nullability(
                    substrait_type.precision_timestamp.nullability,
                    force_nullable=force_nullable,
                ),
            )
        case "timestamp_tz":
            return _dtype.timestamp(
                unit="us",
                tz="UTC",
                nullable=_nullability(
                    substrait_type.timestamp_tz.nullability,
                    force_nullable=force_nullable,
                ),
            )
        case "precision_timestamp_tz":
            return _dtype.timestamp(
                unit=_precision_to_unit(
                    "precision_timestamp_tz",
                    substrait_type.precision_timestamp_tz.precision,
                ),
                tz="UTC",
                nullable=_nullability(
                    substrait_type.precision_timestamp_tz.nullability,
                    force_nullable=force_nullable,
                ),
            )
        case "list":
            return _dtype.list_(
                _type_to_dtype(
                    substrait_type.list.type,
                    extension_types=extension_types,
                    type_variations=type_variations,
                ),
                nullable=_nullability(substrait_type.list.nullability, force_nullable=force_nullable),
            )
        case "user_defined":
            return _user_defined_dtype(
                substrait_type.user_defined.type_reference,
                substrait_type.user_defined.type_variation_reference,
                substrait_type.user_defined.nullability,
                force_nullable=force_nullable,
                extension_types=extension_types,
                type_variations=type_variations,
            )
        case "struct":
            raise NotImplementedError("Substrait null struct literals are not supported because field names are absent")
        case "map":
            raise NotImplementedError("Substrait map types are not supported")
        case kind:
            raise NotImplementedError(f"Substrait type {kind} is not supported")


def _integer_dtype(
    width: IntWidth,
    type_variation_reference: int,
    nullability: int,
    *,
    force_nullable: bool,
    type_variations: TypeVariations,
) -> _dtype.DType:
    nullable = _nullability(nullability, force_nullable=force_nullable)
    if _is_unsigned_integer_name(type_variations.get(type_variation_reference), width):
        return _dtype.uint(width, nullable=nullable)
    return _dtype.int_(width, nullable=nullable)


def _user_defined_dtype(
    type_reference: int,
    type_variation_reference: int,
    nullability: int,
    *,
    force_nullable: bool,
    extension_types: ExtensionTypes,
    type_variations: TypeVariations,
) -> _dtype.DType:
    nullable = _nullability(nullability, force_nullable=force_nullable)

    # PyArrow's `to_substrait()` currently encodes unsigned integers as Arrow-defined
    # user-defined types named `u8`/`u16`/`u32`/`u64`, even though core Substrait does not
    # define unsigned simple types:
    # https://github.com/apache/arrow/blob/main/format/substrait/extension_types.yaml
    extension_name = extension_types.get(type_reference)
    match extension_name:
        case "u8":
            return _dtype.uint(8, nullable=nullable)
        case "u16":
            return _dtype.uint(16, nullable=nullable)
        case "u32":
            return _dtype.uint(32, nullable=nullable)
        case "u64":
            return _dtype.uint(64, nullable=nullable)
        case _:
            pass

    variation_name = type_variations.get(type_variation_reference)
    match variation_name:
        case "u8":
            return _dtype.uint(8, nullable=nullable)
        case "u16":
            return _dtype.uint(16, nullable=nullable)
        case "u32":
            return _dtype.uint(32, nullable=nullable)
        case "u64":
            return _dtype.uint(64, nullable=nullable)
        case _:
            pass

    raise NotImplementedError(f"Substrait user-defined type {extension_name or type_reference} is not supported")


def _is_unsigned_integer_name(name: str | None, width: IntWidth | None = None) -> bool:
    if name is None:
        return False

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if width is not None and normalized in {f"u{width}", f"uint{width}", f"unsigned_{width}"}:
        return True
    return normalized.startswith("unsigned")


def _nullability(nullability: int, *, force_nullable: bool = False) -> bool:
    if force_nullable:
        return True
    match nullability:
        case 1:
            return True
        case 2:
            return False
        case other:
            raise ValueError(f"Unknown Substrait nullability {other}")


def _precision_to_unit(type_: str, p: int) -> Literal["s", "ms", "us", "ns"]:
    match p:
        case 0:
            return "s"
        case 3:
            return "ms"
        case 6:
            return "us"
        case 9:
            return "ns"
        case other:
            raise ValueError(f"{type_} with a precision of {other} is not supported with Vortex")


def field_reference(substrait_object: Expression.FieldReference, schema: NamedStruct) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L1415
    match substrait_object.WhichOneof("reference_type"):
        case "direct_reference":
            segments = reference_segment(substrait_object.direct_reference)
            if len(segments) == 0 or len(segments) > 1:
                raise NotImplementedError
            arrow_field_name = schema.names[segments[0]]
            return _expr.column(arrow_field_name)
        case "masked_reference":
            raise NotImplementedError
        case reference_type:
            raise ValueError(f"unknown reference_type {reference_type}")


def reference_segment(substrait_object: Expression.ReferenceSegment) -> list[int]:
    # NB: The field ids are returned in reverse order i.e. [deepest, next_deepest, ..., top_level]
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L1312
    match substrait_object.WhichOneof("reference_type"):
        case "map_key":
            raise NotImplementedError
        case "struct_field":
            return struct_field(substrait_object.struct_field)
        case "list_element":
            raise NotImplementedError
        case reference_type:
            raise ValueError(f"unknown reference_type {reference_type}")


def struct_field(substrait_object: Expression.ReferenceSegment.StructField) -> list[int]:
    if substrait_object.HasField("child"):
        segment = reference_segment(substrait_object.child)
        segment.append(substrait_object.field)
        return segment
    else:
        return [substrait_object.field]


def scalar_function(
    substrait_object: Expression.ScalarFunction,
    functions: list[Callable[..., _expr.Expr]],
    schema: NamedStruct,
    *,
    extension_types: ExtensionTypes,
    type_variations: TypeVariations,
) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extensions/extensions.proto#L57
    function = functions[substrait_object.function_reference]
    arguments = [
        function_argument(
            argument,
            functions,
            schema,
            extension_types=extension_types,
            type_variations=type_variations,
        )
        for argument in substrait_object.arguments
    ]
    return function(*arguments)


def function_argument(
    substrait_object: FunctionArgument,
    functions: list[Callable[..., _expr.Expr]],
    schema: NamedStruct,
    *,
    extension_types: ExtensionTypes,
    type_variations: TypeVariations,
) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L832
    match substrait_object.WhichOneof("arg_type"):
        case "enum":
            raise NotImplementedError
        case "type":
            raise NotImplementedError
        case "value":
            return expression(
                substrait_object.value,
                functions,
                schema,
                extension_types=extension_types,
                type_variations=type_variations,
            )
        case arg_type:
            raise ValueError(f"unknown arg_type {arg_type}")


def extension_function(
    substrait_object: SimpleExtensionDeclaration.ExtensionFunction,
    extension_uris: RepeatedCompositeFieldContainer[SimpleExtensionURI],  # pyright: ignore[reportDeprecated]
) -> Callable[..., _expr.Expr]:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extensions/extensions.proto#L57
    match extension_uris[substrait_object.extension_uri_reference].uri:
        case "https://github.com/substrait-io/substrait/blob/main/extensions/functions_boolean.yaml":
            match substrait_object.name:
                case "or":
                    return operator.or_
                case "and":
                    return operator.and_
                case "xor":
                    return operator.__xor__
                case "not":
                    return _expr.not_
                case name:
                    raise NotImplementedError(f"Function name {name} not supported")
        case "https://github.com/substrait-io/substrait/blob/main/extensions/functions_comparison.yaml":
            match substrait_object.name:
                case "equal":
                    return operator.__eq__
                case "not_equal":
                    return operator.__ne__
                case "lt":
                    return operator.__lt__
                case "lte":
                    return operator.__le__
                case "gt":
                    return operator.__gt__
                case "gte":
                    return operator.__ge__
                case "is_null":
                    return _expr.is_null
                case "is_not_null":
                    return _is_not_null
                case name:
                    raise NotImplementedError(f"Function name {name} not supported")
        case "https://github.com/substrait-io/substrait/blob/main/extensions/functions_arithmetic.yaml":
            match substrait_object.name:
                case "add":
                    return operator.__add__
                case "subtract":
                    return operator.__sub__
                case "multiply":
                    return operator.__mul__
                case "divide":
                    return operator.__truediv__
                case name:
                    raise NotImplementedError(f"Arithmetic function {name} not supported")
        case uri:
            raise NotImplementedError(f"Extension URI {uri} not supported")


def _is_not_null(e: _expr.Expr) -> _expr.Expr:
    """
    Helper function to have a well-typed callable to return
    """
    return _expr.not_(_expr.is_null(e))


def expression(
    substrait_object: Expression,
    functions: list[Callable[..., _expr.Expr]],
    schema: NamedStruct,
    *,
    extension_types: ExtensionTypes,
    type_variations: TypeVariations,
) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L857
    match substrait_object.WhichOneof("rex_type"):
        case "literal":
            return _literal(
                substrait_object.literal,
                extension_types=extension_types,
                type_variations=type_variations,
            )
        case "selection":
            return field_reference(substrait_object.selection, schema)
        case "scalar_function":
            return scalar_function(
                substrait_object.scalar_function,
                functions,
                schema,
                extension_types=extension_types,
                type_variations=type_variations,
            )
        case "window_function":
            raise NotImplementedError
        case "if_then":
            raise NotImplementedError
        case "cast":
            raise NotImplementedError
        case "subquery":
            raise NotImplementedError
        case "nested":
            raise NotImplementedError
        case rex_type:
            raise ValueError(f"unknown rex_type {rex_type}")


def expression_reference(
    substrait_object: ExpressionReference,
    functions: list[Callable[..., _expr.Expr]],
    schema: NamedStruct,
    *,
    extension_types: ExtensionTypes,
    type_variations: TypeVariations,
) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extended__expression.proto#L16
    match substrait_object.WhichOneof("expr_type"):
        case "expression":
            return expression(
                substrait_object.expression,
                functions,
                schema,
                extension_types=extension_types,
                type_variations=type_variations,
            )
        case _:
            raise ValueError("unknown expr_type: {}")


def extended_expression(substrait_object: ExtendedExpression) -> list[_expr.Expr]:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extended__expression.proto#L27
    functions: list[Callable[..., _expr.Expr]] = []
    extension_types: ExtensionTypes = {}
    type_variations: TypeVariations = {}

    substrait_schema = substrait_object.base_schema
    extension_uris = substrait_object.extension_uris
    extensions = substrait_object.extensions
    expressions = substrait_object.referred_expr

    for extension in extensions:
        # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extensions/extensions.proto#L25
        match extension.WhichOneof("mapping_type"):
            case "extension_type":
                extension_types[extension.extension_type.type_anchor] = extension.extension_type.name
                # Arrow can declare extension-backed types like `null` in the schema even when the
                # expression only references functions. Those declarations do not participate in
                # function_reference indexing.
                continue
            case "extension_type_variation":
                type_variations[extension.extension_type_variation.type_variation_anchor] = (
                    extension.extension_type_variation.name
                )
                continue
            case "extension_function":
                functions.append(extension_function(extension.extension_function, extension_uris))
            case mapping_type:
                raise ValueError(f"unsupported extension mapping_type {mapping_type}")

    return [
        expression_reference(
            expression,
            functions,
            substrait_schema,
            extension_types=extension_types,
            type_variations=type_variations,
        )
        for expression in expressions
    ]
