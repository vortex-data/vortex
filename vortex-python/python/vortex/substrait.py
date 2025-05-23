import operator
from collections.abc import Callable

import substrait
from substrait.proto import ExtendedExpression, NamedStruct

import vortex as vx
from vortex._lib import expr as _expr


def literal(substrait_object) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L890
    match substrait_object.WhichOneof("literal_type"):
        case "boolean":
            return _expr.literal(vx.bool_(nullable=False), substrait_object.boolean)
        case "i8":
            return _expr.literal(vx.int_(8, nullable=False), substrait_object.i8)
        case "i16":
            return _expr.literal(vx.int_(16, nullable=False), substrait_object.i16)
        case "i32":
            return _expr.literal(vx.int_(32, nullable=False), substrait_object.i32)
        case "i64":
            return _expr.literal(vx.int_(64, nullable=False), substrait_object.i64)
        case "fp32":
            return _expr.literal(vx.float_(32, nullable=False), substrait_object.fp32)
        case "fp64":
            return _expr.literal(vx.float_(64, nullable=False), substrait_object.fp64)
        case "string":
            return _expr.literal(vx.utf8(nullable=False), substrait_object.string)
        case "binary":
            return _expr.literal(vx.binary(nullable=False), substrait_object.binary)
        case "timestamp":
            raise NotImplementedError
        case "date":
            raise NotImplementedError
        case "time":
            raise NotImplementedError
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
        case "decimal":
            raise NotImplementedError
        case "precision_timestamp":
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
        case "null":
            # substrait_object.null is a Type which needs to be converted
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


def field_reference(substrait_object, schema: NamedStruct) -> _expr.Expr:
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


def reference_segment(substrait_object) -> list[int]:
    # NB: The field ids are returned in reverse order i.e. [deepest, next_deepest, ..., top_level]
    #
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


def struct_field(substrait_object) -> list[int]:
    if substrait_object.HasField("child"):
        segment = reference_segment(substrait_object.child)
        segment.append(substrait_object.field)
        return segment
    else:
        return [substrait_object.field]


def scalar_function(substrait_object, functions: list[Callable[..., _expr.Expr]], schema: NamedStruct) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extensions/extensions.proto#L57
    function = functions[substrait_object.function_reference]
    if len(substrait_object.options) != 0:
        raise NotImplementedError(substrait_object.options)
    arguments = [function_argument(argument, functions, schema) for argument in substrait_object.arguments]
    return function(*arguments)


def function_argument(substrait_object, functions: list[Callable[..., _expr.Expr]], schema: NamedStruct) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L832
    match substrait_object.WhichOneof("arg_type"):
        case "enum":
            raise NotImplementedError
        case "type":
            raise NotImplementedError
        case "value":
            return expression(substrait_object.value, functions, schema)
        case arg_type:
            raise ValueError(f"unknown arg_type {arg_type}")


def extension_function(
    substrait_object, extension_uris: list["substrait.proto.extensions.SimpleExtensionURI"]
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
                    return operator.__not__
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
                    raise NotImplementedError
                case "is_not_null":
                    raise NotImplementedError
                case name:
                    raise NotImplementedError(f"Function name {name} not supported")
        case uri:
            raise NotImplementedError(f"Extension URI {uri} not supported")


def expression(substrait_object, functions: list[Callable[..., _expr.Expr]], schema: NamedStruct) -> _expr.Expr:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto#L857
    match substrait_object.WhichOneof("rex_type"):
        case "literal":
            return literal(substrait_object.literal)
        case "selection":
            return field_reference(substrait_object.selection, schema)
        case "scalar_function":
            return scalar_function(substrait_object.scalar_function, functions, schema)
        case "window_function":
            raise NotImplementedError
        case "if_then":
            raise NotImplementedError
        case "switch":
            raise NotImplementedError
        case "singular":
            raise NotImplementedError
        case "multi":
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
    substrait_object, functions: list[Callable[..., _expr.Expr]], schema: NamedStruct
) -> _expr.Expr:
    print(substrait_object)
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extended__expression.proto#L16
    match substrait_object.WhichOneof("expr_type"):
        case "expression":
            return expression(substrait_object.expression, functions, schema)
        case _:
            raise ValueError("unknown expr_type: {}")


def extended_expression(substrait_object: ExtendedExpression) -> list[_expr.Expr]:
    # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extended__expression.proto#L27
    functions = []

    substrait_schema = substrait_object.base_schema
    extension_uris = substrait_object.extension_uris
    extensions = substrait_object.extensions
    expressions = substrait_object.referred_expr

    for extension in extensions:
        # https://github.com/substrait-io/substrait/blob/main/proto/substrait/extensions/extensions.proto#L25
        match extension.WhichOneof("mapping_type"):
            case "extension_function":
                functions.append(extension_function(extension.extension_function, extension_uris))
            case mapping_type:
                raise ValueError(f"unsupported extension mapping_type {mapping_type}")

    return [expression_reference(expression, functions, substrait_schema) for expression in expressions]
