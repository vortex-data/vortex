/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.api.expressions.proto;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import dev.vortex.api.Expression;
import dev.vortex.api.expressions.*;
import dev.vortex.proto.DTypeProtos;
import dev.vortex.proto.ExprProtos;
import dev.vortex.proto.ScalarProtos;
import java.util.List;
import java.util.Optional;

public final class ExpressionProtoDeserializer {
    private ExpressionProtoDeserializer() {}

    public static Expression deserialize(ExprProtos.Expr expr) {
        switch (expr.getKind().getKindCase()) {
            case LITERAL:
                return deserializeLiteral(expr.getKind().getLiteral(), expr.getChildrenList());
            case BINARY_OP:
                return deserializeBinaryOp(expr.getKind().getBinaryOp(), expr.getChildrenList());
            case IDENTITY:
                return deserializeIdentity(expr.getKind().getIdentity());
            case NOT:
                return deserializeNot(expr.getKind().getNot(), expr.getChildrenList());
            case GET_ITEM:
                return deserializeGetItem(expr.getKind().getGetItem(), expr.getChildrenList());
            default:
                throw new UnsupportedOperationException("Unsupported expression type encountered: "
                        + expr.getKind().getKindCase());
        }
    }

    private static Expression deserializeIdentity(ExprProtos.Kind.Identity identity) {
        return Identity.INSTANCE;
    }

    private static Expression deserializeBinaryOp(ExprProtos.Kind.BinaryOp binaryOp, List<ExprProtos.Expr> children) {
        switch (binaryOp) {
            case Eq: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.eq(left, right);
            }
            case NotEq: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.notEq(left, right);
            }
            case Gt: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.gt(left, right);
            }
            case Gte: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.gtEq(left, right);
            }
            case Lt: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.lt(left, right);
            }
            case Lte: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.ltEq(left, right);
            }
            case And: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.and(left, right);
            }
            case Or: {
                Expression left = deserialize(children.get(0));
                Expression right = deserialize(children.get(1));
                return Binary.or(left, right);
            }
            default:
                throw new UnsupportedOperationException("Unsupported BinaryOp encountered: " + binaryOp);
        }
    }

    private static Expression deserializeLiteral(ExprProtos.Kind.Literal literal, List<ExprProtos.Expr> children) {

        ScalarProtos.Scalar literalScalar = literal.getValue();

        DTypeProtos.DType dtype = literalScalar.getDtype();

        // Special handling of extension types
        if (dtype.hasExtension()) {
            return deserializeExtensionLiteral(literal);
        }

        ScalarProtos.ScalarValue scalarValue = literalScalar.getValue();

        switch (scalarValue.getKindCase()) {
            case NULL_VALUE:
                return nullLiteral(dtype);
            case BOOL_VALUE:
                return Literal.bool(scalarValue.getBoolValue());
            case INT64_VALUE:
                return Literal.int64(scalarValue.getInt64Value());
            case UINT64_VALUE:
                return Literal.int64(scalarValue.getUint64Value());
            case F32_VALUE:
                return Literal.float32(scalarValue.getF32Value());
            case F64_VALUE:
                return Literal.float64(scalarValue.getF64Value());
            case STRING_VALUE:
                return Literal.string(scalarValue.getStringValue());
            case BYTES_VALUE:
                return Literal.bytes(scalarValue.getBytesValue().toByteArray());
            default:
                throw new UnsupportedOperationException("Unsupported ScalarValue type encountered: " + scalarValue);
        }
    }

    private static Expression deserializeExtensionLiteral(ExprProtos.Kind.Literal literal) {
        ScalarProtos.Scalar scalar = literal.getValue();
        DTypeProtos.DType scalarType = scalar.getDtype();

        Preconditions.checkArgument(scalarType.hasExtension());

        DTypeProtos.Extension extType = scalarType.getExtension();
        String extId = scalarType.getExtension().getId();

        switch (extId) {
            case "vortex.time": {
                byte timeUnit =
                        TemporalMetadatas.getTimeUnit(extType.getMetadata().toByteArray());
                if (timeUnit == TemporalMetadatas.TIME_UNIT_SECONDS) {
                    return Literal.timeSeconds(Math.toIntExact(scalar.getValue().getInt64Value()));
                } else if (timeUnit == TemporalMetadatas.TIME_UNIT_MILLIS) {
                    return Literal.timeMillis(Math.toIntExact(scalar.getValue().getInt64Value()));
                } else if (timeUnit == TemporalMetadatas.TIME_UNIT_MICROS) {
                    return Literal.timeMicros(scalar.getValue().getInt64Value());
                } else if (timeUnit == TemporalMetadatas.TIME_UNIT_NANOS) {
                    return Literal.timeNanos(scalar.getValue().getInt64Value());
                } else {
                    throw new UnsupportedOperationException("Unsupported TIME time unit: " + timeUnit);
                }
            }
            case "vortex.date": {
                byte timeUnit =
                        TemporalMetadatas.getTimeUnit(extType.getMetadata().toByteArray());
                if (timeUnit == TemporalMetadatas.TIME_UNIT_DAYS) {
                    return Literal.dateDays(Math.toIntExact(scalar.getValue().getInt64Value()));
                } else if (timeUnit == TemporalMetadatas.TIME_UNIT_MILLIS) {
                    return Literal.dateMillis(scalar.getValue().getInt64Value());
                } else {
                    throw new UnsupportedOperationException("Unsupported DATE time unit: " + timeUnit);
                }
            }
            case "vortex.timestamp": {
                byte timeUnit =
                        TemporalMetadatas.getTimeUnit(extType.getMetadata().toByteArray());
                Optional<String> timeZone =
                        TemporalMetadatas.getTimeZone(extType.getMetadata().toByteArray());
                if (timeUnit == TemporalMetadatas.TIME_UNIT_MILLIS) {
                    return Literal.timestampMillis(scalar.getValue().getInt64Value(), timeZone);
                } else if (timeUnit == TemporalMetadatas.TIME_UNIT_MICROS) {
                    return Literal.timestampMicros(scalar.getValue().getInt64Value(), timeZone);
                } else if (timeUnit == TemporalMetadatas.TIME_UNIT_NANOS) {
                    return Literal.timestampNanos(scalar.getValue().getInt64Value(), timeZone);
                } else {
                    throw new UnsupportedOperationException("Unsupported TIMESTAMP time unit: " + timeUnit);
                }
            }
            default:
                throw new UnsupportedOperationException("Unsupported extension type: " + extId);
        }
    }

    private static Expression deserializeNot(ExprProtos.Kind.Not not, List<ExprProtos.Expr> children) {
        ExprProtos.Expr child = Iterables.getOnlyElement(children);
        Expression childExpr = deserialize(child);

        return Not.of(childExpr);
    }

    private static Expression deserializeGetItem(ExprProtos.Kind.GetItem getItem, List<ExprProtos.Expr> children) {
        ExprProtos.Expr child = Iterables.getOnlyElement(children);
        Expression childExpr = deserialize(child);

        return GetItem.of(childExpr, getItem.getPath());
    }

    private static Literal<?> nullLiteral(DTypeProtos.DType type) {
        switch (type.getDtypeTypeCase()) {
            case NULL:
                return Literal.nullLit();
            case BOOL:
                return Literal.bool(null);
            case PRIMITIVE:
                switch (type.getPrimitive().getType()) {
                    case U8:
                    case I8:
                        return Literal.int8(null);
                    case U16:
                    case I16:
                        return Literal.int16(null);
                    case U32:
                    case I32:
                        return Literal.int32(null);
                    case U64:
                    case I64:
                        return Literal.int64(null);
                    case F32:
                        return Literal.float32(null);
                    case F64:
                        return Literal.float64(null);
                    default:
                        throw new UnsupportedOperationException("Unsupported ScalarValue type encountered: " + type);
                }
            case UTF8:
                return Literal.string(null);
            case BINARY:
                return Literal.bytes(null);
            default:
                throw new UnsupportedOperationException("Unsupported ScalarValue type encountered: " + type);
        }
    }
}
