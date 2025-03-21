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

import com.google.common.collect.Iterables;
import dev.vortex.api.Expression;
import dev.vortex.api.expressions.*;
import dev.vortex.proto.DTypeProtos;
import dev.vortex.proto.ExprProtos;
import dev.vortex.proto.ScalarProtos;
import java.util.List;

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

        ScalarProtos.ScalarValue scalarValue = literalScalar.getValue();

        switch (scalarValue.getKindCase()) {
            case NULL_VALUE:
                return nullLiteral(dtype);
            case BOOL_VALUE:
                return Literal.bool(scalarValue.getBoolValue());
            case INT8_VALUE:
                return Literal.int8(Casts.toByte(scalarValue.getInt8Value()));
            case INT16_VALUE:
                return Literal.int16(Casts.toShort(scalarValue.getInt16Value()));
            case INT32_VALUE:
                return Literal.int32(scalarValue.getInt32Value());
            case INT64_VALUE:
                return Literal.int64(scalarValue.getInt64Value());
            case UINT8_VALUE:
                return Literal.int8(Casts.toUnsignedByte(scalarValue.getUint8Value()));
            case UINT16_VALUE:
                return Literal.int16(Casts.toShort(scalarValue.getUint16Value()));
            case UINT32_VALUE:
                return Literal.int32(scalarValue.getUint32Value());
            case UINT64_VALUE:
                return Literal.int64(scalarValue.getUint64Value());
            case F16_VALUE:
                throw new IllegalArgumentException("F16 is not supported yet");
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
                // TODO(aduffy): fix timestamps/dates support
                // TODO(aduffy): struct/list support
            default:
                throw new UnsupportedOperationException("Unsupported ScalarValue type encountered: " + type);
        }
    }
}
