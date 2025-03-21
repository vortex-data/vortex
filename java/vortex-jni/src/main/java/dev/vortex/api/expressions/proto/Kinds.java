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

import dev.vortex.api.expressions.*;
import dev.vortex.proto.ExprProtos;

final class Kinds {
    private Kinds() {}

    static ExprProtos.Kind identity(Identity identity) {
        return ExprProtos.Kind.newBuilder()
                .setIdentity(ExprProtos.Kind.Identity.newBuilder().build())
                .build();
    }

    static ExprProtos.Kind not(Not _not) {
        return ExprProtos.Kind.newBuilder()
                .setNot(ExprProtos.Kind.Not.newBuilder().build())
                .build();
    }

    static ExprProtos.Kind literal(Literal<?> lit) {
        return ExprProtos.Kind.newBuilder()
                .setLiteral(ExprProtos.Kind.Literal.newBuilder()
                        .setValue(lit.acceptLiteralVisitor(LiteralToScalar.INSTANCE))
                        .build())
                .build();
    }

    static ExprProtos.Kind binary(Binary binary) {
        ExprProtos.Kind.BinaryOp op;
        switch (binary.getOperator()) {
            case EQ:
                op = ExprProtos.Kind.BinaryOp.Eq;
                break;
            case NOT_EQ:
                op = ExprProtos.Kind.BinaryOp.NotEq;
                break;
            case GT:
                op = ExprProtos.Kind.BinaryOp.Gt;
                break;
            case GT_EQ:
                op = ExprProtos.Kind.BinaryOp.Gte;
                break;
            case LT:
                op = ExprProtos.Kind.BinaryOp.Lt;
                break;
            case LT_EQ:
                op = ExprProtos.Kind.BinaryOp.Lte;
                break;
            case AND:
                op = ExprProtos.Kind.BinaryOp.And;
                break;
            case OR:
                op = ExprProtos.Kind.BinaryOp.Or;
                break;
            default:
                throw new IllegalArgumentException("Unsupported binary operator: " + binary.getOperator());
        }

        return ExprProtos.Kind.newBuilder().setBinaryOp(op).build();
    }

    static ExprProtos.Kind getItem(GetItem getItem) {
        return ExprProtos.Kind.newBuilder()
                .setGetItem(ExprProtos.Kind.GetItem.newBuilder()
                        .setPath(getItem.getPath())
                        .build())
                .build();
    }
}
