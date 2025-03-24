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
package dev.vortex.api.expressions;

import dev.vortex.api.Expression;
import java.util.Objects;
import java.util.stream.Stream;

public final class Binary implements Expression {
    private final BinaryOp operator;
    private final Expression left;
    private final Expression right;

    private Binary(BinaryOp operator, Expression left, Expression right) {
        this.operator = operator;
        this.left = left;
        this.right = right;
    }

    public static Binary of(BinaryOp operator, Expression left, Expression right) {
        return new Binary(operator, left, right);
    }

    public static Binary and(Expression first, Expression... rest) {
        Expression rhs = Stream.of(rest).reduce(Binary::and).orElse(Literal.bool(true));
        return new Binary(BinaryOp.AND, first, rhs);
    }

    public static Binary or(Expression first, Expression... rest) {
        Expression rhs = Stream.of(rest).reduce(Binary::or).orElse(Literal.bool(false));
        return new Binary(BinaryOp.OR, first, rhs);
    }

    public static Binary eq(Expression left, Expression right) {
        return new Binary(BinaryOp.EQ, left, right);
    }

    public static Binary notEq(Expression left, Expression right) {
        return new Binary(BinaryOp.NOT_EQ, left, right);
    }

    public static Binary gt(Expression left, Expression right) {
        return new Binary(BinaryOp.GT, left, right);
    }

    public static Binary gtEq(Expression left, Expression right) {
        return new Binary(BinaryOp.GT_EQ, left, right);
    }

    public static Binary lt(Expression left, Expression right) {
        return new Binary(BinaryOp.LT, left, right);
    }

    public static Binary ltEq(Expression left, Expression right) {
        return new Binary(BinaryOp.LT_EQ, left, right);
    }

    @Override
    public String type() {
        return "binary";
    }

    @Override
    public String toString() {
        return "(" + left + " " + operator + " " + right + ")";
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        Binary binary = (Binary) o;
        return operator == binary.operator && Objects.equals(left, binary.left) && Objects.equals(right, binary.right);
    }

    @Override
    public int hashCode() {
        return Objects.hash(operator, left, right);
    }

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visitBinary(this);
    }

    public BinaryOp getOperator() {
        return operator;
    }

    public Expression getLeft() {
        return left;
    }

    public Expression getRight() {
        return right;
    }

    public enum BinaryOp {
        // comparison
        EQ,
        NOT_EQ,
        GT,
        GT_EQ,
        LT,
        LT_EQ,
        // boolean algebra
        AND,
        OR,
        ;

        @Override
        public String toString() {
            switch (this) {
                case EQ:
                    return "==";
                case NOT_EQ:
                    return "!=";
                case GT:
                    return ">";
                case GT_EQ:
                    return ">=";
                case LT:
                    return "<";
                case LT_EQ:
                    return "<=";
                case AND:
                    return "&&";
                case OR:
                    return "||";
                default:
                    throw new IllegalStateException("Unknown Operator: " + this);
            }
        }
    }
}
