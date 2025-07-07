// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions.proto;

import dev.vortex.api.Expression;
import dev.vortex.api.expressions.*;
import dev.vortex.proto.ExprProtos;

/**
 * Generate a protocol buffers representation of an {@link Expression}.
 */
public final class ExpressionProtoSerializer implements Expression.Visitor<ExprProtos.Expr> {
    public static final ExpressionProtoSerializer INSTANCE = new ExpressionProtoSerializer();

    private ExpressionProtoSerializer() {}

    /**
     * Serialize an {@link Expression} to a protocol buffer.
     */
    public static ExprProtos.Expr serialize(Expression expression) {
        return expression.accept(INSTANCE);
    }

    @Override
    public ExprProtos.Expr visitLiteral(Literal<?> literal) {
        return ExprProtos.Expr.newBuilder()
                .setId(literal.type())
                .setKind(Kinds.literal(literal))
                .build();
    }

    @Override
    public ExprProtos.Expr visitIdentity(Identity identity) {
        return ExprProtos.Expr.newBuilder()
                .setId(identity.type())
                .setKind(Kinds.identity(identity))
                .build();
    }

    @Override
    public ExprProtos.Expr visitBinary(Binary binary) {
        ExprProtos.Expr.Builder builder =
                ExprProtos.Expr.newBuilder().setId(binary.type()).setKind(Kinds.binary(binary));

        ExprProtos.Expr leftChild = serialize(binary.getLeft());
        ExprProtos.Expr rightChild = serialize(binary.getRight());

        builder.addChildren(leftChild);
        builder.addChildren(rightChild);

        return builder.build();
    }

    @Override
    public ExprProtos.Expr visitNot(Not not) {
        ExprProtos.Expr child = serialize(not.getChild());

        return ExprProtos.Expr.newBuilder()
                .setId(not.type())
                .setKind(Kinds.not(not))
                .addChildren(child)
                .build();
    }

    @Override
    public ExprProtos.Expr visitGetItem(GetItem getItem) {
        ExprProtos.Expr child = serialize(getItem.getChild());

        return ExprProtos.Expr.newBuilder()
                .setId(getItem.type())
                .setKind(Kinds.getItem(getItem))
                .addChildren(child)
                .build();
    }
}
