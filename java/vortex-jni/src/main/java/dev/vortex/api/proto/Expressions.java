// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.proto;

import com.google.protobuf.ByteString;
import dev.vortex.api.Expression;
import dev.vortex.api.expressions.*;
import dev.vortex.proto.ExprProtos;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Generate a protocol buffers representation of an {@link Expression}.
 */
public final class Expressions {
    /**
     * Serialize an {@link Expression} to a protocol buffer.
     */
    public static ExprProtos.Expr serialize(Expression expression) {
        ByteString metadata = ByteString.copyFrom(expression
                .metadata()
                .orElseThrow(() -> new IllegalArgumentException("Expression is not serializable: " + expression.id())));

        return ExprProtos.Expr.newBuilder()
                .setId(expression.id())
                .addAllChildren(expression.children().stream()
                        .map(Expressions::serialize)
                        .collect(Collectors.toList()))
                .setMetadata(metadata)
                .build();
    }

    /**
     * Deserialize a protocol buffer representation back into an {@link Expression} object.
     * The method examines the expression ID and creates the appropriate concrete expression type
     * based on the registered expression types (binary, get_item, root, literal, not).
     * If the expression ID is not recognized, an {@link Unknown} expression is created.
     *
     * @param expr the protocol buffer expression to deserialize
     * @return the deserialized Expression object
     */
    public static Expression deserialize(ExprProtos.Expr expr) {
        byte[] metadata = expr.getMetadata().toByteArray();
        List<Expression> children =
                expr.getChildrenList().stream().map(Expressions::deserialize).collect(Collectors.toList());

        switch (expr.getId()) {
            case "vortex.binary":
                return Binary.parse(metadata, children);
            case "vortex.get_item":
                return GetItem.parse(metadata, children);
            case "vortex.root":
                return Root.parse(metadata, children);
            case "vortex.literal":
                return Literal.parse(metadata, children);
            case "vortex.not":
                return Not.parse(metadata, children);
            case "vortex.is_null":
                return IsNull.parse(metadata, children);
            default:
                return new Unknown(expr.getId(), children, expr.getMetadata().toByteArray());
        }
    }

    private Expressions() {}
}
