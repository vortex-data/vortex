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

    public static Expression deserialize(ExprProtos.Expr expr) {
        byte[] metadata = expr.getMetadata().toByteArray();
        List<Expression> children =
                expr.getChildrenList().stream().map(Expressions::deserialize).collect(Collectors.toList());

        switch (expr.getId()) {
            case "binary":
                return Binary.parse(metadata, children);
            case "get_item":
                return GetItem.parse(metadata, children);
            case "identity":
                return Identity.parse(metadata, children);
            case "literal":
                return Literal.parse(metadata, children);
            case "not":
                return Not.parse(metadata, children);
            default:
                return new Unknown(expr.getId(), children, expr.getMetadata().toByteArray());
        }
    }

    private Expressions() {}
}
