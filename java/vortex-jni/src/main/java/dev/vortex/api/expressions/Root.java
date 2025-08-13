// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;
import java.util.List;
import java.util.Optional;

/**
 * Represents the root expression in a Vortex expression tree.
 * This is a singleton expression that serves as the starting point for expression evaluation,
 * typically representing the root context or input data from which other expressions derive values.
 */
public final class Root implements Expression {
    /**
     * The singleton instance of the Root expression.
     * Since Root expressions have no state or parameters, a single instance is shared.
     */
    public static final Root INSTANCE = new Root();

    private Root() {}

    /**
     * Parses a Root expression from serialized metadata and child expressions.
     * This method is used during deserialization of Vortex expressions.
     *
     * @param _metadata the serialized metadata (ignored for Root expressions)
     * @param children the child expressions, must be empty for Root expressions
     * @return the singleton Root instance
     * @throws RuntimeException if any children are provided
     */
    public static Root parse(byte[] _metadata, List<Expression> children) {
        if (!children.isEmpty()) {
            throw new IllegalArgumentException("Root expression must have no children, found: " + children.size());
        }
        return INSTANCE;
    }

    @Override
    public String id() {
        return "root";
    }

    @Override
    public List<Expression> children() {
        return List.of();
    }

    @Override
    public Optional<byte[]> metadata() {
        return Optional.of(new byte[] {});
    }

    @Override
    public String toString() {
        return "$";
    }

    // equals and hashCode depend on address equality to INSTANCE.

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visitRoot(this);
    }
}
