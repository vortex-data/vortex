// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Represents a logical NOT expression that negates the boolean result of its child expression.
 * This expression applies the logical NOT operation to the result of evaluating its single child expression.
 */
public final class Not implements Expression {
    private final Expression child;

    private Not(Expression child) {
        this.child = child;
    }

    /**
     * Parses a Not expression from serialized metadata and child expressions.
     * This method is used during deserialization of Vortex expressions.
     *
     * @param metadata the serialized metadata, must be empty for Not expressions
     * @param children the child expressions, must contain exactly one element
     * @return a new Not expression parsed from the provided data
     * @throws RuntimeException if the number of children is not exactly one,
     *                                  or if metadata is not empty
     */
    public static Not parse(byte[] metadata, List<Expression> children) {
        if (children.size() != 1) {
            throw new IllegalArgumentException("Not expression must have exactly one child, found: " + children.size());
        }
        if (metadata.length > 0) {
            throw new IllegalArgumentException("Not expression must not have metadata, found: " + metadata.length);
        }
        return new Not(children.get(0));
    }

    /**
     * Creates a new Not expression that negates the given child expression.
     *
     * @param child the expression to negate
     * @return a new Not expression
     */
    public static Not of(Expression child) {
        return new Not(child);
    }

    @Override
    public boolean equals(Object o) {
        if (o == null || getClass() != o.getClass()) return false;
        Not other = (Not) o;
        return Objects.equals(child, other.child);
    }

    @Override
    public int hashCode() {
        return Objects.hash(child);
    }

    @Override
    public String id() {
        return "not";
    }

    @Override
    public List<Expression> children() {
        return List.of(child);
    }

    @Override
    public Optional<byte[]> metadata() {
        return Optional.of(new byte[] {});
    }

    @Override
    public String toString() {
        return "not(" + child + ")";
    }

    /**
     * Returns the child expression that will be negated by this Not expression.
     *
     * @return the child expression
     */
    public Expression getChild() {
        return child;
    }

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visitNot(this);
    }
}
