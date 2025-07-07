// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;
import java.util.Objects;

public final class Not implements Expression {
    private final Expression child;

    private Not(Expression child) {
        this.child = child;
    }

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
    public String type() {
        return "not";
    }

    @Override
    public String toString() {
        return "not(" + child + ")";
    }

    public Expression getChild() {
        return child;
    }

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visitNot(this);
    }
}
