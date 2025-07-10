// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;

import java.util.List;
import java.util.Optional;

public final class Root implements Expression {
    public static final Root INSTANCE = new Root();

    private Root() {}

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
