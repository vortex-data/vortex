// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;

import java.util.List;
import java.util.Optional;

public final class Identity implements Expression {
    public static final Identity INSTANCE = new Identity();

    private Identity() {}

    public static Identity parse(byte[] metadata, List<Expression> children) {
        if (!children.isEmpty()) {
            throw new IllegalArgumentException("Identity expression must have no children, found: " + children.size());
        }
        if (metadata.length > 0) {
            throw new IllegalArgumentException("Identity expression must not have metadata, found: " + metadata.length);
        }
        return INSTANCE;
    }

    @Override
    public String id() {
        return "identity";
    }

    @Override
    public List<Expression> children() {
        return List.of();
    }

    @Override
    public Optional<byte[]> metadata() {
        return Optional.of(new byte[] {}); // No metadata, but still serializable
    }

    @Override
    public String toString() {
        return "identity";
    }

    // equals and hashCode depend on address equality to INSTANCE.

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visitIdentity(this);
    }
}
