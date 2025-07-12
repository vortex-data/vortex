// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;

import java.util.List;
import java.util.Optional;

/**
 * Represents a generic expression deserialized from a Vortex expression without a concrete Java type.
 */
public final class Unknown implements Expression {
    private final String id;
    private final List<Expression> children;
    private final byte[] metadata;

    public Unknown(String id, List<Expression> children, byte[] metadata) {
        this.id = id;
        this.children = children;
        this.metadata = metadata;
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public List<Expression> children() {
        return children;
    }

    @Override
    public Optional<byte[]> metadata() {
        return Optional.of(metadata);
    }
}
