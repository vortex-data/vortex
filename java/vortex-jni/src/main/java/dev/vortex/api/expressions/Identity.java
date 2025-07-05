// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;
import dev.vortex.proto.ExprProtos;

import java.util.List;
import java.util.Optional;

public final class Identity implements Expression {
    public static final Identity INSTANCE = new Identity();

    private Identity() {}

    public static Identity parse(byte[] metadata, List<Expression> children) {
        if (!children.isEmpty()) {
            throw new IllegalArgumentException("Identity expression must have no children, found: " + children.size());
        }
        try {
            ExprProtos.VarOpts varOpts = ExprProtos.VarOpts.parseFrom(metadata);
            if (!varOpts.getVar().isEmpty()) {
                throw new IllegalArgumentException(
                        "Identity expression must have empty var name, found: " + varOpts.getVar());
            }
            return INSTANCE;
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to parse metadata for Identity expression", e);
        }
    }

    @Override
    public String id() {
        return "var";
    }

    @Override
    public List<Expression> children() {
        return List.of();
    }

    @Override
    public Optional<byte[]> metadata() {
        // TODO(ngates): currently Vortex has a Var expression, but this will be reverted back
        //  to an Identity expression in the future.
        return Optional.of(ExprProtos.VarOpts.newBuilder()
                // Identity has empty var name
                .setVar("")
                .build()
                .toByteArray());
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
