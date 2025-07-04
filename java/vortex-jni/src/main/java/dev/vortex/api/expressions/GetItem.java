// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import com.google.protobuf.InvalidProtocolBufferException;
import dev.vortex.api.Expression;
import dev.vortex.proto.ExprProtos;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

public final class GetItem implements Expression {
    private final String path;
    private final Expression child;

    private GetItem(Expression child, String path) {
        this.child = child;
        this.path = path;
    }

    public static GetItem of(Expression child, String path) {
        return new GetItem(child, path);
    }

    public static GetItem parse(byte[] metadata, List<Expression> children) {
        if (children.size() != 1) {
            throw new IllegalArgumentException(
                    "GetItem expression must have exactly one child, found: " + children.size());
        }
        try {
            ExprProtos.GetItemOpts opts = ExprProtos.GetItemOpts.parseFrom(metadata);
            return new GetItem(children.get(0), opts.getPath());
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalArgumentException("Failed to parse GetItem metadata", e);
        }
    }

    public Expression getChild() {
        return child;
    }

    public String getPath() {
        return path;
    }

    @Override
    public String id() {
        return "get_item";
    }

    @Override
    public List<Expression> children() {
        return List.of(child);
    }

    @Override
    public Optional<byte[]> metadata() {
        return Optional.of(
                ExprProtos.GetItemOpts.newBuilder().setPath(path).build().toByteArray());
    }

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visitGetItem(this);
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof GetItem)) return false;
        GetItem getItem = (GetItem) o;
        return Objects.equals(path, getItem.path);
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(path);
    }
}
