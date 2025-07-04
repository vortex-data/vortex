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
