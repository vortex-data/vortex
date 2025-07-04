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

import com.google.protobuf.ByteString;
import dev.vortex.api.Expression;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

public final class Not implements Expression {
    private final Expression child;

    private Not(Expression child) {
        this.child = child;
    }

    public static Not parse(byte[] metadata, List<Expression> children) {
        if (children.size() != 1) {
            throw new IllegalArgumentException("Not expression must have exactly one child, found: " + children.size());
        }
        if (metadata.length > 0) {
            throw new IllegalArgumentException("Not expression must not have metadata, found: " + metadata.length);
        }
        return new Not(children.get(0));
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

    public Expression getChild() {
        return child;
    }

    @Override
    public <T> T accept(Visitor<T> visitor) {
        return visitor.visitNot(this);
    }
}
