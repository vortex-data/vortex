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
