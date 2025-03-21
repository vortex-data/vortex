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
