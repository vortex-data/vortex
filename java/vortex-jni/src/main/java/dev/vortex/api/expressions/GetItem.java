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

    public Expression getChild() {
        return child;
    }

    public String getPath() {
        return path;
    }

    @Override
    public String type() {
        return "get_item";
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
