// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
