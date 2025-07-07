// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import dev.vortex.api.Expression;

public final class Identity implements Expression {
    public static final Identity INSTANCE = new Identity();

    private Identity() {}

    @Override
    public String type() {
        return "identity";
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
