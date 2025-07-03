// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.api.expressions.*;

/**
 * Vortex expression language.
 */
public interface Expression {
    String type();

    <T> T accept(Visitor<T> visitor);

    interface Visitor<T> {
        T visitLiteral(Literal<?> literal);

        T visitIdentity(Identity identity);

        T visitBinary(Binary binary);

        T visitNot(Not not);

        T visitGetItem(GetItem getItem);
    }
}
