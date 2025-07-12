// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.api.expressions.*;

import java.util.List;
import java.util.Optional;

/**
 * Vortex expression language.
 */
public interface Expression {
    /**
     * The globally unique identifier for this type of expression.
     */
    String id();

    /**
     * Returns the children of this expression.
     */
    List<Expression> children();

    /**
     * Returns the serialized metadata for this expression, or empty if serialization is not supported.
     */
    Optional<byte[]> metadata();

    default <T> T accept(Visitor<T> visitor) {
        return visitor.visitOther(this);
    }

    interface Visitor<T> {
        T visitLiteral(Literal<?> literal);

        T visitRoot(Root root);

        T visitBinary(Binary binary);

        T visitNot(Not not);

        T visitGetItem(GetItem getItem);

        /**
         * For expressions that do not have a specific visitor method.
         */
        T visitOther(Expression expression);
    }
}
