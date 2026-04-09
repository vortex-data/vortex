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

    /**
     * Accepts a visitor and dispatches to the appropriate visit method based on the expression type.
     * This method implements the visitor pattern, allowing different operations to be performed
     * on expressions without modifying the expression classes themselves.
     *
     * @param <T> the return type of the visitor
     * @param visitor the visitor to accept
     * @return the result of the visitor's operation on this expression
     */
    default <T> T accept(Visitor<T> visitor) {
        return visitor.visitOther(this);
    }

    /**
     * Visitor interface for implementing the visitor pattern on expressions.
     * This interface defines methods for visiting different types of expressions,
     * allowing for type-safe operations across the expression hierarchy.
     *
     * @param <T> the return type of the visitor methods
     */
    interface Visitor<T> {
        /**
         * Visits a literal expression.
         *
         * @param literal the literal expression to visit
         * @return the result of visiting the literal expression
         */
        T visitLiteral(Literal<?> literal);

        /**
         * Visits a root expression.
         *
         * @param root the root expression to visit
         * @return the result of visiting the root expression
         */
        T visitRoot(Root root);

        /**
         * Visits a binary expression.
         *
         * @param binary the binary expression to visit
         * @return the result of visiting the binary expression
         */
        T visitBinary(Binary binary);

        /**
         * Visits a not expression (logical negation).
         *
         * @param not the not expression to visit
         * @return the result of visiting the not expression
         */
        T visitNot(Not not);

        /**
         * Visits a get item expression (array/object indexing).
         *
         * @param getItem the get item expression to visit
         * @return the result of visiting the get item expression
         */
        T visitGetItem(GetItem getItem);

        /**
         * Visits an is null expression (null check).
         *
         * @param isNull the is null expression to visit
         * @return the result of visiting the is null expression
         */
        T visitIsNull(IsNull isNull);

        /**
         * For expressions that do not have a specific visitor method.
         */
        T visitOther(Expression expression);
    }
}
