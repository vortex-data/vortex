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

        T visitIdentity(Identity identity);

        T visitBinary(Binary binary);

        T visitNot(Not not);

        T visitGetItem(GetItem getItem);

        /**
         * For expressions that do not have a specific visitor method.
         */
        T visitOther(Expression expression);
    }
}
