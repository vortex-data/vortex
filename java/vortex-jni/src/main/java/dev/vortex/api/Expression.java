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
