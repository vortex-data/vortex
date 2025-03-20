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
package dev.vortex.api.expressions.proto;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.api.Expression;
import dev.vortex.api.expressions.*;
import dev.vortex.proto.ExprProtos;
import org.junit.jupiter.api.Test;

public final class TestExpressionProtos {
    @Test
    public void testRoundTrip() {
        Expression expression = Binary.and(
                GetItem.of(Identity.INSTANCE, "a.b.c"),
                Binary.or(Identity.INSTANCE, Not.of(Literal.bool(null)), Literal.bool(false)),
                Binary.eq(Literal.bool(true), Not.of(Literal.bool(false))));
        ExprProtos.Expr proto = ExpressionProtoSerializer.serialize(expression);
        Expression deserialized = ExpressionProtoDeserializer.deserialize(proto);
        assertEquals(expression, deserialized);
    }
}
