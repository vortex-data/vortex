// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
