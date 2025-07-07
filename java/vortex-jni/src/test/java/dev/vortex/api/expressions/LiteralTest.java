// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.api.Expression;
import dev.vortex.api.expressions.proto.ExpressionProtoDeserializer;
import dev.vortex.api.expressions.proto.ExpressionProtoSerializer;
import dev.vortex.proto.ExprProtos;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.junit.jupiter.api.Test;

public final class LiteralTest {
    @Test
    public void testLiteral_decimals() {
        Literal<BigDecimal> lit = Literal.decimal(new BigDecimal(BigInteger.valueOf(-1234L), 3), 38, 3);
        ExprProtos.Expr serialized = ExpressionProtoSerializer.serialize(lit);
        Expression out = ExpressionProtoDeserializer.deserialize(serialized);
        assertEquals(lit, out);
    }
}
