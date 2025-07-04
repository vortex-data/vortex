// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.api.Expression;
import dev.vortex.api.proto.Expressions;
import dev.vortex.proto.ExprProtos;
import java.math.BigDecimal;
import java.math.BigInteger;
import org.junit.jupiter.api.Test;

public final class LiteralTest {
    @Test
    public void testLiteral_decimals() {
        Literal<BigDecimal> lit = Literal.decimal(new BigDecimal(BigInteger.valueOf(-1234L), 3), 38, 3);
        ExprProtos.Expr serialized = Expressions.serialize(lit);
        Expression out = Expressions.deserialize(serialized);
        assertEquals(lit, out);
    }
}
