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
