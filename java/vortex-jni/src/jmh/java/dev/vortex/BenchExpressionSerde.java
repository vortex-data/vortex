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
package dev.vortex;

import com.jakewharton.nopen.annotation.Open;
import dev.vortex.api.Expression;
import dev.vortex.api.expressions.Binary;
import dev.vortex.api.expressions.Literal;
import dev.vortex.api.expressions.Not;
import dev.vortex.api.expressions.proto.ExpressionProtoDeserializer;
import dev.vortex.api.expressions.proto.ExpressionProtoSerializer;
import dev.vortex.proto.ExprProtos;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Results on macOS M2 Max:
 * <p>
 * Benchmark                    Mode  Cnt    Score    Error  Units
 * BenchExpressionSerde.decode  avgt    3   75.539 ±  9.966  ns/op
 * BenchExpressionSerde.encode  avgt    3  275.764 ± 16.858  ns/op
 * </p>
 */
@BenchmarkMode(value = Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Open
public class BenchExpressionSerde {
    static Expression decoded;
    static ExprProtos.Expr encoded;

    @Setup(Level.Trial)
    public static void setup() {
        decoded = Binary.and(
                Binary.or(Literal.bool(false), Literal.bool(true), Binary.eq(Literal.int8((byte) 1), Literal.int8((byte)
                        20))),
                Not.of(Literal.bool(false)));
        encoded = ExpressionProtoSerializer.serialize(decoded);
    }

    @Benchmark
    public void encode(Blackhole bh) {
        bh.consume(ExpressionProtoSerializer.serialize(decoded));
    }

    @Benchmark
    public void decode(Blackhole bh) {
        bh.consume(ExpressionProtoDeserializer.deserialize(encoded));
    }
}
