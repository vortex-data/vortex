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

import com.google.common.base.Objects;
import dev.vortex.api.Expression;

public abstract class Literal<T> implements Expression {
    private final T value;

    private Literal(T value) {
        this.value = value;
    }

    public T getValue() {
        return this.value;
    }

    @Override
    public String type() {
        return "literal";
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(getValue());
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Literal)) return false;
        Literal<?> literal = (Literal<?>) o;
        return java.util.Objects.equals(value, literal.value);
    }

    public static Literal<Void> nullLit() {
        return NullLiteral.INSTANCE;
    }

    public static Literal<Boolean> bool(Boolean value) {
        return new BooleanLiteral(value);
    }

    public static Literal<Byte> int8(Byte value) {
        return new Int8Literal(value);
    }

    public static Literal<Short> int16(Short value) {
        return new Int16Literal(value);
    }

    public static Literal<Integer> int32(Integer value) {
        return new Int32Literal(value);
    }

    public static Literal<Long> int64(Long value) {
        return new Int64Literal(value);
    }

    public static Literal<Float> float32(Float value) {
        return new Float32Literal(value);
    }

    public static Literal<Double> float64(Double value) {
        return new Float64Literal(value);
    }

    public static Literal<String> string(String value) {
        return new StringLiteral(value);
    }

    public static Literal<byte[]> bytes(byte[] value) {
        return new BytesLiteral(value);
    }

    @Override
    public <R> R accept(Expression.Visitor<R> visitor) {
        return visitor.visitLiteral(this);
    }

    public abstract <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor);

    public interface LiteralVisitor<U> {
        U visitNull();

        U visitBoolean(Boolean literal);

        U visitInt8(Byte literal);

        U visitInt16(Short literal);

        U visitInt32(Integer literal);

        U visitInt64(Long literal);

        U visitFloat32(Float literal);

        U visitFloat64(Double literal);

        U visitString(String literal);

        U visitBytes(byte[] literal);
    }

    static final class NullLiteral extends Literal<Void> {
        static final NullLiteral INSTANCE = new NullLiteral();

        private NullLiteral() {
            super(null);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitNull();
        }
    }

    static final class BooleanLiteral extends Literal<Boolean> {
        BooleanLiteral(Boolean value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitBoolean(getValue());
        }
    }

    static final class Int8Literal extends Literal<Byte> {
        Int8Literal(Byte value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitInt8(getValue());
        }
    }

    static final class Int16Literal extends Literal<Short> {
        Int16Literal(Short value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitInt16(getValue());
        }
    }

    static final class Int32Literal extends Literal<Integer> {
        Int32Literal(Integer value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitInt32(getValue());
        }
    }

    static final class Int64Literal extends Literal<Long> {
        Int64Literal(Long value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitInt64(getValue());
        }
    }

    static final class Float32Literal extends Literal<Float> {
        Float32Literal(Float value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitFloat32(getValue());
        }
    }

    static final class Float64Literal extends Literal<Double> {
        Float64Literal(Double value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitFloat64(getValue());
        }
    }

    static final class StringLiteral extends Literal<String> {
        StringLiteral(String value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitString(getValue());
        }
    }

    static final class BytesLiteral extends Literal<byte[]> {
        BytesLiteral(byte[] value) {
            super(value);
        }

        @Override
        public <U> U acceptLiteralVisitor(LiteralVisitor<U> visitor) {
            return visitor.visitBytes(getValue());
        }
    }
}
