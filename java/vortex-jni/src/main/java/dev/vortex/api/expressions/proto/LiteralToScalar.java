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

import dev.vortex.api.expressions.Literal;
import dev.vortex.proto.ScalarProtos;
import java.util.Objects;

final class LiteralToScalar implements Literal.LiteralVisitor<ScalarProtos.Scalar> {
    static final LiteralToScalar INSTANCE = new LiteralToScalar();

    private LiteralToScalar() {}

    @Override
    public ScalarProtos.Scalar visitNull() {
        return Scalars.nullNull();
    }

    @Override
    public ScalarProtos.Scalar visitBoolean(Boolean literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullBool();
        } else {
            return Scalars.bool(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitInt8(Byte literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullInt8();
        } else {
            return Scalars.int8(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitInt16(Short literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullInt16();
        } else {
            return Scalars.int16(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitInt32(Integer literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullInt32();
        } else {
            return Scalars.int32(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitInt64(Long literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullInt64();
        } else {
            return Scalars.int64(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitFloat32(Float literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullFloat32();
        } else {
            return Scalars.float32(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitFloat64(Double literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullFloat64();
        } else {
            return Scalars.float64(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitString(String literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullString();
        } else {
            return Scalars.string(literal);
        }
    }

    @Override
    public ScalarProtos.Scalar visitBytes(byte[] literal) {
        if (Objects.isNull(literal)) {
            return Scalars.nullBytes();
        } else {
            return Scalars.bytes(literal);
        }
    }
}
