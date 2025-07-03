// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions.proto;

import dev.vortex.api.expressions.Literal;
import dev.vortex.proto.ScalarProtos;
import java.math.BigDecimal;
import java.util.Objects;
import java.util.Optional;

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
    public ScalarProtos.Scalar visitDateDays(Integer days) {
        if (Objects.isNull(days)) {
            return Scalars.nullDateDays();
        } else {
            return Scalars.dateDays(days);
        }
    }

    @Override
    public ScalarProtos.Scalar visitDateMillis(Long millis) {
        if (Objects.isNull(millis)) {
            return Scalars.nullDateMillis();
        } else {
            return Scalars.dateMillis(millis);
        }
    }

    @Override
    public ScalarProtos.Scalar visitTimeSeconds(Integer seconds) {
        if (Objects.isNull(seconds)) {
            return Scalars.nullTimeSeconds();
        } else {
            return Scalars.timeSeconds(seconds);
        }
    }

    @Override
    public ScalarProtos.Scalar visitTimeMillis(Integer seconds) {
        if (Objects.isNull(seconds)) {
            return Scalars.nullTimeMillis();
        } else {
            return Scalars.timeMillis(seconds);
        }
    }

    @Override
    public ScalarProtos.Scalar visitTimeMicros(Long seconds) {
        if (Objects.isNull(seconds)) {
            return Scalars.nullTimeMicros();
        } else {
            return Scalars.timeMicros(seconds);
        }
    }

    @Override
    public ScalarProtos.Scalar visitTimeNanos(Long seconds) {
        if (Objects.isNull(seconds)) {
            return Scalars.nullTimeNanos();
        } else {
            return Scalars.timeNanos(seconds);
        }
    }

    @Override
    public ScalarProtos.Scalar visitTimestampMillis(Long epochMillis, Optional<String> timeZone) {
        if (Objects.isNull(epochMillis)) {
            return Scalars.nullTimestampMillis(timeZone);
        } else {
            return Scalars.timestampMillis(epochMillis, timeZone);
        }
    }

    @Override
    public ScalarProtos.Scalar visitTimestampMicros(Long epochMicros, Optional<String> timeZone) {
        if (Objects.isNull(epochMicros)) {
            return Scalars.nullTimestampMicros(timeZone);
        } else {
            return Scalars.timestampMicros(epochMicros, timeZone);
        }
    }

    @Override
    public ScalarProtos.Scalar visitTimestampNanos(Long epochNanos, Optional<String> timeZone) {
        if (Objects.isNull(epochNanos)) {
            return Scalars.nullTimestampNanos(timeZone);
        } else {
            return Scalars.timestampNanos(epochNanos, timeZone);
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
    public ScalarProtos.Scalar visitDecimal(BigDecimal decimal, int precision, int scale) {
        if (Objects.isNull(decimal)) {
            return Scalars.nullDecimal(precision, scale);
        } else {
            return Scalars.decimal(decimal, precision, scale);
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
