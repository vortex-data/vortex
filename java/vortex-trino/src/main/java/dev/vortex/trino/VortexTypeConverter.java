/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import dev.vortex.api.DType;
import io.trino.spi.type.BigintType;
import io.trino.spi.type.BooleanType;
import io.trino.spi.type.DateType;
import io.trino.spi.type.DecimalType;
import io.trino.spi.type.DoubleType;
import io.trino.spi.type.IntegerType;
import io.trino.spi.type.RealType;
import io.trino.spi.type.SmallintType;
import io.trino.spi.type.TimeType;
import io.trino.spi.type.TimestampType;
import io.trino.spi.type.TimestampWithTimeZoneType;
import io.trino.spi.type.TinyintType;
import io.trino.spi.type.Type;
import io.trino.spi.type.VarbinaryType;
import io.trino.spi.type.VarcharType;

/**
 * Converts Vortex data types to Trino types.
 */
public final class VortexTypeConverter {

    private VortexTypeConverter() {
    }

    /**
     * Convert a Vortex DType to a Trino Type.
     *
     * @param dtype the Vortex data type
     * @return the corresponding Trino type
     */
    public static Type toTrinoType(DType dtype) {
        DType.Variant variant = dtype.getVariant();
        switch (variant) {
            case NULL:
                throw new UnsupportedOperationException("NULL type not supported");

            case BOOL:
                return BooleanType.BOOLEAN;

            case PRIMITIVE_I8:
            case PRIMITIVE_U8:
                return TinyintType.TINYINT;

            case PRIMITIVE_I16:
            case PRIMITIVE_U16:
                return SmallintType.SMALLINT;

            case PRIMITIVE_I32:
            case PRIMITIVE_U32:
                return IntegerType.INTEGER;

            case PRIMITIVE_I64:
            case PRIMITIVE_U64:
                return BigintType.BIGINT;

            case PRIMITIVE_F16:
            case PRIMITIVE_F32:
                return RealType.REAL;

            case PRIMITIVE_F64:
                return DoubleType.DOUBLE;

            case UTF8:
                return VarcharType.VARCHAR;

            case BINARY:
                return VarbinaryType.VARBINARY;

            case DECIMAL:
                return DecimalType.createDecimalType(dtype.getPrecision(), dtype.getScale());

            case EXTENSION:
                // Handle temporal types
                if (dtype.isDate()) {
                    return DateType.DATE;
                }
                if (dtype.isTime()) {
                    return timeTypeFromUnit(dtype.getTimeUnit());
                }
                if (dtype.isTimestamp()) {
                    return timestampTypeFromDType(dtype);
                }
                throw new UnsupportedOperationException("Unsupported extension type: " + dtype);
            case STRUCT:
            case LIST:
            default:
                throw new UnsupportedOperationException("Unsupported Vortex type: " + variant);
        }
    }

    private static Type timeTypeFromUnit(DType.TimeUnit unit) {
        switch (unit) {
            case SECONDS:
                return TimeType.createTimeType(0);
            case MILLISECONDS:
                return TimeType.createTimeType(3);
            case MICROSECONDS:
                return TimeType.createTimeType(6);
            case NANOSECONDS:
                return TimeType.createTimeType(9);
            default:
                throw new UnsupportedOperationException("Unsupported time unit: " + unit);
        }
    }

    private static Type timestampTypeFromDType(DType dtype) {
        int precision;
        switch (dtype.getTimeUnit()) {
            case SECONDS:
                precision = 0;
                break;
            case MILLISECONDS:
                precision = 3;
                break;
            case MICROSECONDS:
                precision = 6;
                break;
            case NANOSECONDS:
                precision = 9;
                break;
            default:
                throw new UnsupportedOperationException("Unsupported timestamp time unit: " + dtype.getTimeUnit());
        }

        if (dtype.getTimeZone().isPresent()) {
            return TimestampWithTimeZoneType.createTimestampWithTimeZoneType(precision);
        } else {
            return TimestampType.createTimestampType(precision);
        }
    }
}
