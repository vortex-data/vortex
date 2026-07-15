// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.relocated.org.apache.arrow.vector.types.DateUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.FloatingPointPrecision;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import java.util.List;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link ArrowUtils}, which maps Arrow types to Spark SQL {@link DataType}s.
 *
 * <p>Characterizes both the supported mappings and the type configurations that are explicitly rejected, so that
 * regressions in either direction are caught.
 */
final class ArrowUtilsTest {
    @Test
    @DisplayName("Bool maps to BooleanType")
    void boolMapsToBoolean() {
        assertEquals(DataTypes.BooleanType, ArrowUtils.fromArrowType(new ArrowType.Bool()));
    }

    @Test
    @DisplayName("Signed integers map to the matching Spark integral types by bit width")
    void signedIntegersMapByWidth() {
        assertEquals(DataTypes.ByteType, ArrowUtils.fromArrowType(new ArrowType.Int(8, true)));
        assertEquals(DataTypes.ShortType, ArrowUtils.fromArrowType(new ArrowType.Int(16, true)));
        assertEquals(DataTypes.IntegerType, ArrowUtils.fromArrowType(new ArrowType.Int(32, true)));
        assertEquals(DataTypes.LongType, ArrowUtils.fromArrowType(new ArrowType.Int(64, true)));
    }

    @Test
    @DisplayName("Single/double floating point map to Float/Double")
    void floatingPointMapsByPrecision() {
        assertEquals(
                DataTypes.FloatType,
                ArrowUtils.fromArrowType(new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)));
        assertEquals(
                DataTypes.DoubleType,
                ArrowUtils.fromArrowType(new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)));
    }

    @Test
    @DisplayName("Decimal preserves precision and scale")
    void decimalPreservesPrecisionAndScale() {
        assertEquals(DataTypes.createDecimalType(20, 4), ArrowUtils.fromArrowType(new ArrowType.Decimal(20, 4, 128)));
    }

    @Test
    @DisplayName("Utf8 and LargeUtf8 map to StringType")
    void utf8MapsToString() {
        assertEquals(DataTypes.StringType, ArrowUtils.fromArrowType(new ArrowType.Utf8()));
        assertEquals(DataTypes.StringType, ArrowUtils.fromArrowType(new ArrowType.LargeUtf8()));
    }

    @Test
    @DisplayName("Binary and LargeBinary map to BinaryType")
    void binaryMapsToBinary() {
        assertEquals(DataTypes.BinaryType, ArrowUtils.fromArrowType(new ArrowType.Binary()));
        assertEquals(DataTypes.BinaryType, ArrowUtils.fromArrowType(new ArrowType.LargeBinary()));
    }

    @Test
    @DisplayName("Date(DAY) maps to DateType")
    void dateDayMapsToDate() {
        assertEquals(DataTypes.DateType, ArrowUtils.fromArrowType(new ArrowType.Date(DateUnit.DAY)));
    }

    @Test
    @DisplayName("Timestamp(MICROSECOND) maps to Timestamp with tz, TimestampNTZ without")
    void timestampMapsByTimezonePresence() {
        assertEquals(
                DataTypes.TimestampType,
                ArrowUtils.fromArrowType(new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC")));
        assertEquals(
                DataTypes.TimestampNTZType,
                ArrowUtils.fromArrowType(new ArrowType.Timestamp(TimeUnit.MICROSECOND, null)));
    }

    @Test
    @DisplayName("Null maps to NullType")
    void nullMapsToNull() {
        assertEquals(DataTypes.NullType, ArrowUtils.fromArrowType(new ArrowType.Null()));
    }

    @Test
    @DisplayName("fromArrowField builds a StructType from nested children")
    void structFieldBuildsStructType() {
        Field struct = new Field(
                "s",
                FieldType.nullable(new ArrowType.Struct()),
                List.of(
                        new Field("a", FieldType.nullable(new ArrowType.Int(32, true)), null),
                        new Field("b", FieldType.notNullable(new ArrowType.Utf8()), null)));

        DataType expected = DataTypes.createStructType(new org.apache.spark.sql.types.StructField[] {
            DataTypes.createStructField("a", DataTypes.IntegerType, true),
            DataTypes.createStructField("b", DataTypes.StringType, false)
        });
        assertEquals(expected, ArrowUtils.fromArrowField(struct));
    }

    @Test
    @DisplayName("fromArrowField builds an ArrayType carrying the element's nullability")
    void listFieldBuildsArrayType() {
        Field list = new Field(
                "l",
                FieldType.nullable(new ArrowType.List()),
                List.of(new Field("element", FieldType.nullable(new ArrowType.Int(32, true)), null)));

        assertEquals(DataTypes.createArrayType(DataTypes.IntegerType, true), ArrowUtils.fromArrowField(list));
    }

    @Test
    @DisplayName("Unsigned integers are unsupported")
    void unsignedIntegerIsUnsupported() {
        assertThrows(UnsupportedOperationException.class, () -> ArrowUtils.fromArrowType(new ArrowType.Int(32, false)));
    }

    @Test
    @DisplayName("Half-precision floating point is unsupported")
    void halfPrecisionFloatIsUnsupported() {
        assertThrows(
                UnsupportedOperationException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.FloatingPoint(FloatingPointPrecision.HALF)));
    }

    @Test
    @DisplayName("Non-DAY date units are unsupported")
    void millisecondDateIsUnsupported() {
        assertThrows(
                UnsupportedOperationException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.Date(DateUnit.MILLISECOND)));
    }

    @Test
    @DisplayName("Non-microsecond timestamp units are unsupported")
    void secondTimestampIsUnsupported() {
        assertThrows(
                UnsupportedOperationException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.Timestamp(TimeUnit.SECOND, "UTC")));
    }

    @Test
    @DisplayName("Unrecognized Arrow types raise IllegalArgumentException")
    void unrecognizedTypeIsIllegalArgument() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.Duration(TimeUnit.SECOND)));
    }
}
