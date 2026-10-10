// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.relocated.org.apache.arrow.vector.types.DateUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.FloatingPointPrecision;
import dev.vortex.relocated.org.apache.arrow.vector.types.IntervalUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.UnionMode;
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
    @DisplayName("Decimal preserves precision and scale regardless of bit width")
    void decimalPreservesPrecisionAndScale() {
        assertEquals(DataTypes.createDecimalType(20, 4), ArrowUtils.fromArrowType(new ArrowType.Decimal(20, 4, 128)));
        assertEquals(DataTypes.createDecimalType(20, 4), ArrowUtils.fromArrowType(new ArrowType.Decimal(20, 4, 256)));
    }

    @Test
    @DisplayName("Utf8, LargeUtf8 and Utf8View map to StringType")
    void utf8MapsToString() {
        assertEquals(DataTypes.StringType, ArrowUtils.fromArrowType(new ArrowType.Utf8()));
        assertEquals(DataTypes.StringType, ArrowUtils.fromArrowType(new ArrowType.LargeUtf8()));
        assertEquals(DataTypes.StringType, ArrowUtils.fromArrowType(new ArrowType.Utf8View()));
    }

    @Test
    @DisplayName("Binary, LargeBinary and BinaryView map to BinaryType")
    void binaryMapsToBinary() {
        assertEquals(DataTypes.BinaryType, ArrowUtils.fromArrowType(new ArrowType.Binary()));
        assertEquals(DataTypes.BinaryType, ArrowUtils.fromArrowType(new ArrowType.LargeBinary()));
        assertEquals(DataTypes.BinaryType, ArrowUtils.fromArrowType(new ArrowType.BinaryView()));
    }

    @Test
    @DisplayName("Date(DAY) maps to DateType")
    void dateDayMapsToDate() {
        assertEquals(DataTypes.DateType, ArrowUtils.fromArrowType(new ArrowType.Date(DateUnit.DAY)));
    }

    @Test
    @DisplayName("Timestamps of every unit map to Timestamp with tz, TimestampNTZ without")
    void timestampMapsByTimezonePresence() {
        for (TimeUnit unit :
                new TimeUnit[] {TimeUnit.SECOND, TimeUnit.MILLISECOND, TimeUnit.MICROSECOND, TimeUnit.NANOSECOND}) {
            assertEquals(DataTypes.TimestampType, ArrowUtils.fromArrowType(new ArrowType.Timestamp(unit, "UTC")));
            assertEquals(DataTypes.TimestampNTZType, ArrowUtils.fromArrowType(new ArrowType.Timestamp(unit, null)));
        }
    }

    @Test
    @DisplayName("Null maps to NullType")
    void nullMapsToNull() {
        assertEquals(DataTypes.NullType, ArrowUtils.fromArrowType(new ArrowType.Null()));
    }

    @Test
    @DisplayName("Interval(YEAR_MONTH) maps to YearMonthIntervalType")
    void yearMonthIntervalMapsToYearMonthIntervalType() {
        assertEquals(
                DataTypes.createYearMonthIntervalType(),
                ArrowUtils.fromArrowType(new ArrowType.Interval(IntervalUnit.YEAR_MONTH)));
    }

    @Test
    @DisplayName("Interval(MONTH_DAY_NANO) maps to CalendarIntervalType")
    void monthDayNanoIntervalMapsToCalendarIntervalType() {
        assertEquals(
                DataTypes.CalendarIntervalType,
                ArrowUtils.fromArrowType(new ArrowType.Interval(IntervalUnit.MONTH_DAY_NANO)));
    }

    @Test
    @DisplayName("Duration(MICROSECOND) maps to DayTimeIntervalType")
    void microsecondDurationMapsToDayTimeIntervalType() {
        assertEquals(
                DataTypes.createDayTimeIntervalType(),
                ArrowUtils.fromArrowType(new ArrowType.Duration(TimeUnit.MICROSECOND)));
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
    @DisplayName("fromArrowField builds an ArrayType from a ListView field")
    void listViewFieldBuildsArrayType() {
        Field listView = new Field(
                "lv",
                FieldType.nullable(new ArrowType.ListView()),
                List.of(new Field("element", FieldType.notNullable(new ArrowType.Utf8View()), null)));

        assertEquals(DataTypes.createArrayType(DataTypes.StringType, false), ArrowUtils.fromArrowField(listView));
    }

    @Test
    @DisplayName("fromArrowField builds a MapType carrying the value's nullability")
    void mapFieldBuildsMapType() {
        assertEquals(
                DataTypes.createMapType(DataTypes.StringType, DataTypes.LongType, true),
                ArrowUtils.fromArrowField(mapField(FieldType.nullable(new ArrowType.Int(64, true)))));
        assertEquals(
                DataTypes.createMapType(DataTypes.StringType, DataTypes.LongType, false),
                ArrowUtils.fromArrowField(mapField(FieldType.notNullable(new ArrowType.Int(64, true)))));
    }

    private static Field mapField(FieldType valueType) {
        Field key = new Field("key", FieldType.notNullable(new ArrowType.Utf8()), null);
        Field value = new Field("value", valueType, null);
        Field entries = new Field("entries", FieldType.notNullable(new ArrowType.Struct()), List.of(key, value));
        return new Field("m", FieldType.nullable(new ArrowType.Map(false)), List.of(entries));
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
    @DisplayName("Non-microsecond duration units are unsupported")
    void nonMicrosecondDurationsAreUnsupported() {
        for (TimeUnit unit : new TimeUnit[] {TimeUnit.SECOND, TimeUnit.MILLISECOND, TimeUnit.NANOSECOND}) {
            assertThrows(
                    UnsupportedOperationException.class, () -> ArrowUtils.fromArrowType(new ArrowType.Duration(unit)));
        }
    }

    @Test
    @DisplayName("Time is unsupported: Spark's TimeType only exists in Spark 4.1+")
    void timeIsUnsupported() {
        assertThrows(
                UnsupportedOperationException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.Time(TimeUnit.NANOSECOND, 64)));
        assertThrows(
                UnsupportedOperationException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.Time(TimeUnit.MILLISECOND, 32)));
    }

    @Test
    @DisplayName("Interval(DAY_TIME) is unsupported")
    void dayTimeIntervalIsUnsupported() {
        assertThrows(
                UnsupportedOperationException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.Interval(IntervalUnit.DAY_TIME)));
    }

    @Test
    @DisplayName("FixedSizeBinary is unsupported")
    void fixedSizeBinaryIsUnsupported() {
        assertThrows(
                UnsupportedOperationException.class, () -> ArrowUtils.fromArrowType(new ArrowType.FixedSizeBinary(16)));
    }

    @Test
    @DisplayName("Union is unsupported")
    void unionIsUnsupported() {
        assertThrows(
                UnsupportedOperationException.class,
                () -> ArrowUtils.fromArrowType(new ArrowType.Union(UnionMode.Sparse, new int[0])));
    }

    @Test
    @DisplayName("LargeList and FixedSizeList fields are unsupported")
    void largeAndFixedSizeListFieldsAreUnsupported() {
        Field element = new Field("element", FieldType.nullable(new ArrowType.Int(32, true)), null);
        Field largeList = new Field("ll", FieldType.nullable(new ArrowType.LargeList()), List.of(element));
        Field fixedSizeList = new Field("fsl", FieldType.nullable(new ArrowType.FixedSizeList(2)), List.of(element));

        assertThrows(UnsupportedOperationException.class, () -> ArrowUtils.fromArrowField(largeList));
        assertThrows(UnsupportedOperationException.class, () -> ArrowUtils.fromArrowField(fixedSizeList));
    }
}
