// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.BitVector;
import dev.vortex.relocated.org.apache.arrow.vector.DateDayVector;
import dev.vortex.relocated.org.apache.arrow.vector.DateMilliVector;
import dev.vortex.relocated.org.apache.arrow.vector.Decimal256Vector;
import dev.vortex.relocated.org.apache.arrow.vector.DecimalVector;
import dev.vortex.relocated.org.apache.arrow.vector.DurationVector;
import dev.vortex.relocated.org.apache.arrow.vector.Float2Vector;
import dev.vortex.relocated.org.apache.arrow.vector.Float4Vector;
import dev.vortex.relocated.org.apache.arrow.vector.Float8Vector;
import dev.vortex.relocated.org.apache.arrow.vector.IntVector;
import dev.vortex.relocated.org.apache.arrow.vector.IntervalMonthDayNanoVector;
import dev.vortex.relocated.org.apache.arrow.vector.IntervalYearVector;
import dev.vortex.relocated.org.apache.arrow.vector.LargeVarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.LargeVarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.NullVector;
import dev.vortex.relocated.org.apache.arrow.vector.SmallIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeMicroVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMicroTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMicroVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMilliTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMilliVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampNanoTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampNanoVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampSecTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampSecVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampVector;
import dev.vortex.relocated.org.apache.arrow.vector.TinyIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.UInt4Vector;
import dev.vortex.relocated.org.apache.arrow.vector.VarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.ViewVarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.ViewVarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListViewVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.MapVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.impl.NullableStructWriter;
import dev.vortex.relocated.org.apache.arrow.vector.complex.impl.UnionListViewWriter;
import dev.vortex.relocated.org.apache.arrow.vector.complex.impl.UnionListWriter;
import dev.vortex.relocated.org.apache.arrow.vector.complex.impl.UnionMapWriter;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import dev.vortex.relocated.org.apache.arrow.vector.util.Text;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.CalendarInterval;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VortexArrowColumnVector} covering every Arrow vector type it supports (Spark 4.1's own
 * ArrowColumnVector set plus the Arrow view types): the Spark type mapping, the value conversion, and null handling.
 */
final class VortexArrowColumnVectorTest {

    private static final BufferAllocator ALLOCATOR = new RootAllocator();

    @AfterAll
    static void closeAllocator() {
        ALLOCATOR.close();
    }

    @Test
    @DisplayName("BitVector maps to BooleanType")
    void booleanVector() {
        try (BitVector vector = new BitVector("bool", ALLOCATOR)) {
            vector.allocateNew(3);
            vector.set(0, 1);
            vector.set(2, 0);
            vector.setValueCount(3);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.BooleanType, column.dataType());
            assertTrue(column.getBoolean(0));
            assertTrue(column.isNullAt(1));
            assertFalse(column.getBoolean(2));
            assertTrue(column.hasNull());
            assertEquals(1, column.numNulls());
        }
    }

    @Test
    @DisplayName("Signed integer vectors map to Byte/Short/Integer/LongType")
    void signedIntegers() {
        try (TinyIntVector i8 = new TinyIntVector("i8", ALLOCATOR);
                SmallIntVector i16 = new SmallIntVector("i16", ALLOCATOR);
                IntVector i32 = new IntVector("i32", ALLOCATOR);
                BigIntVector i64 = new BigIntVector("i64", ALLOCATOR)) {
            i8.allocateNew(2);
            i8.set(0, Byte.MIN_VALUE);
            i8.setValueCount(2);
            i16.allocateNew(2);
            i16.set(0, Short.MIN_VALUE);
            i16.setValueCount(2);
            i32.allocateNew(2);
            i32.set(0, Integer.MIN_VALUE);
            i32.setValueCount(2);
            i64.allocateNew(2);
            i64.set(0, Long.MIN_VALUE);
            i64.setValueCount(2);

            VortexArrowColumnVector byteColumn = new VortexArrowColumnVector(i8);
            assertEquals(DataTypes.ByteType, byteColumn.dataType());
            assertEquals(Byte.MIN_VALUE, byteColumn.getByte(0));
            assertTrue(byteColumn.isNullAt(1));

            VortexArrowColumnVector shortColumn = new VortexArrowColumnVector(i16);
            assertEquals(DataTypes.ShortType, shortColumn.dataType());
            assertEquals(Short.MIN_VALUE, shortColumn.getShort(0));

            VortexArrowColumnVector intColumn = new VortexArrowColumnVector(i32);
            assertEquals(DataTypes.IntegerType, intColumn.dataType());
            assertEquals(Integer.MIN_VALUE, intColumn.getInt(0));

            VortexArrowColumnVector longColumn = new VortexArrowColumnVector(i64);
            assertEquals(DataTypes.LongType, longColumn.dataType());
            assertEquals(Long.MIN_VALUE, longColumn.getLong(0));
        }
    }

    @Test
    @DisplayName("Float vectors map to Float/DoubleType")
    void floats() {
        try (Float4Vector f32 = new Float4Vector("f32", ALLOCATOR);
                Float8Vector f64 = new Float8Vector("f64", ALLOCATOR)) {
            f32.allocateNew(2);
            f32.set(0, 2.5f);
            f32.setValueCount(2);
            f64.allocateNew(2);
            f64.set(0, 3.5d);
            f64.setValueCount(2);

            VortexArrowColumnVector floatColumn = new VortexArrowColumnVector(f32);
            assertEquals(DataTypes.FloatType, floatColumn.dataType());
            assertEquals(2.5f, floatColumn.getFloat(0));
            assertTrue(floatColumn.isNullAt(1));

            VortexArrowColumnVector doubleColumn = new VortexArrowColumnVector(f64);
            assertEquals(DataTypes.DoubleType, doubleColumn.dataType());
            assertEquals(3.5d, doubleColumn.getDouble(0));
        }
    }

    @Test
    @DisplayName("Decimal128 vectors map to DecimalType")
    void decimal() {
        try (DecimalVector d128 = new DecimalVector("d128", ALLOCATOR, 10, 2)) {
            d128.allocateNew(2);
            d128.set(0, new BigDecimal("12345678.90"));
            d128.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(d128);
            assertEquals(DataTypes.createDecimalType(10, 2), column.dataType());
            assertEquals(
                    new BigDecimal("12345678.90"), column.getDecimal(0, 10, 2).toJavaBigDecimal());
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("String vectors (regular, large, view) map to StringType")
    void strings() {
        try (VarCharVector utf8 = new VarCharVector("utf8", ALLOCATOR);
                LargeVarCharVector largeUtf8 = new LargeVarCharVector("large_utf8", ALLOCATOR);
                ViewVarCharVector utf8View = new ViewVarCharVector("utf8_view", ALLOCATOR)) {
            utf8.allocateNew(2);
            utf8.setSafe(0, "hello".getBytes(StandardCharsets.UTF_8));
            utf8.setValueCount(2);
            largeUtf8.allocateNew(2);
            largeUtf8.setSafe(0, "world".getBytes(StandardCharsets.UTF_8));
            largeUtf8.setValueCount(2);
            utf8View.allocateNew(2);
            utf8View.setSafe(0, new Text("a string long enough to not be inlined"));
            utf8View.setValueCount(2);

            VortexArrowColumnVector utf8Column = new VortexArrowColumnVector(utf8);
            assertEquals(DataTypes.StringType, utf8Column.dataType());
            assertEquals(UTF8String.fromString("hello"), utf8Column.getUTF8String(0));
            assertTrue(utf8Column.isNullAt(1));

            VortexArrowColumnVector largeUtf8Column = new VortexArrowColumnVector(largeUtf8);
            assertEquals(DataTypes.StringType, largeUtf8Column.dataType());
            assertEquals(UTF8String.fromString("world"), largeUtf8Column.getUTF8String(0));

            VortexArrowColumnVector utf8ViewColumn = new VortexArrowColumnVector(utf8View);
            assertEquals(DataTypes.StringType, utf8ViewColumn.dataType());
            assertEquals(
                    UTF8String.fromString("a string long enough to not be inlined"), utf8ViewColumn.getUTF8String(0));
            assertTrue(utf8ViewColumn.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Binary vectors (regular, large, view) map to BinaryType")
    void binary() {
        byte[] payload = new byte[] {1, 2, 3, 4};
        try (VarBinaryVector bin = new VarBinaryVector("bin", ALLOCATOR);
                LargeVarBinaryVector largeBin = new LargeVarBinaryVector("large_bin", ALLOCATOR);
                ViewVarBinaryVector binView = new ViewVarBinaryVector("bin_view", ALLOCATOR)) {
            bin.allocateNew(2);
            bin.setSafe(0, payload);
            bin.setValueCount(2);
            largeBin.allocateNew(2);
            largeBin.setSafe(0, payload);
            largeBin.setValueCount(2);
            binView.allocateNew(2);
            binView.setSafe(0, payload);
            binView.setValueCount(2);

            for (VortexArrowColumnVector column : new VortexArrowColumnVector[] {
                new VortexArrowColumnVector(bin),
                new VortexArrowColumnVector(largeBin),
                new VortexArrowColumnVector(binView),
            }) {
                assertEquals(DataTypes.BinaryType, column.dataType());
                assertArrayEquals(payload, column.getBinary(0));
                assertTrue(column.isNullAt(1));
            }
        }
    }

    @Test
    @DisplayName("Day-unit date vectors map to DateType")
    void date() {
        try (DateDayVector vector = new DateDayVector("date_day", ALLOCATOR)) {
            vector.allocateNew(2);
            vector.set(0, 19000);
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.DateType, column.dataType());
            assertEquals(19000, column.getInt(0));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Timestamp vectors of every unit normalize to microseconds")
    void timestampsWithoutTimezone() {
        try (TimeStampSecVector sec = new TimeStampSecVector("ts_s", ALLOCATOR);
                TimeStampMilliVector milli = new TimeStampMilliVector("ts_ms", ALLOCATOR);
                TimeStampMicroVector micro = new TimeStampMicroVector("ts_us", ALLOCATOR);
                TimeStampNanoVector nano = new TimeStampNanoVector("ts_ns", ALLOCATOR)) {
            sec.allocateNew(3);
            sec.set(0, 1_700_000_000L);
            sec.setValueCount(3);
            milli.allocateNew(3);
            milli.set(0, 1_700_000_000_123L);
            milli.setValueCount(3);
            micro.allocateNew(3);
            micro.set(0, 1_700_000_000_123_456L);
            micro.setValueCount(3);
            nano.allocateNew(3);
            nano.set(0, 1_700_000_000_123_456_789L);
            // Negative nanos floor towards negative infinity when reduced to micros.
            nano.set(2, -1_500L);
            nano.setValueCount(3);

            assertTimestamp(sec, DataTypes.TimestampNTZType, 1_700_000_000_000_000L);
            assertTimestamp(milli, DataTypes.TimestampNTZType, 1_700_000_000_123_000L);
            assertTimestamp(micro, DataTypes.TimestampNTZType, 1_700_000_000_123_456L);
            VortexArrowColumnVector nanoColumn =
                    assertTimestamp(nano, DataTypes.TimestampNTZType, 1_700_000_000_123_456L);
            assertEquals(-2L, nanoColumn.getLong(2));
        }
    }

    @Test
    @DisplayName("Timezone-aware timestamp vectors of every unit map to TimestampType")
    void timestampsWithTimezone() {
        try (TimeStampSecTZVector sec = new TimeStampSecTZVector("ts_s", ALLOCATOR, "UTC");
                TimeStampMilliTZVector milli = new TimeStampMilliTZVector("ts_ms", ALLOCATOR, "UTC");
                TimeStampMicroTZVector micro = new TimeStampMicroTZVector("ts_us", ALLOCATOR, "UTC");
                TimeStampNanoTZVector nano = new TimeStampNanoTZVector("ts_ns", ALLOCATOR, "UTC")) {
            sec.allocateNew(1);
            sec.set(0, 1_700_000_000L);
            sec.setValueCount(1);
            milli.allocateNew(1);
            milli.set(0, 1_700_000_000_123L);
            milli.setValueCount(1);
            micro.allocateNew(1);
            micro.set(0, 1_700_000_000_123_456L);
            micro.setValueCount(1);
            nano.allocateNew(1);
            nano.set(0, 1_700_000_000_123_456_789L);
            nano.setValueCount(1);

            assertTimestamp(sec, DataTypes.TimestampType, 1_700_000_000_000_000L);
            assertTimestamp(milli, DataTypes.TimestampType, 1_700_000_000_123_000L);
            assertTimestamp(micro, DataTypes.TimestampType, 1_700_000_000_123_456L);
            assertTimestamp(nano, DataTypes.TimestampType, 1_700_000_000_123_456L);
        }
    }

    private static VortexArrowColumnVector assertTimestamp(
            TimeStampVector vector, org.apache.spark.sql.types.DataType expectedType, long expectedMicros) {
        VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
        assertEquals(expectedType, column.dataType());
        assertEquals(expectedMicros, column.getLong(0));
        return column;
    }

    @Test
    @DisplayName("Microsecond duration vectors map to DayTimeIntervalType")
    void duration() {
        try (DurationVector vector = new DurationVector(
                "dur_us", FieldType.nullable(new ArrowType.Duration(TimeUnit.MICROSECOND)), ALLOCATOR)) {
            vector.allocateNew(2);
            vector.set(0, 1_500L);
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.createDayTimeIntervalType(), column.dataType());
            assertEquals(1_500L, column.getLong(0));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Year-month interval vectors map to YearMonthIntervalType as months")
    void intervalYear() {
        try (IntervalYearVector vector = new IntervalYearVector("interval_ym", ALLOCATOR)) {
            vector.allocateNew(2);
            vector.set(0, 14);
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.createYearMonthIntervalType(), column.dataType());
            assertEquals(14, column.getInt(0));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Month-day-nano interval vectors map to CalendarIntervalType")
    void intervalMonthDayNano() {
        try (IntervalMonthDayNanoVector vector = new IntervalMonthDayNanoVector("interval_mdn", ALLOCATOR)) {
            vector.allocateNew(2);
            vector.set(0, 1, 2, 3_500L);
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.CalendarIntervalType, column.dataType());
            // Nanoseconds truncate to microseconds, matching Spark's ArrowColumnVector.
            assertEquals(new CalendarInterval(1, 2, 3L), column.getInterval(0));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Null vectors map to NullType")
    void nullVector() {
        try (NullVector vector = new NullVector("null_col")) {
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.NullType, column.dataType());
            assertTrue(column.isNullAt(0));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("List vectors map to ArrayType")
    void list() {
        try (ListVector vector = ListVector.empty("list", ALLOCATOR)) {
            UnionListWriter writer = vector.getWriter();
            writer.allocate();
            writer.setPosition(0);
            writer.startList();
            writer.writeInt(1);
            writer.writeInt(2);
            writer.writeInt(3);
            writer.endList();
            writer.setPosition(2);
            writer.startList();
            writer.endList();
            vector.setValueCount(3);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.IntegerType, ((ArrayType) column.dataType()).elementType());
            ColumnarArray array = column.getArray(0);
            assertEquals(3, array.numElements());
            assertEquals(1, array.getInt(0));
            assertEquals(3, array.getInt(2));
            assertTrue(column.isNullAt(1));
            assertEquals(0, column.getArray(2).numElements());
        }
    }

    @Test
    @DisplayName("List view vectors map to ArrayType")
    void listView() {
        try (ListViewVector vector = ListViewVector.empty("list_view", ALLOCATOR)) {
            UnionListViewWriter writer = vector.getWriter();
            writer.allocate();
            writer.setPosition(0);
            writer.startListView();
            writer.writeInt(7);
            writer.writeInt(8);
            writer.endListView();
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(DataTypes.IntegerType, ((ArrayType) column.dataType()).elementType());
            ColumnarArray array = column.getArray(0);
            assertEquals(2, array.numElements());
            assertEquals(7, array.getInt(0));
            assertEquals(8, array.getInt(1));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Map vectors map to MapType")
    void map() {
        try (MapVector vector = MapVector.empty("map", ALLOCATOR, false)) {
            UnionMapWriter writer = vector.getWriter();
            writer.allocate();
            writer.setPosition(0);
            writer.startMap();
            writer.startEntry();
            writer.key().integer().writeInt(1);
            writer.value().bigInt().writeBigInt(100L);
            writer.endEntry();
            writer.startEntry();
            writer.key().integer().writeInt(2);
            writer.value().bigInt().writeBigInt(200L);
            writer.endEntry();
            writer.endMap();
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            MapType mapType = (MapType) column.dataType();
            assertEquals(DataTypes.IntegerType, mapType.keyType());
            assertEquals(DataTypes.LongType, mapType.valueType());

            ColumnarMap columnarMap = column.getMap(0);
            assertEquals(2, columnarMap.numElements());
            assertEquals(1, columnarMap.keyArray().getInt(0));
            assertEquals(100L, columnarMap.valueArray().getLong(0));
            assertEquals(2, columnarMap.keyArray().getInt(1));
            assertEquals(200L, columnarMap.valueArray().getLong(1));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Struct vectors map to StructType with child columns")
    void struct() {
        try (StructVector vector = StructVector.empty("struct", ALLOCATOR)) {
            NullableStructWriter writer = vector.getWriter();
            writer.allocate();
            writer.setPosition(0);
            writer.start();
            writer.integer("a").writeInt(5);
            writer.bigInt("b").writeBigInt(7L);
            writer.end();
            vector.setValueCount(2);

            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            StructType structType = (StructType) column.dataType();
            assertEquals(2, structType.fields().length);
            assertEquals(DataTypes.IntegerType, structType.fields()[0].dataType());
            assertEquals(DataTypes.LongType, structType.fields()[1].dataType());
            assertEquals(5, column.getChild(0).getInt(0));
            assertEquals(7L, column.getChild(1).getLong(0));
            assertTrue(column.isNullAt(1));
        }
    }

    @Test
    @DisplayName("Arrow types outside the supported set are rejected with descriptive errors")
    void unsupportedTypes() {
        try (UInt4Vector unsignedInt = new UInt4Vector("u32", ALLOCATOR);
                Float2Vector halfFloat = new Float2Vector("f16", ALLOCATOR);
                Decimal256Vector decimal256 = new Decimal256Vector("d256", ALLOCATOR, 38, 2);
                DateMilliVector dateMilli = new DateMilliVector("date_ms", ALLOCATOR);
                TimeMicroVector time = new TimeMicroVector("time_us", ALLOCATOR);
                DurationVector durationSec = new DurationVector(
                        "dur_s", FieldType.nullable(new ArrowType.Duration(TimeUnit.SECOND)), ALLOCATOR)) {
            assertUnsupported(unsignedInt, "unsigned");
            assertUnsupported(halfFloat, "float precision");
            // Decimal256 maps to DecimalType but has no accessor, matching Spark's ArrowColumnVector.
            assertUnsupported(decimal256, "Decimal256Vector");
            assertUnsupported(dateMilli, "date unit");
            assertUnsupported(time, "Time");
            assertUnsupported(durationSec, "duration unit");
        }
    }

    private static void assertUnsupported(
            dev.vortex.relocated.org.apache.arrow.vector.ValueVector vector, String expectedMessagePart) {
        UnsupportedOperationException e =
                assertThrows(UnsupportedOperationException.class, () -> new VortexArrowColumnVector(vector));
        assertTrue(
                e.getMessage().contains(expectedMessagePart),
                "expected error message to contain \"" + expectedMessagePart + "\" but was: " + e.getMessage());
    }
}
