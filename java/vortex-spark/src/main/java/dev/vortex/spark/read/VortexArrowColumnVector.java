// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// spotless:disabled
package dev.vortex.spark.read;

import com.jakewharton.nopen.annotation.Open;
import dev.vortex.relocated.org.apache.arrow.vector.*;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListViewVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.MapVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.relocated.org.apache.arrow.vector.holders.NullableLargeVarCharHolder;
import dev.vortex.relocated.org.apache.arrow.vector.holders.NullableVarCharHolder;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.spark.ArrowUtils;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * Spark ColumnVector implementation that wraps Apache Arrow vectors from Vortex data.
 * <p>
 * This class provides a bridge between Vortex's Arrow-based data representation and Spark's
 * ColumnVector interface. It supports the same Arrow vector types as Spark 4.1's own
 * {@code ArrowColumnVector} — booleans, signed integers, single/double floats, 128-bit decimals,
 * strings and binary (regular and large), day dates, timestamps of every unit with and without
 * timezone (normalized to microseconds), microsecond durations, year-month and month-day-nano
 * intervals, nulls, lists, maps, and structs — plus the Arrow view types (Utf8View, BinaryView,
 * ListView) that Vortex produces natively.
 * <p>
 * Arrow types outside that set (unsigned integers, half floats, 256-bit decimals,
 * non-microsecond durations, Time, fixed-size binary and lists, large lists, unions, run-end and
 * dictionary encodings) are rejected with a descriptive {@link UnsupportedOperationException}.
 *
 * @see ColumnVector
 * @see ValueVector
 */
@Open
public class VortexArrowColumnVector extends ColumnVector {
    VortexArrowColumnVector.ArrowVectorAccessor accessor;
    VortexArrowColumnVector[] childColumns;

    /**
     * Returns the underlying Apache Arrow ValueVector wrapped by this column vector.
     *
     * @return the Arrow ValueVector containing the actual data
     */
    public ValueVector getValueVector() {
        return accessor.vector;
    }

    /**
     * Returns whether this column contains any null values.
     *
     * @return true if the column contains at least one null value, false otherwise
     */
    @Override
    public boolean hasNull() {
        return accessor.getNullCount() > 0;
    }

    /**
     * Returns the total number of null values in this column.
     *
     * @return the count of null values
     */
    @Override
    public int numNulls() {
        return accessor.getNullCount();
    }

    /**
     * No-op: the underlying Arrow {@link ValueVector}s are owned by the
     * {@link dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader} that produced
     * this batch and are released when that reader is closed.
     */
    @Override
    public void close() {}

    /**
     * Returns whether the value at the specified row is null.
     *
     * @param rowId the row index to check
     * @return true if the value at rowId is null, false otherwise
     */
    @Override
    public boolean isNullAt(int rowId) {
        return accessor.isNullAt(rowId);
    }

    /**
     * Returns the boolean value at the specified row.
     *
     * @param rowId the row index
     * @return the boolean value at rowId
     * @throws UnsupportedOperationException if this column is not of boolean type
     */
    @Override
    public boolean getBoolean(int rowId) {
        return accessor.getBoolean(rowId);
    }

    /**
     * Returns the byte value at the specified row.
     *
     * @param rowId the row index
     * @return the byte value at rowId
     * @throws UnsupportedOperationException if this column is not of byte type
     */
    @Override
    public byte getByte(int rowId) {
        return accessor.getByte(rowId);
    }

    /**
     * Returns the short value at the specified row.
     *
     * @param rowId the row index
     * @return the short value at rowId
     * @throws UnsupportedOperationException if this column is not of short type
     */
    @Override
    public short getShort(int rowId) {
        return accessor.getShort(rowId);
    }

    /**
     * Returns the int value at the specified row.
     *
     * @param rowId the row index
     * @return the int value at rowId
     * @throws UnsupportedOperationException if this column is not of int type
     */
    @Override
    public int getInt(int rowId) {
        return accessor.getInt(rowId);
    }

    /**
     * Returns the long value at the specified row.
     *
     * @param rowId the row index
     * @return the long value at rowId
     * @throws UnsupportedOperationException if this column is not of long type
     */
    @Override
    public long getLong(int rowId) {
        return accessor.getLong(rowId);
    }

    /**
     * Returns the float value at the specified row.
     *
     * @param rowId the row index
     * @return the float value at rowId
     * @throws UnsupportedOperationException if this column is not of float type
     */
    @Override
    public float getFloat(int rowId) {
        return accessor.getFloat(rowId);
    }

    /**
     * Returns the double value at the specified row.
     *
     * @param rowId the row index
     * @return the double value at rowId
     * @throws UnsupportedOperationException if this column is not of double type
     */
    @Override
    public double getDouble(int rowId) {
        return accessor.getDouble(rowId);
    }

    /**
     * Returns the decimal value at the specified row with the given precision and scale.
     *
     * @param rowId the row index
     * @param precision the precision of the decimal
     * @param scale the scale of the decimal
     * @return the Decimal value at rowId, or null if the value is null
     * @throws UnsupportedOperationException if this column is not of decimal type
     */
    @Override
    public Decimal getDecimal(int rowId, int precision, int scale) {
        if (isNullAt(rowId)) return null;
        return accessor.getDecimal(rowId, precision, scale);
    }

    /**
     * Returns the UTF8String value at the specified row.
     *
     * @param rowId the row index
     * @return the UTF8String value at rowId, or null if the value is null
     * @throws UnsupportedOperationException if this column is not of string type
     */
    @Override
    public UTF8String getUTF8String(int rowId) {
        if (isNullAt(rowId)) return null;
        return accessor.getUTF8String(rowId);
    }

    /**
     * Returns the binary data (byte array) at the specified row.
     *
     * @param rowId the row index
     * @return the byte array at rowId, or null if the value is null
     * @throws UnsupportedOperationException if this column is not of binary type
     */
    @Override
    public byte[] getBinary(int rowId) {
        if (isNullAt(rowId)) return null;
        return accessor.getBinary(rowId);
    }

    /**
     * Returns the array value at the specified row.
     *
     * @param rowId the row index
     * @return the ColumnarArray at rowId, or null if the value is null
     * @throws UnsupportedOperationException if this column is not of array type
     */
    @Override
    public ColumnarArray getArray(int rowId) {
        if (isNullAt(rowId)) return null;
        return accessor.getArray(rowId);
    }

    /**
     * Returns the map value at the specified row.
     *
     * @param rowId the row index
     * @return the ColumnarMap at rowId, or null if the value is null
     * @throws UnsupportedOperationException if this column is not of map type
     */
    @Override
    public ColumnarMap getMap(int rowId) {
        if (isNullAt(rowId)) return null;
        return accessor.getMap(rowId);
    }

    /**
     * Returns the child column at the specified ordinal.
     * <p>
     * This is used for complex types like structs where each field is represented
     * as a child column. For month-day-nano interval columns, the children follow the
     * months/days/microseconds protocol documented on {@link ColumnVector#getInterval(int)}.
     *
     * @param ordinal the index of the child column
     * @return the child VortexArrowColumnVector at the specified ordinal
     * @throws ArrayIndexOutOfBoundsException if ordinal is out of bounds
     */
    @Override
    public VortexArrowColumnVector getChild(int ordinal) {
        return childColumns[ordinal];
    }

    /**
     * Creates a new VortexArrowColumnVector with the specified Spark DataType.
     * <p>
     * This constructor is used internally for creating column vectors before
     * the underlying Arrow vector is available.
     *
     * @param type the Spark DataType for this column
     */
    VortexArrowColumnVector(DataType type) {
        super(type);
    }

    /**
     * Creates a new VortexArrowColumnVector wrapping the specified Arrow ValueVector.
     * <p>
     * This constructor automatically determines the appropriate Spark DataType from
     * the Arrow field and initializes the type-specific accessor.
     *
     * @param vector the Arrow ValueVector to wrap
     * @throws UnsupportedOperationException if the vector type is not supported
     */
    public VortexArrowColumnVector(ValueVector vector) {
        this(ArrowUtils.fromArrowField(vector.getField()));
        initAccessor(vector);
    }

    private static VortexArrowColumnVector withAccessor(DataType type, ArrowVectorAccessor accessor) {
        VortexArrowColumnVector column = new VortexArrowColumnVector(type);
        column.accessor = accessor;
        return column;
    }

    void initAccessor(ValueVector vector) {
        if (vector instanceof BitVector bitVector) {
            accessor = new VortexArrowColumnVector.BooleanAccessor(bitVector);
        } else if (vector instanceof TinyIntVector tinyIntVector) {
            accessor = new VortexArrowColumnVector.ByteAccessor(tinyIntVector);
        } else if (vector instanceof SmallIntVector smallIntVector) {
            accessor = new VortexArrowColumnVector.ShortAccessor(smallIntVector);
        } else if (vector instanceof IntVector intVector) {
            accessor = new VortexArrowColumnVector.IntAccessor(intVector);
        } else if (vector instanceof BigIntVector bigIntVector) {
            accessor = new VortexArrowColumnVector.LongAccessor(bigIntVector);
        } else if (vector instanceof Float4Vector float4Vector) {
            accessor = new VortexArrowColumnVector.FloatAccessor(float4Vector);
        } else if (vector instanceof Float8Vector float8Vector) {
            accessor = new VortexArrowColumnVector.DoubleAccessor(float8Vector);
        } else if (vector instanceof DecimalVector decimalVector) {
            accessor = new VortexArrowColumnVector.DecimalAccessor(decimalVector);
        } else if (vector instanceof VarCharVector varCharVector) {
            accessor = new VortexArrowColumnVector.StringAccessor(varCharVector);
        } else if (vector instanceof LargeVarCharVector largeVarCharVector) {
            accessor = new VortexArrowColumnVector.LargeStringAccessor(largeVarCharVector);
        } else if (vector instanceof ViewVarCharVector viewVarCharVector) {
            accessor = new VortexArrowColumnVector.StringViewAccessor(viewVarCharVector);
        } else if (vector instanceof VarBinaryVector varBinaryVector) {
            accessor = new VortexArrowColumnVector.BinaryAccessor(varBinaryVector);
        } else if (vector instanceof LargeVarBinaryVector largeVarBinaryVector) {
            accessor = new VortexArrowColumnVector.LargeBinaryAccessor(largeVarBinaryVector);
        } else if (vector instanceof ViewVarBinaryVector viewVarBinaryVector) {
            accessor = new VortexArrowColumnVector.BinaryViewAccessor(viewVarBinaryVector);
        } else if (vector instanceof DateDayVector dateDayVector) {
            accessor = new VortexArrowColumnVector.DateAccessor(dateDayVector);
        } else if (vector instanceof TimeStampVector timeStampVector) {
            // Covers all eight unit/timezone variants; values are normalized to microseconds.
            accessor = new VortexArrowColumnVector.TimestampAccessor(timeStampVector);
        } else if (vector instanceof MapVector mapVector) {
            // MapVector extends ListVector, so this check must come first.
            accessor = new VortexArrowColumnVector.MapAccessor(mapVector);
        } else if (vector instanceof ListVector listVector) {
            accessor = new VortexArrowColumnVector.ArrayAccessor(listVector);
        } else if (vector instanceof ListViewVector listViewVector) {
            accessor = new VortexArrowColumnVector.ListViewAccessor(listViewVector);
        } else if (vector instanceof StructVector structVector) {
            accessor = new VortexArrowColumnVector.StructAccessor(structVector);

            childColumns = new VortexArrowColumnVector[structVector.size()];
            for (int i = 0; i < childColumns.length; ++i) {
                childColumns[i] = new VortexArrowColumnVector(structVector.getVectorById(i));
            }
        } else if (vector instanceof NullVector nullVector) {
            accessor = new VortexArrowColumnVector.NullAccessor(nullVector);
        } else if (vector instanceof IntervalYearVector intervalYearVector) {
            accessor = new VortexArrowColumnVector.IntervalYearAccessor(intervalYearVector);
        } else if (vector instanceof IntervalMonthDayNanoVector intervalMonthDayNanoVector) {
            accessor = new VortexArrowColumnVector.IntervalMonthDayNanoAccessor(intervalMonthDayNanoVector);
            // CalendarInterval values are read through ColumnVector.getInterval, which uses the
            // months/days/microseconds child column protocol.
            childColumns = new VortexArrowColumnVector[] {
                withAccessor(DataTypes.IntegerType, new IntervalMonthsAccessor(intervalMonthDayNanoVector)),
                withAccessor(DataTypes.IntegerType, new IntervalDaysAccessor(intervalMonthDayNanoVector)),
                withAccessor(DataTypes.LongType, new IntervalMicrosAccessor(intervalMonthDayNanoVector)),
            };
        } else if (vector instanceof DurationVector durationVector) {
            accessor = new VortexArrowColumnVector.DurationAccessor(durationVector);
        } else {
            throw new UnsupportedOperationException("Unsupported Arrow vector type: "
                    + vector.getClass().getSimpleName() + " for field " + vector.getField());
        }
    }

    abstract static class ArrowVectorAccessor {

        final ValueVector vector;

        ArrowVectorAccessor(ValueVector vector) {
            this.vector = vector;
        }

        final boolean isNullAt(int rowId) {
            return vector.isNull(rowId);
        }

        final int getNullCount() {
            return vector.getNullCount();
        }

        boolean getBoolean(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        byte getByte(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        short getShort(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        int getInt(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        long getLong(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        float getFloat(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        double getDouble(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        Decimal getDecimal(int rowId, int precision, int scale) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        UTF8String getUTF8String(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        byte[] getBinary(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        ColumnarArray getArray(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }

        ColumnarMap getMap(int rowId) {
            throw new UnsupportedOperationException(getClass().getName());
        }
    }

    @Open
    static class BooleanAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final BitVector accessor;

        BooleanAccessor(BitVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final boolean getBoolean(int rowId) {
            return accessor.get(rowId) == 1;
        }
    }

    @Open
    static class ByteAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final TinyIntVector accessor;

        ByteAccessor(TinyIntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final byte getByte(int rowId) {
            return accessor.get(rowId);
        }
    }

    @Open
    static class ShortAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final SmallIntVector accessor;

        ShortAccessor(SmallIntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final short getShort(int rowId) {
            return accessor.get(rowId);
        }
    }

    @Open
    static class IntAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final IntVector accessor;

        IntAccessor(IntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final int getInt(int rowId) {
            return accessor.get(rowId);
        }
    }

    @Open
    static class LongAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final BigIntVector accessor;

        LongAccessor(BigIntVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    @Open
    static class FloatAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final Float4Vector accessor;

        FloatAccessor(Float4Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final float getFloat(int rowId) {
            return accessor.get(rowId);
        }
    }

    @Open
    static class DoubleAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final Float8Vector accessor;

        DoubleAccessor(Float8Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final double getDouble(int rowId) {
            return accessor.get(rowId);
        }
    }

    @Open
    static class DecimalAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final DecimalVector accessor;

        DecimalAccessor(DecimalVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final Decimal getDecimal(int rowId, int precision, int scale) {
            if (isNullAt(rowId)) return null;
            return Decimal.apply(accessor.getObject(rowId), precision, scale);
        }
    }

    @Open
    static class StringAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final VarCharVector accessor;
        private final NullableVarCharHolder stringResult = new NullableVarCharHolder();

        StringAccessor(VarCharVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final UTF8String getUTF8String(int rowId) {
            accessor.get(rowId, stringResult);
            if (stringResult.isSet == 0) {
                return null;
            } else {
                return UTF8String.fromAddress(
                        null,
                        stringResult.buffer.memoryAddress() + stringResult.start,
                        stringResult.end - stringResult.start);
            }
        }
    }

    @Open
    static class LargeStringAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final LargeVarCharVector accessor;
        private final NullableLargeVarCharHolder stringResult = new NullableLargeVarCharHolder();

        LargeStringAccessor(LargeVarCharVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final UTF8String getUTF8String(int rowId) {
            accessor.get(rowId, stringResult);
            if (stringResult.isSet == 0) {
                return null;
            } else {
                return UTF8String.fromAddress(
                        null,
                        stringResult.buffer.memoryAddress() + stringResult.start,
                        // A single string cannot be larger than the max integer size, so the conversion is safe
                        (int) (stringResult.end - stringResult.start));
            }
        }
    }

    @Open
    static class StringViewAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final ViewVarCharVector accessor;

        StringViewAccessor(ViewVarCharVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final UTF8String getUTF8String(int rowId) {
            // View values may be inlined in the view buffer rather than contiguous in a data
            // buffer, so copy out rather than aliasing vector memory.
            return UTF8String.fromBytes(accessor.get(rowId));
        }
    }

    @Open
    static class BinaryAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final VarBinaryVector accessor;

        BinaryAccessor(VarBinaryVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final byte[] getBinary(int rowId) {
            return accessor.getObject(rowId);
        }
    }

    @Open
    static class LargeBinaryAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final LargeVarBinaryVector accessor;

        LargeBinaryAccessor(LargeVarBinaryVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final byte[] getBinary(int rowId) {
            return accessor.getObject(rowId);
        }
    }

    @Open
    static class BinaryViewAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final ViewVarBinaryVector accessor;

        BinaryViewAccessor(ViewVarBinaryVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final byte[] getBinary(int rowId) {
            return accessor.getObject(rowId);
        }
    }

    @Open
    static class DateAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final DateDayVector accessor;

        DateAccessor(DateDayVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final int getInt(int rowId) {
            return accessor.get(rowId);
        }
    }

    /**
     * Reads any of the eight timestamp vector variants (four units, with or without timezone),
     * normalizing values to the microseconds Spark expects for TimestampType and TimestampNTZType.
     * Seconds and milliseconds fail on overflow rather than silently wrapping; nanoseconds floor
     * towards negative infinity, matching Spark's nanosecond-to-microsecond conversions.
     */
    @Open
    static class TimestampAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final TimeStampVector accessor;
        private final TimeUnit unit;

        TimestampAccessor(TimeStampVector vector) {
            super(vector);
            this.accessor = vector;
            this.unit = ((ArrowType.Timestamp) vector.getField().getType()).getUnit();
        }

        @Override
        final long getLong(int rowId) {
            long value = accessor.get(rowId);
            return switch (unit) {
                case SECOND -> Math.multiplyExact(value, 1_000_000L);
                case MILLISECOND -> Math.multiplyExact(value, 1_000L);
                case MICROSECOND -> value;
                case NANOSECOND -> Math.floorDiv(value, 1_000L);
            };
        }
    }

    @Open
    static class ArrayAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final ListVector accessor;
        private final VortexArrowColumnVector arrayData;

        ArrayAccessor(ListVector vector) {
            super(vector);
            this.accessor = vector;
            this.arrayData = new VortexArrowColumnVector(vector.getDataVector());
        }

        @Override
        final ColumnarArray getArray(int rowId) {
            int start = accessor.getElementStartIndex(rowId);
            int end = accessor.getElementEndIndex(rowId);
            return new ColumnarArray(arrayData, start, end - start);
        }
    }

    @Open
    static class ListViewAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final ListViewVector accessor;
        private final VortexArrowColumnVector arrayData;

        ListViewAccessor(ListViewVector vector) {
            super(vector);
            this.accessor = vector;
            this.arrayData = new VortexArrowColumnVector(vector.getDataVector());
        }

        @Override
        final ColumnarArray getArray(int rowId) {
            int start = accessor.getElementStartIndex(rowId);
            int end = accessor.getElementEndIndex(rowId);
            return new ColumnarArray(arrayData, start, end - start);
        }
    }

    /**
     * Any call to "get" method will throw UnsupportedOperationException.
     * <p>
     * Access struct values in a ArrowColumnVector doesn't use this accessor. Instead, it uses
     * getStruct() method defined in the parent class. Any call to "get" method in this class is a
     * bug in the code.
     */
    @Open
    static class StructAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        StructAccessor(StructVector vector) {
            super(vector);
        }
    }

    @Open
    static class MapAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {
        private final MapVector accessor;
        private final VortexArrowColumnVector keys;
        private final VortexArrowColumnVector values;

        MapAccessor(MapVector vector) {
            super(vector);
            this.accessor = vector;
            StructVector entries = (StructVector) vector.getDataVector();
            this.keys = new VortexArrowColumnVector(entries.getChild(MapVector.KEY_NAME));
            this.values = new VortexArrowColumnVector(entries.getChild(MapVector.VALUE_NAME));
        }

        @Override
        final ColumnarMap getMap(int rowId) {
            int index = rowId * MapVector.OFFSET_WIDTH;
            int offset = accessor.getOffsetBuffer().getInt(index);
            int length = accessor.getInnerValueCountAt(rowId);
            return new ColumnarMap(keys, values, offset, length);
        }
    }

    @Open
    static class NullAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        NullAccessor(NullVector vector) {
            super(vector);
        }
    }

    @Open
    static class IntervalYearAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final IntervalYearVector accessor;

        IntervalYearAccessor(IntervalYearVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        int getInt(int rowId) {
            return accessor.get(rowId);
        }
    }

    /**
     * Accessor for the {@code MONTH_DAY_NANO} interval vector itself. Values are read through
     * {@link ColumnVector#getInterval(int)}, which uses the months/days/microseconds child
     * columns; this accessor only provides null tracking.
     */
    @Open
    static class IntervalMonthDayNanoAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        IntervalMonthDayNanoAccessor(IntervalMonthDayNanoVector vector) {
            super(vector);
        }
    }

    @Open
    static class IntervalMonthsAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final IntervalMonthDayNanoVector accessor;

        IntervalMonthsAccessor(IntervalMonthDayNanoVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final int getInt(int rowId) {
            return IntervalMonthDayNanoVector.getMonths(accessor.getDataBuffer(), rowId);
        }
    }

    @Open
    static class IntervalDaysAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final IntervalMonthDayNanoVector accessor;

        IntervalDaysAccessor(IntervalMonthDayNanoVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final int getInt(int rowId) {
            return IntervalMonthDayNanoVector.getDays(accessor.getDataBuffer(), rowId);
        }
    }

    @Open
    static class IntervalMicrosAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final IntervalMonthDayNanoVector accessor;

        IntervalMicrosAccessor(IntervalMonthDayNanoVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final long getLong(int rowId) {
            // Truncating division matches Spark's ArrowColumnVector month-day-nano handling.
            return IntervalMonthDayNanoVector.getNanoseconds(accessor.getDataBuffer(), rowId) / 1_000L;
        }
    }

    @Open
    static class DurationAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final DurationVector accessor;

        DurationAccessor(DurationVector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        final long getLong(int rowId) {
            return DurationVector.get(accessor.getDataBuffer(), rowId);
        }
    }
}
