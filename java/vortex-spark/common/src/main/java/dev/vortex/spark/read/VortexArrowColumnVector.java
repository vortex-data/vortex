// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// spotless:disabled
package dev.vortex.spark.read;

import com.jakewharton.nopen.annotation.Open;
import dev.vortex.relocated.org.apache.arrow.memory.ArrowBuf;
import dev.vortex.relocated.org.apache.arrow.vector.*;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListViewVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.MapVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.spark.ArrowUtils;
import dev.vortex.spark.SparkVariants;
import java.math.BigDecimal;
import java.nio.ByteOrder;
import java.util.Arrays;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.Platform;
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

    /**
     * Wraps an unsigned 64-bit vector as a Spark {@code LONG} column, for values known to stay below 2^63, such as the
     * row indexes of a file.
     *
     * <p>Spark has no unsigned type, so schema inference leaves unsigned columns unmapped, and only a column the reader
     * generates itself comes through here.
     */
    public static VortexArrowColumnVector unsignedAsLong(UInt8Vector vector) {
        return withAccessor(DataTypes.LongType, new UnsignedLongAccessor(vector));
    }

    /**
     * Wraps the unscaled values of a decimal column of at most 18 digits as a Spark {@code DECIMAL(precision, scale)}
     * column, which reads each value as Spark's own vectorized readers do.
     */
    public static VortexArrowColumnVector unscaledDecimal(BigIntVector vector, DecimalType type) {
        return withAccessor(type, new UnscaledDecimalAccessor(vector));
    }

    private static VortexArrowColumnVector withAccessor(DataType type, ArrowVectorAccessor accessor) {
        VortexArrowColumnVector column = new VortexArrowColumnVector(type);
        column.accessor = accessor;
        return column;
    }

    /** A column of {@code type} that is null in every one of its {@code rowCount} rows. */
    static VortexArrowColumnVector nulls(DataType type, int rowCount) {
        VortexArrowColumnVector column = withAccessor(type, new NullAccessor(new NullVector("", rowCount)));
        if (type instanceof StructType struct) {
            column.childColumns = Arrays.stream(struct.fields())
                    .map(field -> nulls(field.dataType(), rowCount))
                    .toArray(VortexArrowColumnVector[]::new);
        }
        return column;
    }

    /** This struct column read as {@code type}, with {@code fields} as its fields. */
    VortexArrowColumnVector withFields(DataType type, VortexArrowColumnVector[] fields) {
        VortexArrowColumnVector column = withAccessor(type, accessor);
        column.childColumns = fields;
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
            int precision = decimalVector.getPrecision();
            if (precision > 0 && precision <= Decimal.MAX_LONG_DIGITS()) {
                accessor = new VortexArrowColumnVector.SmallDecimalAccessor(decimalVector, ByteOrder.nativeOrder());
            } else {
                accessor = new VortexArrowColumnVector.DecimalAccessor(decimalVector);
            }
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

            if (SparkVariants.isVariant(structVector.getField())) {
                // Spark reads a variant's value from child 0 and its metadata from child 1.
                childColumns = new VortexArrowColumnVector[] {
                    new VortexArrowColumnVector(structVector.getChild("value")),
                    new VortexArrowColumnVector(structVector.getChild("metadata")),
                };
            } else {
                childColumns = new VortexArrowColumnVector[structVector.size()];
                for (int i = 0; i < childColumns.length; ++i) {
                    childColumns[i] = new VortexArrowColumnVector(structVector.getVectorById(i));
                }
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
        /** Marks a vector whose validity {@link #isNullAt} asks the vector for, as it is not a plain bitmap. */
        private static final long VALIDITY_FROM_VECTOR = -1;

        final ValueVector vector;

        private final int nullCount;

        /**
         * The address of the vector's validity bitmap, 0 when every value is valid, or {@link #VALIDITY_FROM_VECTOR}.
         *
         * <p>Spark reads every Vortex column as nullable, so generated code checks each value for null. Reading the
         * bit directly, or nothing at all, keeps that check off Arrow's interface calls.
         */
        private final long validityAddress;

        ArrowVectorAccessor(ValueVector vector) {
            this.vector = vector;
            this.nullCount = vector.getNullCount();
            this.validityAddress = validityAddress(vector, nullCount);
        }

        private static long validityAddress(ValueVector vector, int nullCount) {
            if (nullCount == 0) {
                return 0;
            }
            if (vector instanceof BaseFixedWidthVector
                    || vector instanceof BaseVariableWidthVector
                    || vector instanceof BaseLargeVariableWidthVector
                    || vector instanceof BaseVariableWidthViewVector) {
                return vector.getValidityBuffer().memoryAddress();
            }
            return VALIDITY_FROM_VECTOR;
        }

        final boolean isNullAt(int rowId) {
            long address = validityAddress;
            if (address == 0) {
                return false;
            }
            if (address == VALIDITY_FROM_VECTOR) {
                return vector.isNull(rowId);
            }
            return (Platform.getByte(null, address + (rowId >>> 3)) & (1 << (rowId & 7))) == 0;
        }

        final int getNullCount() {
            return nullCount;
        }

        /** The address of a fixed-width vector's values. */
        static long valuesAddress(BaseFixedWidthVector vector) {
            return vector.getDataBuffer().memoryAddress();
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

        private final long values;

        BooleanAccessor(BitVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final boolean getBoolean(int rowId) {
            return (Platform.getByte(null, values + (rowId >>> 3)) & (1 << (rowId & 7))) != 0;
        }
    }

    @Open
    static class ByteAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long values;

        ByteAccessor(TinyIntVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final byte getByte(int rowId) {
            return Platform.getByte(null, values + rowId);
        }
    }

    @Open
    static class ShortAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long values;

        ShortAccessor(SmallIntVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final short getShort(int rowId) {
            return Platform.getShort(null, values + ((long) rowId << 1));
        }
    }

    @Open
    static class IntAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long values;

        IntAccessor(IntVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final int getInt(int rowId) {
            return Platform.getInt(null, values + ((long) rowId << 2));
        }
    }

    @Open
    static class LongAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long values;

        LongAccessor(BigIntVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final long getLong(int rowId) {
            return Platform.getLong(null, values + ((long) rowId << 3));
        }
    }

    static final class UnsignedLongAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final UInt8Vector accessor;

        UnsignedLongAccessor(UInt8Vector vector) {
            super(vector);
            this.accessor = vector;
        }

        @Override
        long getLong(int rowId) {
            return accessor.get(rowId);
        }
    }

    @Open
    static class FloatAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long values;

        FloatAccessor(Float4Vector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final float getFloat(int rowId) {
            return Platform.getFloat(null, values + ((long) rowId << 2));
        }
    }

    @Open
    static class DoubleAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long values;

        DoubleAccessor(Float8Vector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final double getDouble(int rowId) {
            return Platform.getDouble(null, values + ((long) rowId << 3));
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

    /** Reads a decimal of at most 18 digits from the long of its unscaled value. */
    static final class UnscaledDecimalAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {
        private final long values;

        UnscaledDecimalAccessor(BigIntVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        Decimal getDecimal(int rowId, int precision, int scale) {
            if (isNullAt(rowId)) {
                return null;
            }
            return Decimal.createUnsafe(Platform.getLong(null, values + ((long) rowId << 3)), precision, scale);
        }
    }

    static final class SmallDecimalAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {
        private static final long[] LONG_POWERS_OF_TEN = longPowersOfTen();
        private final long values;

        private final DecimalVector accessor;
        private final int lowWordOffset;
        private final int highWordOffset;

        SmallDecimalAccessor(DecimalVector vector, ByteOrder byteOrder) {
            super(vector);
            this.accessor = vector;
            this.values = valuesAddress(vector);
            // Arrow Java stores Decimal128 in native byte order, including the order of its two words.
            this.lowWordOffset = byteOrder == ByteOrder.LITTLE_ENDIAN ? 0 : Long.BYTES;
            this.highWordOffset = byteOrder == ByteOrder.LITTLE_ENDIAN ? Long.BYTES : 0;
        }

        private static long[] longPowersOfTen() {
            long[] powers = new long[Decimal.MAX_LONG_DIGITS() + 1];
            powers[0] = 1;
            for (int i = 1; i < powers.length; i++) {
                powers[i] = powers[i - 1] * 10;
            }
            return powers;
        }

        /**
         * Whether every value of the source type is a value of {@code DECIMAL(precision, scale)} with the same unscaled
         * value, which then fits a long.
         */
        private boolean fitsRequestedType(int precision, int scale) {
            return scale >= 0
                    && scale == accessor.getScale()
                    && accessor.getPrecision() <= precision
                    && precision <= Decimal.MAX_LONG_DIGITS();
        }

        /** Whether {@code unscaled} has at most the source precision's digits, as every valid value does. */
        private boolean fitsSourcePrecision(long unscaled) {
            long bound = LONG_POWERS_OF_TEN[accessor.getPrecision()];
            return unscaled < bound && unscaled > -bound;
        }

        @Override
        final Decimal getDecimal(int rowId, int precision, int scale) {
            if (isNullAt(rowId)) {
                return null;
            }
            long value = values + (long) rowId * DecimalVector.TYPE_WIDTH;
            long unscaled = Platform.getLong(null, value + lowWordOffset);
            long high = Platform.getLong(null, value + highWordOffset);
            if (high == (unscaled >> 63)) {
                if (fitsRequestedType(precision, scale) && fitsSourcePrecision(unscaled)) {
                    // The value is a value of the requested type as it is. Spark's own vectorized readers return
                    // such decimals as a long, without a BigDecimal per value.
                    return Decimal.createUnsafe(unscaled, precision, scale);
                }
                // Decode with the source scale; Spark performs the requested rescaling and overflow checks.
                // Keep the expanded Decimal representation for Spark's checked integer casts.
                return Decimal.apply(BigDecimal.valueOf(unscaled, accessor.getScale()), precision, scale);
            }
            // Preserve all 128 bits when the stored integer does not fit in a long.
            return Decimal.apply(accessor.getObject(rowId), precision, scale);
        }
    }

    /**
     * Reads strings straight from the vector's buffers, whose addresses it keeps. The column checks for null before it
     * asks for a value, so the accessor does not check again.
     */
    @Open
    static class StringAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long offsets;
        private final long data;

        StringAccessor(VarCharVector vector) {
            super(vector);
            this.offsets = vector.getOffsetBuffer().memoryAddress();
            this.data = vector.getDataBuffer().memoryAddress();
        }

        @Override
        final UTF8String getUTF8String(int rowId) {
            long offset = offsets + ((long) rowId << 2);
            int start = Platform.getInt(null, offset);
            int end = Platform.getInt(null, offset + Integer.BYTES);
            return UTF8String.fromAddress(null, data + start, end - start);
        }
    }

    /** Reads large strings like {@link StringAccessor}, through 64-bit offsets. */
    @Open
    static class LargeStringAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long offsets;
        private final long data;

        LargeStringAccessor(LargeVarCharVector vector) {
            super(vector);
            this.offsets = vector.getOffsetBuffer().memoryAddress();
            this.data = vector.getDataBuffer().memoryAddress();
        }

        @Override
        final UTF8String getUTF8String(int rowId) {
            long offset = offsets + ((long) rowId << 3);
            long start = Platform.getLong(null, offset);
            long end = Platform.getLong(null, offset + Long.BYTES);
            // A single string cannot be larger than the max integer size, so the conversion is safe.
            return UTF8String.fromAddress(null, data + start, (int) (end - start));
        }
    }

    @Open
    static class StringViewAccessor extends VortexArrowColumnVector.ArrowVectorAccessor {

        private final long views;

        /** The address of each data buffer, by the index a view gives. */
        private final long[] data;

        StringViewAccessor(ViewVarCharVector vector) {
            super(vector);
            this.views = vector.getDataBuffer().memoryAddress();
            this.data = vector.getDataBuffers().stream()
                    .mapToLong(ArrowBuf::memoryAddress)
                    .toArray();
        }

        /**
         * Aliases the value's bytes, like {@link StringAccessor}. A view holds its length in its first four bytes, and
         * then either the value itself when it is short, or a prefix, the index of a data buffer, and an offset into it.
         */
        @Override
        final UTF8String getUTF8String(int rowId) {
            long view = views + (long) rowId * BaseVariableWidthViewVector.ELEMENT_SIZE;
            int length = Platform.getInt(null, view);
            if (length <= BaseVariableWidthViewVector.INLINE_SIZE) {
                return UTF8String.fromAddress(null, view + Integer.BYTES, length);
            }

            long buffer = data[Platform.getInt(null, view + 2L * Integer.BYTES)];
            long offset = Integer.toUnsignedLong(Platform.getInt(null, view + 3L * Integer.BYTES));
            return UTF8String.fromAddress(null, buffer + offset, length);
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

        private final long values;

        DateAccessor(DateDayVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
        }

        @Override
        final int getInt(int rowId) {
            return Platform.getInt(null, values + ((long) rowId << 2));
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

        private final long values;
        private final TimeUnit unit;

        TimestampAccessor(TimeStampVector vector) {
            super(vector);
            this.values = valuesAddress(vector);
            this.unit = ((ArrowType.Timestamp) vector.getField().getType()).getUnit();
        }

        @Override
        final long getLong(int rowId) {
            long value = Platform.getLong(null, values + ((long) rowId << 3));
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
