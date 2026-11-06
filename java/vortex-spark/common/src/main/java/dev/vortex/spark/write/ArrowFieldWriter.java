// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.relocated.org.apache.arrow.vector.BigIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.BitVector;
import dev.vortex.relocated.org.apache.arrow.vector.DateDayVector;
import dev.vortex.relocated.org.apache.arrow.vector.DecimalVector;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.Float4Vector;
import dev.vortex.relocated.org.apache.arrow.vector.Float8Vector;
import dev.vortex.relocated.org.apache.arrow.vector.IntVector;
import dev.vortex.relocated.org.apache.arrow.vector.SmallIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMicroTZVector;
import dev.vortex.relocated.org.apache.arrow.vector.TimeStampMicroVector;
import dev.vortex.relocated.org.apache.arrow.vector.TinyIntVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarBinaryVector;
import dev.vortex.relocated.org.apache.arrow.vector.VarCharVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.MapVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.catalyst.expressions.SpecializedGetters;
import org.apache.spark.sql.catalyst.util.ArrayData;
import org.apache.spark.sql.catalyst.util.MapData;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.BinaryType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * Appends one Spark column of a row to an Arrow vector, the way Spark's own
 * {@code org.apache.spark.sql.execution.arrow.ArrowWriter} does (that class cannot be used here because the connector
 * relocates Arrow).
 *
 * <p>Each writer counts the values it has appended to its vector, so nested writers (list elements, map entries) know
 * their next slot without consulting the vector. The type dispatch happens once, in {@link #create}, instead of once
 * per value.
 */
abstract class ArrowFieldWriter<V extends FieldVector> {
    final V vector;
    int count;

    private ArrowFieldWriter(V vector) {
        this.vector = vector;
    }

    static ArrowFieldWriter<?> create(FieldVector vector, DataType dataType) {
        if (dataType instanceof BooleanType) {
            return new BooleanWriter((BitVector) vector);
        } else if (dataType instanceof ByteType) {
            return new ByteWriter((TinyIntVector) vector);
        } else if (dataType instanceof ShortType) {
            return new ShortWriter((SmallIntVector) vector);
        } else if (dataType instanceof IntegerType) {
            return new IntegerWriter((IntVector) vector);
        } else if (dataType instanceof LongType) {
            return new LongWriter((BigIntVector) vector);
        } else if (dataType instanceof FloatType) {
            return new FloatWriter((Float4Vector) vector);
        } else if (dataType instanceof DoubleType) {
            return new DoubleWriter((Float8Vector) vector);
        } else if (dataType instanceof StringType) {
            return new StringWriter((VarCharVector) vector);
        } else if (dataType instanceof BinaryType) {
            return new BinaryWriter((VarBinaryVector) vector);
        } else if (dataType instanceof DateType) {
            return new DateWriter((DateDayVector) vector);
        } else if (dataType instanceof TimestampType) {
            return new TimestampWriter((TimeStampMicroTZVector) vector);
        } else if (dataType instanceof TimestampNTZType) {
            return new TimestampNTZWriter((TimeStampMicroVector) vector);
        } else if (dataType instanceof DecimalType decimalType) {
            return new DecimalWriter((DecimalVector) vector, decimalType.precision(), decimalType.scale());
        } else if (dataType instanceof StructType structType) {
            StructVector structVector = (StructVector) vector;
            StructField[] fields = structType.fields();
            ArrowFieldWriter<?>[] children = new ArrowFieldWriter<?>[fields.length];
            for (int i = 0; i < fields.length; i++) {
                children[i] = create((FieldVector) structVector.getChildByOrdinal(i), fields[i].dataType());
            }
            return new StructWriter(structVector, children);
        } else if (dataType instanceof ArrayType arrayType) {
            ListVector listVector = (ListVector) vector;
            return new ArrayWriter(listVector, create(listVector.getDataVector(), arrayType.elementType()));
        } else if (dataType instanceof MapType mapType) {
            MapVector mapVector = (MapVector) vector;
            StructVector entries = (StructVector) mapVector.getDataVector();
            return new MapWriter(
                    mapVector,
                    entries,
                    create(entries.getChild(MapVector.KEY_NAME), mapType.keyType()),
                    create(entries.getChild(MapVector.VALUE_NAME), mapType.valueType()));
        } else {
            throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }
    }

    /**
     * Appends the value at {@code ordinal} of {@code row}, or a null, to the vector.
     *
     * <p>A null slot is left as allocated, so a null struct costs nothing per leaf below it. That relies on the vectors
     * being freshly allocated for each batch: Arrow zeroes their validity and offsets, and fills the offsets of skipped
     * slots at the next value or at {@code setValueCount}.
     */
    final void write(SpecializedGetters row, int ordinal) {
        if (!row.isNullAt(ordinal)) {
            setValue(row, ordinal);
        }
        count++;
    }

    abstract void setValue(SpecializedGetters row, int ordinal);

    /** Forgets the values appended so far. Called once their buffers have been handed to the native writer. */
    void reset() {
        count = 0;
    }

    private static final class BooleanWriter extends ArrowFieldWriter<BitVector> {
        BooleanWriter(BitVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getBoolean(ordinal) ? 1 : 0);
        }
    }

    private static final class ByteWriter extends ArrowFieldWriter<TinyIntVector> {
        ByteWriter(TinyIntVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getByte(ordinal));
        }
    }

    private static final class ShortWriter extends ArrowFieldWriter<SmallIntVector> {
        ShortWriter(SmallIntVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getShort(ordinal));
        }
    }

    private static final class IntegerWriter extends ArrowFieldWriter<IntVector> {
        IntegerWriter(IntVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getInt(ordinal));
        }
    }

    private static final class LongWriter extends ArrowFieldWriter<BigIntVector> {
        LongWriter(BigIntVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getLong(ordinal));
        }
    }

    private static final class FloatWriter extends ArrowFieldWriter<Float4Vector> {
        FloatWriter(Float4Vector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getFloat(ordinal));
        }
    }

    private static final class DoubleWriter extends ArrowFieldWriter<Float8Vector> {
        DoubleWriter(Float8Vector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getDouble(ordinal));
        }
    }

    private static final class StringWriter extends ArrowFieldWriter<VarCharVector> {
        StringWriter(VarCharVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            UTF8String value = row.getUTF8String(ordinal);
            int length = value.numBytes();
            // Reserve the slot first: it may reallocate the data buffer. Then copy straight from the row
            // (on- or off-heap) into the Arrow buffer, without staging the bytes in a byte[].
            vector.setValueLengthSafe(count, length);
            value.writeToMemory(null, vector.getDataBufferAddress() + vector.getStartOffset(count));
            vector.setIndexDefined(count);
        }
    }

    private static final class BinaryWriter extends ArrowFieldWriter<VarBinaryVector> {
        BinaryWriter(VarBinaryVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getBinary(ordinal));
        }
    }

    private static final class DateWriter extends ArrowFieldWriter<DateDayVector> {
        DateWriter(DateDayVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getInt(ordinal));
        }
    }

    private static final class TimestampWriter extends ArrowFieldWriter<TimeStampMicroTZVector> {
        TimestampWriter(TimeStampMicroTZVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getLong(ordinal));
        }
    }

    private static final class TimestampNTZWriter extends ArrowFieldWriter<TimeStampMicroVector> {
        TimestampNTZWriter(TimeStampMicroVector vector) {
            super(vector);
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            vector.setSafe(count, row.getLong(ordinal));
        }
    }

    private static final class DecimalWriter extends ArrowFieldWriter<DecimalVector> {
        private final int precision;
        private final int scale;

        DecimalWriter(DecimalVector vector, int precision, int scale) {
            super(vector);
            this.precision = precision;
            this.scale = scale;
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            Decimal value = row.getDecimal(ordinal, precision, scale);
            if (precision <= Decimal.MAX_LONG_DIGITS()) {
                vector.setSafe(count, value.toUnscaledLong());
            } else {
                vector.setSafe(count, value.toJavaBigDecimal());
            }
        }
    }

    private static final class StructWriter extends ArrowFieldWriter<StructVector> {
        private final ArrowFieldWriter<?>[] children;

        StructWriter(StructVector vector, ArrowFieldWriter<?>[] children) {
            super(vector);
            this.children = children;
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            InternalRow value = row.getStruct(ordinal, children.length);
            vector.setIndexDefined(count);
            for (int i = 0; i < children.length; i++) {
                // Children skip the slots of null structs, so catch them up to this one.
                children[i].count = count;
                children[i].write(value, i);
            }
        }

        @Override
        void reset() {
            super.reset();
            for (ArrowFieldWriter<?> child : children) {
                child.reset();
            }
        }
    }

    private static final class ArrayWriter extends ArrowFieldWriter<ListVector> {
        private final ArrowFieldWriter<?> elements;

        ArrayWriter(ListVector vector, ArrowFieldWriter<?> elements) {
            super(vector);
            this.elements = elements;
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            ArrayData value = row.getArray(ordinal);
            int length = value.numElements();
            vector.startNewValue(count);
            for (int i = 0; i < length; i++) {
                elements.write(value, i);
            }
            vector.endValue(count, length);
        }

        @Override
        void reset() {
            super.reset();
            elements.reset();
        }
    }

    private static final class MapWriter extends ArrowFieldWriter<MapVector> {
        private final StructVector entries;
        private final ArrowFieldWriter<?> keys;
        private final ArrowFieldWriter<?> values;

        MapWriter(MapVector vector, StructVector entries, ArrowFieldWriter<?> keys, ArrowFieldWriter<?> values) {
            super(vector);
            this.entries = entries;
            this.keys = keys;
            this.values = values;
        }

        @Override
        void setValue(SpecializedGetters row, int ordinal) {
            MapData value = row.getMap(ordinal);
            ArrayData keyArray = value.keyArray();
            ArrayData valueArray = value.valueArray();
            int length = value.numElements();
            vector.startNewValue(count);
            for (int i = 0; i < length; i++) {
                entries.setIndexDefined(keys.count);
                keys.write(keyArray, i);
                values.write(valueArray, i);
            }
            vector.endValue(count, length);
        }

        @Override
        void reset() {
            super.reset();
            keys.reset();
            values.reset();
        }
    }
}
