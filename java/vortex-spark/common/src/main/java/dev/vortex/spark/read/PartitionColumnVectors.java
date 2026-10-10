// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.types.BinaryType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DayTimeIntervalType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.sql.types.YearMonthIntervalType;
import org.apache.spark.sql.vectorized.ColumnVector;

/** Creates constant column vectors from Spark's typed partition values. */
final class PartitionColumnVectors {
    private PartitionColumnVectors() {}

    static ColumnVector[] create(int rowCount, StructType schema, InternalRow values) {
        StructField[] fields = schema.fields();
        ColumnVector[] vectors = new ColumnVector[fields.length];
        for (int i = 0; i < fields.length; i++) {
            vectors[i] = create(rowCount, fields[i], values, i);
        }
        return vectors;
    }

    static ConstantColumnVector create(int rowCount, StructField field, InternalRow values, int ordinal) {
        ConstantColumnVector vector = new ConstantColumnVector(rowCount, field.dataType());
        if (values.isNullAt(ordinal)) {
            vector.setNull();
            return vector;
        }
        vector.setNotNull();
        DataType type = field.dataType();
        if (type instanceof BooleanType) {
            vector.setBoolean(values.getBoolean(ordinal));
        } else if (type instanceof ByteType) {
            vector.setByte(values.getByte(ordinal));
        } else if (type instanceof ShortType) {
            vector.setShort(values.getShort(ordinal));
        } else if (type instanceof IntegerType || type instanceof DateType || type instanceof YearMonthIntervalType) {
            vector.setInt(values.getInt(ordinal));
        } else if (type instanceof LongType
                || type instanceof TimestampType
                || type instanceof TimestampNTZType
                || type instanceof DayTimeIntervalType) {
            vector.setLong(values.getLong(ordinal));
        } else if (type instanceof FloatType) {
            vector.setFloat(values.getFloat(ordinal));
        } else if (type instanceof DoubleType) {
            vector.setDouble(values.getDouble(ordinal));
        } else if (type instanceof StringType) {
            vector.setUtf8String(values.getUTF8String(ordinal));
        } else if (type instanceof BinaryType) {
            vector.setBinary(values.getBinary(ordinal));
        } else if (type instanceof DecimalType decimalType) {
            vector.setDecimal(
                    values.getDecimal(ordinal, decimalType.precision(), decimalType.scale()), decimalType.precision());
        } else {
            vector.close();
            throw new UnsupportedOperationException("Unsupported partition column type: " + type);
        }
        return vector;
    }
}
