// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.relocated.org.apache.arrow.vector.complex.MapVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.DateUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.FloatingPointPrecision;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.BinaryType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
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

/**
 * Utility class for converting Spark SQL schemas to Arrow schemas.
 *
 * <p>This enables the conversion of Spark DataFrames to Arrow format for writing to Vortex files.
 */
public final class SparkToArrowSchema {

    private SparkToArrowSchema() {}

    /**
     * Converts a Spark StructType schema to an Arrow Schema.
     *
     * @param sparkSchema the Spark schema to convert
     * @return the corresponding Arrow schema
     */
    public static Schema convert(StructType sparkSchema) {
        List<Field> fields = new ArrayList<>();
        for (StructField sparkField : sparkSchema.fields()) {
            fields.add(convertField(sparkField));
        }
        return new Schema(fields);
    }

    /**
     * Converts a Spark StructField to an Arrow Field.
     *
     * @param sparkField the Spark field to convert
     * @return the corresponding Arrow field
     */
    private static Field convertField(StructField sparkField) {
        return convertField(sparkField.name(), sparkField.dataType(), sparkField.nullable());
    }

    private static Field convertField(String name, DataType sparkType, boolean nullable) {
        if (sparkType instanceof StructType structType) {
            List<Field> children = new ArrayList<>();
            for (StructField childField : structType.fields()) {
                children.add(convertField(childField));
            }
            return new Field(name, new FieldType(nullable, new ArrowType.Struct(), null), children);
        } else if (sparkType instanceof ArrayType arrayType) {
            Field elementField = convertField("element", arrayType.elementType(), arrayType.containsNull());
            return new Field(name, new FieldType(nullable, new ArrowType.List(), null), List.of(elementField));
        } else if (sparkType instanceof MapType mapType) {
            Field keyField = convertField(MapVector.KEY_NAME, mapType.keyType(), false);
            Field valueField = convertField(MapVector.VALUE_NAME, mapType.valueType(), mapType.valueContainsNull());
            Field entries = new Field(
                    MapVector.DATA_VECTOR_NAME,
                    FieldType.notNullable(new ArrowType.Struct()),
                    List.of(keyField, valueField));
            return new Field(name, new FieldType(nullable, new ArrowType.Map(false), null), List.of(entries));
        } else {
            return new Field(name, new FieldType(nullable, convertType(sparkType), null), null);
        }
    }

    /**
     * Converts a Spark DataType to an Arrow ArrowType.
     *
     * @param sparkType the Spark data type to convert
     * @return the corresponding Arrow type
     * @throws UnsupportedOperationException if the type is not supported
     */
    private static ArrowType convertType(DataType sparkType) {
        if (sparkType instanceof BooleanType) {
            return new ArrowType.Bool();
        } else if (sparkType instanceof ByteType) {
            return new ArrowType.Int(8, true);
        } else if (sparkType instanceof ShortType) {
            return new ArrowType.Int(16, true);
        } else if (sparkType instanceof IntegerType) {
            return new ArrowType.Int(32, true);
        } else if (sparkType instanceof LongType) {
            return new ArrowType.Int(64, true);
        } else if (sparkType instanceof FloatType) {
            return new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE);
        } else if (sparkType instanceof DoubleType) {
            return new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE);
        } else if (sparkType instanceof StringType) {
            return new ArrowType.Utf8();
        } else if (sparkType instanceof BinaryType) {
            return new ArrowType.Binary();
        } else if (sparkType instanceof DateType) {
            return new ArrowType.Date(DateUnit.DAY);
        } else if (sparkType instanceof TimestampType) {
            return new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC");
        } else if (sparkType instanceof TimestampNTZType) {
            return new ArrowType.Timestamp(TimeUnit.MICROSECOND, null);
        } else if (sparkType instanceof DecimalType) {
            DecimalType decimal = (DecimalType) sparkType;
            return new ArrowType.Decimal(decimal.precision(), decimal.scale(), 128);
        } else if (sparkType instanceof ArrayType) {
            return new ArrowType.List();
        } else if (sparkType instanceof StructType) {
            return new ArrowType.Struct();
        } else {
            throw new UnsupportedOperationException("Unsupported Spark type for Arrow conversion: "
                    + sparkType.getClass().getName());
        }
    }
}
