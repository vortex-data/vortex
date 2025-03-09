/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.spark;

import com.google.common.collect.Streams;
import dev.vortex.api.DType;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampType$;

/**
 * Helpers for converting between Spark and Vortex type systems.
 */
public final class SparkTypes {
    private SparkTypes() {}

    /**
     * Convert a STRUCT Vortex type to a Spark {@link DataType}.
     */
    public static DataType toDataType(DType dType) {
        switch (dType.getVariant()) {
            case NULL:
                return DataTypes.NullType;
            case BOOL:
                return DataTypes.BooleanType;
            case PRIMITIVE_U8:
            case PRIMITIVE_I8:
                return DataTypes.ByteType;
            case PRIMITIVE_U16:
            case PRIMITIVE_I16:
                return DataTypes.ShortType;
            case PRIMITIVE_U32:
            case PRIMITIVE_I32:
                return DataTypes.IntegerType;
            case PRIMITIVE_U64:
            case PRIMITIVE_I64:
                return DataTypes.LongType;
            case PRIMITIVE_F16:
                throw new IllegalArgumentException("Spark does not support f16");
            case PRIMITIVE_F32:
                return DataTypes.FloatType;
            case PRIMITIVE_F64:
                return DataTypes.DoubleType;
            case UTF8:
                return DataTypes.StringType;
            case BINARY:
                return DataTypes.BinaryType;
            case STRUCT:
                // For each of the inner struct fields, we capture them together here.
                var struct = new StructType();

                // TODO(aduffy): FieldTypes leaks. DType_field_dtype will box a new value, we need to iterate
                //  and free each of the boxed values.

                Streams.forEachPair(
                        dType.getFieldNames().stream(),
                        dType.getFieldTypes().stream(),
                        (name, type) -> struct.add(name, toDataType(type)));
                return struct;
            case LIST:
                return DataTypes.createArrayType(toDataType(dType.getElementType()), dType.isNullable());
            case EXTENSION:
                /*
                 * Spark does not have a direct equivalent for many of the temporal types we support in Vortex or Arrow.
                 * Notably, there is no DATE type, and timestamps can have at most µs-level precision.
                 *
                 * This means that we need to "cheat" a little in how we convert into Spark's type system. We support
                 * the following conversions:
                 *
                 *  1. Vortex DATE -> Spark TIMESTAMP (with 00:00:00 time and local timezone)
                 *  2. Vortex TIMSTAMP -> Spark TIMESTAMP, with precision truncated to µs
                 *  3. Vortex TIME -> not supported
                 */

                if (dType.isTime()) {
                    throw new IllegalArgumentException("Spark does not support Vortex TIME data type");
                }

                if (dType.isDate() || dType.isTimestamp()) {
                    return TimestampType$.MODULE$;
                }

                // TODO(aduffy): other extension types
                throw new IllegalArgumentException("Unsupported non-temporal extension type");
            default:
                throw new IllegalArgumentException("unreachable");
        }
    }

    /**
     * Convert a STRUCT Vortex type to a Spark {@link Column}.
     */
    public static Column[] toColumns(DType dType) {
        return Streams.zip(dType.getFieldNames().stream(), dType.getFieldTypes().stream(), (name, fieldType) -> {
                    var dataType = toDataType(fieldType);
                    return Column.create(name, dataType);
                })
                .toArray(Column[]::new);
    }
}
