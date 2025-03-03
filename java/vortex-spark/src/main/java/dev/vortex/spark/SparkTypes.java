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
import org.apache.spark.sql.types.*;

/**
 * Helpers for converting between Spark and Vortex type systems.
 */
public final class SparkTypes {
    private SparkTypes() {}

    /**
     * Convert a STRUCT Vortex type to a Spark {@link DataType}.
     */
    public static DataType toDataType(DType dType) {
        return switch (dType.getVariant()) {
            case NULL -> DataTypes.NullType;
            case BOOL -> DataTypes.BooleanType;
            case PRIMITIVE_U8, PRIMITIVE_I8 -> DataTypes.ByteType;
            case PRIMITIVE_U16, PRIMITIVE_I16 -> DataTypes.ShortType;
            case PRIMITIVE_U32, PRIMITIVE_I32 -> DataTypes.IntegerType;
            case PRIMITIVE_U64, PRIMITIVE_I64 -> DataTypes.LongType;
            case PRIMITIVE_F16 -> {
                throw new IllegalArgumentException("Spark does not support f16");
            }
            case PRIMITIVE_F32 -> DataTypes.FloatType;
            case PRIMITIVE_F64 -> DataTypes.DoubleType;
            case UTF8 -> DataTypes.StringType;
            case BINARY -> DataTypes.BinaryType;
            case STRUCT -> {
                // For each of the inner struct fields, we capture them together here.
                var struct = new StructType();

                // TODO(aduffy): FieldTypes leaks. DType_field_dtype will box a new value, we need to iterate
                //  and free each of the boxed values.

                Streams.forEachPair(
                        dType.getFieldNames().stream(),
                        dType.getFieldTypes().stream(),
                        (name, type) -> struct.add(name, toDataType(type)));
                yield struct;
            }
            case LIST -> DataTypes.createArrayType(toDataType(dType.getElementType()), dType.isNullable());
            case EXTENSION -> {
                // TODO(aduffy): temporal types
                throw new UnsupportedOperationException("TODO(aduffy): implement extension types for temporal");
            }
        };
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
