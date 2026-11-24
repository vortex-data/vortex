// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import java.sql.Timestamp;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.sources.EqualTo;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.sources.GreaterThan;
import org.apache.spark.sql.sources.IsNotNull;
import org.apache.spark.sql.sources.IsNull;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;

final class ReadTypesTest {
    private static final StructType SCHEMA = new StructType()
            .add("id", DataTypes.LongType)
            .add("ts", DataTypes.TimestampType)
            .add("st", new StructType().add("a", DataTypes.LongType).add("b", DataTypes.StringType))
            .add("s", DataTypes.StringType)
            .add("geom", DataTypes.BinaryType)
            .add("v", new StructType().add("metadata", DataTypes.BinaryType).add("value", DataTypes.BinaryType));

    private static final List<Field> FILE = List.of(
            Field.nullable("id", new ArrowType.Int(64, true)),
            Field.nullable("ts", new ArrowType.Timestamp(TimeUnit.MILLISECOND, "UTC")),
            new Field(
                    "st",
                    FieldType.nullable(ArrowType.Struct.INSTANCE),
                    List.of(
                            Field.nullable("a", new ArrowType.Int(64, true)),
                            Field.nullable("b", ArrowType.Utf8.INSTANCE))),
            Field.nullable("s", ArrowType.Utf8.INSTANCE),
            new Field("geom", extension(ArrowType.Binary.INSTANCE, "geoarrow.wkb"), null),
            new Field(
                    "v",
                    extension(ArrowType.Struct.INSTANCE, "arrow.parquet.variant"),
                    List.of(
                            Field.nullable("metadata", ArrowType.Binary.INSTANCE),
                            Field.nullable("value", ArrowType.Binary.INSTANCE))));

    @Test
    void nullChecksArePushedOnATimestampInAnotherUnit() {
        Filter[] filters = {
            new IsNull("ts"), new IsNotNull("ts"), new GreaterThan("ts", Timestamp.valueOf("2026-01-01 00:00:00"))
        };
        StructType filterable = ReadTypes.filterable(SCHEMA, FILE, filters);
        assertTrue(SparkFilterToVortexExpression.isPushable(filters[0], filterable));
        assertTrue(SparkFilterToVortexExpression.isPushable(filters[1], filterable));
        assertFalse(SparkFilterToVortexExpression.isPushable(filters[2], filterable));
    }

    @Test
    void comparisonsAreNotPushedOnAnExtensionLeaf() {
        Filter[] filters = {new IsNotNull("geom"), new EqualTo("geom", new byte[] {1})};
        StructType filterable = ReadTypes.filterable(SCHEMA, FILE, filters);
        assertTrue(SparkFilterToVortexExpression.isPushable(filters[0], filterable));
        assertFalse(SparkFilterToVortexExpression.isPushable(filters[1], filterable));
    }

    @Test
    void fieldsOfAnExtensionStructAreNotFiltered() {
        Filter[] filters = {new IsNotNull("v"), new IsNotNull("v.metadata"), new EqualTo("v.metadata", new byte[] {1})};
        StructType filterable = ReadTypes.filterable(SCHEMA, FILE, filters);
        assertTrue(SparkFilterToVortexExpression.isPushable(filters[0], filterable));
        assertFalse(SparkFilterToVortexExpression.isPushable(filters[1], filterable));
        assertFalse(SparkFilterToVortexExpression.isPushable(filters[2], filterable));
    }

    private static FieldType extension(ArrowType storage, String name) {
        return new FieldType(true, storage, null, Map.of(ArrowType.ExtensionType.EXTENSION_METADATA_KEY_NAME, name));
    }
}
