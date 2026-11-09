// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.relocated.org.apache.arrow.vector.types.FloatingPointPrecision;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.FieldType;
import dev.vortex.spark.write.SparkToArrowSchema;
import java.util.List;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Unit tests for VortexDataSourceV2 basic functionality. */
public final class VortexDataSourceBasicTest {

    @Test
    @DisplayName("VortexDataSourceV2 should provide correct short name")
    public void testShortName() {
        VortexDataSourceV2 dataSource = new VortexDataSourceV2();
        assertEquals("vortex", dataSource.shortName(), "Data source should register with short name 'vortex'");
    }

    @Test
    @DisplayName("SparkToArrowSchema should convert basic types")
    public void testSparkToArrowSchemaConversion() {
        // Create a simple Spark schema
        StructType sparkSchema = DataTypes.createStructType(new StructField[] {
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, true),
            DataTypes.createStructField("value", DataTypes.DoubleType, false),
            DataTypes.createStructField("active", DataTypes.BooleanType, true)
        });

        // Convert to Arrow schema
        var arrowSchema = SparkToArrowSchema.convert(sparkSchema);

        assertEquals(
                List.of(
                        Field.notNullable("id", new ArrowType.Int(32, true)),
                        Field.nullable("name", new ArrowType.Utf8()),
                        Field.notNullable("value", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                        Field.nullable("active", new ArrowType.Bool())),
                arrowSchema.getFields());
    }

    @Test
    @DisplayName("SparkToArrowSchema should convert nested types")
    public void testNestedSparkToArrowSchemaConversion() {
        // Create a more complex spark schema
        StructType sparkSchema = DataTypes.createStructType(new StructField[] {
            DataTypes.createStructField(
                    "inner",
                    DataTypes.createStructType(new StructField[] {
                        DataTypes.createStructField("id", DataTypes.IntegerType, false),
                        DataTypes.createStructField("name", DataTypes.StringType, true),
                        DataTypes.createStructField("value", DataTypes.DoubleType, false),
                        DataTypes.createStructField("active", DataTypes.BooleanType, true)
                    }),
                    false)
        });

        // Convert to Arrow schema
        var arrowSchema = SparkToArrowSchema.convert(sparkSchema);

        assertEquals(
                List.of(new Field(
                        "inner",
                        FieldType.notNullable(new ArrowType.Struct()),
                        List.of(
                                Field.notNullable("id", new ArrowType.Int(32, true)),
                                Field.nullable("name", new ArrowType.Utf8()),
                                Field.notNullable("value", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                                Field.nullable("active", new ArrowType.Bool())))),
                arrowSchema.getFields());
    }
}
