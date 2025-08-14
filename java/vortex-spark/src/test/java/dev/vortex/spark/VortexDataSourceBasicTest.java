// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.*;

import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for VortexDataSourceV2 basic functionality.
 */
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
        var arrowSchema = dev.vortex.spark.write.SparkToArrowSchema.convert(sparkSchema);

        // Verify conversion
        assertNotNull(arrowSchema, "Arrow schema should not be null");
        assertEquals(4, arrowSchema.getFields().size(), "Arrow schema should have same number of fields");

        // Verify field names are preserved
        assertEquals("id", arrowSchema.getFields().get(0).getName());
        assertEquals("name", arrowSchema.getFields().get(1).getName());
        assertEquals("value", arrowSchema.getFields().get(2).getName());
        assertEquals("active", arrowSchema.getFields().get(3).getName());
    }

    @Test
    @DisplayName("VortexWriterCommitMessage should store metadata correctly")
    public void testWriterCommitMessage() {
        String testPath = "/test/path/file.vortex";
        long recordCount = 1000;
        long bytesWritten = 50000;

        var message = new dev.vortex.spark.write.VortexWriterCommitMessage(testPath, recordCount, bytesWritten);

        assertEquals(testPath, message.getFilePath());
        assertEquals(recordCount, message.getRecordCount());
        assertEquals(bytesWritten, message.getBytesWritten());
    }
}
