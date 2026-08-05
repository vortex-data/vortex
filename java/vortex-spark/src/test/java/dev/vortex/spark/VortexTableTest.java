// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.google.common.collect.ImmutableList;
import dev.vortex.spark.read.VortexScanBuilder;
import dev.vortex.spark.write.VortexWriteBuilder;
import java.util.Map;
import java.util.NoSuchElementException;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VortexTable}, the Spark V2 table of Vortex files.
 *
 * <p>Characterizes the table-level contract Spark relies on: which capabilities are advertised, how the table name is
 * rendered from its paths, and the read/write builders it hands back. Notably, writes accept exactly one path, so
 * {@link VortexTable#newWriteBuilder} rejects tables built from several paths.
 */
final class VortexTableTest {

    private static final StructType SCHEMA = new StructType(new StructField[] {
        new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
        new StructField("name", DataTypes.StringType, true, Metadata.empty())
    });

    private static VortexTable tableFor(String... paths) {
        return new VortexTable(ImmutableList.copyOf(paths), SCHEMA, VortexOptions.empty(), new Transform[0]);
    }

    @Test
    @DisplayName("Advertises exactly batch read, batch write and truncate")
    void capabilitiesAreBatchReadWriteAndTruncate() {
        assertEquals(
                java.util.Set.of(TableCapability.BATCH_READ, TableCapability.BATCH_WRITE, TableCapability.TRUNCATE),
                tableFor("/data/a.vortex").capabilities());
    }

    @Test
    @DisplayName("Table name joins every path with commas under the vortex prefix")
    void nameJoinsAllPaths() {
        assertEquals("vortex.\"/data/a.vortex\"", tableFor("/data/a.vortex").name());
        assertEquals(
                "vortex.\"/data/a.vortex,/data/b.vortex\"",
                tableFor("/data/a.vortex", "/data/b.vortex").name());
    }

    @Test
    @DisplayName("Schema and partitioning are returned as supplied")
    void schemaAndPartitioningRoundTrip() {
        Transform[] transforms = new Transform[] {Expressions.identity("year")};
        VortexTable table = new VortexTable(ImmutableList.of("/tbl"), SCHEMA, VortexOptions.empty(), transforms);

        assertEquals(SCHEMA, table.schema());
        assertArrayEquals(transforms, table.partitioning());
    }

    @Test
    @DisplayName("Scan builder projects the full table schema by default")
    void scanBuilderStartsFromTableSchema() {
        VortexTable table = tableFor("/data/a.vortex");

        var builder = table.newScanBuilder(new CaseInsensitiveStringMap(Map.of()));

        assertInstanceOf(VortexScanBuilder.class, builder);
        Scan scan = builder.build();
        assertEquals(SCHEMA, scan.readSchema());
    }

    @Test
    @DisplayName("Scan description carries every table path")
    void scanDescriptionCarriesPaths() {
        VortexTable table = tableFor("/data/a.vortex", "/data/b.vortex");

        Scan scan = table.newScanBuilder(new CaseInsensitiveStringMap(Map.of())).build();

        String description = scan.description();
        assertTrue(description.startsWith("VortexScan{"), description);
        assertTrue(description.contains("/data/a.vortex"), description);
        assertTrue(description.contains("/data/b.vortex"), description);
        assertTrue(description.contains("pushedPredicates=[]"), description);
    }

    @Test
    @DisplayName("Write builder is created for a single-path table")
    void writeBuilderAcceptsSinglePath() {
        VortexTable table = tableFor("/data/out");

        assertInstanceOf(VortexWriteBuilder.class, table.newWriteBuilder(writeInfo()));
    }

    @Test
    @DisplayName("Writing a table of several paths is rejected: there is no single output path")
    void writeBuilderRejectsMultiplePaths() {
        VortexTable table = tableFor("/data/a.vortex", "/data/b.vortex");

        assertThrows(IllegalArgumentException.class, () -> table.newWriteBuilder(writeInfo()));
    }

    @Test
    @DisplayName("Writing a table with no path is rejected")
    void writeBuilderRejectsNoPaths() {
        VortexTable table = tableFor();

        assertThrows(NoSuchElementException.class, () -> table.newWriteBuilder(writeInfo()));
    }

    @Test
    @DisplayName("Scan options override the table's own options, whatever the spelling")
    void scanOptionsOverrideTableOptions() {
        VortexTable table = new VortexTable(
                ImmutableList.of("/data/a.vortex"),
                SCHEMA,
                VortexOptions.of(Map.of(VortexOptions.WORKER_THREADS, "4", "aws_region", "us-east-1")),
                new Transform[0]);

        var builder = table.newScanBuilder(new CaseInsensitiveStringMap(Map.of(VortexOptions.WORKER_THREADS, "16")));
        VortexFilePartition partition =
                (VortexFilePartition) ((org.apache.spark.sql.connector.read.SupportsReportStatistics) builder.build())
                        .toBatch()
                        .planInputPartitions()[0];

        assertEquals(16, partition.formatOptions().workerThreads());
        // The unrelated table option survives, and the overridden one is not left behind twice.
        assertEquals("us-east-1", partition.formatOptions().asMap().get("aws_region"));
        assertEquals(2, partition.formatOptions().asMap().size());
    }

    private static LogicalWriteInfo writeInfo() {
        return new LogicalWriteInfo() {
            @Override
            public String queryId() {
                return "query-1";
            }

            @Override
            public StructType schema() {
                return SCHEMA;
            }

            @Override
            public CaseInsensitiveStringMap options() {
                return new CaseInsensitiveStringMap(Map.of());
            }
        };
    }
}
