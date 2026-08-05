// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexOptions;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.InputPartition;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VortexBatchExec} input-partition planning.
 *
 * <p>Characterizes how explicit {@code .vortex} paths are planned: one {@link VortexFilePartition} per file, each
 * carrying the requested read schema and the Hive-style partition values parsed from its own path. All paths used here
 * end in {@code .vortex} so planning never lists directories through native I/O.
 */
final class VortexBatchExecTest {

    private static final List<Column> COLUMNS = List.of(
            Column.create("id", org.apache.spark.sql.types.DataTypes.IntegerType),
            Column.create("name", org.apache.spark.sql.types.DataTypes.StringType));

    private static VortexBatchExec execFor(List<String> paths) {
        return new VortexBatchExec(paths, COLUMNS, VortexOptions.empty(), new Predicate[0]);
    }

    @Test
    @DisplayName("Plans one input partition per .vortex file")
    void plansOnePartitionPerFile() {
        VortexBatchExec exec = execFor(List.of("/data/a.vortex", "/data/b.vortex", "/data/c.vortex"));

        InputPartition[] partitions = exec.planInputPartitions();

        assertEquals(3, partitions.length);
        for (InputPartition partition : partitions) {
            assertInstanceOf(VortexFilePartition.class, partition);
            assertEquals(1, ((VortexFilePartition) partition).paths().size());
        }
    }

    @Test
    @DisplayName("Each partition carries exactly its own file path, in input order")
    void partitionsKeepInputOrder() {
        VortexBatchExec exec = execFor(List.of("/data/first.vortex", "/data/second.vortex"));

        InputPartition[] partitions = exec.planInputPartitions();

        assertEquals(List.of("/data/first.vortex"), ((VortexFilePartition) partitions[0]).paths());
        assertEquals(List.of("/data/second.vortex"), ((VortexFilePartition) partitions[1]).paths());
    }

    @Test
    @DisplayName("Partitions carry the requested read schema")
    void partitionsCarryReadSchema() {
        VortexBatchExec exec = execFor(List.of("/data/a.vortex"));

        VortexFilePartition partition = (VortexFilePartition) exec.planInputPartitions()[0];

        assertEquals(2, partition.readSchema().size());
        assertEquals("id", partition.readSchema().fields()[0].name());
        assertEquals("name", partition.readSchema().fields()[1].name());
    }

    @Test
    @DisplayName("Hive-style partition values are parsed from each file's own path")
    void partitionValuesParsedPerFile() {
        VortexBatchExec exec = execFor(
                List.of("/tbl/year=2024/month=01/a.vortex", "/tbl/year=2025/month=02/b.vortex", "/tbl/plain.vortex"));

        InputPartition[] partitions = exec.planInputPartitions();

        assertEquals(Map.of("year", "2024", "month", "01"), ((VortexFilePartition) partitions[0]).partitionValues());
        assertEquals(Map.of("year", "2025", "month", "02"), ((VortexFilePartition) partitions[1]).partitionValues());
        assertEquals(Map.of(), ((VortexFilePartition) partitions[2]).partitionValues());
    }

    @Test
    @DisplayName("Format options are propagated to every partition")
    void formatOptionsPropagated() {
        VortexOptions options = VortexOptions.of(Map.of(VortexOptions.WORKER_THREADS, "8"));
        VortexBatchExec exec = new VortexBatchExec(List.of("/data/a.vortex"), COLUMNS, options, new Predicate[0]);

        VortexFilePartition partition = (VortexFilePartition) exec.planInputPartitions()[0];

        assertEquals(8, partition.formatOptions().workerThreads());
        // Also pin the raw spelling: the native bindings receive asMap(), so a key mangled in
        // transit would still resolve through the case-insensitive accessor above.
        assertEquals("8", partition.formatOptions().asMap().get(VortexOptions.WORKER_THREADS));
    }
}
