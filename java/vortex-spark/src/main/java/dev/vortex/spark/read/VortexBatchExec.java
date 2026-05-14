// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.collect.ImmutableMap;
import dev.vortex.jni.NativeFiles;
import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexSparkSession;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.types.StructType;

/** Execution source for batch scans of Vortex file tables. */
public final class VortexBatchExec implements Batch {
    private final List<String> paths;
    private final StructType readSchema;
    private final Map<String, String> formatOptions;
    private List<String> resolvedPaths;

    /**
     * Creates a new VortexBatchExec for scanning the specified Vortex files.
     *
     * @param paths the list of file paths to scan
     * @param columns the list of columns to read from the files
     */
    public VortexBatchExec(List<String> paths, List<Column> columns, Map<String, String> formatOptions) {
        this.paths = List.copyOf(paths);
        this.readSchema = CatalogV2Util.v2ColumnsToStructType(columns.toArray(new Column[0]));
        this.formatOptions = Map.copyOf(formatOptions);
    }

    /**
     * Plans the input partitions for this batch scan.
     *
     * <p>Directory-like entries are expanded to concrete {@code .vortex} files. Each resolved file becomes its own
     * {@link VortexFilePartition}; the partition carries the paths the reader should open, the requested schema, and
     * any Hive-style partition values parsed out of the path.
     */
    @Override
    public InputPartition[] planInputPartitions() {
        resolvedPaths = resolvePaths();
        return resolvedPaths.stream()
                .map(path -> {
                    Map<String, String> partVals = PartitionPathUtils.parsePartitionValues(path);
                    return new VortexFilePartition(
                            List.of(path), readSchema, formatOptions, ImmutableMap.copyOf(partVals));
                })
                .toArray(InputPartition[]::new);
    }

    @Override
    public PartitionReaderFactory createReaderFactory() {
        List<String> files = resolvedPaths != null ? resolvedPaths : resolvePaths();
        Set<String> partitionColumns = collectPartitionColumnNames(files);
        List<String> dataColumnNames = Arrays.stream(readSchema.fieldNames())
                .filter(name -> !partitionColumns.contains(name))
                .collect(Collectors.toList());
        return new VortexPartitionReaderFactory(dataColumnNames, formatOptions);
    }

    private List<String> resolvePaths() {
        var session = VortexSparkSession.get(formatOptions);
        return paths.stream()
                .flatMap(path -> {
                    if (path.endsWith(".vortex")) {
                        return Stream.of(path);
                    }
                    return NativeFiles.listFiles(session, path, formatOptions).stream();
                })
                .collect(Collectors.toList());
    }

    private static Set<String> collectPartitionColumnNames(List<String> files) {
        Set<String> all = new HashSet<>();
        for (String path : files) {
            all.addAll(PartitionPathUtils.parsePartitionValues(path).keySet());
        }
        return all;
    }
}
