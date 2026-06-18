// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import dev.vortex.spark.read.VortexScanBuilder;
import dev.vortex.spark.write.VortexWriteBuilder;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.SupportsRead;
import org.apache.spark.sql.connector.catalog.SupportsWrite;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCapability;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.connector.write.WriteBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/** Spark V2 {@link Table} of Vortex files that supports both reading and writing. */
public final class VortexTable implements Table, SupportsRead, SupportsWrite {
    private static final String SHORT_NAME = "vortex";

    private final ImmutableList<String> paths;
    private final StructType schema;
    private final Map<String, String> formatOptions;
    private final Transform[] partitionTransforms;

    /** Creates a new VortexTable with read/write support. */
    public VortexTable(
            ImmutableList<String> paths,
            StructType schema,
            Map<String, String> formatOptions,
            Transform[] partitionTransforms) {
        this.paths = paths;
        this.schema = schema;
        this.formatOptions = formatOptions;
        this.partitionTransforms = partitionTransforms;
    }

    /**
     * Creates a new ScanBuilder for this table.
     *
     * <p>The scan builder is pre-configured with all the file paths and columns from this table.
     *
     * @param options scan options
     * @return a new VortexScanBuilder configured for this table
     */
    @Override
    public ScanBuilder newScanBuilder(CaseInsensitiveStringMap options) {
        Map<String, String> opts = Maps.newHashMap();
        opts.putAll(formatOptions);
        opts.putAll(options);
        return new VortexScanBuilder(opts, partitionTransforms)
                .addAllPaths(paths)
                .addAllColumns(Arrays.asList(CatalogV2Util.structTypeToV2Columns(schema)));
    }

    /**
     * Returns the name of this table.
     *
     * <p>The name includes the "vortex" prefix and a comma-separated list of all file paths that comprise this table.
     *
     * @return the table name in the format: vortex."path1,path2,..."
     */
    @Override
    public String name() {
        return String.format("%s.\"%s\"", SHORT_NAME, String.join(",", paths));
    }

    /**
     * Returns the schema of this table.
     *
     * <p>The schema is derived from the columns available for reading, or from the explicit write schema if this table
     * is being used for writing.
     *
     * @return the StructType representing the table schema
     */
    @Override
    public StructType schema() {
        return schema;
    }

    /**
     * Creates a new WriteBuilder for writing data to this table.
     *
     * <p>The WriteBuilder is responsible for configuring and executing write operations to create new Vortex files.
     *
     * @param info logical information about the write operation
     * @return a new VortexWriteBuilder configured for this table
     */
    @Override
    public WriteBuilder newWriteBuilder(LogicalWriteInfo info) {
        // Make sure only one write path was provided.
        String writePath = Iterables.getOnlyElement(paths);
        return new VortexWriteBuilder(writePath, info, formatOptions, partitionTransforms);
    }

    /**
     * Returns the partitioning transforms for this table.
     *
     * @return an array of partition transforms
     */
    @Override
    public Transform[] partitioning() {
        return partitionTransforms;
    }

    /**
     * Returns the capabilities supported by this table.
     *
     * <p>Vortex tables support batch reading and batch writing.
     *
     * @return a set containing TableCapability.BATCH_READ and BATCH_WRITE
     */
    @Override
    public Set<TableCapability> capabilities() {
        return ImmutableSet.of(TableCapability.BATCH_READ, TableCapability.BATCH_WRITE, TableCapability.TRUNCATE);
    }
}
