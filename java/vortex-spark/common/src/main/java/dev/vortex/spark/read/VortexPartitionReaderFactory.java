// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.spark.ArrowProperties;
import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.OptionalLong;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.datasources.v2.EmptyPartitionReader;
import org.apache.spark.sql.execution.datasources.v2.FilePartitionReaderFactory;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Produces one Vortex reader for each file selected by Spark's file index.
 *
 * <p>A listed file without {@link VortexFile#EXTENSION} is not part of the dataset and reads as empty.
 *
 * <p>Runtime filters arrive after Spark has built this factory, so they travel in each {@link VortexFilePartition}.
 * Every task deserializes its own copy of the factory and reads one partition, so the copy notes that partition's
 * filters when Spark asks it for the partition's reader, and applies them to each file reader it builds next.
 */
public final class VortexPartitionReaderFactory extends FilePartitionReaderFactory implements Serializable {
    static {
        // Executors load this class before any Vortex batch, so Arrow sees the settings before it reads them.
        ArrowProperties.configure();
    }

    private static final long serialVersionUID = 1L;

    private final FileSourceOptions fileOptions;
    private final VortexIo io;
    private final VortexOptions formatOptions;
    private final StructType dataSchema;
    private final StructType readDataSchema;
    private final StructType readPartitionSchema;
    private final Filter[] pushedFilters;
    private final boolean caseSensitive;
    /** The most rows a reader needs to return, or 0 for no limit. */
    private final long limit;

    /** The pushed-down top-N, or null. */
    private final VortexTopN topN;

    /** The pushed-down table sample, or null. */
    private final VortexSample sample;

    /** Data filters without a V1 form that the reader converts natively. */
    private final org.apache.spark.sql.catalyst.expressions.Expression[] catalystPredicates;

    /** For the file a pushed-down {@code OFFSET} ends in, the first row to read, by path. */
    private final Map<String, Long> firstRows;

    /** Whether readers keep each file's row order, which the scan reports to Spark. */
    private final boolean ordered;

    /** The pushed filters together with the runtime filters of the partition being read. */
    private transient Filter[] partitionFilters;

    /** The top-N threshold the readers of the partition being read share. */
    private transient TopNThreshold partitionTopN;

    /** The metrics the readers of the partition being read add to. */
    private transient VortexScanMetrics.Task partitionMetrics;

    public VortexPartitionReaderFactory(
            FileSourceOptions fileOptions,
            VortexIo io,
            VortexOptions formatOptions,
            StructType dataSchema,
            StructType readDataSchema,
            StructType readPartitionSchema,
            Filter[] pushedFilters,
            boolean caseSensitive,
            long limit,
            VortexTopN topN,
            VortexSample sample,
            org.apache.spark.sql.catalyst.expressions.Expression[] catalystPredicates,
            Map<String, Long> firstRows,
            boolean ordered) {
        this.fileOptions = fileOptions;
        this.io = io;
        this.formatOptions = formatOptions;
        this.dataSchema = dataSchema;
        this.readDataSchema = readDataSchema;
        this.readPartitionSchema = readPartitionSchema;
        this.pushedFilters = pushedFilters == null ? new Filter[0] : pushedFilters.clone();
        this.caseSensitive = caseSensitive;
        this.limit = limit;
        this.topN = topN;
        this.sample = sample;
        this.catalystPredicates = catalystPredicates.clone();
        this.firstRows = new HashMap<>(firstRows);
        this.ordered = ordered;
    }

    @Override
    public FileSourceOptions options() {
        return fileOptions;
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        Filter[] runtimeFilters =
                partition instanceof VortexFilePartition vortexPartition ? vortexPartition.runtimeFilters() : null;
        partitionFilters = runtimeFilters == null || runtimeFilters.length == 0
                ? pushedFilters
                : concat(pushedFilters, runtimeFilters);
        partitionTopN = topNThreshold();
        partitionMetrics = new VortexScanMetrics.Task();
        return VortexScanMetrics.reporting(super.createColumnarReader(partition), partitionMetrics);
    }

    private TopNThreshold topNThreshold() {
        if (topN == null) {
            return null;
        }
        for (StructField field : readDataSchema.fields()) {
            if (field.name().equals(topN.column())) {
                return TopNThreshold.create(topN, field.dataType()).orElse(null);
            }
        }
        return null;
    }

    private static Filter[] concat(Filter[] first, Filter[] second) {
        Filter[] combined = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, combined, first.length, second.length);
        return combined;
    }

    @Override
    public PartitionReader<InternalRow> buildReader(PartitionedFile file) {
        throw new UnsupportedOperationException("row-based V2 reads are not supported");
    }

    @Override
    public PartitionReader<ColumnarBatch> buildColumnarReader(PartitionedFile file) {
        if (!VortexFile.hasVortexExtension(file.toPath().getName())) {
            return new EmptyPartitionReader<>();
        }

        return new VortexPartitionReader(
                file,
                dataSchema,
                readDataSchema,
                readPartitionSchema,
                io,
                formatOptions,
                partitionFilters == null ? pushedFilters : partitionFilters,
                caseSensitive,
                limit > 0 ? OptionalLong.of(limit) : OptionalLong.empty(),
                partitionTopN,
                sample,
                catalystPredicates,
                firstRows.getOrDefault(file.toPath().toString(), 0L),
                ordered,
                partitionMetrics == null ? new VortexScanMetrics.Task() : partitionMetrics);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }
}
