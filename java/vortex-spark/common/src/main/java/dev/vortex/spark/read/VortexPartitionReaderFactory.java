// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.VortexIo;
import java.io.Serializable;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.datasources.v2.FilePartitionReaderFactory;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/** Produces one Vortex reader for each file selected by Spark's file index. */
public final class VortexPartitionReaderFactory extends FilePartitionReaderFactory implements Serializable {
    private static final long serialVersionUID = 1L;

    private final FileSourceOptions fileOptions;
    private final VortexIo io;
    private final VortexOptions formatOptions;
    private final StructType dataSchema;
    private final StructType readDataSchema;
    private final StructType readPartitionSchema;
    private final Filter[] pushedFilters;

    public VortexPartitionReaderFactory(
            FileSourceOptions fileOptions,
            VortexIo io,
            VortexOptions formatOptions,
            StructType dataSchema,
            StructType readDataSchema,
            StructType readPartitionSchema,
            Filter[] pushedFilters) {
        this.fileOptions = fileOptions;
        this.io = io;
        this.formatOptions = formatOptions;
        this.dataSchema = dataSchema;
        this.readDataSchema = readDataSchema;
        this.readPartitionSchema = readPartitionSchema;
        this.pushedFilters = pushedFilters == null ? new Filter[0] : pushedFilters.clone();
    }

    @Override
    public FileSourceOptions options() {
        return fileOptions;
    }

    @Override
    public PartitionReader<InternalRow> buildReader(PartitionedFile file) {
        throw new UnsupportedOperationException("row-based V2 reads are not supported");
    }

    @Override
    public PartitionReader<ColumnarBatch> buildColumnarReader(PartitionedFile file) {
        return new VortexPartitionReader(
                file, dataSchema, readDataSchema, readPartitionSchema, io, formatOptions, pushedFilters);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }
}
