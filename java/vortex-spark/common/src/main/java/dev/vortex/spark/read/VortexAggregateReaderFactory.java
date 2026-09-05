// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.spark.VortexOptions;
import dev.vortex.spark.io.VortexFile;
import dev.vortex.spark.io.VortexIo;
import java.io.Serializable;
import java.util.OptionalLong;
import org.apache.spark.sql.catalyst.FileSourceOptions;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.aggregate.Aggregation;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.datasources.PartitionedFile;
import org.apache.spark.sql.execution.datasources.v2.FilePartitionReaderFactory;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/** Produces one footer-backed partial COUNT(*) row per Vortex file. */
public final class VortexAggregateReaderFactory extends FilePartitionReaderFactory implements Serializable {
    private static final long serialVersionUID = 1L;

    private final FileSourceOptions fileOptions;
    private final VortexIo io;
    private final VortexOptions formatOptions;
    private final StructType aggregateSchema;
    private final int[] groupByOrdinals;

    public VortexAggregateReaderFactory(
            FileSourceOptions fileOptions,
            VortexIo io,
            VortexOptions formatOptions,
            StructType aggregateSchema,
            StructType partitionSchema,
            Aggregation aggregation) {
        this.fileOptions = fileOptions;
        this.io = io;
        this.formatOptions = formatOptions;
        this.aggregateSchema = aggregateSchema;
        Expression[] groupBy = aggregation.groupByExpressions();
        this.groupByOrdinals = new int[groupBy.length];
        for (int i = 0; i < groupBy.length; i++) {
            if (!(groupBy[i] instanceof NamedReference reference) || reference.fieldNames().length != 1) {
                throw new IllegalArgumentException("COUNT(*) group-by must reference partition columns");
            }
            this.groupByOrdinals[i] = partitionSchema.fieldIndex(reference.fieldNames()[0]);
        }
    }

    @Override
    public FileSourceOptions options() {
        return fileOptions;
    }

    @Override
    public PartitionReader<InternalRow> buildReader(PartitionedFile file) {
        throw new UnsupportedOperationException("row-based aggregate reads are not supported");
    }

    @Override
    public PartitionReader<ColumnarBatch> buildColumnarReader(PartitionedFile file) {
        return new PartitionReader<>() {
            private boolean emitted;
            private ColumnarBatch batch;

            @Override
            public boolean next() {
                if (emitted) {
                    return false;
                }
                emitted = true;
                // An estimate is no answer to COUNT(*), so the footer must state the count exactly.
                OptionalLong rowCount = VortexFooterReader.exactRowCount(
                        new VortexFile(file.toPath().toString(), file.fileSize()), io, formatOptions);
                if (rowCount.isEmpty()) {
                    throw new IllegalStateException(String.format(
                            "Vortex footer states no exact row count for %s, so COUNT(*) cannot be answered from it. "
                                    + "Set the vortex.aggregatePushdown option to false to count rows instead.",
                            file.toPath()));
                }
                StructField[] fields = aggregateSchema.fields();
                ColumnVector[] vectors = new ColumnVector[fields.length];
                for (int i = 0; i < groupByOrdinals.length; i++) {
                    vectors[i] =
                            PartitionColumnVectors.create(1, fields[i], file.partitionValues(), groupByOrdinals[i]);
                }
                for (int i = groupByOrdinals.length; i < fields.length; i++) {
                    ConstantColumnVector count = new ConstantColumnVector(1, fields[i].dataType());
                    count.setNotNull();
                    count.setLong(rowCount.getAsLong());
                    vectors[i] = count;
                }
                batch = new ColumnarBatch(vectors, 1);
                return true;
            }

            @Override
            public ColumnarBatch get() {
                if (batch == null) {
                    throw new IllegalStateException("no aggregate row loaded; call next() first");
                }
                return batch;
            }

            @Override
            public void close() {
                if (batch != null) {
                    batch.close();
                    batch = null;
                }
            }
        };
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return true;
    }
}
