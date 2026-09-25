// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.BoundExpression;
import dev.vortex.api.DataSource;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.memory.BufferAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.ipc.ArrowReader;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.spark.VortexFilePartition;
import dev.vortex.spark.VortexSparkSession;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Per-{@link VortexFilePartition} columnar reader.
 *
 * <p>Opens a single Vortex {@link Session}, {@link DataSource} and {@link Scan} spanning all of
 * {@link VortexFilePartition#paths()} and streams every Vortex partition's record batches through the
 * {@link PartitionReader} interface.
 */
final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final VortexFilePartition spark;
    private final BufferAllocator allocator;

    // Held so the DataSource/Scan stay reachable even if the JVM-wide singleton is
    // ever reset during a task; the actual native session is owned by
    // {@link VortexSparkSession} and is not released when this reader closes.
    private Session session;
    private DataSource dataSource;
    private Scan scan;

    private Partition currentPartition;
    private ArrowReader currentReader;
    private boolean currentBatchLoaded;
    private boolean exhausted;

    VortexPartitionReader(
            VortexFilePartition spark,
            List<String> dataColumnNames,
            Map<String, String> formatOptions,
            Predicate[] pushedPredicates) {
        this.spark = spark;
        this.allocator = ArrowAllocation.rootAllocator();

        session = VortexSparkSession.get(formatOptions);
        dataSource = DataSource.open(session, spark.paths(), formatOptions);

        var options = ScanOptions.builder();
        if (!dataColumnNames.isEmpty()) {
            BoundExpression projection =
                    BoundExpression.select(dataColumnNames.toArray(new String[0]), BoundExpression.root(dataSource));
            options.projection(projection);
        }
        if (pushedPredicates != null && pushedPredicates.length > 0) {
            Map<List<String>, ArrowType.Timestamp> timestampTypes =
                    Arrays.stream(pushedPredicates).anyMatch(SparkPredicateToVortexExpression::hasTimestampLiteral)
                            ? SparkPredicateToVortexExpression.timestampTypes(dataSource.arrowSchema(allocator))
                            : Map.of();
            buildFilterExpression(pushedPredicates, dataSource, timestampTypes).ifPresent(options::filter);
        }
        scan = dataSource.scan(options.build());
    }

    private static Optional<BoundExpression> buildFilterExpression(
            Predicate[] predicates,
            DataSource dataSource,
            Map<List<String>, ArrowType.Timestamp> timestampTypes) {
        BoundExpression combined = null;
        for (Predicate predicate : predicates) {
            Optional<BoundExpression> expr =
                    SparkPredicateToVortexExpression.convertBound(predicate, dataSource, timestampTypes);
            if (expr.isEmpty()) {
                throw new IllegalStateException("Spark dropped a predicate that Vortex cannot convert: " + predicate);
            }
            combined = combined == null ? expr.get() : BoundExpression.and(combined, expr.get());
        }
        return Optional.ofNullable(combined);
    }

    @Override
    public boolean next() {
        if (exhausted) {
            return false;
        }
        while (true) {
            if (currentReader != null) {
                try {
                    if (currentReader.loadNextBatch()) {
                        currentBatchLoaded = true;
                        return true;
                    }
                } catch (IOException e) {
                    throw failure("load a batch from", e);
                }
                closeCurrentReader();
            }
            if (!scan.hasNext()) {
                exhausted = true;
                return false;
            }
            currentPartition = scan.next();
            currentReader = currentPartition.scanArrow(allocator);
        }
    }

    @Override
    public ColumnarBatch get() {
        if (!currentBatchLoaded) {
            throw new IllegalStateException("no batch loaded; call next() first");
        }
        currentBatchLoaded = false;

        VectorSchemaRoot root;
        try {
            root = currentReader.getVectorSchemaRoot();
        } catch (IOException e) {
            throw failure("read the loaded batch of", e);
        }

        int rowCount = root.getRowCount();
        Map<String, String> partVals = spark.partitionValues();
        if (partVals.isEmpty()) {
            ColumnVector[] vectors = new ColumnVector[root.getFieldVectors().size()];
            for (int i = 0; i < vectors.length; i++) {
                vectors[i] = new VortexArrowColumnVector(root.getFieldVectors().get(i));
            }
            return new ColumnarBatch(vectors, rowCount);
        }

        StructField[] fields = spark.readSchema().fields();
        ColumnVector[] combined = new ColumnVector[fields.length];
        int dataIdx = 0;
        for (int i = 0; i < fields.length; i++) {
            StructField field = fields[i];
            String partValue = partVals.get(field.name());
            if (partValue != null) {
                combined[i] = PartitionPathUtils.createConstantVector(rowCount, field.dataType(), partValue);
            } else {
                combined[i] = new VortexArrowColumnVector(root.getFieldVectors().get(dataIdx++));
            }
        }
        return new ColumnarBatch(combined, rowCount);
    }

    @Override
    public void close() {
        closeCurrentReader();
        // Scan and DataSource native resources are released by VortexCleaner once
        // references are dropped. Session is the JVM-wide singleton and outlives this reader.
        scan = null;
        dataSource = null;
        session = null;
    }

    /** Wraps a failure with the paths being read, the one fact a stack trace alone does not give. */
    private RuntimeException failure(String what, IOException cause) {
        return new RuntimeException(String.format("Failed to %s %s", what, spark.paths()), cause);
    }

    private void closeCurrentReader() {
        if (currentReader != null) {
            try {
                currentReader.close();
            } catch (IOException e) {
                throw failure("close the reader over", e);
            }
            currentReader = null;
        }
        currentPartition = null;
    }
}
