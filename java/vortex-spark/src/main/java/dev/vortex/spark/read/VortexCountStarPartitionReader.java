// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkState;

import dev.vortex.api.DataSource;
import dev.vortex.spark.VortexSparkSession;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.execution.vectorized.OnHeapColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Reads the exact row count of a single Vortex file from its footer and emits it as a one-row {@link ColumnarBatch}.
 *
 * <p>No column data is decoded: opening a single-file {@link DataSource} reads the file footer eagerly, so
 * {@link DataSource#rowCount()} is {@link DataSource.RowCount.Exact}. The reader fails fast if the count is ever not
 * exact rather than silently returning a wrong result — a single-file source is exact by construction today, and this
 * guard turns any future change of that invariant into a loud error instead of a correctness bug.
 */
public final class VortexCountStarPartitionReader implements PartitionReader<ColumnarBatch> {
    private final String path;
    private final Map<String, String> formatOptions;

    private ColumnarBatch batch;
    private boolean emitted = false;

    /**
     * Creates a new reader for a single file.
     *
     * @param path the resolved {@code .vortex} file path
     * @param formatOptions the format options for opening the file
     */
    VortexCountStarPartitionReader(String path, Map<String, String> formatOptions) {
        this.path = path;
        this.formatOptions = formatOptions;
    }

    @Override
    public boolean next() {
        if (emitted) {
            return false;
        }
        long count = readFooterRowCount();
        OnHeapColumnVector[] vectors = OnHeapColumnVector.allocateColumns(1, VortexCountStarScan.COUNT_SCHEMA);
        vectors[0].putLong(0, count);
        this.batch = new ColumnarBatch(vectors, 1);
        this.emitted = true;
        return true;
    }

    @Override
    public ColumnarBatch get() {
        checkState(batch != null, "next() must return true before get()");
        return batch;
    }

    @Override
    public void close() {
        if (batch != null) {
            batch.close();
            batch = null;
        }
    }

    private long readFooterRowCount() {
        DataSource source = DataSource.open(VortexSparkSession.get(formatOptions), List.of(path), formatOptions);
        DataSource.RowCount rowCount = source.rowCount();
        checkState(
                rowCount instanceof DataSource.RowCount.Exact,
                "expected exact footer row count for single-file data source %s, got %s",
                path,
                rowCount);
        return ((DataSource.RowCount.Exact) rowCount).value();
    }
}
