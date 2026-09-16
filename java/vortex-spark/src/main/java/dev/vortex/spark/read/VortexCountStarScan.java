// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;

/**
 * Spark V2 {@link Scan} that answers a pushed-down global {@code COUNT(*)} from Vortex file footer metadata.
 *
 * <p>Built by {@link VortexScanBuilder} when Spark pushes an aggregation that is exactly one {@code COUNT(*)} with no
 * grouping expressions and no pushed predicates. Instead of scanning data, each input partition opens a single-file
 * {@link dev.vortex.api.DataSource} and reads the exact row count recorded in that file's footer, emitting one partial
 * count per file. This is a partial pushdown: Spark performs the final summation of the per-file counts, following the
 * column-order contract of {@link org.apache.spark.sql.connector.read.SupportsPushDownAggregates}.
 */
public final class VortexCountStarScan implements Scan {
    static final StructType COUNT_SCHEMA = new StructType().add("count(*)", DataTypes.LongType, false);

    private final List<String> paths;
    private final Map<String, String> formatOptions;

    /**
     * Creates a new VortexCountStarScan for the specified file paths. The caller is responsible for passing immutable
     * collections; the constructor does not copy.
     *
     * @param paths the list of Vortex file paths to count over
     * @param formatOptions the format options for opening the files
     */
    VortexCountStarScan(List<String> paths, Map<String, String> formatOptions) {
        this.paths = paths;
        this.formatOptions = formatOptions;
    }

    @Override
    public StructType readSchema() {
        return COUNT_SCHEMA;
    }

    @Override
    public Batch toBatch() {
        return new VortexCountStarBatchExec(paths, formatOptions);
    }

    @Override
    public String description() {
        return "VortexCountStarScan PushedAggregation: [COUNT(*)]";
    }

    @Override
    public ColumnarSupportMode columnarSupportMode() {
        return ColumnarSupportMode.SUPPORTED;
    }
}
