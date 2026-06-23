// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DataSource;
import dev.vortex.api.Session;
import dev.vortex.spark.VortexSparkSession;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.OptionalLong;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.Statistics;
import org.apache.spark.sql.connector.read.SupportsReportStatistics;
import org.apache.spark.sql.connector.read.colstats.ColumnStatistics;
import org.apache.spark.sql.internal.SQLConf;
import org.apache.spark.sql.types.StructType;

/**
 * Spark V2 {@link Scan} over a table of Vortex files.
 *
 * <p>Implements {@link SupportsReportStatistics} to surface both the row count Vortex records in each file footer and a
 * Spark scan-size estimate. The byte estimate starts from the on-storage file sizes collected by
 * {@code MultiFileDataSource}, then follows Spark's file scan convention by applying the SQL file-compression factor
 * and scaling by the pushed read schema's default size relative to the full table schema's default size. When the
 * listing did not return a size for one or more files the file-byte total is extrapolated before Spark scaling is
 * applied.
 */
public final class VortexScan implements Scan, SupportsReportStatistics {

    private final List<String> paths;
    private final List<Column> tableColumns;
    private final List<Column> readColumns;
    private final Map<String, String> formatOptions;
    private final Predicate[] pushedPredicates;

    private volatile Statistics cachedStatistics;

    /**
     * Creates a new VortexScan for the specified file paths and columns. The caller is responsible for passing
     * immutable collections; the constructor does not copy.
     *
     * @param paths the list of Vortex file paths to scan
     * @param tableColumns the full table columns before projection pushdown
     * @param readColumns the list of columns to read from the files
     * @param pushedPredicates predicates pushed down by Spark; {@code null} or empty means no pushdown
     */
    public VortexScan(
            List<String> paths,
            List<Column> tableColumns,
            List<Column> readColumns,
            Predicate[] pushedPredicates,
            Map<String, String> formatOptions) {
        this.paths = paths;
        this.tableColumns = tableColumns;
        this.readColumns = readColumns;
        this.formatOptions = formatOptions;
        this.pushedPredicates = pushedPredicates == null ? new Predicate[0] : pushedPredicates.clone();
    }

    /**
     * Returns the schema for the data that will be read by this scan.
     *
     * <p>The schema is constructed from the read columns that were specified when this scan was created.
     *
     * @return the StructType representing the schema of the read data
     */
    @Override
    public StructType readSchema() {
        return CatalogV2Util.v2ColumnsToStructType(readColumns.toArray(new Column[0]));
    }

    /** Logging-friendly readable description of the scan source. */
    @Override
    public String description() {
        return String.format(
                "VortexScan{paths=%s, columns=%s, pushedPredicates=%s}",
                paths, readColumns, Arrays.toString(pushedPredicates));
    }

    /**
     * Converts this scan to a Batch for execution.
     *
     * <p>Creates a VortexBatchExec that will handle the actual reading of the specified files and columns.
     *
     * @return a Batch implementation for executing this scan
     */
    @Override
    public Batch toBatch() {
        return new VortexBatchExec(paths, readColumns, formatOptions, pushedPredicates);
    }

    /**
     * Returns the columnar support mode for this scan.
     *
     * <p>Vortex always provides columnar data access, so this method always returns SUPPORTED.
     *
     * @return ColumnarSupportMode.SUPPORTED
     */
    @Override
    public ColumnarSupportMode columnarSupportMode() {
        return ColumnarSupportMode.SUPPORTED;
    }

    /**
     * Returns statistics for this scan.
     *
     * <p>Opens the Vortex {@link DataSource} on first invocation and caches the result. The row count is taken from the
     * data source (sum of file-footer row counts; extrapolated from the first opened file when other files are
     * deferred). {@link Statistics#sizeInBytes()} is derived from the per-file sizes reported by the filesystem
     * listing, then adjusted by Spark's compression factor and the ratio between the pushed read schema and the full
     * table schema. When a listing did not return a size for some file the file-byte total is extrapolated. When no
     * file size is known at all the value is left empty so Spark falls back to its default heuristic.
     *
     * @return statistics with row-count and Spark scan-size estimates
     */
    @Override
    public Statistics estimateStatistics() {
        Statistics local = cachedStatistics;
        if (local != null) {
            return local;
        }
        synchronized (this) {
            if (cachedStatistics == null) {
                cachedStatistics = computeStatistics();
            }
            return cachedStatistics;
        }
    }

    private Statistics computeStatistics() {
        Session session = VortexSparkSession.get(formatOptions);
        List<String> resolvedPaths = VortexBatchExec.resolveVortexPaths(session, paths, formatOptions);
        if (resolvedPaths.isEmpty()) {
            return new VortexStatistics(OptionalLong.empty(), OptionalLong.empty());
        }

        DataSource source = DataSource.open(session, resolvedPaths, formatOptions);
        return new VortexStatistics(
                source.rowCount().asOptional(),
                scaleSizeInBytes(source.byteSize().asOptional()));
    }

    private OptionalLong scaleSizeInBytes(OptionalLong fileBytes) {
        if (fileBytes.isEmpty()) {
            return OptionalLong.empty();
        }

        StructType tableSchema = CatalogV2Util.v2ColumnsToStructType(tableColumns.toArray(new Column[0]));
        StructType readSchema = readSchema();
        int tableDefaultSize = tableSchema.defaultSize();
        if (tableDefaultSize <= 0) {
            return fileBytes;
        }

        double scaled = SQLConf.get().fileCompressionFactor()
                * fileBytes.getAsLong()
                / tableDefaultSize
                * readSchema.defaultSize();
        return OptionalLong.of((long) scaled);
    }

    private record VortexStatistics(OptionalLong numRows, OptionalLong sizeInBytes) implements Statistics {

        @Override
        public Map<NamedReference, ColumnStatistics> columnStats() {
            return Map.of();
        }
    }
}
