// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.connector.read.SupportsPushDownRequiredColumns;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Spark V2 {@link ScanBuilder} for table scans over Vortex files.
 */
public final class VortexScanBuilder implements ScanBuilder, SupportsPushDownRequiredColumns {
    private final ImmutableList.Builder<String> paths;
    private final List<Column> columns;
    private final Map<String, String> formatOptions;

    /**
     * Creates a new VortexScanBuilder with empty paths and columns.
     */
    public VortexScanBuilder(Map<String, String> formatOptions) {
        this.paths = ImmutableList.builder();
        this.columns = new ArrayList<>();
        this.formatOptions = Map.copyOf(formatOptions);
    }

    /**
     * Adds a file path to scan.
     *
     * @param path the file path to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addPath(String path) {
        this.paths.add(path);
        return this;
    }

    /**
     * Adds a column to read.
     *
     * @param column the column to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addColumn(Column column) {
        this.columns.add(column);
        return this;
    }

    /**
     * Adds multiple file paths to scan.
     *
     * @param paths the iterable of file paths to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addAllPaths(Iterable<String> paths) {
        this.paths.addAll(paths);
        return this;
    }

    /**
     * Adds multiple columns to read.
     *
     * @param columns the iterable of columns to add
     * @return this builder for method chaining
     */
    public VortexScanBuilder addAllColumns(Iterable<Column> columns) {
        for (Column column : columns) {
            this.columns.add(column);
        }
        return this;
    }

    /**
     * Builds a VortexScan with the configured paths and columns.
     *
     * @return a new VortexScan instance
     * @throws IllegalStateException if no paths or columns have been added
     */
    @Override
    public Scan build() {
        var paths = this.paths.build();

        checkState(!paths.isEmpty(), "paths cannot be empty");
        // Allow empty columns for operations like count() that don't need actual column data
        // If no columns are specified, we'll read the minimal schema needed

        return new VortexScan(paths, List.copyOf(this.columns), this.formatOptions);
    }

    /**
     * Prunes the columns to only include those specified in the required schema.
     * <p>
     * This method clears the current column list and replaces it with columns
     * derived from the required schema. Currently only supports top-level schema
     * pruning - deeply nested schema pruning is not yet implemented.
     *
     * @param requiredSchema the schema specifying which columns are required
     */
    @Override
    public void pruneColumns(StructType requiredSchema) {
        // TODO(aduffy): support deeply nested schema prunes
        columns.clear();
        for (StructField field : requiredSchema.fields()) {
            columns.add(Column.create(field.name(), field.dataType()));
        }
    }
}
