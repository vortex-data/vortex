// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.List;
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

    public VortexScanBuilder() {
        this.paths = ImmutableList.builder();
        this.columns = new ArrayList<>();
    }

    public VortexScanBuilder addPath(String path) {
        this.paths.add(path);
        return this;
    }

    public VortexScanBuilder addColumn(Column column) {
        this.columns.add(column);
        return this;
    }

    public VortexScanBuilder addAllPaths(Iterable<String> paths) {
        this.paths.addAll(paths);
        return this;
    }

    public VortexScanBuilder addAllColumns(Iterable<Column> columns) {
        for (Column column : columns) {
            this.columns.add(column);
        }
        return this;
    }

    @Override
    public Scan build() {
        var paths = this.paths.build();
        var columns = ImmutableList.copyOf(this.columns);

        checkState(!paths.isEmpty(), "paths cannot be empty");
        checkState(!columns.isEmpty(), "readColumns cannot be empty");

        return new VortexScan(paths, columns);
    }

    @Override
    public void pruneColumns(StructType requiredSchema) {
        // TODO(aduffy): support deeply nested schema prunes
        columns.clear();
        for (StructField field : requiredSchema.fields()) {
            columns.add(Column.create(field.name(), field.dataType()));
        }
    }
}
