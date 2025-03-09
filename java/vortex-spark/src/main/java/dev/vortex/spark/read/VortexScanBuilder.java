/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.connector.read.ScanBuilder;

/**
 * Spark V2 {@link ScanBuilder} for table scans over Vortex files.
 */
public final class VortexScanBuilder implements ScanBuilder {
    private final ImmutableList.Builder<String> paths;
    private final ImmutableList.Builder<Column> columns;

    public VortexScanBuilder() {
        this.paths = ImmutableList.builder();
        this.columns = ImmutableList.builder();
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
        this.columns.addAll(columns);
        return this;
    }

    @Override
    public Scan build() {
        var paths = this.paths.build();
        var columns = this.columns.build();

        checkState(!paths.isEmpty(), "paths cannot be empty");
        checkState(!columns.isEmpty(), "readColumns cannot be empty");

        return new VortexScan(paths, columns);
    }
}
