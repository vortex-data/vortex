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

import com.google.common.collect.ImmutableList;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.Batch;
import org.apache.spark.sql.connector.read.Scan;
import org.apache.spark.sql.types.StructType;

/**
 * Spark V2 {@link Scan} over a table of Vortex files.
 */
public final class VortexScan implements Scan {

    private final ImmutableList<String> paths;
    private final ImmutableList<Column> readColumns;

    public VortexScan(ImmutableList<String> paths, ImmutableList<Column> readColumns) {
        this.paths = paths;
        this.readColumns = readColumns;
    }

    @Override
    public StructType readSchema() {
        return CatalogV2Util.v2ColumnsToStructType(readColumns.toArray(new Column[0]));
    }

    /**
     * Logging-friendly readable description of the scan source.
     */
    @Override
    public String description() {
        return String.format("VortexScan{paths=%s, columns=%s}", paths, readColumns);
    }

    @Override
    public Batch toBatch() {
        return new VortexBatchExec(paths, readColumns);
    }

    // We always provide columnar scans.
    @Override
    public ColumnarSupportMode columnarSupportMode() {
        return ColumnarSupportMode.SUPPORTED;
    }
}
