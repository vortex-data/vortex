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
import com.google.common.collect.ImmutableSet;
import java.util.Set;
import org.apache.spark.sql.connector.catalog.*;
import org.apache.spark.sql.connector.read.ScanBuilder;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Spark V2 {@link Table} of Vortex files.
 */
public final class VortexTable implements Table, SupportsRead {
    private static final String SHORT_NAME = "vortex";

    private final ImmutableList<String> paths;
    private final ImmutableList<Column> readColumns;

    public VortexTable(ImmutableList<String> paths, ImmutableList<Column> readColumns) {
        this.paths = paths;
        this.readColumns = readColumns;
    }

    @Override
    public ScanBuilder newScanBuilder(CaseInsensitiveStringMap _options) {
        // TODO(aduffy): pass any S3 creds from options down into the scan builder.
        //  Or can those be pulled out in the task-side instead?
        return new VortexScanBuilder().addAllPaths(paths).addAllColumns(readColumns);
    }

    @Override
    public String name() {
        return String.format("%s.\"%s\"", SHORT_NAME, String.join(",", paths));
    }

    @Override
    public StructType schema() {
        return CatalogV2Util$.MODULE$.v2ColumnsToStructType(columns());
    }

    @Override
    public Column[] columns() {
        return readColumns.toArray(new Column[0]);
    }

    @Override
    public Set<TableCapability> capabilities() {
        return ImmutableSet.of(TableCapability.BATCH_READ);
    }
}
