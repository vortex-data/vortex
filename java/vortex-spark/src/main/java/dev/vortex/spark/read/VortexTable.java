// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
