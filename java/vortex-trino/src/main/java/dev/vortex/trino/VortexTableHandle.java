/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import dev.vortex.api.File;
import dev.vortex.relocated.com.google.common.collect.Streams;
import io.trino.spi.connector.ColumnMetadata;
import io.trino.spi.connector.ConnectorTableHandle;
import io.trino.spi.connector.ConnectorTableMetadata;
import io.trino.spi.connector.SchemaTableName;

import java.util.List;
import java.util.stream.Collectors;

// NOTE(ngates): the all handles in Trino need to be JSON serializable.
//  They must be lightweight, so cannot contain the footer, column stats, etc etc.
//  Instead, each worker should read the footer directly themselves.
//  We should however include enough information to help do this, for example the offsets/lengths of the required
//  segments.
public final class VortexTableHandle implements ConnectorTableHandle {
    private final File vxf;

    public VortexTableHandle(File vxf) {
        this.vxf = vxf;
    }

    /**
     * Get the Vortex file.
     */
    public File getFile() {
        return vxf;
    }

    /**
     * Get the table metadata.
     */
    public ConnectorTableMetadata getTableMetadata() {
        List<ColumnMetadata> columns = Streams.zip(
                        vxf.getDType().getFieldNames().stream(),
                        vxf.getDType().getFieldTypes().stream(),
                        (name, type) -> new ColumnMetadata(name, VortexTypeConverter.toTrinoType(type))
                )
                .collect(Collectors.toList());

        return new ConnectorTableMetadata(
                // FIXME(ngates): derive these from the file path perhaps?
                SchemaTableName.schemaTableName("schema", "tableName"),
                columns
        );
    }
}
