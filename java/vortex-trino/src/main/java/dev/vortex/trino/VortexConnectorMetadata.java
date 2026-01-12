/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.*;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public final class VortexConnectorMetadata implements ConnectorMetadata {
    /**
     * This function converts the TableFunctionHandle into a regular TableHandle.
     * This makes it eligible for push-down and other table-based optimizations.
     */
    @Override
    public Optional<TableFunctionApplicationResult<ConnectorTableHandle>> applyTableFunction(ConnectorSession session, ConnectorTableFunctionHandle handle) {
        VortexTableFunctionHandle tableFunctionHandle = (VortexTableFunctionHandle) handle;

        VortexTableHandle tableHandle = new VortexTableHandle(tableFunctionHandle.getFile());

        List<ColumnHandle> columns = IntStream.range(0, tableHandle.getFile().getDType().getFieldNames().size())
                .boxed()
                .map(idx -> new VortexColumnHandle(tableHandle, idx))
                .collect(Collectors.toList());

        return Optional.of(new TableFunctionApplicationResult<>(tableHandle, columns));
    }

    @Override
    public ConnectorTableMetadata getTableMetadata(ConnectorSession session, ConnectorTableHandle table) {
        VortexTableHandle handle = (VortexTableHandle) table;
        // FIXME(ngateS): inline
        return handle.getTableMetadata();
    }

    @Override
    public ColumnMetadata getColumnMetadata(ConnectorSession session, ConnectorTableHandle tableHandle, ColumnHandle columnHandle) {
        VortexTableHandle table = (VortexTableHandle) tableHandle;
        VortexColumnHandle column = (VortexColumnHandle) columnHandle;
        return new ColumnMetadata(
                table.getFile().getDType().getFieldNames().get(column.getColumnIndex()),
                VortexTypeConverter.toTrinoType(table.getFile().getDType().getFieldTypes().get(column.getColumnIndex()))
        );
    }
}
