/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.*;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

public enum VortexSplitManager implements ConnectorSplitManager {
    INSTANCE;

    @Override
    public ConnectorSplitSource getSplits(ConnectorTransactionHandle transaction, ConnectorSession session, ConnectorTableHandle table, DynamicFilter dynamicFilter, Constraint constraint) {
        VortexTableHandle handle = (VortexTableHandle) table;
        return new VortexSplitSource(handle, dynamicFilter, constraint);
    }

    @Override
    public ConnectorSplitSource getSplits(ConnectorTransactionHandle transaction, ConnectorSession session, ConnectorTableFunctionHandle function) {
        throw new IllegalArgumentException("TableFunctionHandle should have been converted to TableHandle by dev.vortex.trino.VortexConnectorMetadata.applyTableFunction");
    }
}
