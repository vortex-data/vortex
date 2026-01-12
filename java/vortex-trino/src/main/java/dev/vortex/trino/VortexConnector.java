/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.*;
import io.trino.spi.function.table.ConnectorTableFunction;
import io.trino.spi.transaction.IsolationLevel;

import java.util.Set;

public final class VortexConnector implements Connector {
    private final Set<ConnectorTableFunction> tableFunctions;

    public VortexConnector() {
        this.tableFunctions = Set.of(new VortexTableFunction());
    }

    @Override
    public ConnectorMetadata getMetadata(ConnectorSession session, ConnectorTransactionHandle transactionHandle) {
        return new VortexConnectorMetadata();
    }

    @Override
    public Set<ConnectorTableFunction> getTableFunctions() {
        return tableFunctions;
    }

    @Override
    public ConnectorSplitManager getSplitManager() {
        return VortexSplitManager.INSTANCE;
    }

    @Override
    public ConnectorTransactionHandle beginTransaction(IsolationLevel isolationLevel, boolean readOnly, boolean autoCommit) {
        return VortexTransactionHandle.INSTANCE;
    }
}
