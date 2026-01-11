/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.Connector;
import io.trino.spi.function.table.ConnectorTableFunction;

import java.util.Set;

public final class VortexConnector implements Connector {
    private final Set<ConnectorTableFunction> tableFunctions;

    public VortexConnector() {
        this.tableFunctions = Set.of(new VortexTableFunction());
    }

    @Override
    public Set<ConnectorTableFunction> getTableFunctions() {
        return tableFunctions;
    }
}
