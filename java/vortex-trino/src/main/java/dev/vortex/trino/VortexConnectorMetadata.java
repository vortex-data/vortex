/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.*;
import io.trino.spi.expression.ConnectorExpression;

import java.util.List;
import java.util.Map;
import java.util.Optional;

public final class VortexConnectorMetadata implements ConnectorMetadata {
    @Override
    public Optional<ProjectionApplicationResult<ConnectorTableHandle>> applyProjection(ConnectorSession session, ConnectorTableHandle handle, List<ConnectorExpression> projections, Map<String, ColumnHandle> assignments) {
        VortexTableHandle vortexHandle = (VortexTableHandle) handle;

        for (ConnectorExpression expression : projections) {
            // Process each projection expression as needed
        }
        
        return null;
    }
}
