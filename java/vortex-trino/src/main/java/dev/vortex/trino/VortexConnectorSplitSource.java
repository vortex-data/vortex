/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.ConnectorSplitSource;

public final class VortexConnectorSplitSource implements ConnectorSplitSource {

    @Override
    public void close() {

    }

    @Override
    public boolean isFinished() {
        return false;
    }
}
