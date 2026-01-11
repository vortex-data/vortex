/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.Plugin;
import io.trino.spi.connector.ConnectorFactory;

/**
 * Entry point for the Vortex Trino plugin.
 */
public final class VortexPlugin implements Plugin {
    @Override
    public Iterable<ConnectorFactory> getConnectorFactories() {
        return java.util.List.of(new VortexConnectorFactory());
    }
}
