/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import com.google.common.collect.ImmutableSet;
import io.trino.spi.connector.Connector;
import io.trino.spi.connector.ConnectorCapabilities;
import io.trino.spi.connector.ConnectorPageSourceProvider;

import java.util.Set;

public final class VortexConnector implements Connector {

    @Override
    public ConnectorPageSourceProvider getPageSourceProvider() {
        return Connector.super.getPageSourceProvider();
    }
}
