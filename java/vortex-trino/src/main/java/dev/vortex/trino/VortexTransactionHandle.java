/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.ConnectorTransactionHandle;

/**
 * Dummy transaction handle for Vortex connector since we do not support transactions.
 */
public enum VortexTransactionHandle implements ConnectorTransactionHandle {
    INSTANCE
}
