/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.ColumnHandle;

public final class VortexColumnHandle implements ColumnHandle {
    private final VortexTableHandle tableHandle;
    private final int columnIndex;

    public VortexColumnHandle(VortexTableHandle tableHandle, int columnIndex) {
        this.tableHandle = tableHandle;
        this.columnIndex = columnIndex;
    }


    public int getColumnIndex() {
        return columnIndex;
    }
}
