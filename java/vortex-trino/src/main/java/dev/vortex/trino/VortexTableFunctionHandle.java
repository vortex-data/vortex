/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import dev.vortex.api.File;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

public final class VortexTableFunctionHandle implements ConnectorTableFunctionHandle {
    private final File vxf;

    public VortexTableFunctionHandle(File vxf) {
        this.vxf = vxf;
    }
}
