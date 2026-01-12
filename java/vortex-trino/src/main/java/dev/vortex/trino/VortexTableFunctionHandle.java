/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import dev.vortex.api.DType;
import dev.vortex.api.File;
import io.trino.spi.function.table.ConnectorTableFunctionHandle;

public final class VortexTableFunctionHandle implements ConnectorTableFunctionHandle {
    // FIXME(ngates): this should hold the parsed Footer, but not an open File since this object is not Closeable.
    private final File vxf;

    public VortexTableFunctionHandle(File vxf) {
        this.vxf = vxf;
    }

    /**
     * Get the Vortex file.
     */
    public File getFile() {
        return vxf;
    }

    /**
     * Get the DType of the Vortex file.
     */
    public DType getDType() {
        return vxf.getDType();
    }
}
