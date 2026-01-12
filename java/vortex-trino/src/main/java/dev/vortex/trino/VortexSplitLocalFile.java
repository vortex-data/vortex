/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import com.fasterxml.jackson.annotation.JsonCreator;
import dev.vortex.api.File;
import io.trino.spi.connector.ConnectorSplit;

/**
 * A split representing a local Vortex file.
 * <p>
 * FIXME(ngates): in theory, this should be not be "remote accessible" since it's a local file. But we're only testing.
 */
public final class VortexSplitLocalFile implements ConnectorSplit {
    // FIXME(ngates): obviously this is not serializable as-is.
    private final File file;

    @JsonCreator
    public VortexSplitLocalFile(File file) {
        this.file = file;
    }

    public File getFile() {
        return file;
    }
}
