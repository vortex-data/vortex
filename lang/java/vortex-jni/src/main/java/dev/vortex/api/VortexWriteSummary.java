// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.util.List;

/** Immutable metadata returned after a Vortex writer has finalized its file. */
public final class VortexWriteSummary {
    private final long fileSize;
    private final long rowCount;
    private final List<VortexColumnStatistics> columnStatistics;

    /**
     * Construct a native write summary.
     *
     * <p>This constructor is public so the JNI implementation can instantiate the value without reflective access.
     * Applications should obtain summaries from {@link VortexWriter#finish()}.
     */
    public VortexWriteSummary(long fileSize, long rowCount, VortexColumnStatistics[] columnStatistics) {
        this.fileSize = fileSize;
        this.rowCount = rowCount;
        this.columnStatistics = List.of(columnStatistics.clone());
    }

    /** Exact size of the completed file in bytes. */
    public long fileSize() {
        return fileSize;
    }

    /** Exact number of rows written to the file. */
    public long rowCount() {
        return rowCount;
    }

    /** Per-column statistics in Arrow schema order. */
    public List<VortexColumnStatistics> columnStatistics() {
        return columnStatistics;
    }
}
