// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.api.VortexWriteSummary;
import java.util.Objects;

/** What one {@link VortexOutputWriter} wrote, available once it closes successfully. */
public final class VortexOutputWriteSummary {
    private final VortexWriteSummary fileSummary;
    private final long batchCount;
    private final long peakBatchBytes;
    private final long peakBufferedBytes;

    VortexOutputWriteSummary(
            VortexWriteSummary fileSummary, long batchCount, long peakBatchBytes, long peakBufferedBytes) {
        this.fileSummary = Objects.requireNonNull(fileSummary, "fileSummary");
        this.batchCount = batchCount;
        this.peakBatchBytes = peakBatchBytes;
        this.peakBufferedBytes = peakBufferedBytes;
    }

    /** Exact size of the completed file in bytes. */
    public long fileSize() {
        return fileSummary.fileSize();
    }

    /** Exact number of rows in the file. */
    public long rowCount() {
        return fileSummary.rowCount();
    }

    /** Number of Arrow batches handed to the native writer. */
    public long batchCount() {
        return batchCount;
    }

    /** Arrow bytes of the largest batch: its values, offsets and validity bitmaps, nested ones included. */
    public long peakBatchBytes() {
        return peakBatchBytes;
    }

    /**
     * The largest {@link dev.vortex.api.VortexWriter#bufferedBytes()} seen after a batch was handed over: compressed
     * bytes that the native writer held and had not written to the file yet.
     */
    public long peakBufferedBytes() {
        return peakBufferedBytes;
    }

    /** The native writer's summary, including per-column statistics. */
    public VortexWriteSummary fileSummary() {
        return fileSummary;
    }

    @Override
    public String toString() {
        return "VortexOutputWriteSummary{fileSize=" + fileSize() + ", rowCount=" + rowCount() + ", batchCount="
                + batchCount + ", peakBatchBytes=" + peakBatchBytes + ", peakBufferedBytes=" + peakBufferedBytes
                + "}";
    }
}
