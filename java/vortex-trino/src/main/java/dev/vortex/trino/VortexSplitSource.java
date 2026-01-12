/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.connector.ConnectorSplitSource;
import io.trino.spi.connector.Constraint;
import io.trino.spi.connector.DynamicFilter;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * The split source for a Vortex table.
 */
public final class VortexSplitSource implements ConnectorSplitSource {
    private final VortexTableHandle handle;
    // We can use the dynamic filter to prune splits as we generate them.
    // We can further use the DynamicFilter in VortexPageSourceProvider when reading data on the worker nodes.
    private final DynamicFilter dynamicFilter;
    private final Constraint constraint;

    public VortexSplitSource(VortexTableHandle handle, DynamicFilter dynamicFilter, Constraint constraint) {
        this.handle = handle;
        this.dynamicFilter = dynamicFilter;
        this.constraint = constraint;
    }

    // NOTE(ngates): maxSize is the maximum number of splits to return in this batch.
    //  We can return fewer, but *should* not return more.
    //  Trino will invoke this function and adjust maxSize depending on how fast the workers are processing splits.
    //  Therefore, if we have lots of splits, it's plausible that the dynamic filter becomes applicable during this
    //  process, so we should check the dynamic filter before returning splits.
    @Override
    public CompletableFuture<ConnectorSplitBatch> getNextBatch(int maxSize) {
        // Parquet probably returns 1 split per row group? If not 1 split per file.
        // So in theory we just need to partition the row count by maxSize and return that many splits.

        // For now, we return a single split for the entire table.
        return CompletableFuture.completedFuture(new ConnectorSplitBatch(
                List.of(new VortexSplitLocalFile(this.handle.getFile())),
                true
        ));
    }

    @Override
    public void close() {
        // Since this is closeable, we should be holding onto a scan object here.
    }

    @Override
    public boolean isFinished() {
        return true;
    }
}
