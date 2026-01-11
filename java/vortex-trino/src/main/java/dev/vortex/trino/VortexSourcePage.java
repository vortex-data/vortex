/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import io.trino.spi.Page;
import io.trino.spi.PageBuilder;
import io.trino.spi.block.Block;
import io.trino.spi.connector.SourcePage;

import java.util.List;
import java.util.function.ObjLongConsumer;

/**
 * A Vortex implementation of SourcePage.
 * <p>
 * SourcePage represents a page of data read from the underlying data source.
 * It was introduced to support late materialization (perfect for Vortex!).
 * <p>
 * Page ~= Arrow RecordBatch
 * Block ~= Arrow Array (column)
 * Position Count ~= Row Count
 */
public final class VortexSourcePage implements SourcePage {
    // NOTE(ngates): position count ~= row count
    @Override
    public int getPositionCount() {
        return 0;
    }

    @Override
    public long getSizeInBytes() {
        return 0;
    }

    @Override
    public long getRetainedSizeInBytes() {
        return 0;
    }

    @Override
    public void retainedBytesForEachPart(ObjLongConsumer<Object> consumer) {

    }

    @Override
    public int getChannelCount() {
        return 0;
    }

    // NOTE(ngates): a Block ~= Arrow Array, where channel ~= column
    //  Trino supports DictionaryBlock, RunLengthEncodedBlock, and ValueBlock.
    @Override
    public Block getBlock(int channel) {
        return null;
    }

    // NOTE(ngates): a Page ~= Arrow RecordBatch
    @Override
    public Page getPage() {
        return new PageBuilder(List.of()).build();
    }

    @Override
    public void selectPositions(int[] positions, int offset, int size) {

    }
}
