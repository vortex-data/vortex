/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.spark.read;

import dev.vortex.api.Array;
import dev.vortex.api.ArrayIterator;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import java.util.Iterator;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

public final class VortexColumnarBatchIterator implements Iterator<ColumnarBatch>, AutoCloseable {
    public static final long MAX_BUFFER_BYTES = 16 * 1024 * 1024; // 16MB
    private final ArrayIterator backing;
    private final PrefetchingIterator<Array> prefetching;

    // Reusable root
    private VectorSchemaRoot root = null;

    public VortexColumnarBatchIterator(ArrayIterator backing) {
        this.backing = backing;
        this.prefetching = new PrefetchingIterator<>(backing, MAX_BUFFER_BYTES, Array::nbytes);
    }

    @Override
    public boolean hasNext() {
        return prefetching.hasNext();
    }

    @Override
    public ColumnarBatch next() {
        Array next = prefetching.next();

        root = next.exportToArrow(ArrowAllocation.rootAllocator(), root);

        int rowCount = root.getRowCount();
        ColumnVector[] vectors = new ColumnVector[root.getFieldVectors().size()];
        for (int i = 0; i < root.getFieldVectors().size(); i++) {
            vectors[i] = new VortexArrowColumnVector(root.getFieldVectors().get(i));
        }
        return new VortexColumnarBatch(next, vectors, rowCount);
    }

    @Override
    public void close() {
        this.prefetching.close();
        this.backing.close();
        if (root != null) {
            root.close();
            root = null;
        }
    }
}
