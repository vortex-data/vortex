// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Array;
import dev.vortex.api.ArrayIterator;
import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import java.util.Iterator;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * Iterator that converts Vortex Arrays into Spark ColumnarBatch objects.
 * <p>
 * This iterator wraps a Vortex ArrayIterator and converts each Array into a Spark ColumnarBatch
 * by exporting the data to Arrow format and wrapping it with VortexArrowColumnVector instances.
 * The iterator uses prefetching to optimize memory usage and performance by batching arrays
 * up to a maximum buffer size.
 * <p>
 * The iterator maintains a reusable VectorSchemaRoot to minimize allocation overhead when
 * converting between Vortex and Arrow formats.
 *
 * @see ArrayIterator
 * @see ColumnarBatch
 * @see VortexArrowColumnVector
 */
public final class VortexColumnarBatchIterator implements Iterator<ColumnarBatch>, AutoCloseable {
    /**
     * Maximum buffer size in bytes for prefetching arrays.
     * <p>
     * The iterator will prefetch and batch arrays until this size limit is reached,
     * which helps optimize memory usage and reduces the overhead of converting
     * small arrays individually.
     */
    public static final long MAX_BUFFER_BYTES = 16 * 1024 * 1024; // 16MB

    private final ArrayIterator backing;
    private final PrefetchingIterator<Array> prefetching;

    // Reusable root
    private VectorSchemaRoot root = null;

    /**
     * Creates a new VortexColumnarBatchIterator that wraps the given ArrayIterator.
     * <p>
     * The iterator will use prefetching to batch arrays up to MAX_BUFFER_BYTES
     * to optimize memory usage and conversion performance.
     *
     * @param backing the underlying ArrayIterator to wrap
     */
    public VortexColumnarBatchIterator(ArrayIterator backing) {
        this.backing = backing;
        this.prefetching = new PrefetchingIterator<>(backing, MAX_BUFFER_BYTES, Array::nbytes);
    }

    /**
     * Returns whether there are more columnar batches available.
     *
     * @return true if there are more batches to iterate over, false otherwise
     */
    @Override
    public boolean hasNext() {
        return prefetching.hasNext();
    }

    /**
     * Returns the next columnar batch from the iterator.
     * <p>
     * This method retrieves the next Array from the prefetching iterator,
     * exports it to Arrow format using a reusable VectorSchemaRoot,
     * and wraps each field vector in a VortexArrowColumnVector to create
     * a VortexColumnarBatch.
     *
     * @return the next ColumnarBatch containing the data from the next Array
     * @throws java.util.NoSuchElementException if there are no more elements
     */
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

    /**
     * Closes this iterator and releases all associated resources.
     * <p>
     * This method closes the prefetching iterator, the backing ArrayIterator,
     * and the reusable VectorSchemaRoot if it exists.
     */
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
