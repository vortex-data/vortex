// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Array;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A {@link ColumnarBatch} that returns Vortex-managed memory with Arrow format, shared over the C Data Interface.
 */
public final class VortexColumnarBatch extends ColumnarBatch {
    private Array backingArray;

    /**
     * Creates a new VortexColumnarBatch with the specified backing array and column vectors.
     * <p>
     * The backing array holds the native memory that contains the actual data,
     * while the column vectors provide the Spark API for accessing that data.
     *
     * @param backingArray the Vortex Array that holds the native memory
     * @param columns the array of ColumnVector objects for data access
     * @param numRows the number of rows in this batch
     */
    public VortexColumnarBatch(Array backingArray, ColumnVector[] columns, int numRows) {
        super(columns, numRows);
        this.backingArray = backingArray;
    }

    /**
     * Closes this columnar batch and releases all associated resources.
     * <p>
     * This method frees the native memory held by the backing Vortex array
     * and then delegates to the parent class to close the column vectors.
     */
    @Override
    public void close() {
        freeNativeMemory();
        super.close();
    }

    /**
     * Closes this columnar batch if it is freeable and releases all associated resources.
     * <p>
     * This method frees the native memory held by the backing Vortex array
     * and then delegates to the parent class to close the column vectors if freeable.
     */
    @Override
    public void closeIfFreeable() {
        freeNativeMemory();
        super.closeIfFreeable();
    }

    private void freeNativeMemory() {
        backingArray.close();
        backingArray = null;
    }
}
