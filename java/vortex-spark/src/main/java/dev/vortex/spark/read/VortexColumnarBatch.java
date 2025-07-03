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

    public VortexColumnarBatch(Array backingArray, ColumnVector[] columns, int numRows) {
        super(columns, numRows);
        this.backingArray = backingArray;
    }

    @Override
    public void close() {
        freeNativeMemory();
        super.close();
    }

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
