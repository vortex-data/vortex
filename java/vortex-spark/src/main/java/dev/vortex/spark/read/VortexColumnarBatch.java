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
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.spark.sql.util.ArrowUtils;
import org.apache.spark.sql.vectorized.ArrowColumnVector;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A {@link ColumnarBatch} that returns Vortex-managed memory with Arrow format, shared over the C Data Interface.
 */
public final class VortexColumnarBatch extends ColumnarBatch {
    private Array backingArray;

    private VortexColumnarBatch(Array backingArray, ColumnVector[] columns, int numRows) {
        super(columns, numRows);
        this.backingArray = backingArray;
    }

    /**
     * Create a new columnar batch backed by the fields of a Vortex array.
     */
    public static ColumnarBatch fromVortex(Array vortex) {
        VectorSchemaRoot root = vortex.exportToArrow(ArrowUtils.rootAllocator());
        int rowCount = root.getRowCount();
        ColumnVector[] vectors = new ColumnVector[root.getFieldVectors().size()];
        for (int i = 0; i < root.getFieldVectors().size(); i++) {
            vectors[i] = new ArrowColumnVector(root.getFieldVectors().get(i));
        }
        return new VortexColumnarBatch(vortex, vectors, rowCount);
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
