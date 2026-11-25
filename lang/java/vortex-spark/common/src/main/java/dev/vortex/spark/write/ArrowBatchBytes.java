// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import dev.vortex.relocated.org.apache.arrow.vector.BaseFixedWidthVector;
import dev.vortex.relocated.org.apache.arrow.vector.BaseVariableWidthVector;
import dev.vortex.relocated.org.apache.arrow.vector.FieldVector;
import dev.vortex.relocated.org.apache.arrow.vector.VectorSchemaRoot;
import dev.vortex.relocated.org.apache.arrow.vector.complex.ListVector;
import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import java.util.ArrayList;
import java.util.List;

/**
 * Measures the Arrow memory that the buffers of one batch hold, nested vectors included.
 *
 * <p>The measure is the capacity of the batch's own buffers, not an allocator total: the native writer holds and
 * releases the buffers of earlier batches on its own schedule, so the allocator cannot tell which bytes belong to the
 * open batch.
 *
 * <p>The buffers are read through their plain getters. {@link FieldVector#getFieldBuffers()} is not used because it
 * brings a vector to a consistent state for export, which would disturb a batch that is still being written.
 */
final class ArrowBatchBytes {
    private final FieldVector[] vectors;

    ArrowBatchBytes(VectorSchemaRoot root) {
        List<FieldVector> all = new ArrayList<>();
        for (FieldVector vector : root.getFieldVectors()) {
            collect(vector, all);
        }
        this.vectors = all.toArray(new FieldVector[0]);
    }

    private static void collect(FieldVector vector, List<FieldVector> into) {
        into.add(vector);
        for (FieldVector child : vector.getChildrenFromFields()) {
            collect(child, into);
        }
    }

    /** The bytes that the buffers of the batch hold now. */
    long measure() {
        long bytes = 0;
        for (FieldVector vector : vectors) {
            bytes += bufferBytes(vector);
        }
        return bytes;
    }

    private static long bufferBytes(FieldVector vector) {
        if (vector instanceof BaseFixedWidthVector fixed) {
            return fixed.getValidityBuffer().capacity() + fixed.getDataBuffer().capacity();
        }
        if (vector instanceof BaseVariableWidthVector variable) {
            return variable.getValidityBuffer().capacity()
                    + variable.getOffsetBuffer().capacity()
                    + variable.getDataBuffer().capacity();
        }
        // MapVector is a ListVector.
        if (vector instanceof ListVector list) {
            return list.getValidityBuffer().capacity() + list.getOffsetBuffer().capacity();
        }
        if (vector instanceof StructVector struct) {
            return struct.getValidityBuffer().capacity();
        }
        // SparkToArrowSchema produces none of the other vector kinds.
        return 0;
    }
}
