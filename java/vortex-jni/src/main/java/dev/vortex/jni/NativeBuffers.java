// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.ReferenceManager;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/** Batch metadata with independently retained allocations for the native zero-copy importer. */
// The private accessors are only called from native code.
@SuppressWarnings("UnusedMethod")
public final class NativeBuffers {
    private final long[] addresses;
    private final long[] capacities;
    private final ReferenceManager[] managers;

    private NativeBuffers(List<ArrowBuf> buffers) {
        addresses = new long[buffers.size()];
        capacities = new long[buffers.size()];
        managers = new ReferenceManager[buffers.size()];
        for (int index = 0; index < buffers.size(); index++) {
            ArrowBuf buffer = buffers.get(index);
            addresses[index] = buffer.memoryAddress();
            capacities[index] = buffer.capacity();
            managers[index] = buffer.getReferenceManager();
        }
    }

    /** Capture buffer ownership metadata for one native import while the root remains alive. */
    public static NativeBuffers collect(VectorSchemaRoot root) {
        List<ArrowBuf> buffers = new ArrayList<>();
        for (FieldVector vector : root.getFieldVectors()) {
            collect(vector, buffers);
        }
        return new NativeBuffers(buffers);
    }

    private static void collect(FieldVector vector, List<ArrowBuf> buffers) {
        for (ArrowBuf buffer : vector.getFieldBuffers()) {
            if (buffer.capacity() != 0) {
                buffers.add(buffer);
            }
        }
        for (FieldVector child : vector.getChildrenFromFields()) {
            collect(child, buffers);
        }
    }

    private long[] addresses() {
        return addresses;
    }

    private long[] capacities() {
        return capacities;
    }

    /** Acquire all independent native references in one Java call, rolling back on failure. */
    private void retain() {
        int retained = 0;
        try {
            for (; retained < managers.length; retained++) {
                managers[retained].retain();
            }
        } finally {
            if (retained != managers.length) {
                for (int index = 0; index < retained; index++) {
                    managers[index].release();
                }
            }
        }
    }

    /** Release one native reference, possibly from a writer worker thread. */
    private void release(int index) {
        ReferenceManager manager = managers[index];
        managers[index] = null;
        manager.release();
    }
}
