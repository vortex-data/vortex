// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.ReferenceManager;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/** Per-allocation ownership used by the native zero-copy importer. */
public final class NativeBuffer {
    private final ReferenceManager manager;
    private final long address;
    private final long capacity;

    private NativeBuffer(ArrowBuf buffer) {
        manager = buffer.getReferenceManager();
        address = buffer.memoryAddress();
        capacity = buffer.capacity();
    }

    /** Capture buffer ownership metadata while the root remains alive. */
    public static NativeBuffer[] collect(VectorSchemaRoot root) {
        List<NativeBuffer> buffers = new ArrayList<>();
        for (FieldVector vector : root.getFieldVectors()) {
            collect(vector, buffers);
        }
        return buffers.toArray(new NativeBuffer[0]);
    }

    private static void collect(FieldVector vector, List<NativeBuffer> buffers) {
        for (ArrowBuf buffer : vector.getFieldBuffers()) {
            if (buffer.capacity() != 0) {
                buffers.add(new NativeBuffer(buffer));
            }
        }
        for (FieldVector child : vector.getChildrenFromFields()) {
            collect(child, buffers);
        }
    }

    /** Buffer address, called by JNI. */
    public long address() {
        return address;
    }

    /** Buffer capacity, called by JNI. */
    public long capacity() {
        return capacity;
    }

    /** Acquire an independent native reference, called by JNI. */
    public void retain() {
        manager.retain();
    }

    /** Release the native reference, possibly from a writer worker thread. */
    public void release() {
        manager.release();
    }
}
