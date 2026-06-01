// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativePartition;
import dev.vortex.jni.NativePointer;
import java.lang.ref.Cleaner;
import java.util.OptionalLong;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A unit of scan work that materializes into an Arrow stream. Partitions are single-pass: calling
 * {@link #scanArrow(BufferAllocator)} consumes the partition and transfers ownership of its native memory to the
 * returned {@link ArrowReader}. If the partition is never consumed it can be released via {@link #close()}, or
 * automatically via {@link VortexCleaner} when it becomes unreachable. After consume/close, any method that accesses
 * the native pointer throws {@link IllegalStateException} rather than risking use-after-free.
 */
public final class Partition implements AutoCloseable {
    private final Session session;
    private final NativePointer pointer;
    private final Cleaner.Cleanable closeHandle;

    private Partition(Session session, long pointer) {
        this.session = session;
        this.pointer = NativePointer.of(pointer);
        NativePointer pointerRef = this.pointer;
        // scanArrow() may have already consumed the pointer; use tryTake so a successful
        // scanArrow followed by close() / GC is a no-op rather than throwing.
        this.closeHandle = VortexCleaner.register(this, () -> {
            long ptr = pointerRef.tryTake();
            if (ptr != 0) {
                NativePartition.free(ptr);
            }
        });
    }

    static Partition fromPointer(Session session, long pointer) {
        return new Partition(session, pointer);
    }

    private long nativePointer() {
        return pointer.read();
    }

    /** Estimated row count of the partition. Empty when unknown. */
    public OptionalLong rowCount() {
        long[] out = new long[2];
        NativePartition.rowCount(nativePointer(), out);
        if (out[1] == 0) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(out[0]);
    }

    /**
     * Consume the partition and return an {@link ArrowReader} that yields record batches. The caller must close the
     * reader when finished; doing so releases the native partition resources as well.
     */
    public ArrowReader scanArrow(BufferAllocator allocator) {
        // Native side unconditionally takes ownership of the partition, regardless of
        // whether the call subsequently throws, so clear the pointer before invoking JNI.
        long ptr = pointer.take();
        ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator);
        try {
            NativePartition.scanArrow(session.nativePointer(), ptr, stream.memoryAddress());
        } catch (RuntimeException ex) {
            stream.close();
            throw ex;
        }
        // Unregister the cleaner: native took ownership above, so we don't want GC to run the action later.
        closeHandle.clean();
        return Data.importArrayStream(allocator, stream);
    }

    /** Release the native partition. Idempotent; no-op after {@link #scanArrow(BufferAllocator)}. */
    @Override
    public void close() {
        closeHandle.clean();
    }
}
