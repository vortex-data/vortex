// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativePartition;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A unit of scan work that materializes into an Arrow stream. Partitions are single-pass:
 * calling {@link #scanArrow(BufferAllocator)} consumes the partition and transfers
 * ownership of its native memory to the returned {@link ArrowReader}. If the partition
 * is never consumed, its native memory is released automatically via {@link VortexCleaner}.
 */
public final class Partition {
    private final Session session;
    private final long pointer;
    private final AtomicBoolean consumed = new AtomicBoolean(false);

    private Partition(Session session, long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid partition pointer");
        this.session = session;
        this.pointer = pointer;
        AtomicBoolean consumedRef = this.consumed;
        VortexCleaner.register(this, () -> {
            if (!consumedRef.get()) {
                NativePartition.free(pointer);
            }
        });
    }

    static Partition fromPointer(Session session, long pointer) {
        return new Partition(session, pointer);
    }

    /** Estimated row count of the partition. Empty when unknown. */
    public OptionalLong rowCount() {
        Preconditions.checkState(!consumed.get(), "partition already consumed");
        long[] out = new long[2];
        NativePartition.rowCount(pointer, out);
        if (out[1] == 0) {
            return OptionalLong.empty();
        }
        return OptionalLong.of(out[0]);
    }

    /**
     * Consume the partition and return an {@link ArrowReader} that yields record batches.
     * The caller must close the reader when finished; doing so releases the native partition
     * resources as well.
     */
    public ArrowReader scanArrow(BufferAllocator allocator) {
        if (!consumed.compareAndSet(false, true)) {
            throw new IllegalStateException("partition already consumed");
        }
        // Native side unconditionally takes ownership of the partition, regardless of
        // whether the call subsequently throws, so it is correct to flip `consumed`
        // before invoking JNI. The cleaner then always skips free for this handle.
        ArrowArrayStream stream = ArrowArrayStream.allocateNew(allocator);
        try {
            NativePartition.scanArrow(session.nativePointer(), pointer, stream.memoryAddress());
        } catch (RuntimeException ex) {
            stream.close();
            throw ex;
        }
        return Data.importArrayStream(allocator, stream);
    }
}
