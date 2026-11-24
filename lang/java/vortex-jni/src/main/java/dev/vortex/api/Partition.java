// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativePartition;
import java.lang.ref.Reference;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.arrow.c.ArrowArrayStream;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.ipc.ArrowReader;

/**
 * A unit of scan work that materializes into an Arrow stream. Partitions are single-pass: calling
 * {@link #scanArrow(BufferAllocator)} consumes the partition and transfers ownership of its native memory to the
 * returned {@link ArrowReader}. If the partition is never consumed, its native memory is released automatically via
 * {@link VortexCleaner}.
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
        try {
            long[] out = new long[2];
            NativePartition.rowCount(pointer, out);
            if (out[1] == 0) {
                return OptionalLong.empty();
            }
            return OptionalLong.of(out[0]);
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    /**
     * Consume the partition and return an {@link ArrowReader} that yields record batches. The caller must close the
     * reader when finished; doing so releases the native partition resources as well.
     */
    public ArrowReader scanArrow(BufferAllocator allocator) {
        if (!consumed.compareAndSet(false, true)) {
            throw new IllegalStateException("partition already consumed");
        }
        try {
            ArrowArrayStream stream;
            try {
                stream = ArrowArrayStream.allocateNew(allocator);
            } catch (RuntimeException | Error failure) {
                // JNI has not taken the partition yet, and the Cleaner skips consumed partitions.
                NativePartition.free(pointer);
                throw failure;
            }
            try {
                // JNI consumes the partition even when it throws.
                NativePartition.scanArrow(session.nativePointer(), pointer, stream.memoryAddress());
                // Closes the stream struct once its reader has taken the stream.
                return Data.importArrayStream(allocator, stream);
            } catch (RuntimeException | Error failure) {
                // Import can fail before moving the stream into its reader.
                if (stream.snapshot().release != 0) {
                    stream.release();
                }
                stream.close();
                throw failure;
            }
        } finally {
            Reference.reachabilityFence(this);
        }
    }
}
