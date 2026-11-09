// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativePartition;
import dev.vortex.jni.NativeScan;
import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.atomic.AtomicLong;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * A lazy handle to a set of {@link Partition partitions}.
 *
 * <p>Call {@link #close()} to release the scan and any partition prefetched by {@link #hasNext()} but not yet returned
 * by {@link #next()}. Partitions already returned remain valid. {@link VortexCleaner} releases an unclosed scan when it
 * becomes unreachable.
 */
public final class Scan implements Iterator<Partition>, AutoCloseable {
    private final Session session;
    private final long pointer;

    private final AtomicLong nextPartitionPointer = new AtomicLong();
    private final Cleaner.Cleanable cleanable;
    private boolean primed;
    private boolean closed;

    private Scan(Session session, long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid scan pointer");
        this.session = session;
        this.pointer = pointer;
        AtomicLong pending = nextPartitionPointer;
        cleanable = VortexCleaner.register(this, () -> {
            try {
                long partition = pending.getAndSet(0);
                if (partition != 0) {
                    NativePartition.free(partition);
                }
            } finally {
                NativeScan.free(pointer);
            }
        });
    }

    static Scan fromPointer(Session session, long pointer) {
        return new Scan(session, pointer);
    }

    /**
     * Arrow schema produced by this scan. Must be called before the first call to {@link #hasNext()}/{@link #next()}.
     */
    public synchronized Schema arrowSchema(BufferAllocator allocator) {
        Preconditions.checkState(!closed, "scan is closed");
        try (ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
            NativeScan.arrowSchema(session.nativePointer(), pointer, schema.memoryAddress());
            return Data.importSchema(allocator, schema, null);
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    @Override
    public synchronized boolean hasNext() {
        if (closed) {
            return false;
        }
        try {
            if (!primed) {
                nextPartitionPointer.set(NativeScan.nextPartition(pointer));
                primed = true;
            }
            return nextPartitionPointer.get() != 0;
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    @Override
    public synchronized Partition next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        long ptr = nextPartitionPointer.getAndSet(0);
        primed = false;
        return Partition.fromPointer(session, ptr);
    }

    /** Releases this scan once, without closing partitions already returned to the caller. */
    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            cleanable.clean();
        }
    }
}
