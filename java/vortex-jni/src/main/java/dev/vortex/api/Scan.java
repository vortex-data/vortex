// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativePartition;
import dev.vortex.jni.NativePointer;
import dev.vortex.jni.NativeScan;
import java.lang.ref.Cleaner;
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
 * <p>Once a scan has produced its last partition it is effectively exhausted; native resources are released
 * automatically via {@link VortexCleaner} when the scan becomes unreachable, or eagerly via {@link #close()}. After
 * close, any method that accesses the native pointer throws {@link IllegalStateException} rather than risking
 * use-after-free.
 */
public final class Scan implements Iterator<Partition>, AutoCloseable {
    private final Session session;
    private final NativePointer pointer;
    private final AtomicLong nextPartitionPointer = new AtomicLong(0);
    private final Cleaner.Cleanable closeHandle;

    private boolean primed;

    private Scan(Session session, long pointer) {
        this.session = session;
        this.pointer = NativePointer.of(pointer);
        NativePointer pointerRef = this.pointer;
        AtomicLong pendingRef = this.nextPartitionPointer;
        this.closeHandle = VortexCleaner.register(this, () -> {
            long pending = pendingRef.getAndSet(0);
            if (pending != 0) {
                NativePartition.free(pending);
            }
            NativeScan.free(pointerRef.take());
        });
    }

    static Scan fromPointer(Session session, long pointer) {
        return new Scan(session, pointer);
    }

    private long nativePointer() {
        return pointer.read();
    }

    /**
     * Arrow schema produced by this scan. Must be called before the first call to {@link #hasNext()}/{@link #next()}.
     */
    public Schema arrowSchema(BufferAllocator allocator) {
        long ptr = nativePointer();
        try (ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
            NativeScan.arrowSchema(ptr, schema.memoryAddress());
            return Data.importSchema(allocator, schema, null);
        }
    }

    @Override
    public boolean hasNext() {
        long ptr = nativePointer();
        if (primed) {
            return nextPartitionPointer.get() != 0;
        }
        long next = NativeScan.nextPartition(ptr);
        nextPartitionPointer.set(next);
        primed = true;
        return next != 0;
    }

    @Override
    public Partition next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        long ptr = nextPartitionPointer.getAndSet(0);
        primed = false;
        return Partition.fromPointer(session, ptr);
    }

    /** Release the native scan and any unconsumed pending partition. Idempotent. */
    @Override
    public void close() {
        closeHandle.clean();
    }
}
