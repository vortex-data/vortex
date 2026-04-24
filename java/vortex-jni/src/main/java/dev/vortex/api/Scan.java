// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeScan;
import java.util.Iterator;
import java.util.NoSuchElementException;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * A lazy handle to a set of {@link Partition partitions}.
 *
 * <p>Once a scan has produced its last partition it is effectively exhausted; native
 * resources are released automatically via {@link VortexCleaner} when the scan becomes
 * unreachable.
 */
public final class Scan implements Iterator<Partition> {
    private final Session session;
    private final long pointer;

    private long nextPartitionPointer;
    private boolean primed;

    private Scan(Session session, long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid scan pointer");
        this.session = session;
        this.pointer = pointer;
        VortexCleaner.register(this, () -> NativeScan.free(pointer));
    }

    static Scan fromPointer(Session session, long pointer) {
        return new Scan(session, pointer);
    }

    /**
     * Arrow schema produced by this scan. Must be called before the first call to
     * {@link #hasNext()}/{@link #next()}.
     */
    public Schema arrowSchema(BufferAllocator allocator) {
        try (ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
            NativeScan.arrowSchema(pointer, schema.memoryAddress());
            return Data.importSchema(allocator, schema, null);
        }
    }

    @Override
    public boolean hasNext() {
        if (primed) {
            return nextPartitionPointer != 0;
        }
        nextPartitionPointer = NativeScan.nextPartition(pointer);
        primed = true;
        return nextPartitionPointer != 0;
    }

    @Override
    public Partition next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        long ptr = nextPartitionPointer;
        nextPartitionPointer = 0;
        primed = false;
        return Partition.fromPointer(session, ptr);
    }
}
