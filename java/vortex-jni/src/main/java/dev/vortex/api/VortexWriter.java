// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeWriter;
import java.io.IOException;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Writer for Vortex files.
 *
 * <p>Batches are accepted via the Arrow C Data Interface: callers export an Arrow record batch to an {@code ArrowArray}
 * / {@code ArrowSchema} pair and pass the addresses to {@link #writeBatch(long, long)}. The writer accepts up to four
 * in-flight batches on the session's runtime thread before back-pressuring the caller.
 *
 * <p>Call {@link #close()} to flush remaining batches and finalize the file. If the writer becomes unreachable without
 * an explicit {@code close()}, {@link VortexCleaner} will flush and release native resources as a backstop — but
 * callers should always finalize explicitly so that I/O errors surface through the normal call path.
 */
public final class VortexWriter implements AutoCloseable {
    private final long pointer;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private VortexWriter(long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid writer pointer");
        this.pointer = pointer;
        AtomicBoolean closedRef = this.closed;
        VortexCleaner.register(this, () -> {
            if (closedRef.compareAndSet(false, true)) {
                NativeWriter.close(pointer);
            }
        });
    }

    /**
     * Create a writer that streams records into the file at {@code uri}. The Arrow schema describes the exact layout of
     * every batch written.
     */
    public static VortexWriter create(
            Session session, String uri, Schema arrowSchema, Map<String, String> options, BufferAllocator allocator)
            throws IOException {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(uri, "uri");
        Objects.requireNonNull(arrowSchema, "arrowSchema");
        Objects.requireNonNull(allocator, "allocator");
        ArrowSchema ffi = ArrowSchema.allocateNew(allocator);
        try {
            Data.exportSchema(allocator, arrowSchema, null, ffi);
            long ptr = NativeWriter.create(session.nativePointer(), uri, ffi.memoryAddress(), options);
            if (ptr <= 0) {
                throw new IOException("failed to create writer for uri " + uri + " (ptr=" + ptr + ")");
            }
            return new VortexWriter(ptr);
        } finally {
            ffi.close();
        }
    }

    /** Write a batch directly from Arrow C Data Interface addresses. */
    public void writeBatch(long arrowArrayAddr, long arrowSchemaAddr) throws IOException {
        Preconditions.checkState(!closed.get(), "writer already closed");
        final boolean ok;
        try {
            ok = NativeWriter.writeBatch(pointer, arrowArrayAddr, arrowSchemaAddr);
        } catch (RuntimeException e) {
            throw new IOException("failed to write batch", e);
        }
        if (!ok) {
            throw new IOException("failed to write batch");
        }
    }

    /** Flush any pending batches and finalize the file. Idempotent. */
    @Override
    public void close() throws IOException {
        if (closed.compareAndSet(false, true)) {
            try {
                NativeWriter.close(pointer);
            } catch (RuntimeException e) {
                throw new IOException("failed to close writer", e);
            }
        }
    }
}
