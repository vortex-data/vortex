// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativePointer;
import dev.vortex.jni.NativeWriter;
import java.io.IOException;
import java.lang.ref.Cleaner;
import java.util.Map;
import java.util.Objects;
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
 * callers should always finalize explicitly so that I/O errors surface through the normal call path. After close, any
 * method that accesses the native pointer throws {@link IllegalStateException} rather than risking use-after-free.
 */
public final class VortexWriter implements AutoCloseable {
    private final NativePointer pointer;
    private final Cleaner.Cleanable closeHandle;

    private VortexWriter(long pointer) {
        this.pointer = NativePointer.of(pointer);
        NativePointer pointerRef = this.pointer;
        this.closeHandle = VortexCleaner.register(this, () -> NativeWriter.close(pointerRef.take()));
    }

    /**
     * Create a writer that streams records into the file at {@code uri}. The path may be a full URI or a plain local
     * filesystem path. The Arrow schema describes the exact layout of every batch written.
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

    private long nativePointer() {
        return pointer.read();
    }

    /** Write a batch directly from Arrow C Data Interface addresses. */
    public void writeBatch(long arrowArrayAddr, long arrowSchemaAddr) throws IOException {
        long ptr = nativePointer();
        final boolean ok;
        try {
            ok = NativeWriter.writeBatch(ptr, arrowArrayAddr, arrowSchemaAddr);
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
        try {
            closeHandle.clean();
        } catch (RuntimeException e) {
            throw new IOException("failed to close writer", e);
        }
    }
}
