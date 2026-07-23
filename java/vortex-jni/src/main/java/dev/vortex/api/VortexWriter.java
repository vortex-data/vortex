// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.io.NativeWritable;
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
    private volatile VortexWriteSummary summary;

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

    /**
     * Create a writer that streams the file into a caller-provided byte sink instead of a native storage client. This
     * is the integration point for external I/O abstractions (for example Iceberg's {@code PositionOutputStream}).
     *
     * <p>The native side writes and flushes the sink but never closes it: after {@link #close()} returns, all bytes
     * have been written and flushed, and the caller must close the sink to finalize the file.
     */
    public static VortexWriter create(
            Session session, NativeWritable writable, Schema arrowSchema, BufferAllocator allocator)
            throws IOException {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(writable, "writable");
        Objects.requireNonNull(arrowSchema, "arrowSchema");
        Objects.requireNonNull(allocator, "allocator");
        ArrowSchema ffi = ArrowSchema.allocateNew(allocator);
        try {
            Data.exportSchema(allocator, arrowSchema, null, ffi);
            long ptr = NativeWriter.createStream(session.nativePointer(), writable, ffi.memoryAddress());
            if (ptr <= 0) {
                throw new IOException("failed to create stream writer (ptr=" + ptr + ")");
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

    /**
     * Return the number of bytes successfully written to the underlying sink so far.
     *
     * <p>This count does not include queued batches or data still buffered by layout strategies, so it may lag the
     * amount of input accepted by {@link #writeBatch(long, long)}. After {@link #finish()}, it is the exact completed
     * file size and is equal to {@link VortexWriteSummary#fileSize()}.
     */
    public synchronized long bytesWritten() {
        if (summary != null) {
            return summary.fileSize();
        }
        Preconditions.checkState(!closed.get(), "writer closed without a write summary");
        long bytesWritten = NativeWriter.bytesWritten(pointer);
        Preconditions.checkState(bytesWritten >= 0, "native writer returned an invalid byte count");
        return bytesWritten;
    }

    /**
     * Return the number of uncompressed bytes accepted by the writer but not yet written to the sink.
     *
     * <p>Together with {@link #bytesWritten()}, this lets callers estimate the in-progress file size: bytes that
     * reached the sink are already compressed, while buffered bytes are still uncompressed and will shrink by roughly
     * the file's observed compression ratio once flushed. After {@link #finish()}, this is zero.
     */
    public synchronized long bufferedBytes() {
        if (summary != null) {
            return 0;
        }
        Preconditions.checkState(!closed.get(), "writer closed without a write summary");
        long bufferedBytes = NativeWriter.bufferedBytes(pointer);
        Preconditions.checkState(bufferedBytes >= 0, "native writer returned an invalid buffered byte count");
        return bufferedBytes;
    }

    /**
     * Flush pending batches, finalize the file, and return its statistics and physical sizes.
     *
     * <p>This method is idempotent. Later calls return the same immutable summary.
     */
    public synchronized VortexWriteSummary finish() throws IOException {
        if (closed.compareAndSet(false, true)) {
            try {
                summary = NativeWriter.finish(pointer);
            } catch (RuntimeException e) {
                throw new IOException("failed to close writer", e);
            }
        }
        if (summary == null) {
            throw new IOException("writer was closed without retaining its write summary");
        }
        return summary;
    }

    /** Flush any pending batches and finalize the file. Idempotent. */
    @Override
    public void close() throws IOException {
        finish();
    }
}
