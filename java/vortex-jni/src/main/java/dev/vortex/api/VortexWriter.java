// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.io.NativeWritable;
import dev.vortex.jni.NativeBuffers;
import dev.vortex.jni.NativeWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Writer for Vortex files.
 *
 * <p>Open one with {@link #builder(Session, String, Schema, BufferAllocator)} to write to a URI, or
 * {@link #builder(Session, NativeWritable, Schema, BufferAllocator)} to write into a caller-provided byte sink.
 *
 * <p>Use {@link #writeBatch(VectorSchemaRoot, BufferAllocator)} for zero-copy Java batches with independent buffer
 * lifetimes. Batches are also accepted via the Arrow C Data Interface: callers export an Arrow record batch to an
 * {@code ArrowArray} / {@code ArrowSchema} pair and pass the addresses to {@link #writeBatch(long, long)}, or export
 * only the {@code ArrowArray} and pass its address to {@link #writeBatch(long)}. The writer accepts up to four
 * in-flight batches on the session's runtime thread before back-pressuring the caller.
 *
 * <p>Call {@link #close()} to flush remaining batches and finalize the file. Call {@link #finish(Map)} instead to add
 * metadata segments that are known only after the last batch, for example a bloom filter of the written keys.
 *
 * <p>If the writer becomes unreachable without an explicit {@code close()}, {@link VortexCleaner} will flush and
 * release native resources as a backstop — but callers should always finalize explicitly so that I/O errors surface
 * through the normal call path.
 */
public final class VortexWriter implements AutoCloseable {
    private final long pointer;
    private final Schema arrowSchema;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private volatile VortexWriteSummary summary;

    private VortexWriter(long pointer, Schema arrowSchema) {
        Preconditions.checkArgument(pointer != 0, "invalid writer pointer");
        this.pointer = pointer;
        this.arrowSchema = arrowSchema;
        AtomicBoolean closedRef = this.closed;
        VortexCleaner.register(this, () -> {
            if (closedRef.compareAndSet(false, true)) {
                NativeWriter.close(pointer);
            }
        });
    }

    /**
     * Start configuring a writer that streams records into the file at {@code uri} through a native storage client. The
     * path may be a full URI or a plain local filesystem path.
     *
     * @param arrowSchema describes the exact layout of every batch written
     */
    public static Builder builder(Session session, String uri, Schema arrowSchema, BufferAllocator allocator) {
        return new Builder(session, Objects.requireNonNull(uri, "uri"), null, arrowSchema, allocator);
    }

    /**
     * Start configuring a writer that streams the file into a caller-provided byte sink instead of a native storage
     * client. This is the integration point for external I/O abstractions (for example Iceberg's
     * {@code PositionOutputStream}).
     *
     * <p>The native side writes and flushes the sink but never closes it: after {@link #close()} returns, all bytes
     * have been written and flushed, and the caller must close the sink to finalize the file.
     *
     * @param arrowSchema describes the exact layout of every batch written
     */
    public static Builder builder(
            Session session, NativeWritable writable, Schema arrowSchema, BufferAllocator allocator) {
        return new Builder(session, null, Objects.requireNonNull(writable, "writable"), arrowSchema, allocator);
    }

    /**
     * Configures and opens a {@link VortexWriter}. Everything required to open one is fixed by
     * {@link VortexWriter#builder}; the setters here are all optional.
     *
     * <p>Not thread-safe, and not reusable: each {@link #build()} opens one file.
     */
    public static final class Builder {
        private final Session session;
        private final String uri;
        private final NativeWritable writable;
        private final Schema arrowSchema;
        private final BufferAllocator allocator;
        private final Map<String, String> options = new HashMap<>();
        private final Map<String, byte[]> metadata = new HashMap<>();
        private boolean compact;

        private Builder(
                Session session, String uri, NativeWritable writable, Schema arrowSchema, BufferAllocator allocator) {
            this.session = Objects.requireNonNull(session, "session");
            this.uri = uri;
            this.writable = writable;
            this.arrowSchema = Objects.requireNonNull(arrowSchema, "arrowSchema");
            this.allocator = Objects.requireNonNull(allocator, "allocator");
        }

        /**
         * Object-store credentials and options, replacing any set so far. Only the URI destination reaches a storage
         * client, so this is rejected on a writer built over a {@link NativeWritable}.
         */
        public Builder options(Map<String, String> newOptions) {
            checkOptionsApply();
            Objects.requireNonNull(newOptions, "options");
            options.clear();
            newOptions.forEach(this::putOption);
            return this;
        }

        /** Add a single object-store option. See {@link #options(Map)}. */
        public Builder putOption(String key, String value) {
            checkOptionsApply();
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
            options.put(key, value);
            return this;
        }

        /**
         * User-defined metadata segments to store in the file footer, replacing any set so far.
         *
         * <p>Values are opaque bytes, returned verbatim by {@code NativeFiles.readMetadata}. The native writer caps
         * both the number of segments per file and the length of each key; the limits are enforced (and named in the
         * error) by {@link #build()}, rather than when the file is finalized.
         */
        public Builder metadata(Map<String, byte[]> newMetadata) {
            Objects.requireNonNull(newMetadata, "metadata");
            metadata.clear();
            newMetadata.forEach(this::putMetadata);
            return this;
        }

        /** Add a single metadata segment. See {@link #metadata(Map)}. */
        public Builder putMetadata(String key, byte[] value) {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
            metadata.put(key, value);
            return this;
        }

        /**
         * Write with the compact strategy, which prefers a smaller file over read throughput and latency. It is what
         * {@code vortex.io.VortexWriteOptions.compact()} selects in Python, and is off by default. The file is read
         * like any other.
         */
        public Builder compact(boolean compact) {
            this.compact = compact;
            return this;
        }

        /**
         * Open the writer.
         *
         * @throws IOException if the writer cannot be opened, including when the native side rejects the configured
         *     metadata
         */
        public VortexWriter build() throws IOException {
            ArrowSchema ffi = ArrowSchema.allocateNew(allocator);
            try {
                Data.exportSchema(allocator, arrowSchema, null, ffi);
                final long pointer;
                try {
                    pointer = uri != null
                            ? NativeWriter.create(
                                    session.nativePointer(), uri, ffi.memoryAddress(), options, metadata, compact)
                            : NativeWriter.createStream(
                                    session.nativePointer(), writable, ffi.memoryAddress(), metadata, compact);
                } catch (RuntimeException e) {
                    // Native failures arrive as RuntimeException. Wrap them so callers can handle every
                    // failure to open a writer as an IOException, as writeBatch and finish already do.
                    throw new IOException("failed to create writer for " + destination(), e);
                }
                if (pointer <= 0) {
                    throw new IOException("failed to create writer for " + destination() + " (ptr=" + pointer + ")");
                }
                return new VortexWriter(pointer, arrowSchema);
            } finally {
                // Native code takes and releases the schema. If it failed before it took the schema, release it here.
                if (ffi.snapshot().release != 0) {
                    ffi.release();
                }
                ffi.close();
            }
        }

        private String destination() {
            return uri != null ? "uri " + uri : "stream";
        }

        private void checkOptionsApply() {
            Preconditions.checkState(uri != null, "options apply to uri destinations only");
        }
    }

    /**
     * Write a Java Arrow batch without copying its data. Native buffers retain their allocations independently, so a
     * retained column does not pin the entire batch. The caller may close or clear the root after this method returns,
     * but must not overwrite its retained buffers.
     *
     * @param root batch to write; dictionary-encoded fields require the address-based API
     * @param allocator allocator for temporary C Data Interface metadata
     */
    public synchronized void writeBatch(VectorSchemaRoot root, BufferAllocator allocator) throws IOException {
        Preconditions.checkState(!closed.get(), "writer already closed");
        if (!arrowSchema.equals(root.getSchema())) {
            throw new IOException("batch schema does not match writer schema");
        }
        try (ArrowArray array = ArrowArray.allocateNew(allocator)) {
            try {
                Data.exportVectorSchemaRoot(allocator, root, null, array);
                if (!NativeWriter.writeBatchBuffers(pointer, array.memoryAddress(), NativeBuffers.collect(root))) {
                    throw new IOException("failed to write batch");
                }
            } finally {
                array.release();
            }
        } catch (RuntimeException e) {
            throw new IOException("failed to write batch", e);
        }
    }

    /**
     * Write a batch directly from Arrow C Data Interface addresses. Native code takes both the array and the schema,
     * and releases them. These buffers share the producer's batch release callback; use
     * {@link #writeBatch(VectorSchemaRoot, BufferAllocator)} for independently retained Java allocations.
     */
    public synchronized void writeBatch(long arrowArrayAddr, long arrowSchemaAddr) throws IOException {
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
     * Write a batch from an Arrow C Data Interface array. Native code imports it with the schema of this writer.
     *
     * <p>The array must be a struct array that agrees with the writer schema. Native code rejects incorrect buffer or
     * child counts with an {@link IOException}, but it cannot find all type mismatches (for example {@code Int64} in
     * the place of {@code Int32}), and such an array can cause incorrect data or a crash.
     *
     * <p>Native code takes the array struct and releases it, also when the call fails.
     *
     * @param arrowArrayAddr address of an exported {@code ArrowArray} struct
     */
    public synchronized void writeBatch(long arrowArrayAddr) throws IOException {
        Preconditions.checkState(!closed.get(), "writer already closed");
        Preconditions.checkArgument(arrowArrayAddr != 0, "null arrow array address");

        final boolean ok;
        try {
            ok = NativeWriter.writeArray(pointer, arrowArrayAddr);
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
        return finishWith(null);
    }

    /**
     * Flush pending batches, add {@code metadata} to the file footer, finalize the file, and return its statistics and
     * physical sizes.
     *
     * <p>The segments merge with the segments from {@link Builder#metadata(Map)}, and a key in {@code metadata}
     * replaces the builder segment with the same key. The native limits apply to the merged set.
     *
     * <p>This method is not idempotent: a call after the writer is finished or closed throws
     * {@link IllegalStateException}.
     *
     * @param metadata the segments to add; keys and values must not be null
     * @throws IllegalStateException if the writer is already finished or closed
     * @throws IOException if the file cannot be finalized
     */
    public synchronized VortexWriteSummary finish(Map<String, byte[]> metadata) throws IOException {
        Objects.requireNonNull(metadata, "metadata");
        Preconditions.checkState(!closed.get(), "writer already finished; metadata must be given to the first finish");

        // Check and copy the entries before the native call consumes the writer.
        Map<String, byte[]> segments = new HashMap<>();
        metadata.forEach((key, value) -> segments.put(
                Objects.requireNonNull(key, "metadata key"), Objects.requireNonNull(value, "metadata value")));

        return finishWith(segments);
    }

    private VortexWriteSummary finishWith(Map<String, byte[]> metadata) throws IOException {
        if (closed.compareAndSet(false, true)) {
            try {
                summary = NativeWriter.finish(pointer, metadata);
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
