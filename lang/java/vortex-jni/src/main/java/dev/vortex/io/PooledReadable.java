// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import com.google.common.base.Preconditions;
import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * A {@link NativeReadable} that serves each of Vortex's concurrent reads from its own stream.
 *
 * <p>Streams are opened on demand and reused until this readable is closed. Subclasses supply only
 * {@link #openStream()}.
 */
public abstract class PooledReadable implements NativeReadable {
    /** Largest scratch array retained for reuse on a pooled stream; bigger reads get a one-shot array. */
    private static final int SCRATCH_RETAIN_LIMIT = 1 << 20;

    private final String name;
    private final long length;
    private final Queue<PooledStream> pool = new ConcurrentLinkedQueue<>();
    private volatile boolean closed = false;

    /**
     * @param name stable unique name for this source, as required by {@link NativeReadable#name()}
     * @param length total size of the source in bytes
     */
    protected PooledReadable(String name, long length) {
        Preconditions.checkArgument(name != null && !name.isEmpty(), "name is required");
        Preconditions.checkArgument(length >= 0, "length must not be negative: %s", length);
        this.name = name;
        this.length = length;
    }

    /** Opens one more handle on this source. Called whenever a read finds no free stream in the pool. */
    protected abstract PositionalStream openStream() throws IOException;

    @Override
    public final String name() {
        return name;
    }

    @Override
    public final long length() {
        return length;
    }

    @Override
    public final void readFully(long position, ByteBuffer buffer) throws IOException {
        Preconditions.checkState(!closed, "Cannot read %s: already closed", name);
        int requested = buffer.remaining();
        if (position < 0 || position + requested > length) {
            throw new EOFException(String.format(
                    Locale.ROOT,
                    "Cannot read %d bytes at position %d: %s is %d bytes long",
                    requested,
                    position,
                    name,
                    length));
        }

        PooledStream pooled = pool.poll();
        if (pooled == null) {
            pooled = new PooledStream(openStream());
        }

        try {
            pooled.stream.readFully(position, buffer, pooled);
        } catch (IOException | RuntimeException e) {
            try {
                pooled.close();
            } catch (IOException suppressed) {
                e.addSuppressed(suppressed);
            }
            throw e;
        }

        if (buffer.hasRemaining()) {
            EOFException failure = new EOFException(String.format(
                    Locale.ROOT,
                    "Read of %d bytes at position %d in %s left %d bytes unfilled",
                    requested,
                    position,
                    name,
                    buffer.remaining()));
            try {
                pooled.close();
            } catch (IOException e) {
                failure.addSuppressed(e);
            }
            throw failure;
        }

        pool.add(pooled);
        if (closed) {
            // Racing with close(): make sure nothing this read returned to the pool leaks. The read
            // itself succeeded, so a failure to close an idle stream must not fail it retroactively.
            try {
                closeAllPooled();
            } catch (IOException ignored) {
                // Best-effort: the streams were drained from the pool and close was attempted on each.
            }
        }
    }

    @Override
    public void close() throws IOException {
        closed = true;
        closeAllPooled();
    }

    private void closeAllPooled() throws IOException {
        List<PooledStream> drained = new ArrayList<>();
        PooledStream pooled;
        while ((pooled = pool.poll()) != null) {
            drained.add(pooled);
        }
        Closeables.closeAll(drained);
    }

    /** One handle on the source, used by at most one read at a time. */
    public interface PositionalStream extends Closeable {
        /**
         * Fills {@code buffer} completely from absolute {@code position}, in as few requests to storage as possible:
         * Vortex has already coalesced neighbouring reads into this one.
         *
         * <p>Implementations limited to {@code byte[]} APIs should stage through {@code scratch} rather than allocate.
         */
        void readFully(long position, ByteBuffer buffer, ScratchBytes scratch) throws IOException;
    }

    /** A staging array, reused across the reads served by one stream. */
    public interface ScratchBytes {
        /** Returns an array of at least {@code length} bytes, whose contents are undefined. */
        byte[] bytes(int length);
    }

    private static final class PooledStream implements ScratchBytes, Closeable {
        private final PositionalStream stream;
        private byte[] scratch = new byte[0];

        private PooledStream(PositionalStream stream) {
            this.stream = stream;
        }

        @Override
        public byte[] bytes(int length) {
            if (scratch.length >= length) {
                return scratch;
            } else if (length > SCRATCH_RETAIN_LIMIT) {
                return new byte[length];
            }

            scratch = new byte[length];
            return scratch;
        }

        @Override
        public void close() throws IOException {
            stream.close();
        }
    }
}
