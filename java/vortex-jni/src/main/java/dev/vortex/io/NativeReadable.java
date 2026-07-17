// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;

/**
 * A random-access byte source supplied by the caller and read from native code.
 *
 * <p>Implementations bridge external I/O abstractions — for example Iceberg's {@code FileIO} input streams — into the
 * Vortex native reader. The native side invokes {@link #readFully(long, ByteBuffer)} from its own worker threads,
 * potentially many calls concurrently, so implementations must be safe for concurrent positional reads (typically by
 * opening one underlying stream per concurrent call, or delegating to a positional-read API).
 *
 * <p>Lifecycle: the creator owns this object. Native code never calls {@link #close()}; close it only after every data
 * source or scan built on top of it has been closed.
 */
public interface NativeReadable extends Closeable {
    /**
     * Unique name of this source, typically its original location (URI or file path). Used to key per-session caches
     * and in native error messages, so it must be stable and unique across sources; it must not contain glob characters
     * ({@code *?[}).
     */
    String name();

    /** Total length of the source in bytes. Must be cheap; called once when the source is registered. */
    long length();

    /**
     * Read exactly {@code buffer.remaining()} bytes starting at absolute {@code position} into {@code buffer}.
     *
     * <p>Unlike {@link java.io.InputStream#read}, short reads are not permitted: implementations must either fill the
     * buffer completely ({@code buffer.remaining() == 0} on return) or throw.
     *
     * <p>{@code buffer} is a <em>direct</em> {@link ByteBuffer} wrapping native memory owned by the caller, so bytes
     * written to it land in the native reader without an extra copy. It is valid only for the duration of this call:
     * implementations must not retain a reference to it after returning. Implementations backed by {@code byte[]} APIs
     * can bulk-{@link ByteBuffer#put(byte[], int, int) put} into it.
     *
     * @throws java.io.EOFException if the range extends past the end of the source
     * @throws IOException if the underlying storage fails
     */
    void readFully(long position, ByteBuffer buffer) throws IOException;
}
