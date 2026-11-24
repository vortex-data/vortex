// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import java.io.Closeable;
import java.io.IOException;

/**
 * A sequential byte sink supplied by the caller and written to from native code.
 *
 * <p>Implementations bridge external output abstractions — for example Iceberg's {@code PositionOutputStream} — into
 * the Vortex native writer. Writes are sequential and single-threaded: the native side never issues concurrent
 * {@code write} calls against the same instance.
 *
 * <p>Lifecycle: the creator owns this object. Native code calls {@link #write} and {@link #flush} but never
 * {@link #close()}; after {@link dev.vortex.api.VortexWriter#close()} returns, all bytes have been written and flushed,
 * and the creator must close the underlying stream to finalize it.
 */
public interface NativeWritable extends Closeable {
    /** Append {@code length} bytes from {@code buffer} starting at {@code offset}. */
    void write(byte[] buffer, int offset, int length) throws IOException;

    /** Flush buffered bytes to the underlying storage. */
    void flush() throws IOException;
}
