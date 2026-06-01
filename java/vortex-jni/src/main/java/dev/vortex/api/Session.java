// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativePointer;
import dev.vortex.jni.NativeSession;
import java.lang.ref.Cleaner;

/**
 * Handle to a native Vortex session. The session owns a current-thread async runtime and is the entry point for opening
 * {@link DataSource data sources} and {@link VortexWriter writers}.
 *
 * <p>Sessions are safe to share across threads within a process, though concrete operations (scans, writes) remain
 * single-threaded on the session's runtime thread.
 *
 * <p>Callers should close the session explicitly to release native resources promptly. If the session becomes
 * unreachable without {@link #close()}, {@link VortexCleaner} releases it as a backstop. After close, any method that
 * accesses the native pointer throws {@link IllegalStateException} rather than risking use-after-free.
 */
public final class Session implements AutoCloseable {
    private final NativePointer pointer;
    private final Cleaner.Cleanable closeHandle;

    private Session(long pointer) {
        this.pointer = NativePointer.of(pointer);
        NativePointer pointerRef = this.pointer;
        this.closeHandle = VortexCleaner.register(this, () -> NativeSession.free(pointerRef.take()));
    }

    /** Create a new session. */
    public static Session create() {
        return new Session(NativeSession.newSession());
    }

    /** Internal: returns the native pointer. Do not free directly. */
    public long nativePointer() {
        return pointer.read();
    }

    /** Release the native session. Idempotent. */
    @Override
    public void close() {
        closeHandle.clean();
    }
}
