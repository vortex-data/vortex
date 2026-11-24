// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeSession;

/**
 * Handle to a native Vortex session. The session owns a current-thread async runtime and is the entry point for opening
 * {@link DataSource data sources} and {@link VortexWriter writers}.
 *
 * <p>Sessions are safe to share across threads within a process, though concrete operations (scans, writes) remain
 * single-threaded on the session's runtime thread.
 *
 * <p>Native resources are released automatically when the session becomes unreachable, via {@link VortexCleaner}.
 */
public final class Session {
    private final long pointer;

    private Session(long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid session pointer");
        this.pointer = pointer;
        VortexCleaner.register(this, () -> NativeSession.free(pointer));
    }

    /** Create a new session. */
    public static Session create() {
        return new Session(NativeSession.newSession());
    }

    /**
     * Enable the edition with the name {@code edition}, for example {@code core2025.10.0}. The edition replaces the
     * enabled edition of the same family. Only writers that start after this call use it.
     *
     * <p>Select an older edition to write files that older Vortex readers can read.
     */
    public void enableEdition(String edition) {
        NativeSession.enableEdition(pointer, edition);
    }

    /** Internal: returns the native pointer. Do not free directly. */
    public long nativePointer() {
        return pointer;
    }
}
