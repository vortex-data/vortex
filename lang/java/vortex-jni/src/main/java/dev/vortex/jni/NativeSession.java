// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/** JNI boundary for {@link dev.vortex.api.Session}. */
public final class NativeSession {
    static {
        NativeLoader.loadJni();
    }

    private NativeSession() {}

    /** Allocate a fresh native session. Free with {@link #free(long)}. */
    public static native long newSession();

    /** Free a session previously returned by {@link #newSession()}. */
    public static native void free(long pointer);

    /**
     * Enable the edition with the name {@code edition}, for example {@code core2025.10.0}, for writes that start after
     * this call. Throws a {@link RuntimeException} if the edition is not known.
     */
    public static native void enableEdition(long pointer, String edition);
}
