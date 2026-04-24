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
}
