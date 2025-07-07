// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

public final class NativeLogging {
    static {
        NativeLoader.loadJni();
    }

    private NativeLogging() {}

    public static final int ERROR = 0;
    public static final int WARN = 1;
    public static final int INFO = 2;
    public static final int DEBUG = 3;
    public static final int TRACE = 4;

    /**
     * Initialize logging to the desired level. Must be one of:
     * <ul>
     *  <li>{@link #ERROR}</li>
     *  <li>{@link #WARN}</li>
     *  <li>{@link #INFO}</li>
     *  <li>{@link #DEBUG}</li>
     *  <li>{@link #TRACE}</li>
     * </ul>
     */
    public static native void initLogging(int level);
}
