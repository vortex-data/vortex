// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/**
 * Utility class for configuring native logging levels in the Vortex JNI layer.
 *
 * <p>This class provides constants for different logging levels and methods to initialize native logging to the desired
 * verbosity level. The logging levels correspond to standard logging frameworks with ERROR being the least verbose and
 * TRACE being the most verbose.
 */
public final class NativeLogging {
    static {
        NativeLoader.loadJni();
    }

    private NativeLogging() {}

    /** Logging level constant for error messages only */
    public static final int ERROR = 0;

    /** Logging level constant for warning and error messages */
    public static final int WARN = 1;

    /** Logging level constant for informational, warning, and error messages */
    public static final int INFO = 2;

    /** Logging level constant for debug, informational, warning, and error messages */
    public static final int DEBUG = 3;

    /** Logging level constant for all messages including trace-level debugging */
    public static final int TRACE = 4;

    /**
     * Initialize logging to the desired level. Must be one of:
     *
     * <ul>
     *   <li>{@link #ERROR}
     *   <li>{@link #WARN}
     *   <li>{@link #INFO}
     *   <li>{@link #DEBUG}
     *   <li>{@link #TRACE}
     * </ul>
     */
    public static native void initLogging(int level);
}
