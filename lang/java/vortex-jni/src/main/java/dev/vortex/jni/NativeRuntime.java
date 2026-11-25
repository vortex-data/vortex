// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

/**
 * Controls for the JVM-wide current-thread worker pool that backs every Vortex session.
 *
 * <p>By default the pool has zero background threads: nothing happens until a Java thread calls a blocking Vortex API.
 * Adding workers via {@link #setWorkerThreads(int)} lets the pool drive Vortex futures on behalf of the caller — useful
 * when a single consumer thread cannot keep the executor busy on its own. Spawning Java threads that each poll a
 * thread-safe Vortex iterator is an equivalent alternative.
 */
public final class NativeRuntime {
    static {
        NativeLoader.loadJni();
    }

    private NativeRuntime() {}

    /**
     * Set the number of background worker threads driving Vortex futures. {@code 0} disables background execution; any
     * positive value adjusts the pool up or down.
     */
    public static native void setWorkerThreads(int n);

    /** Size the pool to {@code Runtime.getRuntime().availableProcessors() - 1}. */
    public static native void setWorkerThreadsToAvailableParallelism();

    /** Returns the current background worker-thread count. */
    public static native int workerCount();
}
