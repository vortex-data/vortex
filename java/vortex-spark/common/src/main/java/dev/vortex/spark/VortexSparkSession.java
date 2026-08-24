// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import dev.vortex.api.Session;
import dev.vortex.jni.NativeRuntime;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * JVM-wide holder for one or more Vortex {@link Session}s used by Spark readers and writers. The Rust side multiplexes
 * every session onto one shared current-thread runtime, so sharing a single Java handle per JVM amortises session
 * construction across every Spark task.
 *
 * <p>Three levels of customisation:
 *
 * <ol>
 *   <li><b>Default:</b> call {@link #get()} — returns a lazily-initialised singleton that lives for the life of the
 *       JVM. No configuration required.
 *   <li><b>Driver-side override:</b> call {@link #setDefault(Session)} with a custom session before any Spark action.
 *       Effective on the driver JVM only.
 *   <li><b>Executor-friendly override:</b> pass option {@code vortex.session.provider} with the fully-qualified name of
 *       a {@link VortexSessionProvider} implementation (no-arg constructor). Spark ships the class name to every
 *       executor, which instantiates the provider once per JVM and caches the returned session. Use this when you need
 *       the same custom session on the driver and on executors.
 * </ol>
 *
 * <p>Native resources are released by {@code VortexCleaner} when the JVM shuts down; sessions held here are strongly
 * referenced for the JVM's lifetime.
 */
public final class VortexSparkSession {
    /** Options key used to select a {@link VortexSessionProvider} by class name. */
    public static final String PROVIDER_OPTION = "vortex.session.provider";

    /** Options key sizing the JVM-wide pool of background threads that drive Vortex futures. */
    public static final String WORKER_THREADS_OPTION = "vortex.workerThreads";

    private static final int DEFAULT_WORKER_THREADS = 4;

    private static final ConcurrentHashMap<String, Session> providerCache = new ConcurrentHashMap<>();
    private static final AtomicBoolean runtimeConfigured = new AtomicBoolean();
    private static volatile Session defaultSession;

    private VortexSparkSession() {}

    /** Returns the default JVM-wide session, creating it on first use. */
    public static Session get() {
        Session s = defaultSession;
        if (s != null) {
            return s;
        }
        synchronized (VortexSparkSession.class) {
            if (defaultSession == null) {
                defaultSession = Session.create();
            }
            return defaultSession;
        }
    }

    /**
     * Resolve the session to use for a given set of Spark format options. Honours the {@value #PROVIDER_OPTION} key;
     * falls back to {@link #get()} otherwise.
     */
    public static Session get(VortexOptions options) {
        configureRuntime(options);
        Optional<String> providerClass = options.get(PROVIDER_OPTION).filter(name -> !name.isEmpty());
        if (providerClass.isEmpty()) {
            return get();
        }
        return providerCache.computeIfAbsent(providerClass.get(), VortexSparkSession::loadProvider);
    }

    /**
     * Sizes the shared worker pool from {@value #WORKER_THREADS_OPTION}, once per JVM.
     *
     * <p>The pool is JVM-wide and resizing it starts or stops threads, so every read and write path resolves its
     * session through here and the first caller wins. Repeating the call per file, or per task, would have concurrent
     * tasks resize the pool underneath each other.
     */
    private static void configureRuntime(VortexOptions options) {
        if (!runtimeConfigured.compareAndSet(false, true)) {
            return;
        }
        int workers = options.getInt(WORKER_THREADS_OPTION, DEFAULT_WORKER_THREADS);
        if (workers < 0) {
            runtimeConfigured.set(false);
            throw new IllegalArgumentException(WORKER_THREADS_OPTION + " must be >= 0, got " + workers);
        }
        NativeRuntime.setWorkerThreads(workers);
    }

    /**
     * Replace the default session. Intended for driver-side customisation before any Spark action runs. Does not
     * propagate to executors — use {@link VortexSessionProvider} for that.
     */
    public static void setDefault(Session session) {
        Objects.requireNonNull(session, "session");
        synchronized (VortexSparkSession.class) {
            defaultSession = session;
        }
    }

    private static Session loadProvider(String className) {
        try {
            Class<?> cls = Class.forName(className, true, Thread.currentThread().getContextClassLoader());
            Object instance = cls.getDeclaredConstructor().newInstance();
            if (!(instance instanceof VortexSessionProvider provider)) {
                throw new IllegalArgumentException(
                        className + " does not implement " + VortexSessionProvider.class.getName());
            }
            return Objects.requireNonNull(provider.get(), className + ".get() returned null");
        } catch (ReflectiveOperationException e) {
            throw new IllegalArgumentException("Failed to load Vortex session provider " + className, e);
        }
    }
}
