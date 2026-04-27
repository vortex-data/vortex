// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import dev.vortex.api.Session;

/**
 * User hook for supplying a custom {@link Session} to Vortex Spark readers and writers.
 *
 * <p>Implementations must have a public no-argument constructor. They are instantiated
 * exactly once per JVM (driver or executor) the first time a scan or write references
 * the provider's class name through the {@code vortex.session.provider} option. The
 * returned {@link Session} is then shared across every Vortex task on that JVM.
 *
 * <p>Typical use: install custom encodings, scalar functions, or layouts on a
 * {@link Session} before returning it.
 */
public interface VortexSessionProvider {
    /**
     * Construct (or return a cached) {@link Session}. Called at most once per JVM per
     * provider class. The returned session is retained for the JVM's lifetime.
     */
    Session get();
}
