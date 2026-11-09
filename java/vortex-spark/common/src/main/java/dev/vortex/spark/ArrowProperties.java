// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

/**
 * Turns off the per-value checks of the Arrow that Vortex relocates, as Spark's vectorized readers do for their own
 * Arrow.
 *
 * <p>Arrow reads these settings once, when it first loads the classes that check bounds and nulls. So the Vortex
 * classes that load first in a JVM call {@link #configure()} from their static initializers. A setting the user gives
 * as a system property or an environment variable stays as it is.
 *
 * <p>The readers check bounds and nulls themselves: Spark asks a column for null before it asks for a value, and each
 * value is read at an index below the row count of its batch.
 */
public final class ArrowProperties {
    private static final String UNSAFE_MEMORY_ACCESS = "arrow.enable_unsafe_memory_access";
    private static final String UNSAFE_MEMORY_ACCESS_ENV = "ARROW_ENABLE_UNSAFE_MEMORY_ACCESS";
    private static final String NULL_CHECK_FOR_GET = "arrow.enable_null_check_for_get";
    private static final String NULL_CHECK_FOR_GET_ENV = "ARROW_ENABLE_NULL_CHECK_FOR_GET";

    static {
        setUnlessGiven(UNSAFE_MEMORY_ACCESS, UNSAFE_MEMORY_ACCESS_ENV, "true");
        setUnlessGiven(NULL_CHECK_FOR_GET, NULL_CHECK_FOR_GET_ENV, "false");
    }

    private ArrowProperties() {}

    /** Applies the settings, once per JVM. The class initializer does the work, so a repeated call costs nothing. */
    public static void configure() {}

    private static void setUnlessGiven(String property, String environment, String value) {
        if (System.getProperty(property) == null && System.getenv(environment) == null) {
            System.setProperty(property, value);
        }
    }
}
