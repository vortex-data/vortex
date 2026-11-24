// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

/**
 * Test expectations that depend on the Spark line this module compiles against. Each module's `src` holds its own
 * copy, so shared tests branch at compile time instead of probing the Spark version at runtime.
 */
final class SparkTestLine {
    /** Spark 4 adds string collations such as `UTF8_LCASE`. */
    static final boolean COLLATIONS = false;

    private SparkTestLine() {}
}
