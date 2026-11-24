// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

/**
 * Behaviour of the Spark line this module compiles against. Each module's `src` holds its own copy, so shared code
 * picks the behaviour at compile time instead of probing the Spark version at runtime.
 */
final class SparkLine {
    /** Spark 4's Parquet source widens decimal scales. */
    static final boolean PARQUET_WIDENS_DECIMAL_SCALE = false;

    /** Spark 4's Parquet source reads timestamps with a time zone as timestamps without one. */
    static final boolean PARQUET_READS_TIMESTAMP_AS_NTZ = false;

    private SparkLine() {}
}
