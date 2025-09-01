// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.util.List;
import java.util.Optional;
import org.immutables.value.Value;

/**
 * Create a new set of options for configuring the scan.
 */
@Value.Immutable
public interface ScanOptions {
    /**
     * Columns to project out.
     */
    List<String> columns();

    /**
     * Optional pruning expression that is pushed down to the scan.
     */
    Optional<Expression> predicate();

    /**
     * Optional start (inclusive) and end (exclusive) row indices to select a range of rows
     * in the scan.
     */
    Optional<long[]> rowRange();

    /**
     * Optional row indices to select specific rows.
     * These must be sorted in ascending order.
     */
    Optional<long[]> rowIndices();

    /**
     * Creates a new ScanOptions instance with default values.
     *
     * @return a ScanOptions instance with empty columns list, no predicate, no row range, and no row indices
     */
    static ScanOptions of() {
        return ImmutableScanOptions.builder().build();
    }

    /**
     * Creates a new builder for constructing ScanOptions instances.
     *
     * @return a new builder instance that can be used to configure and build ScanOptions
     */
    static ImmutableScanOptions.Builder builder() {
        return ImmutableScanOptions.builder();
    }
}
