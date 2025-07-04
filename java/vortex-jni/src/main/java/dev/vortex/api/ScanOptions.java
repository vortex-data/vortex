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

    static ScanOptions of() {
        return ImmutableScanOptions.builder().build();
    }

    static ImmutableScanOptions.Builder builder() {
        return ImmutableScanOptions.builder();
    }
}
