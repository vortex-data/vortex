// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.io.Serializable;

/**
 * A pushed-down {@code ORDER BY column ... LIMIT limit}, which readers use to skip rows that cannot reach the result.
 *
 * <p>Spark still sorts and limits the rows the scan returns, so readers only have to return a superset of the result.
 *
 * @param column the top-level data column the query sorts by first
 * @param descending whether the column sorts in descending order
 * @param nullsFirst whether nulls sort before every value
 * @param strict whether a row that ties the current threshold can be dropped, which holds when {@code column} is the
 *     only sort key; with more keys a tie may still win on a later key
 * @param limit the number of rows the query keeps
 * @param seed a Catalyst value the result's last key provably reaches, from footer statistics, or null. Rows equal to
 *     it may still belong to the result.
 */
public record VortexTopN(String column, boolean descending, boolean nullsFirst, boolean strict, int limit, Object seed)
        implements Serializable {
    private static final long serialVersionUID = 1L;

    /** This top-N starting from {@code seed}. */
    public VortexTopN withSeed(Object seed) {
        return new VortexTopN(column, descending, nullsFirst, strict, limit, seed);
    }
}
