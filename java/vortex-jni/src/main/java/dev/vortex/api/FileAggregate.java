// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import java.util.Objects;

/**
 * One ungrouped aggregate over a Vortex file, answered by {@link dev.vortex.jni.NativeFiles#aggregate}.
 *
 * @param kind the aggregate function
 * @param column the top-level column the aggregate reads, or null for {@link Kind#COUNT_STAR}
 */
public record FileAggregate(Kind kind, String column) {
    public FileAggregate {
        Objects.requireNonNull(kind, "kind");
        Preconditions.checkArgument(
                (kind == Kind.COUNT_STAR) == (column == null),
                "%s %s a column",
                kind,
                column == null ? "needs" : "takes no");
    }

    /** {@code COUNT(*)}. */
    public static FileAggregate countStar() {
        return new FileAggregate(Kind.COUNT_STAR, null);
    }

    /** {@code COUNT(column)}. */
    public static FileAggregate count(String column) {
        return new FileAggregate(Kind.COUNT, column);
    }

    /** {@code MIN(column)}. */
    public static FileAggregate min(String column) {
        return new FileAggregate(Kind.MIN, column);
    }

    /** {@code MAX(column)}. */
    public static FileAggregate max(String column) {
        return new FileAggregate(Kind.MAX, column);
    }

    /** Whether {@code column} is sorted, as the footer states. */
    public static FileAggregate isSorted(String column) {
        return new FileAggregate(Kind.IS_SORTED, column);
    }

    /** {@code SUM(column)} over an integer column. */
    public static FileAggregate sum(String column) {
        return new FileAggregate(Kind.SUM, column);
    }

    /**
     * An aggregate function. The tags must match {@code AggregateKind::from_tag} in
     * {@code vortex-jni/src/aggregate.rs}.
     */
    public enum Kind {
        /** The number of rows, as a 64-bit integer. */
        COUNT_STAR((byte) 0),
        /** The number of non-null values, as a 64-bit integer. */
        COUNT((byte) 1),
        /**
         * The smallest non-null value, in the column's type, or null when there is none. NaN orders above every number,
         * so only a column of nothing but NaN has a NaN minimum.
         */
        MIN((byte) 2),
        /**
         * The largest non-null value, in the column's type, or null when there is none. A column holding a NaN has a
         * NaN maximum.
         */
        MAX((byte) 3),
        /**
         * The sum of the non-null values of an integer column, as a 64-bit integer that wraps on overflow, or null when
         * there are none. Wrapping is exact modulo 2^64, so the sums of several files add up to the sum of all of them.
         */
        SUM((byte) 4),
        /**
         * Whether the column is sorted ascending across the file and holds no nulls, as a boolean the footer states.
         * Never computed from data, so it is unanswered for a file written without that statistic.
         */
        IS_SORTED((byte) 5);

        private final byte tag;

        Kind(byte tag) {
            this.tag = tag;
        }

        public byte tag() {
            return tag;
        }
    }
}
