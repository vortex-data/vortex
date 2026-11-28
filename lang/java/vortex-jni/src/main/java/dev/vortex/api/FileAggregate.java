// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

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
        if ((kind == Kind.COUNT_STAR) != (column == null)) {
            throw new IllegalArgumentException(kind + (column == null ? " needs a column" : " takes no column"));
        }
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
     * An aggregate function. The tags must match {@code FileAggregate::from_tag} in
     * {@code vortex-jni-native/src/aggregate/mod.rs}.
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
         * The Vortex SumV2 result for an integer column, widened to 64 bits with the input signedness.
         * The result is null on overflow or when every input is null.
         */
        SUM((byte) 4),
        /**
         * Non-strict ascending order for this column in this file. The footer must record sortedness and establish
         * that the column has no nulls or NaNs. Missing evidence leaves the request unanswered.
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
