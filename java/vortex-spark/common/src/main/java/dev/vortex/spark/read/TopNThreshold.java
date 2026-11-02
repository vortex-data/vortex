// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.DynamicValue;
import dev.vortex.api.Expression;
import dev.vortex.api.Expression.BinaryOp;
import java.util.Comparator;
import java.util.Objects;
import java.util.Optional;
import java.util.PriorityQueue;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.sql.vectorized.ColumnVector;

/**
 * The best {@code limit} sort keys a task has read so far for a {@link VortexTopN}, published as a native dynamic
 * value.
 *
 * <p>Once the task has read {@code limit} keys, a row whose key sorts after the worst of them cannot reach the result.
 * Every reader of the task filters on {@code column < threshold} (or {@code >} when descending) against the shared
 * value, which Vortex also checks against zone statistics, so each tightening skips more of what is left to read. The
 * readers of one task share one threshold, so a file read later starts where the files before it left off.
 *
 * <p>Not thread-safe: a task reads its files one after another.
 */
final class TopNThreshold {
    private final VortexTopN topN;
    private final DataType dataType;
    private final DynamicValue value;

    /** The footer-proven bound, inclusive because rows equal to it may still reach the result, or null. */
    private final DynamicValue seed;

    /** The best keys read so far, worst first. */
    private final PriorityQueue<Comparable<Object>> best;

    private Comparable<Object> published;

    private TopNThreshold(VortexTopN topN, DataType dataType, Expression prototype) {
        this.topN = topN;
        this.dataType = dataType;
        this.value = DynamicValue.create(prototype);
        this.seed = topN.seed() == null
                ? null
                : SparkFilterToVortexExpression.literal(topN.seed(), dataType)
                        .map(literal -> {
                            DynamicValue bound = DynamicValue.create(prototype);
                            bound.set(literal);
                            return bound;
                        })
                        .orElse(null);
        Comparator<Comparable<Object>> natural = Comparable::compareTo;
        // The head is the worst key kept: the largest when ascending, the smallest when descending.
        this.best = new PriorityQueue<>(topN.limit() + 1, topN.descending() ? natural : natural.reversed());
    }

    /** Whether the reader can track keys of {@code dataType}, whose Catalyst values order as Spark orders them. */
    static boolean supports(DataType dataType) {
        return dataType instanceof ByteType
                || dataType instanceof ShortType
                || dataType instanceof IntegerType
                || dataType instanceof LongType
                || dataType instanceof DateType
                || dataType instanceof TimestampType
                || dataType instanceof TimestampNTZType
                || dataType instanceof DecimalType
                || DataTypes.StringType.equals(dataType);
    }

    static Optional<TopNThreshold> create(VortexTopN topN, DataType dataType) {
        if (topN.limit() <= 0 || !supports(dataType)) {
            return Optional.empty();
        }
        return SparkFilterToVortexExpression.literal(null, dataType)
                .map(prototype -> new TopNThreshold(topN, dataType, prototype));
    }

    VortexTopN topN() {
        return topN;
    }

    /** The filter keeping every row that can still reach the result, over the file's own {@code column}. */
    Expression filter(Expression column) {
        BinaryOp op = topN.descending()
                ? (topN.strict() ? BinaryOp.GT : BinaryOp.GTE)
                : (topN.strict() ? BinaryOp.LT : BinaryOp.LTE);
        // Unset, the comparison keeps every row.
        Expression keeps = Expression.dynamic(op, column, value, true);
        if (seed != null) {
            BinaryOp inclusive = topN.descending() ? BinaryOp.GTE : BinaryOp.LTE;
            keeps = Expression.and(Expression.dynamic(inclusive, column, seed, true), keeps);
        }
        // Nulls that sort first always reach the result before any value does.
        return topN.nullsFirst() ? Expression.or(Expression.isNull(column), keeps) : keeps;
    }

    /** Takes in the keys of rows a reader returned, and tightens the threshold if they improve on it. */
    void offer(ColumnVector keys, int rowCount) {
        int limit = topN.limit();
        for (int row = 0; row < rowCount; row++) {
            if (keys.isNullAt(row)) {
                continue;
            }
            Comparable<Object> key = key(keys, row);
            if (best.size() < limit) {
                best.add(key);
            } else if (better(key, best.peek())) {
                best.poll();
                best.add(key);
            }
        }

        if (best.size() == limit && !Objects.equals(best.peek(), published)) {
            published = best.peek();
            SparkFilterToVortexExpression.literal(published, dataType).ifPresent(value::set);
        }
    }

    private boolean better(Comparable<Object> key, Comparable<Object> worst) {
        int order = key.compareTo(worst);
        return topN.descending() ? order > 0 : order < 0;
    }

    @SuppressWarnings("unchecked")
    private Comparable<Object> key(ColumnVector keys, int row) {
        Object key;
        if (dataType instanceof ByteType) {
            key = keys.getByte(row);
        } else if (dataType instanceof ShortType) {
            key = keys.getShort(row);
        } else if (dataType instanceof IntegerType || dataType instanceof DateType) {
            key = keys.getInt(row);
        } else if (dataType instanceof LongType
                || dataType instanceof TimestampType
                || dataType instanceof TimestampNTZType) {
            key = keys.getLong(row);
        } else if (dataType instanceof DecimalType decimal) {
            key = keys.getDecimal(row, decimal.precision(), decimal.scale());
        } else {
            // The vector reuses its memory for the next batch, so the kept key needs its own copy.
            key = keys.getUTF8String(row).copy();
        }
        return (Comparable<Object>) key;
    }
}
