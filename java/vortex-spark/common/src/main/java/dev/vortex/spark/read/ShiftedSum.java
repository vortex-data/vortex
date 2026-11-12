// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.util.Optional;
import org.apache.spark.sql.connector.expressions.Cast;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.GeneralScalarExpression;
import org.apache.spark.sql.connector.expressions.Literal;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;

/**
 * The argument {@code column + offset} of a {@code SUM}, answered as {@code SUM(column) + offset * COUNT(column)}.
 *
 * <p>Spark states the argument in the type of the addition: the column, widened by a cast when it is narrower, plus a
 * literal of that type. ClickBench's {@code SUM(ResolutionWidth + 1)} over a {@code SMALLINT} column arrives as
 * {@code CAST(ResolutionWidth AS INT) + 1}.
 */
final class ShiftedSum {
    private final String column;
    private final long offset;
    private final DataType additionType;
    private final boolean widened;

    private ShiftedSum(String column, long offset, DataType additionType, boolean widened) {
        this.column = column;
        this.offset = offset;
        this.additionType = additionType;
        this.widened = widened;
    }

    /** The top-level column the sum reads. */
    String column() {
        return column;
    }

    /** The constant added to each value. */
    long offset() {
        return offset;
    }

    /** {@code expression} as a column plus a constant, in either order, or empty when it has another form. */
    static Optional<ShiftedSum> parse(Expression expression) {
        if (!(expression instanceof GeneralScalarExpression addition)
                || !addition.name().equals("+")
                || addition.children().length != 2) {
            return Optional.empty();
        }

        Expression[] operands = addition.children();
        Optional<ShiftedSum> shifted = parse(operands[0], operands[1]);
        return shifted.isPresent() ? shifted : parse(operands[1], operands[0]);
    }

    private static Optional<ShiftedSum> parse(Expression operand, Expression constant) {
        if (!(constant instanceof Literal<?> literal) || !isInteger(literal.value())) {
            return Optional.empty();
        }

        Expression column = operand;
        boolean widened = false;
        if (operand instanceof Cast cast) {
            if (!cast.dataType().equals(literal.dataType())) {
                return Optional.empty();
            }
            column = cast.expression();
            widened = true;
        }

        if (!(column instanceof NamedReference reference) || reference.fieldNames().length != 1) {
            return Optional.empty();
        }
        return Optional.of(new ShiftedSum(
                reference.fieldNames()[0], ((Number) literal.value()).longValue(), literal.dataType(), widened));
    }

    /**
     * Whether {@code SUM(column) + offset * COUNT(column)} gives what Spark gives for a column of {@code columnType}.
     *
     * <p>That is so when no value overflows the addition. Then each value of the argument is the column value plus the
     * offset, and the sums agree. Spark fails on such an overflow under ANSI rules and wraps it otherwise, and a sum of
     * the column cannot tell. So the column must be a {@code TINYINT}, {@code SMALLINT}, or {@code INT} narrower than
     * the addition, and the offset must leave room for each of its values.
     */
    boolean isExact(DataType columnType) {
        boolean narrow = columnType instanceof ByteType
                || columnType instanceof ShortType
                || (columnType instanceof IntegerType && additionType instanceof LongType);
        boolean wideAddition = additionType instanceof IntegerType || additionType instanceof LongType;
        if (!widened || !narrow || !wideAddition) {
            return false;
        }

        long bound = columnBound(columnType);
        long largest = additionType instanceof LongType ? Long.MAX_VALUE : Integer.MAX_VALUE;
        long smallest = additionType instanceof LongType ? Long.MIN_VALUE : Integer.MIN_VALUE;
        return offset <= largest - bound && offset >= smallest + bound;
    }

    /** The magnitude that no value of an integer column of {@code columnType} exceeds. */
    private static long columnBound(DataType columnType) {
        if (columnType instanceof ByteType) {
            return 1L << 7;
        }
        if (columnType instanceof ShortType) {
            return 1L << 15;
        }
        return 1L << 31;
    }

    private static boolean isInteger(Object value) {
        return value instanceof Byte || value instanceof Short || value instanceof Integer || value instanceof Long;
    }
}
