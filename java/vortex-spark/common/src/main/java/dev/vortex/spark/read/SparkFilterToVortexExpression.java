// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import dev.vortex.api.Expression.BinaryOp;
import dev.vortex.api.Expression.TimeUnit;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.sql.Date;
import java.sql.Timestamp;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.apache.spark.sql.sources.AlwaysFalse;
import org.apache.spark.sql.sources.AlwaysTrue;
import org.apache.spark.sql.sources.And;
import org.apache.spark.sql.sources.EqualNullSafe;
import org.apache.spark.sql.sources.EqualTo;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.sources.GreaterThan;
import org.apache.spark.sql.sources.GreaterThanOrEqual;
import org.apache.spark.sql.sources.In;
import org.apache.spark.sql.sources.IsNotNull;
import org.apache.spark.sql.sources.IsNull;
import org.apache.spark.sql.sources.LessThan;
import org.apache.spark.sql.sources.LessThanOrEqual;
import org.apache.spark.sql.sources.Not;
import org.apache.spark.sql.sources.Or;
import org.apache.spark.sql.sources.StringContains;
import org.apache.spark.sql.sources.StringEndsWith;
import org.apache.spark.sql.sources.StringStartsWith;
import org.apache.spark.sql.types.BinaryType;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.unsafe.types.UTF8String;

/** Converts Spark's stable V1 file filters into Vortex expressions. */
public final class SparkFilterToVortexExpression {
    private SparkFilterToVortexExpression() {}

    /** Returns whether the complete filter can be evaluated by Vortex for {@code dataSchema}. */
    public static boolean isPushable(Filter filter, StructType dataSchema) {
        return convert(filter, dataSchema).isPresent();
    }

    /** Converts a filter, returning empty when its operator, column, or literal is unsupported. */
    public static Optional<Expression> convert(Filter filter, StructType dataSchema) {
        if (filter instanceof AlwaysTrue) {
            return Optional.of(Expression.literal(true));
        }
        if (filter instanceof AlwaysFalse) {
            return Optional.of(Expression.literal(false));
        }
        if (filter instanceof And and) {
            return combine(and.left(), and.right(), dataSchema, true);
        }
        if (filter instanceof Or or) {
            return combine(or.left(), or.right(), dataSchema, false);
        }
        if (filter instanceof Not not) {
            return convert(not.child(), dataSchema).map(Expression::not);
        }
        if (filter instanceof IsNull isNull) {
            return column(isNull, isNull.attribute(), dataSchema).map(Expression::isNull);
        }
        if (filter instanceof IsNotNull isNotNull) {
            return column(isNotNull, isNotNull.attribute(), dataSchema).map(Expression::isNotNull);
        }
        if (filter instanceof EqualTo equal) {
            return comparison(equal, equal.attribute(), equal.value(), dataSchema, BinaryOp.EQ, false);
        }
        if (filter instanceof EqualNullSafe equal) {
            if (equal.value() == null) {
                return column(equal, equal.attribute(), dataSchema).map(Expression::isNull);
            }
            return comparison(equal, equal.attribute(), equal.value(), dataSchema, BinaryOp.EQ, true);
        }
        if (filter instanceof GreaterThan greater) {
            return comparison(greater, greater.attribute(), greater.value(), dataSchema, BinaryOp.GT, false);
        }
        if (filter instanceof GreaterThanOrEqual greater) {
            return comparison(greater, greater.attribute(), greater.value(), dataSchema, BinaryOp.GTE, false);
        }
        if (filter instanceof LessThan less) {
            return comparison(less, less.attribute(), less.value(), dataSchema, BinaryOp.LT, false);
        }
        if (filter instanceof LessThanOrEqual less) {
            return comparison(less, less.attribute(), less.value(), dataSchema, BinaryOp.LTE, false);
        }
        if (filter instanceof In in) {
            return convertIn(in, dataSchema);
        }
        if (filter instanceof StringStartsWith startsWith) {
            return stringMatch(startsWith, startsWith.attribute(), startsWith.value(), dataSchema, false, true);
        }
        if (filter instanceof StringEndsWith endsWith) {
            return stringMatch(endsWith, endsWith.attribute(), endsWith.value(), dataSchema, true, false);
        }
        if (filter instanceof StringContains contains) {
            return stringMatch(contains, contains.attribute(), contains.value(), dataSchema, true, true);
        }
        return Optional.empty();
    }

    private static Optional<Expression> combine(Filter left, Filter right, StructType dataSchema, boolean conjunction) {
        Optional<Expression> convertedLeft = convert(left, dataSchema);
        Optional<Expression> convertedRight = convert(right, dataSchema);
        if (convertedLeft.isEmpty() || convertedRight.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(
                conjunction
                        ? Expression.and(convertedLeft.get(), convertedRight.get())
                        : Expression.or(convertedLeft.get(), convertedRight.get()));
    }

    private static Optional<Expression> comparison(
            Filter filter, String attribute, Object value, StructType dataSchema, BinaryOp op, boolean nullSafe) {
        Optional<ResolvedColumn> column = resolveColumn(filter, attribute, dataSchema);
        if (column.isEmpty() || (value == null && !nullSafe)) {
            return Optional.empty();
        }
        Optional<Expression> literal = literal(value, column.get().dataType());
        return literal.map(expression -> Expression.binary(op, column.get().expression(), expression));
    }

    private static Optional<Expression> convertIn(In in, StructType dataSchema) {
        Optional<ResolvedColumn> column = resolveColumn(in, in.attribute(), dataSchema);
        if (column.isEmpty()) {
            return Optional.empty();
        }
        List<Expression> comparisons = new ArrayList<>();
        for (Object value : in.values()) {
            if (value == null) {
                continue;
            }
            Optional<Expression> literal = literal(value, column.get().dataType());
            if (literal.isEmpty()) {
                return Optional.empty();
            }
            comparisons.add(Expression.binary(BinaryOp.EQ, column.get().expression(), literal.get()));
        }
        if (comparisons.isEmpty()) {
            return Optional.of(Expression.literal(false));
        }
        if (comparisons.size() == 1) {
            return Optional.of(comparisons.get(0));
        }
        return Optional.of(Expression.or(comparisons.toArray(new Expression[0])));
    }

    private static Optional<Expression> stringMatch(
            Filter filter,
            String attribute,
            String value,
            StructType dataSchema,
            boolean leadingWildcard,
            boolean trailingWildcard) {
        Optional<ResolvedColumn> column = resolveColumn(filter, attribute, dataSchema);
        if (column.isEmpty() || !(column.get().dataType() instanceof StringType)) {
            return Optional.empty();
        }
        return Optional.of(Expression.like(
                column.get().expression(),
                Expression.literal(likePattern(value, leadingWildcard, trailingWildcard)),
                false,
                false));
    }

    private static String likePattern(String value, boolean leadingWildcard, boolean trailingWildcard) {
        StringBuilder pattern = new StringBuilder(value.length() + 2);
        if (leadingWildcard) {
            pattern.append('%');
        }
        for (int i = 0; i < value.length(); i++) {
            char character = value.charAt(i);
            if (character == '%' || character == '_' || character == '\\') {
                pattern.append('\\');
            }
            pattern.append(character);
        }
        if (trailingWildcard) {
            pattern.append('%');
        }
        return pattern.toString();
    }

    private static Optional<Expression> column(Filter filter, String attribute, StructType schema) {
        return resolveColumn(filter, attribute, schema).map(ResolvedColumn::expression);
    }

    private static Optional<ResolvedColumn> resolveColumn(Filter filter, String attribute, StructType schema) {
        String[][] references = filter.v2references();
        if (references.length != 1 || references[0].length == 0) {
            return Optional.empty();
        }
        String[] parts = references[0];
        DataType current = schema;
        for (String part : parts) {
            if (!(current instanceof StructType struct)) {
                return Optional.empty();
            }
            StructField field = findField(struct, part);
            if (field == null) {
                return Optional.empty();
            }
            current = field.dataType();
        }
        return Optional.of(new ResolvedColumn(Expression.column(parts), current));
    }

    private static StructField findField(StructType schema, String name) {
        for (StructField field : schema.fields()) {
            if (field.name().equals(name)) {
                return field;
            }
        }
        return null;
    }

    private static Optional<Expression> literal(Object value, DataType dataType) {
        if (dataType instanceof BooleanType) {
            return value == null
                    ? Optional.of(Expression.nullLiteralBool())
                    : value instanceof Boolean booleanValue
                            ? Optional.of(Expression.literal(booleanValue))
                            : Optional.empty();
        }
        if (dataType instanceof ByteType) {
            return numericLiteral(value, Expression.DType.I8, number -> Expression.literal(number.byteValue()));
        }
        if (dataType instanceof ShortType) {
            return numericLiteral(value, Expression.DType.I16, number -> Expression.literal(number.shortValue()));
        }
        if (dataType instanceof IntegerType) {
            return numericLiteral(value, Expression.DType.I32, number -> Expression.literal(number.intValue()));
        }
        if (dataType instanceof LongType) {
            return numericLiteral(value, Expression.DType.I64, number -> Expression.literal(number.longValue()));
        }
        if (dataType instanceof FloatType) {
            return numericLiteral(value, Expression.DType.F32, number -> Expression.literal(number.floatValue()));
        }
        if (dataType instanceof DoubleType) {
            return numericLiteral(value, Expression.DType.F64, number -> Expression.literal(number.doubleValue()));
        }
        if (dataType instanceof StringType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.UTF8));
            }
            if (value instanceof CharSequence || value instanceof UTF8String) {
                return Optional.of(Expression.literal(value.toString()));
            }
            return Optional.empty();
        }
        if (dataType instanceof BinaryType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.BINARY));
            }
            return value instanceof byte[] bytes ? Optional.of(Expression.literal(bytes)) : Optional.empty();
        }
        if (dataType instanceof DateType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteralDate(TimeUnit.DAYS));
            }
            Optional<Long> days = dateDays(value);
            return days.map(dayCount -> Expression.literalDate(dayCount, TimeUnit.DAYS));
        }
        if (dataType instanceof TimestampType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteralTimestamp(TimeUnit.MICROSECONDS, "UTC"));
            }
            return timestampMicros(value, false)
                    .map(micros -> Expression.literalTimestamp(micros, TimeUnit.MICROSECONDS, "UTC"));
        }
        if (dataType instanceof TimestampNTZType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteralTimestamp(TimeUnit.MICROSECONDS, null));
            }
            return timestampMicros(value, true)
                    .map(micros -> Expression.literalTimestamp(micros, TimeUnit.MICROSECONDS, null));
        }
        if (dataType instanceof DecimalType decimalType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteralDecimal(decimalType.precision(), decimalType.scale()));
            }
            BigDecimal decimal;
            if (value instanceof BigDecimal bigDecimal) {
                decimal = bigDecimal;
            } else if (value instanceof Decimal sparkDecimal) {
                decimal = sparkDecimal.toJavaBigDecimal();
            } else {
                return Optional.empty();
            }
            try {
                BigInteger unscaled = decimal.setScale(decimalType.scale()).unscaledValue();
                return Optional.of(Expression.literalDecimal(unscaled, decimalType.precision(), decimalType.scale()));
            } catch (ArithmeticException ignored) {
                return Optional.empty();
            }
        }
        return Optional.empty();
    }

    private static Optional<Expression> numericLiteral(
            Object value, Expression.DType nullType, NumericExpression factory) {
        if (value == null) {
            return Optional.of(Expression.nullLiteral(nullType));
        }
        return value instanceof Number number ? Optional.of(factory.create(number)) : Optional.empty();
    }

    private static Optional<Long> dateDays(Object value) {
        if (value instanceof Date date) {
            return Optional.of(date.toLocalDate().toEpochDay());
        }
        if (value instanceof LocalDate date) {
            return Optional.of(date.toEpochDay());
        }
        if (value instanceof Number number) {
            return Optional.of(number.longValue());
        }
        return Optional.empty();
    }

    private static Optional<Long> timestampMicros(Object value, boolean withoutTimeZone) {
        if (value instanceof Number number) {
            return Optional.of(number.longValue());
        }
        if (withoutTimeZone && value instanceof LocalDateTime localDateTime) {
            return instantMicros(localDateTime.toInstant(ZoneOffset.UTC));
        }
        if (value instanceof Timestamp timestamp) {
            return instantMicros(timestamp.toInstant());
        }
        if (value instanceof Instant instant) {
            return instantMicros(instant);
        }
        if (value instanceof OffsetDateTime offsetDateTime) {
            return instantMicros(offsetDateTime.toInstant());
        }
        if (value instanceof ZonedDateTime zonedDateTime) {
            return instantMicros(zonedDateTime.toInstant());
        }
        return Optional.empty();
    }

    private static Optional<Long> instantMicros(Instant instant) {
        try {
            return Optional.of(Math.addExact(
                    Math.multiplyExact(instant.getEpochSecond(), 1_000_000L), instant.getNano() / 1_000L));
        } catch (ArithmeticException ignored) {
            return Optional.empty();
        }
    }

    private static final class ResolvedColumn {
        private final Expression expression;
        private final DataType dataType;

        private ResolvedColumn(Expression expression, DataType dataType) {
            this.expression = expression;
            this.dataType = dataType;
        }

        private Expression expression() {
            return expression;
        }

        private DataType dataType() {
            return dataType;
        }
    }

    @FunctionalInterface
    private interface NumericExpression {
        Expression create(Number number);
    }
}
