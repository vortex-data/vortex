// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import dev.vortex.api.Expression.BinaryOp;
import java.util.Optional;
import org.apache.spark.sql.catalyst.expressions.And;
import org.apache.spark.sql.catalyst.expressions.AttributeReference;
import org.apache.spark.sql.catalyst.expressions.BinaryComparison;
import org.apache.spark.sql.catalyst.expressions.EqualTo;
import org.apache.spark.sql.catalyst.expressions.GreaterThan;
import org.apache.spark.sql.catalyst.expressions.GreaterThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.LessThan;
import org.apache.spark.sql.catalyst.expressions.LessThanOrEqual;
import org.apache.spark.sql.catalyst.expressions.Like;
import org.apache.spark.sql.catalyst.expressions.Literal;
import org.apache.spark.sql.catalyst.expressions.Not;
import org.apache.spark.sql.catalyst.expressions.Or;
import org.apache.spark.sql.execution.datasources.DataSourceStrategy$;
import org.apache.spark.sql.internal.connector.PredicateUtils$;
import org.apache.spark.sql.sources.Filter;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DecimalType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;

/**
 * Converts the Catalyst predicates that have no V1 filter form into Vortex expressions: {@code LIKE} with any pattern,
 * and comparisons between two columns, alone or under {@code AND}, {@code OR}, and {@code NOT} together with parts that
 * do have a V1 form.
 *
 * <p>Spark evaluates every data filter again above the scan, so a converted predicate only narrows what is read.
 */
public final class CatalystPredicates {
    private CatalystPredicates() {}

    /**
     * Whether Spark translates {@code predicate} to a V1 filter, which the scan pushes as such. Spark keeps that
     * translation to its own packages in Scala, so it is reached from here.
     */
    public static boolean hasV1Form(org.apache.spark.sql.catalyst.expressions.Expression predicate) {
        return DataSourceStrategy$.MODULE$.translateFilter(predicate, true).isDefined();
    }

    /**
     * The V1 filter form of the V2 {@code predicate}, with its literals converted to the Java values V1 filters hold,
     * or empty when it has none. Spark keeps that conversion to its own packages in Scala, so it is reached from here.
     */
    public static Optional<Filter> toV1(org.apache.spark.sql.connector.expressions.filter.Predicate predicate) {
        scala.Option<Filter> filter = PredicateUtils$.MODULE$.toV1(predicate);
        return filter.isDefined() ? Optional.of(filter.get()) : Optional.empty();
    }

    /** Whether {@code predicate} converts in full against {@code dataSchema}. */
    public static boolean isPushable(
            org.apache.spark.sql.catalyst.expressions.Expression predicate, StructType dataSchema) {
        return convert(predicate, dataSchema).isPresent();
    }

    /** The Vortex form of {@code predicate}, or empty when any part of it does not convert. */
    public static Optional<Expression> convert(
            org.apache.spark.sql.catalyst.expressions.Expression predicate, StructType dataSchema) {
        if (predicate instanceof And and) {
            return both(and.left(), and.right(), dataSchema, true);
        }
        if (predicate instanceof Or or) {
            return both(or.left(), or.right(), dataSchema, false);
        }
        if (predicate instanceof Not not) {
            return convert(not.child(), dataSchema).map(Expression::not);
        }
        if (predicate instanceof Like like) {
            return like(like, dataSchema);
        }
        if (predicate instanceof BinaryComparison comparison) {
            Optional<Expression> columns = columnComparison(comparison, dataSchema);
            if (columns.isPresent()) {
                return columns;
            }
        }
        // A part with a V1 form, such as a comparison with a literal, converts as the pushed V1 filters do.
        scala.Option<Filter> filter = DataSourceStrategy$.MODULE$.translateFilter(predicate, true);
        return filter.isDefined() ? SparkFilterToVortexExpression.convert(filter.get(), dataSchema) : Optional.empty();
    }

    private static Optional<Expression> both(
            org.apache.spark.sql.catalyst.expressions.Expression left,
            org.apache.spark.sql.catalyst.expressions.Expression right,
            StructType dataSchema,
            boolean conjunction) {
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

    /** {@code column LIKE 'pattern'} with Spark's default escape, which Vortex shares. */
    private static Optional<Expression> like(Like like, StructType dataSchema) {
        if (like.escapeChar() != '\\'
                || !(like.left() instanceof AttributeReference attribute)
                || !(like.right() instanceof Literal pattern)
                || !(pattern.value() instanceof UTF8String patternValue)) {
            return Optional.empty();
        }
        return field(attribute, dataSchema)
                .filter(field -> DataTypes.StringType.equals(field.dataType()))
                .map(field -> Expression.like(
                        Expression.column(field.name()), Expression.literal(patternValue.toString()), false, false));
    }

    /** {@code a <op> b} over two columns of one type that Vortex orders as Spark does. */
    private static Optional<Expression> columnComparison(BinaryComparison comparison, StructType dataSchema) {
        BinaryOp op;
        if (comparison instanceof EqualTo) {
            op = BinaryOp.EQ;
        } else if (comparison instanceof GreaterThan) {
            op = BinaryOp.GT;
        } else if (comparison instanceof GreaterThanOrEqual) {
            op = BinaryOp.GTE;
        } else if (comparison instanceof LessThan) {
            op = BinaryOp.LT;
        } else if (comparison instanceof LessThanOrEqual) {
            op = BinaryOp.LTE;
        } else {
            return Optional.empty();
        }
        if (!(comparison.left() instanceof AttributeReference left)
                || !(comparison.right() instanceof AttributeReference right)) {
            return Optional.empty();
        }

        Optional<StructField> leftField = field(left, dataSchema);
        Optional<StructField> rightField = field(right, dataSchema);
        if (leftField.isEmpty()
                || rightField.isEmpty()
                || !leftField.get().dataType().equals(rightField.get().dataType())
                || !comparable(leftField.get().dataType())) {
            return Optional.empty();
        }
        return Optional.of(Expression.binary(
                op,
                Expression.column(leftField.get().name()),
                Expression.column(rightField.get().name())));
    }

    /**
     * Types whose values Vortex compares as Spark does. Floats differ on {@code -0.0} and {@code NaN}, and collated
     * strings by their collation.
     */
    private static boolean comparable(DataType type) {
        return DataTypes.BooleanType.equals(type)
                || DataTypes.ByteType.equals(type)
                || DataTypes.ShortType.equals(type)
                || DataTypes.IntegerType.equals(type)
                || DataTypes.LongType.equals(type)
                || DataTypes.DateType.equals(type)
                || DataTypes.TimestampType.equals(type)
                || DataTypes.TimestampNTZType.equals(type)
                || DataTypes.StringType.equals(type)
                || DataTypes.BinaryType.equals(type)
                || type instanceof DecimalType;
    }

    private static Optional<StructField> field(AttributeReference attribute, StructType dataSchema) {
        for (StructField field : dataSchema.fields()) {
            if (field.name().equals(attribute.name())) {
                return Optional.of(field);
            }
        }
        return Optional.empty();
    }
}
