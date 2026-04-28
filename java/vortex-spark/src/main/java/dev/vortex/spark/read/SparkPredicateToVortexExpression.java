// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import dev.vortex.api.Expression.BinaryOp;
import java.util.Optional;
import java.util.Set;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.And;
import org.apache.spark.sql.connector.expressions.filter.Not;
import org.apache.spark.sql.connector.expressions.filter.Or;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.unsafe.types.UTF8String;

/** Translates {@link Predicate Spark V2 predicates} into Vortex {@link Expression}s for predicate pushdown. */
final class SparkPredicateToVortexExpression {

    private SparkPredicateToVortexExpression() {}

    /**
     * Returns true if the given Spark predicate is structurally convertible to a Vortex expression and references only
     * the supplied {@code dataColumns}.
     *
     * <p>This is the cheap check used in {@code SupportsPushDownV2Filters.pushPredicates} to decide which predicates
     * Spark can drop. It does not allocate any native expressions.
     */
    static boolean isPushable(Predicate predicate, Set<String> dataColumns) {
        for (NamedReference ref : predicate.references()) {
            String[] parts = ref.fieldNames();
            if (parts.length != 1) {
                return false;
            }
            if (!dataColumns.contains(parts[0])) {
                return false;
            }
        }
        return isStructurallyPushable(predicate);
    }

    private static boolean isStructurallyPushable(Predicate predicate) {
        if (predicate instanceof And a) {
            return isStructurallyPushable(a.left()) && isStructurallyPushable(a.right());
        }
        if (predicate instanceof Or o) {
            return isStructurallyPushable(o.left()) && isStructurallyPushable(o.right());
        }
        if (predicate instanceof Not) {
            return isStructurallyPushable(((Not) predicate).child());
        }
        org.apache.spark.sql.connector.expressions.Expression[] children = predicate.children();
        switch (predicate.name()) {
            case "=":
            case ">":
            case ">=":
            case "<":
            case "<=":
                return children.length == 2 && isTopLevelFieldRef(children[0]) && isPushableLiteral(children[1]);
            case "IS_NULL":
            case "IS_NOT_NULL":
                return children.length == 1 && isTopLevelFieldRef(children[0]);
            case "IN":
                if (children.length < 2 || !isTopLevelFieldRef(children[0])) {
                    return false;
                }
                for (int i = 1; i < children.length; i++) {
                    if (!isPushableLiteral(children[i])) {
                        return false;
                    }
                }
                return true;
            default:
                return false;
        }
    }

    /**
     * Converts a Spark predicate to a Vortex expression. Returns {@link Optional#empty()} if the predicate is not
     * pushable; callers should normally pre-check with {@link #isPushable}.
     */
    static Optional<Expression> convert(Predicate predicate) {
        if (predicate instanceof And a) {
            Optional<Expression> left = convert(a.left());
            Optional<Expression> right = convert(a.right());
            if (left.isPresent() && right.isPresent()) {
                return Optional.of(Expression.and(left.get(), right.get()));
            }
            return Optional.empty();
        }
        if (predicate instanceof Or o) {
            Optional<Expression> left = convert(o.left());
            Optional<Expression> right = convert(o.right());
            if (left.isPresent() && right.isPresent()) {
                return Optional.of(Expression.or(left.get(), right.get()));
            }
            return Optional.empty();
        }
        if (predicate instanceof Not) {
            return convert(((Not) predicate).child()).map(Expression::not);
        }
        org.apache.spark.sql.connector.expressions.Expression[] children = predicate.children();
        return switch (predicate.name()) {
            case "=", ">", ">=", "<", "<=" -> convertBinary(predicate.name(), children);
            case "IS_NULL" -> {
                if (children.length != 1) {
                    yield Optional.empty();
                }
                yield columnNameOf(children[0]).map(name -> Expression.isNull(Expression.column(name)));
            }
            case "IS_NOT_NULL" -> {
                if (children.length != 1) {
                    yield Optional.empty();
                }
                yield columnNameOf(children[0]).map(name -> Expression.isNotNull(Expression.column(name)));
            }
            case "IN" -> convertIn(children);
            default -> Optional.empty();
        };
    }

    private static Optional<Expression> convertBinary(
            String op, org.apache.spark.sql.connector.expressions.Expression[] children) {
        if (children.length != 2) {
            return Optional.empty();
        }
        Optional<String> column = columnNameOf(children[0]);
        Optional<Expression> literal = literalOf(children[1]);
        if (column.isEmpty() || literal.isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(Expression.binary(toBinaryOp(op), Expression.column(column.get()), literal.get()));
    }

    private static Optional<Expression> convertIn(org.apache.spark.sql.connector.expressions.Expression[] children) {
        if (children.length < 2) {
            return Optional.empty();
        }
        Optional<String> column = columnNameOf(children[0]);
        if (column.isEmpty()) {
            return Optional.empty();
        }
        Expression columnExpr = Expression.column(column.get());
        Expression[] eqs = new Expression[children.length - 1];
        for (int i = 1; i < children.length; i++) {
            Optional<Expression> literal = literalOf(children[i]);
            if (literal.isEmpty()) {
                return Optional.empty();
            }
            eqs[i - 1] = Expression.binary(BinaryOp.EQ, columnExpr, literal.get());
        }
        if (eqs.length == 1) {
            return Optional.of(eqs[0]);
        }
        return Optional.of(Expression.or(eqs));
    }

    private static BinaryOp toBinaryOp(String name) {
        return switch (name) {
            case "=" -> BinaryOp.EQ;
            case ">" -> BinaryOp.GT;
            case ">=" -> BinaryOp.GTE;
            case "<" -> BinaryOp.LT;
            case "<=" -> BinaryOp.LTE;
            default -> throw new IllegalArgumentException("not a pushable binary operator: " + name);
        };
    }

    private static boolean isTopLevelFieldRef(org.apache.spark.sql.connector.expressions.Expression expr) {
        return expr instanceof NamedReference && ((NamedReference) expr).fieldNames().length == 1;
    }

    private static Optional<String> columnNameOf(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof NamedReference)) {
            return Optional.empty();
        }
        String[] parts = ((NamedReference) expr).fieldNames();
        if (parts.length != 1) {
            return Optional.empty();
        }
        return Optional.of(parts[0]);
    }

    private static boolean isPushableLiteral(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof LiteralValue)) {
            return false;
        }
        Object v = ((LiteralValue<?>) expr).value();
        return v instanceof Boolean
                || v instanceof Byte
                || v instanceof Short
                || v instanceof Integer
                || v instanceof Long
                || v instanceof Float
                || v instanceof Double
                || v instanceof UTF8String
                || v instanceof CharSequence;
    }

    private static Optional<Expression> literalOf(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof LiteralValue)) {
            return Optional.empty();
        }
        Object value = ((LiteralValue<?>) expr).value();
        if (value == null) {
            return Optional.empty();
        }
        if (value instanceof Boolean) {
            return Optional.of(Expression.literal((Boolean) value));
        }
        if (value instanceof Byte) {
            return Optional.of(Expression.literal((Byte) value));
        }
        if (value instanceof Short) {
            return Optional.of(Expression.literal((Short) value));
        }
        if (value instanceof Integer) {
            return Optional.of(Expression.literal((Integer) value));
        }
        if (value instanceof Long) {
            return Optional.of(Expression.literal((Long) value));
        }
        if (value instanceof Float) {
            return Optional.of(Expression.literal((Float) value));
        }
        if (value instanceof Double) {
            return Optional.of(Expression.literal((Double) value));
        }
        if (value instanceof UTF8String || value instanceof CharSequence) {
            return Optional.of(Expression.literal(value.toString()));
        }
        return Optional.empty();
    }
}
