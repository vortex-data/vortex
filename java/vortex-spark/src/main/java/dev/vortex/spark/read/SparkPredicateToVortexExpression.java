// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import dev.vortex.api.Expression.BinaryOp;
import dev.vortex.api.Expression.TimeUnit;
import org.apache.spark.sql.connector.expressions.Literal;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.AlwaysFalse;
import org.apache.spark.sql.connector.expressions.filter.AlwaysTrue;
import org.apache.spark.sql.connector.expressions.filter.And;
import org.apache.spark.sql.connector.expressions.filter.Not;
import org.apache.spark.sql.connector.expressions.filter.Or;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
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

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Translates {@link Predicate Spark V2 predicates} into Vortex {@link Expression}s for predicate pushdown.
 *
 * <p>The translator aims to express every Spark predicate Vortex can evaluate. Predicates that cannot be translated
 * (unsupported functions, literals on user-defined types, references to columns not present in the file, etc.) are left
 * to Spark for post-scan evaluation.
 */
final class SparkPredicateToVortexExpression {

    private SparkPredicateToVortexExpression() {
    }

    /**
     * Returns true if the given Spark predicate can be translated to a Vortex expression and every named reference
     * resolves to a real field path under {@code dataColumnTypes}.
     *
     * <p>{@code dataColumnTypes} maps each pushable top-level column name to its top-level Spark {@link DataType};
     * partition columns and columns the scan does not project should not appear in the map. For nested references
     * (for example {@code info.email}) the validator walks the named reference part by part, descending into
     * {@link StructType} fields so that {@code info} must be a struct that contains an {@code email} field.
     *
     * <p>This is the cheap check used in {@code SupportsPushDownV2Filters.pushPredicates} to decide which predicates
     * Spark can drop. It does not allocate any native expressions; if it returns true, {@link #convert(Predicate)} must
     * succeed (otherwise callers would silently drop predicates).
     */
    static boolean isPushable(Predicate predicate, Map<String, DataType> dataColumnTypes) {
        for (NamedReference ref : predicate.references()) {
            if (!resolveFieldPath(ref.fieldNames(), dataColumnTypes)) {
                return false;
            }
        }
        return isStructurallyPushable(predicate);
    }

    /**
     * Walks {@code parts} against {@code dataColumnTypes}, descending through {@link StructType} fields for
     * dot-separated nested references. Returns true only when every part resolves to an actual field in the
     * schema.
     */
    private static boolean resolveFieldPath(String[] parts, Map<String, DataType> dataColumnTypes) {
        if (parts.length == 0) {
            return false;
        }
        DataType current = dataColumnTypes.get(parts[0]);
        if (current == null) {
            return false;
        }
        for (int i = 1; i < parts.length; i++) {
            if (!(current instanceof StructType struct)) {
                return false;
            }
            Optional<StructField> field = findField(struct, parts[i]);
            if (field.isEmpty()) {
                return false;
            }
            current = field.get().dataType();
        }
        return true;
    }

    private static Optional<StructField> findField(StructType struct, String name) {
        return Arrays.stream(struct.fields()).filter(structField -> structField.name().equals(name)).findFirst();
    }

    private static boolean isStructurallyPushable(Predicate predicate) {
        if (predicate instanceof AlwaysTrue || predicate instanceof AlwaysFalse) {
            return true;
        }
        if (predicate instanceof And a) {
            return isStructurallyPushable(a.left()) && isStructurallyPushable(a.right());
        }
        if (predicate instanceof Or o) {
            return isStructurallyPushable(o.left()) && isStructurallyPushable(o.right());
        }
        if (predicate instanceof Not n) {
            return isStructurallyPushable(n.child());
        }

        org.apache.spark.sql.connector.expressions.Expression[] children = predicate.children();
        return switch (predicate.name()) {
            case "=", "<>", "!=", ">", ">=", "<", "<=" -> isPushableComparison(children);
            case "IS_NULL", "IS_NOT_NULL" -> children.length == 1 && isPushableFieldRef(children[0]);
            case "IN" -> {
                if (children.length < 2 || !isPushableFieldRef(children[0])) {
                    yield false;
                }
                for (int i = 1; i < children.length; i++) {
                    if (!isPushableLiteral(children[i])) {
                        yield false;
                    }
                }
                yield true;
            }
            case "STARTS_WITH", "ENDS_WITH", "CONTAINS" ->
                    children.length == 2 && isPushableFieldRef(children[0]) && isPushableStringLiteral(children[1]);
            // `BOOLEAN_EXPRESSION` wraps a bare boolean-valued child. We only handle the case
            // where the child itself is a field reference (e.g. `WHERE bool_col`).
            case "BOOLEAN_EXPRESSION" -> children.length == 1 && isPushableFieldRef(children[0]);
            default -> false;
        };
    }

    /**
     * Converts a Spark predicate to a Vortex expression. Returns {@link Optional#empty()} if the predicate cannot be
     * translated; callers should normally pre-check with {@link #isPushable}.
     */
    static Optional<Expression> convert(Predicate predicate) {
        if (predicate instanceof AlwaysTrue) {
            return Optional.of(Expression.literal(true));
        }
        if (predicate instanceof AlwaysFalse) {
            return Optional.of(Expression.literal(false));
        }
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
        if (predicate instanceof Not n) {
            return convert(n.child()).map(Expression::not);
        }
        org.apache.spark.sql.connector.expressions.Expression[] children = predicate.children();
        return switch (predicate.name()) {
            case "=", "<>", "!=", ">", ">=", "<", "<=" -> convertComparison(predicate.name(), children);
            case "IS_NULL" -> children.length == 1 ? columnOf(children[0]).map(Expression::isNull) : Optional.empty();
            case "IS_NOT_NULL" ->
                    children.length == 1 ? columnOf(children[0]).map(Expression::isNotNull) : Optional.empty();
            case "IN" -> convertIn(children);
            case "STARTS_WITH" ->
                    convertStringMatch(children, /* leadingWildcard= */ false, /* trailingWildcard= */ true);
            case "ENDS_WITH" ->
                    convertStringMatch(children, /* leadingWildcard= */ true, /* trailingWildcard= */ false);
            case "CONTAINS" -> convertStringMatch(children, /* leadingWildcard= */ true, /* trailingWildcard= */ true);
            case "BOOLEAN_EXPRESSION" -> children.length == 1 ? columnOf(children[0]) : Optional.empty();
            default -> Optional.empty();
        };
    }

    private static Optional<Expression> convertComparison(
            String op, org.apache.spark.sql.connector.expressions.Expression[] children) {
        if (children.length != 2) {
            return Optional.empty();
        }
        // Allow either side to be the column; Spark's V2 builder sometimes commutes.
        Optional<Expression> lhs = exprOf(children[0]);
        Optional<Expression> rhs = exprOf(children[1]);
        if (lhs.isEmpty() || rhs.isEmpty()) {
            return Optional.empty();
        }
        // We require at least one side to be a column reference to keep the surface small and to
        // match what Vortex pushdown understands.
        boolean lhsIsCol = isFieldRefExpr(children[0]);
        boolean rhsIsCol = isFieldRefExpr(children[1]);
        if (!lhsIsCol && !rhsIsCol) {
            return Optional.empty();
        }
        BinaryOp binaryOp = toBinaryOp(op);
        // Canonicalize so the column is on the left when only one side is a column.
        if (!lhsIsCol) {
            binaryOp = swap(binaryOp);
            Expression tmp = lhs.get();
            return Optional.of(Expression.binary(binaryOp, rhs.get(), tmp));
        }
        return Optional.of(Expression.binary(binaryOp, lhs.get(), rhs.get()));
    }

    private static Optional<Expression> convertIn(org.apache.spark.sql.connector.expressions.Expression[] children) {
        if (children.length < 2) {
            return Optional.empty();
        }
        Optional<Expression> column = columnOf(children[0]);
        if (column.isEmpty()) {
            return Optional.empty();
        }
        Expression columnExpr = column.get();
        List<Expression> eqs = new ArrayList<>(children.length - 1);
        for (int i = 1; i < children.length; i++) {
            Optional<Expression> literal = literalOf(children[i]);
            if (literal.isEmpty()) {
                return Optional.empty();
            }
            eqs.add(Expression.binary(BinaryOp.EQ, columnExpr, literal.get()));
        }
        if (eqs.size() == 1) {
            return Optional.of(eqs.get(0));
        }
        return Optional.of(Expression.or(eqs.toArray(new Expression[0])));
    }

    private static Optional<Expression> convertStringMatch(
            org.apache.spark.sql.connector.expressions.Expression[] children,
            boolean leadingWildcard,
            boolean trailingWildcard) {
        if (children.length != 2) {
            return Optional.empty();
        }
        Optional<Expression> column = columnOf(children[0]);
        Optional<String> needle = stringValueOf(children[1]);
        if (column.isEmpty() || needle.isEmpty()) {
            return Optional.empty();
        }
        String pattern = buildLikePattern(needle.get(), leadingWildcard, trailingWildcard);
        return Optional.of(Expression.like(
                column.get(), Expression.literal(pattern), /* negated= */ false, /* caseInsensitive= */ false));
    }

    /**
     * Build a LIKE pattern from a literal substring, escaping the {@code %}, {@code _}, and {@code \} meta-characters
     * so the Spark {@code STARTS_WITH}/{@code ENDS_WITH}/{@code CONTAINS} semantics (exact substring match) are
     * preserved.
     */
    private static String buildLikePattern(String literal, boolean leadingWildcard, boolean trailingWildcard) {
        StringBuilder sb = new StringBuilder(literal.length() + 2);
        if (leadingWildcard) {
            sb.append('%');
        }
        for (int i = 0; i < literal.length(); i++) {
            char c = literal.charAt(i);
            if (c == '%' || c == '_' || c == '\\') {
                sb.append('\\');
            }
            sb.append(c);
        }
        if (trailingWildcard) {
            sb.append('%');
        }
        return sb.toString();
    }

    private static BinaryOp toBinaryOp(String name) {
        return switch (name) {
            case "=" -> BinaryOp.EQ;
            case "<>", "!=" -> BinaryOp.NOT_EQ;
            case ">" -> BinaryOp.GT;
            case ">=" -> BinaryOp.GTE;
            case "<" -> BinaryOp.LT;
            case "<=" -> BinaryOp.LTE;
            default -> throw new IllegalArgumentException("not a pushable comparison operator: " + name);
        };
    }

    private static BinaryOp swap(BinaryOp op) {
        return switch (op) {
            case EQ, NOT_EQ -> op;
            case GT -> BinaryOp.LT;
            case GTE -> BinaryOp.LTE;
            case LT -> BinaryOp.GT;
            case LTE -> BinaryOp.GTE;
            default -> throw new IllegalArgumentException("not a comparison operator: " + op);
        };
    }

    private static boolean isPushableComparison(org.apache.spark.sql.connector.expressions.Expression[] children) {
        if (children.length != 2) {
            return false;
        }
        boolean lhsCol = isPushableFieldRef(children[0]);
        boolean lhsLit = isPushableLiteral(children[0]);
        boolean rhsCol = isPushableFieldRef(children[1]);
        boolean rhsLit = isPushableLiteral(children[1]);
        boolean lhsOk = lhsCol || lhsLit;
        boolean rhsOk = rhsCol || rhsLit;
        // We need at least one column reference; otherwise the predicate is comparing two
        // constants — Spark normally folds those, so we don't bother.
        return lhsOk && rhsOk && (lhsCol || rhsCol);
    }

    private static boolean isPushableFieldRef(org.apache.spark.sql.connector.expressions.Expression expr) {
        return expr instanceof NamedReference && ((NamedReference) expr).fieldNames().length >= 1;
    }

    private static boolean isFieldRefExpr(org.apache.spark.sql.connector.expressions.Expression expr) {
        return expr instanceof NamedReference;
    }

    /**
     * Returns the Vortex column expression for a Spark named reference, walking nested struct fields.
     */
    private static Optional<Expression> columnOf(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof NamedReference)) {
            return Optional.empty();
        }
        String[] parts = ((NamedReference) expr).fieldNames();
        if (parts.length == 0) {
            return Optional.empty();
        }
        return Optional.of(Expression.column(parts));
    }

    private static Optional<Expression> exprOf(org.apache.spark.sql.connector.expressions.Expression expr) {
        Optional<Expression> col = columnOf(expr);
        if (col.isPresent()) {
            return col;
        }
        return literalOf(expr);
    }

    private static Optional<String> stringValueOf(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof Literal<?>)) {
            return Optional.empty();
        }
        Object value = ((Literal<?>) expr).value();
        if (value == null) {
            return Optional.empty();
        }
        if (value instanceof UTF8String) {
            return Optional.of(value.toString());
        }
        if (value instanceof CharSequence) {
            return Optional.of(value.toString());
        }
        return Optional.empty();
    }

    private static boolean isPushableStringLiteral(org.apache.spark.sql.connector.expressions.Expression expr) {
        return stringValueOf(expr).isPresent();
    }

    private static boolean isPushableLiteral(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof Literal<?>)) {
            return false;
        }
        Literal<?> lit = (Literal<?>) expr;
        DataType dataType = lit.dataType();
        // Null literals are pushable (we emit a typed null literal).
        if (lit.value() == null) {
            return dataType instanceof BooleanType
                    || dataType instanceof ByteType
                    || dataType instanceof ShortType
                    || dataType instanceof IntegerType
                    || dataType instanceof LongType
                    || dataType instanceof FloatType
                    || dataType instanceof DoubleType
                    || dataType instanceof StringType
                    || dataType instanceof BinaryType
                    || dataType instanceof DateType
                    || dataType instanceof TimestampType
                    || dataType instanceof TimestampNTZType
                    || dataType instanceof DecimalType;
        }
        return literalOf(expr).isPresent();
    }

    private static Optional<Expression> literalOf(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof Literal<?>)) {
            return Optional.empty();
        }
        Literal<?> lit = (Literal<?>) expr;
        Object value = lit.value();
        DataType dataType = lit.dataType();
        return convertLiteral(value, dataType);
    }

    private static Optional<Expression> convertLiteral(Object value, DataType dataType) {
        if (dataType instanceof BooleanType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteralBool());
            }
            return Optional.of(Expression.literal((Boolean) value));
        }
        if (dataType instanceof ByteType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.I8));
            }
            return Optional.of(Expression.literal(((Number) value).byteValue()));
        }
        if (dataType instanceof ShortType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.I16));
            }
            return Optional.of(Expression.literal(((Number) value).shortValue()));
        }
        if (dataType instanceof IntegerType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.I32));
            }
            return Optional.of(Expression.literal(((Number) value).intValue()));
        }
        if (dataType instanceof LongType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.I64));
            }
            return Optional.of(Expression.literal(((Number) value).longValue()));
        }
        if (dataType instanceof FloatType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.F32));
            }
            return Optional.of(Expression.literal(((Number) value).floatValue()));
        }
        if (dataType instanceof DoubleType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.F64));
            }
            return Optional.of(Expression.literal(((Number) value).doubleValue()));
        }
        if (dataType instanceof StringType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.UTF8));
            }
            if (value instanceof UTF8String || value instanceof CharSequence) {
                return Optional.of(Expression.literal(value.toString()));
            }
        }
        if (dataType instanceof BinaryType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteral(Expression.DType.BINARY));
            }
            if (value instanceof byte[]) {
                return Optional.of(Expression.literal((byte[]) value));
            }
        }
        if (dataType instanceof DateType) {
            // Spark stores DateType as a 32-bit int day count since 1970-01-01.
            if (value == null) {
                return Optional.of(Expression.nullLiteralDate(TimeUnit.DAYS));
            }
            return Optional.of(Expression.literalDate(((Number) value).longValue(), TimeUnit.DAYS));
        }
        if (dataType instanceof TimestampType) {
            // Spark stores TimestampType as a 64-bit microseconds-since-epoch in UTC.
            if (value == null) {
                return Optional.of(Expression.nullLiteralTimestamp(TimeUnit.MICROSECONDS, "UTC"));
            }
            return Optional.of(Expression.literalTimestamp(((Number) value).longValue(), TimeUnit.MICROSECONDS, "UTC"));
        }
        if (dataType instanceof TimestampNTZType) {
            if (value == null) {
                return Optional.of(Expression.nullLiteralTimestamp(TimeUnit.MICROSECONDS, null));
            }
            return Optional.of(Expression.literalTimestamp(((Number) value).longValue(), TimeUnit.MICROSECONDS, null));
        }
        if (dataType instanceof DecimalType) {
            DecimalType decimalType = (DecimalType) dataType;
            int precision = decimalType.precision();
            int scale = decimalType.scale();
            if (value == null) {
                return Optional.of(Expression.nullLiteralDecimal(precision, scale));
            }
            BigInteger unscaled = unscaledValueOf(value, scale);
            if (unscaled == null) {
                return Optional.empty();
            }
            return Optional.of(Expression.literalDecimal(unscaled, precision, scale));
        }
        // Some Spark literals (e.g. NullType, GeographyType) have no Vortex representation.
        return Optional.empty();
    }

    /**
     * Extract the unscaled integer value of a Spark decimal literal at the supplied {@code scale}.
     */
    private static BigInteger unscaledValueOf(Object value, int scale) {
        BigDecimal decimal;
        if (value instanceof Decimal) {
            decimal = ((Decimal) value).toJavaBigDecimal();
        } else if (value instanceof BigDecimal) {
            decimal = (BigDecimal) value;
        } else {
            return null;
        }
        try {
            return decimal.setScale(scale).unscaledValue();
        } catch (ArithmeticException ignored) {
            return null;
        }
    }
}
