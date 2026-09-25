// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.BoundExpression;
import dev.vortex.api.DataSource;
import dev.vortex.api.Expression;
import dev.vortex.api.Expression.BinaryOp;
import dev.vortex.api.Expression.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
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

/**
 * Translates {@link Predicate Spark V2 predicates} into typed Vortex expressions for predicate pushdown.
 *
 * <p>The translator aims to express every Spark predicate Vortex can evaluate. Predicates that cannot be translated
 * (unsupported functions, literals on user-defined types, references to columns not present in the file, etc.) are left
 * to Spark for post-scan evaluation.
 */
final class SparkPredicateToVortexExpression {

    private static final BigInteger LONG_MIN = BigInteger.valueOf(Long.MIN_VALUE);
    private static final BigInteger LONG_MAX = BigInteger.valueOf(Long.MAX_VALUE);

    private SparkPredicateToVortexExpression() {}

    /**
     * Returns true if the given Spark predicate can be translated to a Vortex expression and every named reference
     * resolves to a real field path under {@code dataColumnTypes}.
     *
     * <p>{@code dataColumnTypes} maps each pushable top-level column name to its top-level Spark {@link DataType};
     * partition columns and columns the scan does not project should not appear in the map. For nested references (for
     * example {@code info.email}) the validator walks the named reference part by part, descending into
     * {@link StructType} fields so that {@code info} must be a struct that contains an {@code email} field.
     *
     * <p>This check is used in {@code SupportsPushDownV2Filters.pushPredicates} to decide which predicates Spark can
     * drop. It checks resolved Spark argument types and supported predicate structure. The reader
     * handles source timestamp units when it builds the bound filter for each file.
     */
    static boolean isPushable(Predicate predicate, Map<String, DataType> dataColumnTypes) {
        for (NamedReference ref : predicate.references()) {
            if (!resolveFieldPath(ref.fieldNames(), dataColumnTypes)) {
                return false;
            }
        }
        return isStructurallyPushable(predicate) && isTypeCompatible(predicate, dataColumnTypes);
    }

    static boolean hasTimestampLiteral(Predicate predicate) {
        for (org.apache.spark.sql.connector.expressions.Expression child : predicate.children()) {
            if (child instanceof Literal<?> literal && isTimestampType(literal.dataType())) {
                return true;
            }
            if (child instanceof Predicate nested && hasTimestampLiteral(nested)) {
                return true;
            }
        }
        return false;
    }

    static Map<List<String>, ArrowType.Timestamp> timestampTypes(Schema schema) {
        Map<List<String>, ArrowType.Timestamp> result = new HashMap<>();
        for (Field field : schema.getFields()) {
            collectTimestampTypes(field, new ArrayList<>(), result);
        }
        return result;
    }

    private static void collectTimestampTypes(
            Field field, List<String> path, Map<List<String>, ArrowType.Timestamp> result) {
        path.add(field.getName());
        if (field.getType() instanceof ArrowType.Timestamp timestamp) {
            result.put(List.copyOf(path), timestamp);
        }
        for (Field child : field.getChildren()) {
            collectTimestampTypes(child, path, result);
        }
        path.remove(path.size() - 1);
    }

    private static boolean isTimestampType(DataType type) {
        return type instanceof TimestampType || type instanceof TimestampNTZType;
    }

    private static boolean hasMatchingTimestampType(
            org.apache.spark.sql.connector.expressions.Expression expression,
            DataType sparkType,
            Map<List<String>, ArrowType.Timestamp> timestampTypes) {
        if (!(expression instanceof NamedReference reference)) {
            return false;
        }
        ArrowType.Timestamp sourceType = timestampTypes.get(Arrays.asList(reference.fieldNames()));
        return sourceType != null
                && ((sparkType instanceof TimestampType && sourceType.getTimezone() != null)
                        || (sparkType instanceof TimestampNTZType && sourceType.getTimezone() == null));
    }

    /**
     * Walks {@code parts} against {@code dataColumnTypes}, descending through {@link StructType} fields for
     * dot-separated nested references. Returns true only when every part resolves to an actual field in the schema.
     */
    private static boolean resolveFieldPath(String[] parts, Map<String, DataType> dataColumnTypes) {
        return resolveFieldType(parts, dataColumnTypes).isPresent();
    }

    private static Optional<DataType> resolveFieldType(String[] parts, Map<String, DataType> dataColumnTypes) {
        if (parts.length == 0) {
            return Optional.empty();
        }
        DataType current = dataColumnTypes.get(parts[0]);
        if (current == null) {
            return Optional.empty();
        }
        for (int i = 1; i < parts.length; i++) {
            if (!(current instanceof StructType struct)) {
                return Optional.empty();
            }
            Optional<StructField> field = findField(struct, parts[i]);
            if (field.isEmpty()) {
                return Optional.empty();
            }
            current = field.get().dataType();
        }
        return Optional.of(current);
    }

    private static Optional<DataType> typeOf(
            org.apache.spark.sql.connector.expressions.Expression expression, Map<String, DataType> dataColumnTypes) {
        if (expression instanceof NamedReference reference) {
            return resolveFieldType(reference.fieldNames(), dataColumnTypes);
        }
        if (expression instanceof Literal<?> literal && isPushableLiteral(literal)) {
            return Optional.of(literal.dataType());
        }
        return Optional.empty();
    }

    private static boolean isTypeCompatible(Predicate predicate, Map<String, DataType> dataColumnTypes) {
        if (predicate instanceof AlwaysTrue || predicate instanceof AlwaysFalse) {
            return true;
        }
        if (predicate instanceof And and) {
            return isTypeCompatible(and.left(), dataColumnTypes) && isTypeCompatible(and.right(), dataColumnTypes);
        }
        if (predicate instanceof Or or) {
            return isTypeCompatible(or.left(), dataColumnTypes) && isTypeCompatible(or.right(), dataColumnTypes);
        }
        if (predicate instanceof Not not) {
            return isTypeCompatible(not.child(), dataColumnTypes);
        }

        var children = predicate.children();
        return switch (predicate.name()) {
            case "=", "<>", "!=", ">", ">=", "<", "<=" -> {
                if (children.length != 2) {
                    yield false;
                }
                Optional<DataType> leftType = typeOf(children[0], dataColumnTypes);
                if (leftType.isEmpty() || !leftType.equals(typeOf(children[1], dataColumnTypes))) {
                    yield false;
                }
                if (!isTimestampType(leftType.get())) {
                    yield true;
                }
                yield (children[0] instanceof NamedReference && children[1] instanceof Literal<?>)
                        || (children[1] instanceof NamedReference && children[0] instanceof Literal<?>);
            }
            case "IN" -> {
                if (children.length < 2) {
                    yield false;
                }
                Optional<DataType> fieldType = typeOf(children[0], dataColumnTypes);
                boolean compatible = fieldType.isPresent();
                for (int i = 1; i < children.length; i++) {
                    compatible &= fieldType.equals(typeOf(children[i], dataColumnTypes));
                }
                yield compatible;
            }
            case "IS_NULL", "IS_NOT_NULL" -> children.length == 1;
            case "STARTS_WITH", "ENDS_WITH", "CONTAINS" ->
                children.length == 2 && typeOf(children[0], dataColumnTypes).orElse(null) instanceof StringType;
            case "BOOLEAN_EXPRESSION" ->
                children.length == 1 && typeOf(children[0], dataColumnTypes).orElse(null) instanceof BooleanType;
            default -> false;
        };
    }

    private static Optional<StructField> findField(StructType struct, String name) {
        return Arrays.stream(struct.fields())
                .filter(structField -> structField.name().equals(name))
                .findFirst();
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

    /** Lower a Spark-resolved predicate directly to typed Vortex calls. */
    static Optional<BoundExpression> convertBound(
            Predicate predicate,
            DataSource dataSource,
            Map<List<String>, ArrowType.Timestamp> timestampTypes) {
        if (predicate instanceof AlwaysTrue) {
            return Optional.of(BoundExpression.literal(true));
        }
        if (predicate instanceof AlwaysFalse) {
            return Optional.of(BoundExpression.literal(false));
        }
        if (predicate instanceof And and) {
            Optional<BoundExpression> lhs = convertBound(and.left(), dataSource, timestampTypes);
            Optional<BoundExpression> rhs = convertBound(and.right(), dataSource, timestampTypes);
            return lhs.isPresent() && rhs.isPresent()
                    ? Optional.of(BoundExpression.and(lhs.get(), rhs.get()))
                    : Optional.empty();
        }
        if (predicate instanceof Or or) {
            Optional<BoundExpression> lhs = convertBound(or.left(), dataSource, timestampTypes);
            Optional<BoundExpression> rhs = convertBound(or.right(), dataSource, timestampTypes);
            return lhs.isPresent() && rhs.isPresent()
                    ? Optional.of(BoundExpression.or(lhs.get(), rhs.get()))
                    : Optional.empty();
        }
        if (predicate instanceof Not not) {
            return convertBound(not.child(), dataSource, timestampTypes).map(BoundExpression::not);
        }

        var children = predicate.children();
        return switch (predicate.name()) {
            case "=", "<>", "!=", ">", ">=", "<", "<=" ->
                convertComparisonBound(predicate.name(), children, dataSource, timestampTypes);
            case "IS_NULL" ->
                children.length == 1
                        ? boundColumnOf(children[0], dataSource).map(BoundExpression::isNull)
                        : Optional.empty();
            case "IS_NOT_NULL" ->
                children.length == 1
                        ? boundColumnOf(children[0], dataSource).map(BoundExpression::isNotNull)
                        : Optional.empty();
            case "IN" -> convertInBound(children, dataSource, timestampTypes);
            case "STARTS_WITH" -> convertStringMatchBound(children, dataSource, false, true);
            case "ENDS_WITH" -> convertStringMatchBound(children, dataSource, true, false);
            case "CONTAINS" -> convertStringMatchBound(children, dataSource, true, true);
            case "BOOLEAN_EXPRESSION" ->
                children.length == 1 ? boundColumnOf(children[0], dataSource) : Optional.empty();
            default -> Optional.empty();
        };
    }

    private static Optional<BoundExpression> convertComparisonBound(
            String op,
            org.apache.spark.sql.connector.expressions.Expression[] children,
            DataSource dataSource,
            Map<List<String>, ArrowType.Timestamp> timestampTypes) {
        if (children.length != 2) {
            return Optional.empty();
        }
        if (children[0] instanceof NamedReference column
                && children[1] instanceof Literal<?> literal
                && isTimestampType(literal.dataType())) {
            BoundExpression raw = BoundExpression.castToI64(BoundExpression.column(dataSource, column.fieldNames()));
            return Optional.of(timestampComparison(toBinaryOp(op), column, literal, raw, timestampTypes));
        }
        if (children[1] instanceof NamedReference column
                && children[0] instanceof Literal<?> literal
                && isTimestampType(literal.dataType())) {
            BoundExpression raw = BoundExpression.castToI64(BoundExpression.column(dataSource, column.fieldNames()));
            return Optional.of(timestampComparison(swap(toBinaryOp(op)), column, literal, raw, timestampTypes));
        }
        Optional<BoundExpression> lhs = boundExprOf(children[0], dataSource);
        Optional<BoundExpression> rhs = boundExprOf(children[1], dataSource);
        if (lhs.isEmpty() || rhs.isEmpty()) {
            return Optional.empty();
        }
        BinaryOp operator = toBinaryOp(op);
        if (!isFieldRefExpr(children[0])) {
            operator = swap(operator);
            return Optional.of(BoundExpression.binary(operator, rhs.get(), lhs.get()));
        }
        return Optional.of(BoundExpression.binary(operator, lhs.get(), rhs.get()));
    }

    private static Optional<BoundExpression> convertInBound(
            org.apache.spark.sql.connector.expressions.Expression[] children,
            DataSource dataSource,
            Map<List<String>, ArrowType.Timestamp> timestampTypes) {
        if (children.length < 2) {
            return Optional.empty();
        }
        Optional<BoundExpression> column = boundColumnOf(children[0], dataSource);
        if (column.isEmpty()) {
            return Optional.empty();
        }
        BoundExpression combined = null;
        BoundExpression timestampRaw = null;
        for (int i = 1; i < children.length; i++) {
            BoundExpression equal;
            if (children[0] instanceof NamedReference reference
                    && children[i] instanceof Literal<?> timestampLiteral
                    && isTimestampType(timestampLiteral.dataType())) {
                if (timestampRaw == null) {
                    timestampRaw = BoundExpression.castToI64(column.get());
                }
                equal = timestampComparison(BinaryOp.EQ, reference, timestampLiteral, timestampRaw, timestampTypes);
            } else {
                Optional<BoundExpression> literal = boundLiteralOf(children[i]);
                if (literal.isEmpty()) {
                    return Optional.empty();
                }
                equal = BoundExpression.binary(BinaryOp.EQ, column.get(), literal.get());
            }
            combined = combined == null ? equal : BoundExpression.or(combined, equal);
        }
        return Optional.of(combined);
    }

    private static BoundExpression timestampComparison(
            BinaryOp operator,
            NamedReference column,
            Literal<?> literal,
            BoundExpression raw,
            Map<List<String>, ArrowType.Timestamp> timestampTypes) {
        ArrowType.Timestamp sourceType = timestampTypes.get(Arrays.asList(column.fieldNames()));
        if (sourceType == null || !hasMatchingTimestampType(column, literal.dataType(), timestampTypes)) {
            throw new IllegalStateException("Spark accepted a timestamp predicate with an incompatible source type: "
                    + Arrays.toString(column.fieldNames()));
        }
        // Arrow stores epoch counts in the declared unit; timezone metadata does not change those counts.
        // Compare the raw counts against cutoffs that implement Spark's microsecond normalization.
        if (literal.value() == null) {
            return BoundExpression.binary(operator, raw, BoundExpression.nullLiteral(Expression.DType.I64));
        }
        BigInteger micros = BigInteger.valueOf(((Number) literal.value()).longValue());
        BigInteger lower = timestampLowerBound(micros, sourceType.getUnit());
        BigInteger upper = timestampLowerBound(micros.add(BigInteger.ONE), sourceType.getUnit());
        return switch (operator) {
            case EQ -> BoundExpression.and(rawAtLeast(raw, lower), rawBelow(raw, upper));
            case NOT_EQ -> BoundExpression.or(rawBelow(raw, lower), rawAtLeast(raw, upper));
            case GT -> rawAtLeast(raw, upper);
            case GTE -> rawAtLeast(raw, lower);
            case LT -> rawBelow(raw, lower);
            case LTE -> rawBelow(raw, upper);
            default -> throw new IllegalArgumentException("not a timestamp comparison operator: " + operator);
        };
    }

    /** First stored timestamp value whose Spark-normalized value is at least {@code micros}. */
    static BigInteger timestampLowerBound(
            BigInteger micros, dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit unit) {
        return switch (unit) {
            case SECOND -> ceilDivide(micros, BigInteger.valueOf(1_000_000L));
            case MILLISECOND -> ceilDivide(micros, BigInteger.valueOf(1_000L));
            case MICROSECOND -> micros;
            case NANOSECOND -> micros.multiply(BigInteger.valueOf(1_000L));
        };
    }

    private static BigInteger ceilDivide(BigInteger value, BigInteger divisor) {
        BigInteger[] quotientAndRemainder = value.divideAndRemainder(divisor);
        return quotientAndRemainder[1].signum() > 0
                ? quotientAndRemainder[0].add(BigInteger.ONE)
                : quotientAndRemainder[0];
    }

    private static BoundExpression rawAtLeast(BoundExpression raw, BigInteger bound) {
        if (bound.compareTo(LONG_MIN) < 0) {
            return BoundExpression.binary(BinaryOp.GTE, raw, BoundExpression.literal(Long.MIN_VALUE));
        }
        if (bound.compareTo(LONG_MAX) > 0) {
            return BoundExpression.binary(BinaryOp.LT, raw, BoundExpression.literal(Long.MIN_VALUE));
        }
        return BoundExpression.binary(BinaryOp.GTE, raw, BoundExpression.literal(bound.longValueExact()));
    }

    private static BoundExpression rawBelow(BoundExpression raw, BigInteger bound) {
        if (bound.compareTo(LONG_MIN) < 0) {
            return BoundExpression.binary(BinaryOp.LT, raw, BoundExpression.literal(Long.MIN_VALUE));
        }
        if (bound.compareTo(LONG_MAX) > 0) {
            return BoundExpression.binary(BinaryOp.GTE, raw, BoundExpression.literal(Long.MIN_VALUE));
        }
        return BoundExpression.binary(BinaryOp.LT, raw, BoundExpression.literal(bound.longValueExact()));
    }

    private static Optional<BoundExpression> convertStringMatchBound(
            org.apache.spark.sql.connector.expressions.Expression[] children,
            DataSource dataSource,
            boolean leadingWildcard,
            boolean trailingWildcard) {
        if (children.length != 2) {
            return Optional.empty();
        }
        Optional<BoundExpression> column = boundColumnOf(children[0], dataSource);
        Optional<String> needle = stringValueOf(children[1]);
        if (column.isEmpty() || needle.isEmpty()) {
            return Optional.empty();
        }
        String pattern = buildLikePattern(needle.get(), leadingWildcard, trailingWildcard);
        return Optional.of(BoundExpression.like(column.get(), BoundExpression.literal(pattern), false, false));
    }

    private static Optional<BoundExpression> boundColumnOf(
            org.apache.spark.sql.connector.expressions.Expression expression, DataSource dataSource) {
        if (!(expression instanceof NamedReference reference) || reference.fieldNames().length == 0) {
            return Optional.empty();
        }
        return Optional.of(BoundExpression.column(dataSource, reference.fieldNames()));
    }

    private static Optional<BoundExpression> boundLiteralOf(
            org.apache.spark.sql.connector.expressions.Expression expression) {
        return literalOf(expression);
    }

    private static Optional<BoundExpression> boundExprOf(
            org.apache.spark.sql.connector.expressions.Expression expression, DataSource dataSource) {
        Optional<BoundExpression> column = boundColumnOf(expression, dataSource);
        return column.isPresent() ? column : boundLiteralOf(expression);
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

    private static Optional<BoundExpression> literalOf(org.apache.spark.sql.connector.expressions.Expression expr) {
        if (!(expr instanceof Literal<?>)) {
            return Optional.empty();
        }
        Literal<?> lit = (Literal<?>) expr;
        Object value = lit.value();
        DataType dataType = lit.dataType();
        return convertLiteral(value, dataType);
    }

    private static Optional<BoundExpression> convertLiteral(Object value, DataType dataType) {
        if (dataType instanceof BooleanType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteralBool());
            }
            return Optional.of(BoundExpression.literal((Boolean) value));
        }
        if (dataType instanceof ByteType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.I8));
            }
            return Optional.of(BoundExpression.literal(((Number) value).byteValue()));
        }
        if (dataType instanceof ShortType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.I16));
            }
            return Optional.of(BoundExpression.literal(((Number) value).shortValue()));
        }
        if (dataType instanceof IntegerType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.I32));
            }
            return Optional.of(BoundExpression.literal(((Number) value).intValue()));
        }
        if (dataType instanceof LongType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.I64));
            }
            return Optional.of(BoundExpression.literal(((Number) value).longValue()));
        }
        if (dataType instanceof FloatType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.F32));
            }
            return Optional.of(BoundExpression.literal(((Number) value).floatValue()));
        }
        if (dataType instanceof DoubleType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.F64));
            }
            return Optional.of(BoundExpression.literal(((Number) value).doubleValue()));
        }
        if (dataType instanceof StringType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.UTF8));
            }
            if (value instanceof UTF8String || value instanceof CharSequence) {
                return Optional.of(BoundExpression.literal(value.toString()));
            }
        }
        if (dataType instanceof BinaryType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteral(Expression.DType.BINARY));
            }
            if (value instanceof byte[]) {
                return Optional.of(BoundExpression.literal((byte[]) value));
            }
        }
        if (dataType instanceof DateType) {
            // Spark stores DateType as a 32-bit int day count since 1970-01-01.
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteralDate(TimeUnit.DAYS));
            }
            return Optional.of(BoundExpression.literalDate(((Number) value).longValue(), TimeUnit.DAYS));
        }
        if (dataType instanceof TimestampType) {
            // Spark stores TimestampType as a 64-bit microseconds-since-epoch in UTC.
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteralTimestamp(TimeUnit.MICROSECONDS, "UTC"));
            }
            return Optional.of(
                    BoundExpression.literalTimestamp(((Number) value).longValue(), TimeUnit.MICROSECONDS, "UTC"));
        }
        if (dataType instanceof TimestampNTZType) {
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteralTimestamp(TimeUnit.MICROSECONDS, null));
            }
            return Optional.of(
                    BoundExpression.literalTimestamp(((Number) value).longValue(), TimeUnit.MICROSECONDS, null));
        }
        if (dataType instanceof DecimalType) {
            DecimalType decimalType = (DecimalType) dataType;
            int precision = decimalType.precision();
            int scale = decimalType.scale();
            if (value == null) {
                return Optional.of(BoundExpression.nullLiteralDecimal(precision, scale));
            }
            BigInteger unscaled = unscaledValueOf(value, scale);
            if (unscaled == null) {
                return Optional.empty();
            }
            return Optional.of(BoundExpression.literalDecimal(unscaled, precision, scale));
        }
        // Some Spark literals (e.g. NullType, GeographyType) have no Vortex representation.
        return Optional.empty();
    }

    /** Extract the unscaled integer value of a Spark decimal literal at the supplied {@code scale}. */
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
