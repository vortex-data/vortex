// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.jni.NativeLoader;
import java.math.BigDecimal;
import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.AlwaysFalse;
import org.apache.spark.sql.connector.expressions.filter.AlwaysTrue;
import org.apache.spark.sql.connector.expressions.filter.And;
import org.apache.spark.sql.connector.expressions.filter.Not;
import org.apache.spark.sql.connector.expressions.filter.Or;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link SparkPredicateToVortexExpression#isPushable(Predicate, Map)} and
 * {@link SparkPredicateToVortexExpression#convert(Predicate)}.
 *
 * <p>{@code isPushable} decides which predicates {@code VortexScanBuilder.pushPredicates} lets Spark drop, while
 * {@code convert} builds the filter {@code VortexPartitionReader} actually pushes down. A predicate that passes the
 * first but fails the second is silently skipped by the reader, so the scan returns rows the query excluded. The tests
 * below pin that {@code isPushable} implies {@code convert().isPresent()} across every accepted shape.
 */
final class SparkPredicateToVortexExpressionTest {

    private static final StructType ADDRESS = DataTypes.createStructType(new org.apache.spark.sql.types.StructField[] {
        DataTypes.createStructField("city", DataTypes.StringType, true),
        DataTypes.createStructField("zip", DataTypes.IntegerType, true)
    });

    private static final StructType PROFILE = DataTypes.createStructType(new org.apache.spark.sql.types.StructField[] {
        DataTypes.createStructField("email", DataTypes.StringType, true),
        DataTypes.createStructField("address", ADDRESS, true)
    });

    private static final DataType DECIMAL = DataTypes.createDecimalType(10, 2);

    /** One data column per literal type {@code convertLiteral} maps, so every literal meets a same-typed column. */
    private static final Map<String, DataType> SCHEMA = Map.ofEntries(
            Map.entry("id", DataTypes.IntegerType),
            Map.entry("name", DataTypes.StringType),
            Map.entry("active", DataTypes.BooleanType),
            Map.entry("tiny", DataTypes.ByteType),
            Map.entry("small", DataTypes.ShortType),
            Map.entry("big", DataTypes.LongType),
            Map.entry("ratio", DataTypes.FloatType),
            Map.entry("weight", DataTypes.DoubleType),
            Map.entry("payload", DataTypes.BinaryType),
            Map.entry("birthday", DataTypes.DateType),
            Map.entry("createdAt", DataTypes.TimestampType),
            Map.entry("createdLocal", DataTypes.TimestampNTZType),
            Map.entry("amount", DECIMAL),
            Map.entry("profile", PROFILE));

    private static final List<String> COMPARISON_OPERATORS = List.of("=", "<>", "!=", ">", ">=", "<", "<=");

    /** The columns whose type {@code isPushableLiteral} accepts a {@code null} value for. */
    private static final List<String> NULLABLE_LITERAL_COLUMNS = List.of(
            "active",
            "tiny",
            "small",
            "id",
            "big",
            "ratio",
            "weight",
            "name",
            "payload",
            "birthday",
            "createdAt",
            "createdLocal",
            "amount");

    @BeforeAll
    static void loadNativeLibrary() {
        // `convert` allocates native expressions; `isPushable` does not.
        NativeLoader.loadJni();
    }

    @Test
    @DisplayName("Top-level column reference is pushable when present in the schema")
    void topLevelColumnIsPushable() {
        Predicate equality = equality(ref("id"), literal(42));
        assertTrue(SparkPredicateToVortexExpression.isPushable(equality, SCHEMA));
    }

    @Test
    @DisplayName("Top-level column reference is not pushable when absent from the schema")
    void unknownTopLevelColumnIsNotPushable() {
        Predicate equality = equality(ref("missing"), literal(0));
        assertFalse(SparkPredicateToVortexExpression.isPushable(equality, SCHEMA));
    }

    @Test
    @DisplayName("Nested field reference is pushable when every part resolves under struct types")
    void nestedFieldThatExistsIsPushable() {
        Predicate equality = equality(ref("profile", "email"), literal("a@b.com"));
        assertTrue(SparkPredicateToVortexExpression.isPushable(equality, SCHEMA));
    }

    @Test
    @DisplayName("Doubly nested field reference resolves through multiple struct levels")
    void doublyNestedFieldIsPushable() {
        Predicate equality = equality(ref("profile", "address", "zip"), literal(12345));
        assertTrue(SparkPredicateToVortexExpression.isPushable(equality, SCHEMA));
    }

    @Test
    @DisplayName("Nested field that does not exist in the struct is not pushable")
    void nestedFieldThatDoesNotExistIsNotPushable() {
        Predicate equality = equality(ref("profile", "phone"), literal("555"));
        assertFalse(SparkPredicateToVortexExpression.isPushable(equality, SCHEMA));
    }

    @Test
    @DisplayName("Descending past a leaf (non-struct) field is not pushable")
    void descendingPastLeafFieldIsNotPushable() {
        // `name` is a String, not a struct — `name.first` cannot resolve.
        Predicate equality = equality(ref("name", "first"), literal("alice"));
        assertFalse(SparkPredicateToVortexExpression.isPushable(equality, SCHEMA));
    }

    @Test
    @DisplayName("Empty named reference is not pushable")
    void emptyReferenceIsNotPushable() {
        Predicate equality = equality(ref(), literal(1));
        assertFalse(SparkPredicateToVortexExpression.isPushable(equality, SCHEMA));
    }

    @Test
    @DisplayName("Every accepted comparison operator converts, with the column on either side")
    void everyComparisonOperatorConverts() {
        for (String op : COMPARISON_OPERATORS) {
            assertPushableAndConvertible(predicate(op, ref("id"), literal(42)), op + " with column on the left");
            // Spark's V2 builder sometimes commutes; `convertComparison` swaps the operator back.
            assertPushableAndConvertible(predicate(op, literal(42), ref("id")), op + " with column on the right");
        }
    }

    @Test
    @DisplayName("Comparison between two columns converts")
    void columnToColumnComparisonConverts() {
        assertPushableAndConvertible(equality(ref("id"), ref("profile", "address", "zip")));
    }

    @Test
    @DisplayName("Comparison between two literals is rejected by both sides")
    void literalToLiteralComparisonIsRejected() {
        assertNotPushableAndNotConvertible(equality(literal(1), literal(2)));
    }

    @Test
    @DisplayName("Comparison with the wrong number of children is rejected by both sides")
    void comparisonWithWrongArityIsRejected() {
        assertNotPushableAndNotConvertible(predicate("=", ref("id")));
        assertNotPushableAndNotConvertible(predicate("=", ref("id"), literal(1), literal(2)));
    }

    @Test
    @DisplayName("Nested column comparison converts")
    void nestedColumnComparisonConverts() {
        assertPushableAndConvertible(equality(ref("profile", "email"), literal("a@b.com")));
    }

    @Test
    @DisplayName("IS_NULL and IS_NOT_NULL convert for top-level and nested columns")
    void nullChecksConvert() {
        assertPushableAndConvertible(predicate("IS_NULL", ref("name")));
        assertPushableAndConvertible(predicate("IS_NOT_NULL", ref("name")));
        assertPushableAndConvertible(predicate("IS_NULL", ref("profile", "address", "city")));
    }

    @Test
    @DisplayName("IN converts for a single literal and for many literals")
    void inConverts() {
        assertPushableAndConvertible(predicate("IN", ref("id"), literal(1)));
        assertPushableAndConvertible(predicate("IN", ref("id"), literal(1), literal(2), literal(3)));
    }

    @Test
    @DisplayName("IN with no literals is rejected by both sides")
    void inWithoutLiteralsIsRejected() {
        assertNotPushableAndNotConvertible(predicate("IN", ref("id")));
    }

    @Test
    @DisplayName("String matching predicates convert, including LIKE meta-characters in the needle")
    void stringMatchesConvert() {
        for (String name : List.of("STARTS_WITH", "ENDS_WITH", "CONTAINS")) {
            assertPushableAndConvertible(predicate(name, ref("name"), literal("ali")), name);
            // `buildLikePattern` escapes `%`, `_` and `\` so the match stays an exact substring.
            assertPushableAndConvertible(predicate(name, ref("name"), literal("100%_a\\b")), name + " with escapes");
        }
    }

    @Test
    @DisplayName("String matching against a non-string literal is rejected by both sides")
    void stringMatchAgainstNonStringLiteralIsRejected() {
        assertNotPushableAndNotConvertible(predicate("STARTS_WITH", ref("name"), literal(1)));
    }

    @Test
    @DisplayName("BOOLEAN_EXPRESSION over a column reference converts")
    void bareBooleanColumnConverts() {
        assertPushableAndConvertible(predicate("BOOLEAN_EXPRESSION", ref("active")));
    }

    @Test
    @DisplayName("An unrecognised predicate name is rejected by both sides")
    void unknownPredicateNameIsRejected() {
        assertNotPushableAndNotConvertible(predicate("BLOOM_FILTER", ref("id"), literal(1)));
    }

    @Test
    @DisplayName("AlwaysTrue and AlwaysFalse convert to boolean literals")
    void constantPredicatesConvert() {
        assertPushableAndConvertible(new AlwaysTrue());
        assertPushableAndConvertible(new AlwaysFalse());
    }

    @Test
    @DisplayName("AND, OR and NOT convert when every leaf converts")
    void compoundPredicatesConvert() {
        Predicate left = equality(ref("id"), literal(1));
        Predicate right = predicate("IS_NOT_NULL", ref("name"));
        assertPushableAndConvertible(new And(left, right));
        assertPushableAndConvertible(new Or(left, right));
        assertPushableAndConvertible(new Not(left));
        assertPushableAndConvertible(new Not(new And(left, new Or(right, new AlwaysFalse()))));
    }

    @Test
    @DisplayName("A compound predicate with one unconvertible leaf is rejected by both sides")
    void compoundPredicateWithBadLeafIsRejected() {
        Predicate good = equality(ref("id"), literal(1));
        Predicate bad = predicate("BLOOM_FILTER", ref("id"), literal(1));
        assertNotPushableAndNotConvertible(new And(good, bad));
        assertNotPushableAndNotConvertible(new Or(bad, good));
        assertNotPushableAndNotConvertible(new Not(bad));
    }

    @Test
    @DisplayName("Every literal type that accepts a null value converts")
    void nullLiteralsConvert() {
        for (String column : NULLABLE_LITERAL_COLUMNS) {
            assertPushableAndConvertible(
                    equality(ref(column), new LiteralValue<>(null, SCHEMA.get(column))), "null literal for " + column);
        }
    }

    @Test
    @DisplayName("Every non-null literal type the translator maps converts")
    void nonNullLiteralsConvert() {
        assertPushableAndConvertible(equality(ref("active"), new LiteralValue<>(true, DataTypes.BooleanType)));
        assertPushableAndConvertible(equality(ref("tiny"), new LiteralValue<>((byte) 1, DataTypes.ByteType)));
        assertPushableAndConvertible(equality(ref("small"), new LiteralValue<>((short) 1, DataTypes.ShortType)));
        assertPushableAndConvertible(equality(ref("id"), literal(42)));
        assertPushableAndConvertible(equality(ref("big"), new LiteralValue<>(1L, DataTypes.LongType)));
        assertPushableAndConvertible(equality(ref("ratio"), new LiteralValue<>(1.5f, DataTypes.FloatType)));
        assertPushableAndConvertible(equality(ref("weight"), new LiteralValue<>(1.5d, DataTypes.DoubleType)));
        assertPushableAndConvertible(equality(ref("name"), literal("alice")));
        assertPushableAndConvertible(
                equality(ref("payload"), new LiteralValue<>(new byte[] {1, 2, 3}, DataTypes.BinaryType)));
        // Spark encodes DateType as an epoch-day int and both timestamp types as epoch micros.
        assertPushableAndConvertible(equality(ref("birthday"), new LiteralValue<>(19_000, DataTypes.DateType)));
        assertPushableAndConvertible(
                equality(ref("createdAt"), new LiteralValue<>(1_700_000_000L, DataTypes.TimestampType)));
        assertPushableAndConvertible(
                equality(ref("createdLocal"), new LiteralValue<>(1_700_000_000L, DataTypes.TimestampNTZType)));
        assertPushableAndConvertible(equality(ref("amount"), decimalLiteral("12.34")));
    }

    @Test
    @DisplayName("A literal type with no Vortex representation is rejected by both sides")
    void unrepresentableLiteralIsRejected() {
        assertNotPushableAndNotConvertible(equality(ref("id"), new LiteralValue<>(null, DataTypes.NullType)));
    }

    @Test
    @DisplayName("A decimal literal that does not fit the declared scale is rejected by both sides")
    void decimalThatDoesNotFitTheScaleIsRejected() {
        // `unscaledValueOf` calls `setScale(2)` without a rounding mode, so 12.345 throws and
        // `isPushableLiteral` falls through to `literalOf`, which is empty.
        assertNotPushableAndNotConvertible(equality(ref("amount"), decimalLiteral("12.345")));
    }

    @Test
    @DisplayName("An empty named reference is rejected by convert as well as by isPushable")
    void emptyReferenceIsAlsoNotConvertible() {
        // `isFieldRefExpr` on the convert path only checks `instanceof NamedReference`, so the
        // zero-part guard lives in `columnOf`. Without it a pushable-looking predicate would reach
        // the reader and be silently dropped.
        assertNotPushableAndNotConvertible(equality(ref(), literal(1)));
        assertNotPushableAndNotConvertible(predicate("IS_NULL", ref()));
        assertNotPushableAndNotConvertible(predicate("IN", ref(), literal(1)));
        assertNotPushableAndNotConvertible(predicate("STARTS_WITH", ref(), literal("a")));
        assertNotPushableAndNotConvertible(predicate("BOOLEAN_EXPRESSION", ref()));
    }

    /**
     * Asserts the invariant documented on {@link SparkPredicateToVortexExpression#isPushable(Predicate, Map)}: a
     * predicate Spark is allowed to drop must produce a Vortex expression.
     */
    private static void assertPushableAndConvertible(Predicate predicate) {
        assertPushableAndConvertible(predicate, predicate.name());
    }

    private static void assertPushableAndConvertible(Predicate predicate, String what) {
        assertTrue(SparkPredicateToVortexExpression.isPushable(predicate, SCHEMA), () -> "not pushable: " + what);
        assertTrue(
                SparkPredicateToVortexExpression.convert(predicate).isPresent(),
                () -> "pushable but not convertible: " + what);
    }

    private static void assertNotPushableAndNotConvertible(Predicate predicate) {
        assertNotPushableAndNotConvertible(predicate, predicate.name());
    }

    private static void assertNotPushableAndNotConvertible(Predicate predicate, String what) {
        assertFalse(SparkPredicateToVortexExpression.isPushable(predicate, SCHEMA), () -> "pushable: " + what);
        assertFalse(SparkPredicateToVortexExpression.convert(predicate).isPresent(), () -> "convertible: " + what);
    }

    private static Predicate predicate(String name, Expression... children) {
        return new Predicate(name, children);
    }

    private static LiteralValue<Object> decimalLiteral(String value) {
        return new LiteralValue<>(Decimal.apply(new BigDecimal(value)), DECIMAL);
    }

    private static Predicate equality(Expression left, Expression right) {
        return new Predicate("=", new Expression[] {left, right});
    }

    private static NamedReference ref(String... parts) {
        return new TestNamedReference(parts);
    }

    private static LiteralValue<Object> literal(int value) {
        return new LiteralValue<>(value, DataTypes.IntegerType);
    }

    private static LiteralValue<Object> literal(String value) {
        return new LiteralValue<>(org.apache.spark.unsafe.types.UTF8String.fromString(value), DataTypes.StringType);
    }

    private static final class TestNamedReference implements NamedReference {
        private final String[] fieldNames;

        TestNamedReference(String[] fieldNames) {
            this.fieldNames = fieldNames;
        }

        @Override
        public String[] fieldNames() {
            return fieldNames;
        }
    }
}
