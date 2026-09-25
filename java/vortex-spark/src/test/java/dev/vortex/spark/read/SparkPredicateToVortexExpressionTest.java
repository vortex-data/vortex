// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.jni.NativeLoader;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import java.math.BigDecimal;
import java.math.BigInteger;
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
 * Unit tests for Spark predicate pushdown eligibility.
 *
 * <p>{@code isPushable} decides which predicates {@code VortexScanBuilder.pushPredicates} lets Spark drop. The reader
 * uses each file's Arrow timestamp unit to translate accepted timestamp predicates.
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

    /** One data column per supported literal type, so every literal meets a same-typed column. */
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
        // Literal validation allocates native bound expressions.
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
    @DisplayName("Every supported comparison operator is pushable with the column on either side")
    void everyComparisonOperatorIsPushable() {
        for (String op : COMPARISON_OPERATORS) {
            assertPushable(predicate(op, ref("id"), literal(42)), op + " with column on the left");
            assertPushable(predicate(op, literal(42), ref("id")), op + " with column on the right");
        }
    }

    @Test
    @DisplayName("Comparison between two columns is pushable")
    void columnToColumnComparisonIsPushable() {
        assertPushable(equality(ref("id"), ref("profile", "address", "zip")));
    }

    @Test
    @DisplayName("Comparison between two literals is not pushable")
    void literalToLiteralComparisonIsRejected() {
        assertNotPushable(equality(literal(1), literal(2)));
    }

    @Test
    @DisplayName("Comparison with the wrong number of children is not pushable")
    void comparisonWithWrongArityIsRejected() {
        assertNotPushable(predicate("=", ref("id")));
        assertNotPushable(predicate("=", ref("id"), literal(1), literal(2)));
    }

    @Test
    @DisplayName("Nested column comparison is pushable")
    void nestedColumnComparisonIsPushable() {
        assertPushable(equality(ref("profile", "email"), literal("a@b.com")));
    }

    @Test
    @DisplayName("IS_NULL and IS_NOT_NULL are pushable for top-level and nested columns")
    void nullChecksArePushable() {
        assertPushable(predicate("IS_NULL", ref("name")));
        assertPushable(predicate("IS_NOT_NULL", ref("name")));
        assertPushable(predicate("IS_NULL", ref("profile", "address", "city")));
    }

    @Test
    @DisplayName("IN is pushable for a single literal and for many literals")
    void inIsPushable() {
        assertPushable(predicate("IN", ref("id"), literal(1)));
        assertPushable(predicate("IN", ref("id"), literal(1), literal(2), literal(3)));
    }

    @Test
    @DisplayName("IN with no literals is not pushable")
    void inWithoutLiteralsIsRejected() {
        assertNotPushable(predicate("IN", ref("id")));
    }

    @Test
    @DisplayName("String matching predicates are pushable, including LIKE meta-characters in the needle")
    void stringMatchesArePushable() {
        for (String name : List.of("STARTS_WITH", "ENDS_WITH", "CONTAINS")) {
            assertPushable(predicate(name, ref("name"), literal("ali")), name);
            assertPushable(predicate(name, ref("name"), literal("100%_a\\b")), name + " with escapes");
        }
    }

    @Test
    @DisplayName("String matching against a non-string literal is not pushable")
    void stringMatchAgainstNonStringLiteralIsRejected() {
        assertNotPushable(predicate("STARTS_WITH", ref("name"), literal(1)));
    }

    @Test
    @DisplayName("BOOLEAN_EXPRESSION over a column reference is pushable")
    void bareBooleanColumnIsPushable() {
        assertPushable(predicate("BOOLEAN_EXPRESSION", ref("active")));
    }

    @Test
    @DisplayName("An unrecognised predicate name is not pushable")
    void unknownPredicateNameIsRejected() {
        assertNotPushable(predicate("BLOOM_FILTER", ref("id"), literal(1)));
    }

    @Test
    @DisplayName("AlwaysTrue and AlwaysFalse are pushable")
    void constantPredicatesArePushable() {
        assertPushable(new AlwaysTrue());
        assertPushable(new AlwaysFalse());
    }

    @Test
    @DisplayName("AND, OR and NOT are pushable when every leaf is pushable")
    void compoundPredicatesArePushable() {
        Predicate left = equality(ref("id"), literal(1));
        Predicate right = predicate("IS_NOT_NULL", ref("name"));
        assertPushable(new And(left, right));
        assertPushable(new Or(left, right));
        assertPushable(new Not(left));
        assertPushable(new Not(new And(left, new Or(right, new AlwaysFalse()))));
    }

    @Test
    @DisplayName("A compound predicate with one unsupported leaf is not pushable")
    void compoundPredicateWithBadLeafIsRejected() {
        Predicate good = equality(ref("id"), literal(1));
        Predicate bad = predicate("BLOOM_FILTER", ref("id"), literal(1));
        assertNotPushable(new And(good, bad));
        assertNotPushable(new Or(bad, good));
        assertNotPushable(new Not(bad));
    }

    @Test
    @DisplayName("Every supported literal type accepts a null value for pushdown")
    void nullLiteralsArePushable() {
        for (String column : NULLABLE_LITERAL_COLUMNS) {
            assertPushable(
                    equality(ref(column), new LiteralValue<>(null, SCHEMA.get(column))), "null literal for " + column);
        }
    }

    @Test
    @DisplayName("Every supported non-null literal type is pushable")
    void nonNullLiteralsArePushable() {
        assertPushable(equality(ref("active"), new LiteralValue<>(true, DataTypes.BooleanType)));
        assertPushable(equality(ref("tiny"), new LiteralValue<>((byte) 1, DataTypes.ByteType)));
        assertPushable(equality(ref("small"), new LiteralValue<>((short) 1, DataTypes.ShortType)));
        assertPushable(equality(ref("id"), literal(42)));
        assertPushable(equality(ref("big"), new LiteralValue<>(1L, DataTypes.LongType)));
        assertPushable(equality(ref("ratio"), new LiteralValue<>(1.5f, DataTypes.FloatType)));
        assertPushable(equality(ref("weight"), new LiteralValue<>(1.5d, DataTypes.DoubleType)));
        assertPushable(equality(ref("name"), literal("alice")));
        assertPushable(equality(ref("payload"), new LiteralValue<>(new byte[] {1, 2, 3}, DataTypes.BinaryType)));
        // Spark encodes DateType as an epoch-day int and both timestamp types as epoch micros.
        assertPushable(equality(ref("birthday"), new LiteralValue<>(19_000, DataTypes.DateType)));
        assertPushable(equality(ref("createdAt"), new LiteralValue<>(1_700_000_000L, DataTypes.TimestampType)));
        assertPushable(equality(ref("createdLocal"), new LiteralValue<>(1_700_000_000L, DataTypes.TimestampNTZType)));
        assertPushable(equality(ref("amount"), decimalLiteral("12.34")));
    }

    @Test
    void timestampPushdownUsesResolvedSparkTypes() {
        Predicate aware = equality(ref("createdAt"), new LiteralValue<>(1L, DataTypes.TimestampType));
        assertTrue(SparkPredicateToVortexExpression.isPushable(aware, SCHEMA));
        assertTrue(SparkPredicateToVortexExpression.isPushable(
                equality(ref("createdLocal"), new LiteralValue<>(1L, DataTypes.TimestampNTZType)), SCHEMA));
        assertNotPushable(equality(ref("createdAt"), ref("createdAt")));
        assertNotPushable(equality(ref("createdAt"), new LiteralValue<>(1L, DataTypes.TimestampNTZType)));
    }

    @Test
    void timestampCutoffsMatchSparkNormalization() {
        long[] rawValues = {-1_500L, -1_001L, -1_000L, -999L, -1L, 0L, 1L, 999L, 1_000L, 1_500L};
        long[] microsValues = {Long.MIN_VALUE, -1_001L, -1L, 0L, 1L, 1_001L, Long.MAX_VALUE};
        for (TimeUnit unit : TimeUnit.values()) {
            for (long raw : rawValues) {
                long normalized = switch (unit) {
                    case SECOND -> Math.multiplyExact(raw, 1_000_000L);
                    case MILLISECOND -> Math.multiplyExact(raw, 1_000L);
                    case MICROSECOND -> raw;
                    case NANOSECOND -> Math.floorDiv(raw, 1_000L);
                };
                for (long micros : microsValues) {
                    BigInteger cutoff = SparkPredicateToVortexExpression.timestampLowerBound(
                            BigInteger.valueOf(micros), unit);
                    assertEquals(normalized >= micros, BigInteger.valueOf(raw).compareTo(cutoff) >= 0);
                }
            }
        }
    }

    @Test
    @DisplayName("A literal type with no Vortex representation is not pushable")
    void unrepresentableLiteralIsRejected() {
        assertNotPushable(equality(ref("id"), new LiteralValue<>(null, DataTypes.NullType)));
    }

    @Test
    @DisplayName("A decimal literal that does not fit the declared scale is not pushable")
    void decimalThatDoesNotFitTheScaleIsRejected() {
        // `unscaledValueOf` rejects 12.345 because it cannot be represented at scale 2.
        assertNotPushable(equality(ref("amount"), decimalLiteral("12.345")));
    }

    @Test
    @DisplayName("An empty named reference is not pushable")
    void emptyReferenceIsNotPushable() {
        // Reject empty field paths before the scan attempts to resolve them.
        assertNotPushable(equality(ref(), literal(1)));
        assertNotPushable(predicate("IS_NULL", ref()));
        assertNotPushable(predicate("IN", ref(), literal(1)));
        assertNotPushable(predicate("STARTS_WITH", ref(), literal("a")));
        assertNotPushable(predicate("BOOLEAN_EXPRESSION", ref()));
    }

    /** Check whether Spark may drop the predicate after Vortex accepts it for pushdown. */
    private static void assertPushable(Predicate predicate) {
        assertPushable(predicate, predicate.name());
    }

    private static void assertPushable(Predicate predicate, String what) {
        assertTrue(SparkPredicateToVortexExpression.isPushable(predicate, SCHEMA), () -> "not pushable: " + what);
    }

    private static void assertNotPushable(Predicate predicate) {
        assertNotPushable(predicate, predicate.name());
    }

    private static void assertNotPushable(Predicate predicate, String what) {
        assertFalse(SparkPredicateToVortexExpression.isPushable(predicate, SCHEMA), () -> "pushable: " + what);
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
