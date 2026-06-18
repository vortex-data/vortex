// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Unit tests for {@link SparkPredicateToVortexExpression#isPushable(Predicate, Map)}. */
final class SparkPredicateToVortexExpressionTest {

    private static final StructType ADDRESS = DataTypes.createStructType(new org.apache.spark.sql.types.StructField[] {
        DataTypes.createStructField("city", DataTypes.StringType, true),
        DataTypes.createStructField("zip", DataTypes.IntegerType, true)
    });

    private static final StructType PROFILE = DataTypes.createStructType(new org.apache.spark.sql.types.StructField[] {
        DataTypes.createStructField("email", DataTypes.StringType, true),
        DataTypes.createStructField("address", ADDRESS, true)
    });

    private static final Map<String, DataType> SCHEMA =
            Map.of("id", DataTypes.IntegerType, "name", DataTypes.StringType, "profile", PROFILE);

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
