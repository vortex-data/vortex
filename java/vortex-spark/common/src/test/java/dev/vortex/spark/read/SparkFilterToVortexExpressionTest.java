// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
import org.apache.spark.sql.sources.And;
import org.apache.spark.sql.sources.EqualNullSafe;
import org.apache.spark.sql.sources.EqualTo;
import org.apache.spark.sql.sources.GreaterThan;
import org.apache.spark.sql.sources.In;
import org.apache.spark.sql.sources.IsNotNull;
import org.apache.spark.sql.sources.IsNull;
import org.apache.spark.sql.sources.Not;
import org.apache.spark.sql.sources.Or;
import org.apache.spark.sql.sources.StringContains;
import org.apache.spark.sql.sources.StringStartsWith;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.Test;

final class SparkFilterToVortexExpressionTest {
    private static final StructType SCHEMA = new StructType()
            .add("id", DataTypes.IntegerType)
            .add("date", DataTypes.DateType)
            .add("timestamp", DataTypes.TimestampType)
            .add("decimal", DataTypes.createDecimalType(10, 2))
            .add("binary", DataTypes.BinaryType)
            .add("name", DataTypes.StringType)
            .add("nested", new StructType().add("value", DataTypes.LongType));

    @Test
    void convertsExternalTemporalDecimalAndBinaryLiterals() {
        assertPushable(new EqualTo("date", Date.valueOf("2024-01-02")));
        assertPushable(new GreaterThan("timestamp", Timestamp.valueOf("2024-01-02 03:04:05.123456")));
        assertPushable(new EqualTo("decimal", new BigDecimal("123.45")));
        assertPushable(new EqualTo("binary", new byte[] {1, 2, 3}));
    }

    @Test
    void convertsNestedLogicalAndStringFilters() {
        assertPushable(new EqualTo("nested.value", 42L));
        assertPushable(new And(new GreaterThan("id", 1), new StringContains("name", "%literal_")));
        assertPushable(new In("id", new Object[] {1, 2, null}));
    }

    @Test
    void convertsNullSafeEqualityAndNegation() {
        assertPushable(new EqualNullSafe("id", null));
        assertPushable(new EqualNullSafe("id", 7));
        assertPushable(new Not(new EqualTo("id", 7)));
        assertPushable(new IsNull("name"));
        assertPushable(new IsNotNull("name"));
    }

    @Test
    void convertsInListsHoldingNulls() {
        // SQL never matches a null through IN, so the null values drop out of the disjunction.
        assertPushable(new In("id", new Object[] {1, null}));
        // Nothing but nulls can never match, so the filter becomes a constant false.
        assertPushable(new In("id", new Object[] {null}));
    }

    @Test
    void rejectsDecimalLiteralsThatCannotHoldTheirScale() {
        // The column keeps two decimal places, so a third would be lost in the conversion.
        assertFalse(SparkFilterToVortexExpression.isPushable(new EqualTo("decimal", new BigDecimal("1.005")), SCHEMA));
        assertPushable(new EqualTo("decimal", new BigDecimal("1.00")));
    }

    @Test
    void rejectsAPartlyConvertibleConjunctionAndDisjunction() {
        assertFalse(SparkFilterToVortexExpression.isPushable(
                new And(new GreaterThan("id", 1), new EqualTo("missing", 2)), SCHEMA));
        assertFalse(SparkFilterToVortexExpression.isPushable(
                new Or(new GreaterThan("id", 1), new EqualTo("missing", 2)), SCHEMA));
    }

    @Test
    void rejectsStringMatchesOnNonStringColumns() {
        assertFalse(SparkFilterToVortexExpression.isPushable(new StringContains("id", "1"), SCHEMA));
        assertFalse(SparkFilterToVortexExpression.isPushable(new StringStartsWith("id", "1"), SCHEMA));
    }

    @Test
    void rejectsMissingColumnsAndMismatchedLiteralTypes() {
        assertFalse(SparkFilterToVortexExpression.isPushable(new EqualTo("missing", 1), SCHEMA));
        assertFalse(SparkFilterToVortexExpression.isPushable(new EqualTo("id", "one"), SCHEMA));
        assertFalse(SparkFilterToVortexExpression.isPushable(new EqualTo("id", null), SCHEMA));
    }

    private static void assertPushable(org.apache.spark.sql.sources.Filter filter) {
        assertTrue(SparkFilterToVortexExpression.isPushable(filter, SCHEMA));
        assertTrue(SparkFilterToVortexExpression.convert(filter, SCHEMA).isPresent());
    }
}
