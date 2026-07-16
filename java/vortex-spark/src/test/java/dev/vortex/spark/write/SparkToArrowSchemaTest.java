// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.relocated.org.apache.arrow.vector.types.DateUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.FloatingPointPrecision;
import dev.vortex.relocated.org.apache.arrow.vector.types.TimeUnit;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Schema;
import java.util.List;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link SparkToArrowSchema}, which converts Spark SQL schemas to Arrow schemas on the write path.
 *
 * <p>Characterizes the supported conversions (including field-level nullability and nested types) and the Spark types
 * that are explicitly rejected, mirroring the read-path coverage in {@code ArrowUtilsTest}.
 */
final class SparkToArrowSchemaTest {
    private static Field single(String name, org.apache.spark.sql.types.DataType type, boolean nullable) {
        StructType schema = new StructType(new StructField[] {new StructField(name, type, nullable, Metadata.empty())});
        Schema arrowSchema = SparkToArrowSchema.convert(schema);
        assertEquals(1, arrowSchema.getFields().size());
        return arrowSchema.getFields().get(0);
    }

    @Test
    @DisplayName("BooleanType converts to Arrow Bool")
    void booleanConvertsToBool() {
        assertEquals(
                new ArrowType.Bool(), single("b", DataTypes.BooleanType, true).getType());
    }

    @Test
    @DisplayName("Integral types convert to signed Arrow Ints of matching width")
    void integralTypesConvertBySignedWidth() {
        assertEquals(
                new ArrowType.Int(8, true),
                single("v", DataTypes.ByteType, true).getType());
        assertEquals(
                new ArrowType.Int(16, true),
                single("v", DataTypes.ShortType, true).getType());
        assertEquals(
                new ArrowType.Int(32, true),
                single("v", DataTypes.IntegerType, true).getType());
        assertEquals(
                new ArrowType.Int(64, true),
                single("v", DataTypes.LongType, true).getType());
    }

    @Test
    @DisplayName("Float/Double convert to single/double precision Arrow FloatingPoint")
    void floatingPointConvertsByPrecision() {
        assertEquals(
                new ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE),
                single("v", DataTypes.FloatType, true).getType());
        assertEquals(
                new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE),
                single("v", DataTypes.DoubleType, true).getType());
    }

    @Test
    @DisplayName("String and Binary convert to Utf8 and Binary")
    void stringAndBinaryConvert() {
        assertEquals(
                new ArrowType.Utf8(), single("v", DataTypes.StringType, true).getType());
        assertEquals(
                new ArrowType.Binary(), single("v", DataTypes.BinaryType, true).getType());
    }

    @Test
    @DisplayName("DecimalType preserves precision and scale as 128-bit Arrow Decimal")
    void decimalPreservesPrecisionAndScale() {
        assertEquals(
                new ArrowType.Decimal(20, 4, 128),
                single("v", DataTypes.createDecimalType(20, 4), true).getType());
    }

    @Test
    @DisplayName("DateType converts to Date(DAY)")
    void dateConvertsToDayUnit() {
        assertEquals(
                new ArrowType.Date(DateUnit.DAY),
                single("v", DataTypes.DateType, true).getType());
    }

    @Test
    @DisplayName("Timestamp converts to microsecond precision, with UTC tz only for the tz-aware type")
    void timestampConvertsByTimezoneAwareness() {
        assertEquals(
                new ArrowType.Timestamp(TimeUnit.MICROSECOND, "UTC"),
                single("v", DataTypes.TimestampType, true).getType());
        assertEquals(
                new ArrowType.Timestamp(TimeUnit.MICROSECOND, null),
                single("v", DataTypes.TimestampNTZType, true).getType());
    }

    @Test
    @DisplayName("Field nullability is carried over to the Arrow field")
    void nullabilityIsPreserved() {
        assertTrue(single("v", DataTypes.IntegerType, true).isNullable());
        assertFalse(single("v", DataTypes.IntegerType, false).isNullable());
    }

    @Test
    @DisplayName("Field names and ordering are preserved")
    void fieldNamesAndOrderArePreserved() {
        StructType schema = new StructType()
                .add("first", DataTypes.IntegerType)
                .add("second", DataTypes.StringType)
                .add("third", DataTypes.DoubleType);

        Schema arrowSchema = SparkToArrowSchema.convert(schema);
        assertEquals(
                List.of("first", "second", "third"),
                arrowSchema.getFields().stream().map(Field::getName).toList());
    }

    @Test
    @DisplayName("Nested StructType converts to a Struct field with converted children")
    void structConvertsWithChildren() {
        StructType inner =
                new StructType().add("a", DataTypes.IntegerType, true).add("b", DataTypes.StringType, false);
        Field structField = single("s", inner, true);

        assertEquals(new ArrowType.Struct(), structField.getType());
        assertEquals(2, structField.getChildren().size());

        Field a = structField.getChildren().get(0);
        assertEquals("a", a.getName());
        assertEquals(new ArrowType.Int(32, true), a.getType());
        assertTrue(a.isNullable());

        Field b = structField.getChildren().get(1);
        assertEquals("b", b.getName());
        assertEquals(new ArrowType.Utf8(), b.getType());
        assertFalse(b.isNullable());
    }

    @Test
    @DisplayName("ArrayType converts to a List field whose element carries containsNull")
    void arrayConvertsWithElementNullability() {
        Field withNulls = single("l", DataTypes.createArrayType(DataTypes.IntegerType, true), true);
        assertEquals(new ArrowType.List(), withNulls.getType());
        assertEquals(1, withNulls.getChildren().size());

        Field element = withNulls.getChildren().get(0);
        assertEquals("element", element.getName());
        assertEquals(new ArrowType.Int(32, true), element.getType());
        assertTrue(element.isNullable());

        Field withoutNulls = single("l", DataTypes.createArrayType(DataTypes.StringType, false), true);
        assertFalse(withoutNulls.getChildren().get(0).isNullable());
    }

    @Test
    @DisplayName("Unsupported Spark types raise UnsupportedOperationException naming the type")
    void unsupportedTypeIsRejected() {
        StructType schema = new StructType().add("v", DataTypes.CalendarIntervalType);
        UnsupportedOperationException e =
                assertThrows(UnsupportedOperationException.class, () -> SparkToArrowSchema.convert(schema));
        assertTrue(e.getMessage().contains("CalendarIntervalType"));
    }
}
