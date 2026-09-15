// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.relocated.org.apache.arrow.memory.RootAllocator;
import dev.vortex.relocated.org.apache.arrow.vector.DecimalVector;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import org.apache.spark.sql.catalyst.expressions.codegen.UnsafeRowWriter;
import org.apache.spark.sql.internal.SQLConf;
import org.apache.spark.sql.types.Decimal;
import org.junit.jupiter.api.Test;

final class VortexDecimalAccessorTest {
    @Test
    void compactPrecisionsAndScalesMatchArrowValues() {
        Random random = new Random(1024);
        try (RootAllocator allocator = new RootAllocator()) {
            long limit = 1;
            for (int precision = 1; precision <= Decimal.MAX_LONG_DIGITS(); precision++) {
                limit *= 10;
                List<Long> values = new ArrayList<>(List.of(0L, 1L, -1L, limit - 1, 1 - limit));
                for (long power = 10; power < limit; power *= 10) {
                    values.add(power - 1);
                    values.add(power);
                    values.add(power + 1);
                    values.add(1 - power);
                    values.add(-power);
                    values.add(-power - 1);
                }
                for (int i = 0; i < 32; i++) {
                    values.add(random.nextLong() % limit);
                }
                for (int scale = 0; scale <= precision; scale++) {
                    try (DecimalVector vector = new DecimalVector("decimal", allocator, precision, scale)) {
                        vector.allocateNew(values.size() + 1);
                        for (int row = 0; row < values.size(); row++) {
                            vector.set(row, BigDecimal.valueOf(values.get(row), scale));
                        }
                        vector.setNull(values.size());
                        vector.setValueCount(values.size() + 1);
                        VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
                        for (int row = values.size() - 1; row >= 0; row--) {
                            BigDecimal expected = BigDecimal.valueOf(values.get(row), scale);
                            Decimal actual = column.getDecimal(row, precision, scale);
                            assertEquivalent(Decimal.apply(expected, precision, scale), actual, precision, scale);
                        }
                        assertNull(column.getDecimal(values.size(), precision, scale));
                    }
                }
            }
        }
    }

    @Test
    void smallDecimalReadsNativeWordsInEitherOrder() {
        String[] inputs = {"0.00", "123.45", "-123.45", "9999999999999999.99", "-9999999999999999.99"};
        try (RootAllocator allocator = new RootAllocator()) {
            for (ByteOrder order : new ByteOrder[] {ByteOrder.LITTLE_ENDIAN, ByteOrder.BIG_ENDIAN}) {
                try (DecimalVector vector = new DecimalVector("decimal", allocator, 18, 2)) {
                    vector.allocateNew(inputs.length + 1);
                    for (int row = 0; row < inputs.length; row++) {
                        BigDecimal value = new BigDecimal(inputs[row]);
                        vector.set(row, value);
                        long offset = (long) row * DecimalVector.TYPE_WIDTH;
                        long low = value.unscaledValue().longValueExact();
                        long high = value.signum() < 0 ? -1L : 0L;
                        // Model the two native-word positions; ArrowBuf supplies native-endian long reads.
                        vector.getDataBuffer().setLong(offset, order == ByteOrder.LITTLE_ENDIAN ? low : high);
                        vector.getDataBuffer()
                                .setLong(offset + Long.BYTES, order == ByteOrder.LITTLE_ENDIAN ? high : low);
                    }
                    vector.setNull(inputs.length);
                    vector.setValueCount(inputs.length + 1);
                    var accessor = new VortexArrowColumnVector.SmallDecimalAccessor(vector, order);
                    for (int row = 0; row < inputs.length; row++) {
                        BigDecimal expected = new BigDecimal(inputs[row]);
                        assertEquivalent(Decimal.apply(expected, 18, 2), accessor.getDecimal(row, 18, 2), 18, 2);
                    }
                    assertNull(accessor.getDecimal(inputs.length, 18, 2));
                }
            }
        }
    }

    @Test
    void requestedScaleAndPrecisionRetainRoundingAndOverflow() {
        try (RootAllocator allocator = new RootAllocator();
                DecimalVector vector = new DecimalVector("decimal", allocator, 5, 2)) {
            vector.allocateNew(4);
            vector.set(0, new BigDecimal("123.45"));
            vector.set(1, new BigDecimal("-123.45"));
            vector.set(2, new BigDecimal("999.95"));
            vector.setNull(3);
            vector.setValueCount(4);
            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertEquals(new BigDecimal("123.5"), column.getDecimal(0, 5, 1).toJavaBigDecimal());
            assertEquals(new BigDecimal("-123.5"), column.getDecimal(1, 5, 1).toJavaBigDecimal());
            assertEquals(
                    new BigDecimal("123.450000000000000000"),
                    column.getDecimal(0, 38, 18).toJavaBigDecimal());
            for (int row = 0; row < 3; row++) {
                for (int precision : new int[] {2, 4, 5, 8, 19}) {
                    for (int scale : new int[] {0, 1, 2, 3}) {
                        assertMatchesReference(vector, column, row, precision, scale);
                    }
                }
            }
            assertNull(column.getDecimal(3, 2, 0));
        }
    }

    @Test
    void wideVectorsRetainAllBitsEvenWhenRequestedPrecisionIsSmall() {
        try (RootAllocator allocator = new RootAllocator()) {
            for (int precision : new int[] {19, 20, 38}) {
                BigInteger largest = BigInteger.TEN.pow(precision).subtract(BigInteger.ONE);
                try (DecimalVector vector = new DecimalVector("decimal", allocator, precision, 2)) {
                    vector.allocateNew(4);
                    vector.set(0, new BigDecimal(largest, 2));
                    vector.set(1, new BigDecimal(largest.negate(), 2));
                    vector.set(2, new BigDecimal("1.23"));
                    vector.setNull(3);
                    vector.setValueCount(4);
                    VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
                    for (int row = 0; row < 3; row++) {
                        assertMatchesReference(vector, column, row, precision, 2);
                        assertMatchesReference(vector, column, row, 18, 2);
                    }
                    assertNull(column.getDecimal(3, precision, 2));
                }
            }
        }
    }

    @Test
    void outOfPrecisionBufferValuesRetainReferenceResultsAndErrors() {
        try (RootAllocator allocator = new RootAllocator();
                DecimalVector vector = new DecimalVector("decimal", allocator, 18, 0)) {
            // The byte setter permits values inconsistent with the field's precision.
            BigInteger[] invalid = {
                BigInteger.ONE.shiftLeft(64).add(BigInteger.valueOf(7)),
                BigInteger.ONE.shiftLeft(64).negate().add(BigInteger.valueOf(7)),
                BigInteger.TEN.pow(18),
                BigInteger.TEN.pow(18).negate(),
                BigInteger.valueOf(Long.MAX_VALUE),
                BigInteger.valueOf(Long.MIN_VALUE),
                BigInteger.valueOf(Long.MAX_VALUE).add(BigInteger.ONE),
                BigInteger.valueOf(Long.MIN_VALUE).subtract(BigInteger.ONE)
            };
            vector.allocateNew(invalid.length);
            for (int row = 0; row < invalid.length; row++) {
                vector.setBigEndian(row, invalid[row].toByteArray());
            }
            vector.setValueCount(invalid.length);
            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            for (int row = 0; row < invalid.length; row++) {
                assertMatchesReference(vector, column, row, 18, 0);
                assertMatchesReference(vector, column, row, 38, 0);
            }
            vector.setNull(0);
            assertNull(column.getDecimal(0, 18, 0));
        }
    }

    @Test
    void slicedVectorsAndReturnedValuesRemainIndependent() {
        try (RootAllocator allocator = new RootAllocator();
                DecimalVector vector = new DecimalVector("decimal", allocator, 10, 2)) {
            vector.allocateNew(4);
            vector.set(0, new BigDecimal("999.99"));
            vector.set(1, new BigDecimal("-12.34"));
            vector.setNull(2);
            vector.set(3, new BigDecimal("56.78"));
            vector.setValueCount(4);
            var transfer = vector.getTransferPair(allocator);
            transfer.splitAndTransfer(1, 3);
            try (DecimalVector slice = (DecimalVector) transfer.getTo()) {
                VortexArrowColumnVector column = new VortexArrowColumnVector(slice);
                Decimal first = column.getDecimal(0, 10, 2);
                Decimal repeated = column.getDecimal(0, 10, 2);
                Decimal last = column.getDecimal(2, 10, 2);
                assertNotSame(first, repeated);
                assertEquals(new BigDecimal("-12.34"), first.toJavaBigDecimal());
                assertEquals(new BigDecimal("56.78"), last.toJavaBigDecimal());
                first.set(0L);
                assertEquals(new BigDecimal("-12.34"), repeated.toJavaBigDecimal());
                assertEquals(new BigDecimal("56.78"), last.toJavaBigDecimal());
                assertNull(column.getDecimal(1, 10, 2));
            }
        }
    }

    @Test
    void negativeScaleRetainsReferenceConversionsAndValidation() {
        SQLConf conf = SQLConf.get();
        String key = "spark.sql.legacy.allowNegativeScaleOfDecimal";
        String previous = conf.getConfString(key, "false");
        boolean wasSet = conf.getAllConfs().contains(key);
        conf.setConfString(key, "true");
        try (RootAllocator allocator = new RootAllocator();
                DecimalVector vector = new DecimalVector("decimal", allocator, 5, -2)) {
            vector.allocateNew(1);
            vector.set(0, new BigDecimal("1.23E+4"));
            vector.setValueCount(1);
            VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
            assertMatchesReference(vector, column, 0, 5, -2);
            assertMatchesReference(vector, column, 0, 8, 0);
            assertMatchesReference(vector, column, 0, 5, -3);
            assertEquals(12300L, column.getDecimal(0, 5, -2).toLong());
            conf.setConfString(key, "false");
            assertMatchesReference(vector, column, 0, 5, -2);
        } finally {
            if (wasSet) {
                conf.setConfString(key, previous);
            } else {
                conf.unsetConf(key);
            }
        }
    }

    @Test
    void checkedIntegralConversionsRetainResultsAndOverflow() {
        String[] values = {
            "127.999999999999999",
            "-128.999999999999999",
            "32767.9999999999999",
            "-32768.9999999999999",
            "2147483647.99999999",
            "-2147483648.99999999",
            "123456789.999999999"
        };
        try (RootAllocator allocator = new RootAllocator()) {
            for (String value : values) {
                BigDecimal input = new BigDecimal(value);
                int precision = input.precision();
                int scale = input.scale();
                try (DecimalVector vector = new DecimalVector("decimal", allocator, precision, scale)) {
                    vector.allocateNew(1);
                    vector.set(0, input);
                    vector.setValueCount(1);
                    VortexArrowColumnVector column = new VortexArrowColumnVector(vector);
                    Decimal expected = Decimal.apply(vector.getObject(0), precision, scale);
                    Decimal actual = column.getDecimal(0, precision, scale);
                    assertConversion(expected::roundToByte, actual::roundToByte);
                    assertConversion(expected::roundToShort, actual::roundToShort);
                    assertConversion(expected::roundToInt, actual::roundToInt);
                    assertConversion(expected::roundToLong, actual::roundToLong);
                }
            }
        }
    }

    private static void assertConversion(Supplier<Object> expectedConversion, Supplier<Object> actualConversion) {
        Object expected;
        try {
            expected = expectedConversion.get();
        } catch (Exception expectedError) {
            Exception actualError = assertThrows(expectedError.getClass(), actualConversion::get);
            assertEquals(expectedError.getClass(), actualError.getClass());
            assertEquals(expectedError.getMessage(), actualError.getMessage());
            return;
        }
        assertEquals(expected, actualConversion.get());
    }

    private static void assertMatchesReference(
            DecimalVector vector, VortexArrowColumnVector column, int row, int precision, int scale) {
        Decimal expected;
        try {
            expected = Decimal.apply(vector.getObject(row), precision, scale);
        } catch (Exception expectedError) {
            Exception actualError =
                    assertThrows(expectedError.getClass(), () -> column.getDecimal(row, precision, scale));
            assertEquals(expectedError.getClass(), actualError.getClass());
            assertEquals(expectedError.getMessage(), actualError.getMessage());
            return;
        }
        assertEquivalent(expected, column.getDecimal(row, precision, scale), precision, scale);
    }

    private static void assertEquivalent(Decimal expected, Decimal actual, int precision, int scale) {
        assertEquals(expected.toJavaBigDecimal(), actual.toJavaBigDecimal());
        assertEquals(expected.precision(), actual.precision());
        assertEquals(expected.scale(), actual.scale());
        assertEquals(expected.hashCode(), actual.hashCode());
        assertEquals(0, expected.compare(actual));
        assertEquals(expected.toLong(), actual.toLong());
        assertEquals(expected, actual.clone());
        Decimal expectedCopy = expected.clone();
        Decimal actualCopy = actual.clone();
        assertEquals(expectedCopy.changePrecision(precision, 0), actualCopy.changePrecision(precision, 0));
        assertEquals(expectedCopy.toJavaBigDecimal(), actualCopy.toJavaBigDecimal());
        UnsafeRowWriter expectedWriter = new UnsafeRowWriter(1);
        UnsafeRowWriter actualWriter = new UnsafeRowWriter(1);
        expectedWriter.write(0, expected, precision, scale);
        actualWriter.write(0, actual, precision, scale);
        assertEquals(expectedWriter.getRow(), actualWriter.getRow());
    }
}
