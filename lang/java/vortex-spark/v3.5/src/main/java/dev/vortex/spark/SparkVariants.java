// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import dev.vortex.relocated.org.apache.arrow.vector.complex.StructVector;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import java.util.List;
import java.util.Optional;
import org.apache.spark.sql.catalyst.expressions.SpecializedGetters;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnVector;

/**
 * Spark 3.5 has no {@code VARIANT} type. The Spark 4 build carries a class of this name that maps it to Arrow's
 * {@code arrow.parquet.variant} extension.
 */
public final class SparkVariants {
    /** The Arrow extension name Vortex reads as its variant type. */
    public static final String EXTENSION = "arrow.parquet.variant";

    private static final String EXTENSION_KEY = "ARROW:extension:name";

    private SparkVariants() {}

    public static boolean isVariant(DataType dataType) {
        return false;
    }

    public static Optional<DataType> variantType() {
        return Optional.empty();
    }

    public static boolean isVariant(Field field) {
        return EXTENSION.equals(field.getMetadata().get(EXTENSION_KEY));
    }

    // Unreachable: Spark 3.5 reports no variant column, and the shared code only calls this for one.
    @SuppressWarnings("DoNotCallSuggester")
    public static Field arrowField(String name, boolean nullable) {
        throw new UnsupportedOperationException("Spark 3.5 has no VARIANT type");
    }

    // Unreachable: Spark 3.5 reports no variant column, and the shared code only calls this for one.
    @SuppressWarnings("DoNotCallSuggester")
    public static void write(StructVector vector, SpecializedGetters row, int ordinal, int index) {
        throw new UnsupportedOperationException("Spark 3.5 has no VARIANT type");
    }

    // Unreachable: Spark 3.5 reports no variant column, and the shared code only calls this for one.
    @SuppressWarnings("DoNotCallSuggester")
    public static void writeNull(StructVector vector, int index) {
        throw new UnsupportedOperationException("Spark 3.5 has no VARIANT type");
    }

    public static Optional<List<VariantField>> extractedFields(DataType requested) {
        return Optional.empty();
    }

    // Unreachable: Spark 3.5 asks for no extraction, and the shared code only calls this for one.
    @SuppressWarnings("DoNotCallSuggester")
    public static ColumnVector extract(
            StructType type, List<VariantField> fields, ColumnVector[] extracted, ColumnVector present, int rowCount) {
        throw new UnsupportedOperationException("Spark 3.5 has no VARIANT type");
    }

    /** Matches the Spark 4 build's extraction field, which Spark 3.5 never produces. */
    public static final class VariantField {
        private VariantField() {}

        public String[] names() {
            return new String[0];
        }

        public long[] indexes() {
            return new long[0];
        }
    }
}
