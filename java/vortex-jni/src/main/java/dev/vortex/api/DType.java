// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.util.List;
import java.util.Optional;

/**
 * Vortex logical type interface representing the schema and metadata for array data.
 *
 * <p>DType defines the logical type system used by Vortex arrays, including primitive types,
 * complex types like structs and lists, and temporal types with associated metadata.
 * This interface provides methods to inspect type variants, nullability, temporal properties,
 * decimal precision, and structural information for complex types.
 *
 * <p>Implementations of this interface are typically obtained from Vortex arrays and
 * should be properly closed when no longer needed to free native resources.
 */
public interface DType extends AutoCloseable {

    /**
     * Returns the variant of this data type.
     *
     * @return the {@link Variant} enum value representing the specific type category
     */
    Variant getVariant();

    /**
     * Checks if this data type allows null values.
     *
     * @return {@code true} if the type is nullable, {@code false} otherwise
     */
    boolean isNullable();

    /**
     * Get the field names for a STRUCT type.
     */
    List<String> getFieldNames();

    /**
     * Get the field types for a STRUCT type.
     */
    List<DType> getFieldTypes();

    /**
     * Get the element type for a LIST type.
     */
    DType getElementType();

    /**
     * Checks if this data type represents a date.
     *
     * @return {@code true} if this is a date type, {@code false} otherwise
     */
    boolean isDate();

    /**
     * Checks if this data type represents a time.
     *
     * @return {@code true} if this is a time type, {@code false} otherwise
     */
    boolean isTime();

    /**
     * Checks if this data type represents a timestamp.
     *
     * @return {@code true} if this is a timestamp type, {@code false} otherwise
     */
    boolean isTimestamp();

    /**
     * Returns the time unit for temporal data types.
     *
     * @return the {@link TimeUnit} for this temporal type
     * @throws IllegalStateException if this is not a temporal type
     */
    TimeUnit getTimeUnit();

    /**
     * Returns the timezone for timestamp data types.
     *
     * @return an {@link Optional} containing the timezone string if present,
     *         or empty if no timezone is specified
     */
    Optional<String> getTimeZone();

    /**
     * Checks if this data type represents a decimal number.
     *
     * @return {@code true} if this is a decimal type, {@code false} otherwise
     */
    boolean isDecimal();

    /**
     * Returns the precision for decimal data types.
     *
     * @return the precision (total number of digits) for decimal types
     * @throws IllegalStateException if this is not a decimal type
     */
    int getPrecision();

    /**
     * Returns the scale for decimal data types.
     *
     * @return the scale (number of digits after the decimal point) for decimal types
     * @throws IllegalStateException if this is not a decimal type
     */
    byte getScale();

    /**
     * Closes this DType and releases any associated native resources.
     *
     * <p>After calling this method, the DType should not be used again.
     * This method is idempotent and can be called multiple times safely.
     */
    @Override
    void close();

    /**
     * Enumeration of time units supported by Vortex temporal data types.
     *
     * <p>Time units define the granularity of temporal values and are used
     * by date, time, and timestamp data types to specify their precision.
     */
    enum TimeUnit {
        /** Nanosecond precision (10^-9 seconds) */
        NANOSECONDS,

        /** Microsecond precision (10^-6 seconds) */
        MICROSECONDS,

        /** Millisecond precision (10^-3 seconds) */
        MILLISECONDS,

        /** Second precision */
        SECONDS,

        /** Day precision (24-hour periods) */
        DAYS,
        ;

        /**
         * Converts a byte value to the corresponding TimeUnit enum.
         *
         * @param unit the byte value representing the time unit (0-4)
         * @return the corresponding {@link TimeUnit} enum value
         * @throws RuntimeException if the unit value is not recognized
         */
        public static TimeUnit from(byte unit) {
            switch (unit) {
                case 0:
                    return NANOSECONDS;
                case 1:
                    return MICROSECONDS;
                case 2:
                    return MILLISECONDS;
                case 3:
                    return SECONDS;
                case 4:
                    return DAYS;
                default:
                    throw new IllegalArgumentException("Unknown TimeUnit: " + unit);
            }
        }
    }

    /**
     * Enumeration of all supported data type variants in Vortex.
     *
     * <p>Each variant represents a different category of data type, from primitive
     * numeric types to complex structured types. This enum provides a way to
     * categorize and identify the specific type of data stored in a Vortex array.
     */
    enum Variant {
        /** Null type representing absence of value */
        NULL,

        /** Boolean type for true/false values */
        BOOL,

        /** Unsigned 8-bit integer type */
        PRIMITIVE_U8,

        /** Unsigned 16-bit integer type */
        PRIMITIVE_U16,

        /** Unsigned 32-bit integer type */
        PRIMITIVE_U32,

        /** Unsigned 64-bit integer type */
        PRIMITIVE_U64,

        /** Signed 8-bit integer type */
        PRIMITIVE_I8,

        /** Signed 16-bit integer type */
        PRIMITIVE_I16,

        /** Signed 32-bit integer type */
        PRIMITIVE_I32,

        /** Signed 64-bit integer type */
        PRIMITIVE_I64,

        /** 16-bit floating point type */
        PRIMITIVE_F16,

        /** 32-bit floating point type */
        PRIMITIVE_F32,

        /** 64-bit floating point type */
        PRIMITIVE_F64,

        /** UTF-8 encoded string type */
        UTF8,

        /** Binary data type for arbitrary byte sequences */
        BINARY,

        /** Structured type containing named fields */
        STRUCT,

        /** List type containing elements of a single type */
        LIST,

        /** Extension type for custom or domain-specific types */
        EXTENSION,

        /** Decimal type for precise numeric values */
        DECIMAL,
        ;

        /**
         * Converts a byte value to the corresponding Variant enum.
         *
         * @param variant the byte value representing the variant (0-18)
         * @return the corresponding {@link Variant} enum value
         * @throws RuntimeException if the variant value is not recognized
         */
        public static Variant from(byte variant) {
            switch (variant) {
                case 0:
                    return NULL;
                case 1:
                    return BOOL;
                case 2:
                    return PRIMITIVE_U8;
                case 3:
                    return PRIMITIVE_U16;
                case 4:
                    return PRIMITIVE_U32;
                case 5:
                    return PRIMITIVE_U64;
                case 6:
                    return PRIMITIVE_I8;
                case 7:
                    return PRIMITIVE_I16;
                case 8:
                    return PRIMITIVE_I32;
                case 9:
                    return PRIMITIVE_I64;
                case 10:
                    return PRIMITIVE_F16;
                case 11:
                    return PRIMITIVE_F32;
                case 12:
                    return PRIMITIVE_F64;
                case 13:
                    return UTF8;
                case 14:
                    return BINARY;
                case 15:
                    return STRUCT;
                case 16:
                    return LIST;
                case 17:
                    return EXTENSION;
                case 18:
                    return DECIMAL;
                default:
                    throw new IllegalArgumentException("Unknown DType variant: " + variant);
            }
        }
    }
}
