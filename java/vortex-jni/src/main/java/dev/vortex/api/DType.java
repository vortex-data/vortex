/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.api;

import java.util.List;
import java.util.Optional;

/**
 * Vortex logical type.
 */
public interface DType extends AutoCloseable {

    Variant getVariant();

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

    boolean isDate();

    boolean isTime();

    boolean isTimestamp();

    TimeUnit getTimeUnit();

    Optional<String> getTimeZone();

    @Override
    void close();

    enum TimeUnit {
        NANOSECONDS,
        MICROSECONDS,
        MILLISECONDS,
        SECONDS,
        DAYS,
        ;

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

    enum Variant {
        NULL,
        BOOL,
        PRIMITIVE_U8,
        PRIMITIVE_U16,
        PRIMITIVE_U32,
        PRIMITIVE_U64,
        PRIMITIVE_I8,
        PRIMITIVE_I16,
        PRIMITIVE_I32,
        PRIMITIVE_I64,
        PRIMITIVE_F16,
        PRIMITIVE_F32,
        PRIMITIVE_F64,
        UTF8,
        BINARY,
        STRUCT,
        LIST,
        EXTENSION,
        ;

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
                default:
                    throw new IllegalArgumentException("Unknown DType variant: " + variant);
            }
        }
    }
}
