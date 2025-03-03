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

import com.google.common.collect.ImmutableList;

import java.io.Closeable;

/**
 * Vortex logical type.
 */
public interface DType extends Closeable {

    Variant getVariant();

    boolean isNullable();

    ImmutableList<String> getFieldNames();

    ImmutableList<DType> getFieldTypes();

    @Override
    void close();

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
            return switch (variant) {
                case 0 -> NULL;
                case 1 -> BOOL;
                case 2 -> PRIMITIVE_U8;
                case 3 -> PRIMITIVE_U16;
                case 4 -> PRIMITIVE_U32;
                case 5 -> PRIMITIVE_U64;
                case 6 -> PRIMITIVE_I8;
                case 7 -> PRIMITIVE_I16;
                case 8 -> PRIMITIVE_I32;
                case 9 -> PRIMITIVE_I64;
                case 10 -> PRIMITIVE_F16;
                case 11 -> PRIMITIVE_F32;
                case 12 -> PRIMITIVE_F64;
                case 13 -> UTF8;
                case 14 -> BINARY;
                case 15 -> STRUCT;
                case 16 -> LIST;
                case 17 -> EXTENSION;
                default -> throw new IllegalArgumentException("Unknown DType variant: " + variant);
            };
        }
    }
}
