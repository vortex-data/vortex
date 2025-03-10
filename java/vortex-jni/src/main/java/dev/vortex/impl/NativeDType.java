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
package dev.vortex.impl;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableList;
import com.sun.jna.Memory;
import com.sun.jna.ptr.IntByReference;
import dev.vortex.api.DType;
import dev.vortex.jni.FFI;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Optional;

public final class NativeDType extends BaseWrapped<FFI.FFIDType> implements DType {

    // Assumes no field name is > 1KiB
    private static final int MAX_FIELD_LEN = 1024;
    private static final int MAX_TIMEZONE_LEN = 2048;

    private final Variant variant;

    public NativeDType(FFI.FFIDType inner) {
        super(inner);
        this.variant = Variant.from(FFI.DType_get(inner));
    }

    @Override
    public Variant getVariant() {
        return variant;
    }

    /**
     * True if the DType is nullable
     */
    @Override
    public boolean isNullable() {
        checkNotNull(inner);

        return FFI.DType_nullable(inner);
    }

    @Override
    public String toString() {
        checkNotNull(inner);

        var result = new StringBuilder();
        result.append("DType{");
        result.append(this.variant.toString());
        result.append(", nullable=");
        result.append(isNullable());

        if (variant == Variant.STRUCT) {
            result.append(", fields=[");
            result.append(String.join(", ", getFieldNames()));
            result.append("]");
        }

        // TODO: List, Extension type handling.

        result.append("}");

        return result.toString();
    }

    @Override
    public List<String> getFieldNames() {
        checkNotNull(inner);
        checkArgument(Variant.STRUCT == variant, "getStructFieldNames() for non-struct DType");

        ImmutableList.Builder<String> builder = ImmutableList.builder();

        // We assume no field name is >= 1KiB.
        try (Memory memory = new Memory(MAX_FIELD_LEN)) {
            var fieldCount = FFI.DType_field_count(inner);
            for (int i = 0; i < fieldCount; i++) {
                var lenRef = new IntByReference();
                FFI.DType_field_name(inner, i, memory, lenRef);
                var data = memory.getByteArray(0, lenRef.getValue());
                var name = new String(data, StandardCharsets.UTF_8);
                builder.add(name);
            }
        }

        return builder.build();
    }

    @Override
    public List<DType> getFieldTypes() {
        checkNotNull(inner);
        checkArgument(Variant.STRUCT == variant, "getStructFieldNames() for non-struct DType");
        ImmutableList.Builder<DType> builder = ImmutableList.builder();
        var fieldCount = FFI.DType_field_count(inner);
        for (int i = 0; i < fieldCount; i++) {
            var fieldType = FFI.DType_field_dtype(inner, i);
            builder.add(new NativeDType(fieldType));
        }

        return builder.build();
    }

    @Override
    public DType getElementType() {
        var elementType = FFI.DType_element_type(inner);
        return new NativeDType(elementType);
    }

    @Override
    public boolean isDate() {
        checkNotNull(inner);
        return FFI.DType_is_date(inner);
    }

    @Override
    public boolean isTime() {
        checkNotNull(inner);
        return FFI.DType_is_time(inner);
    }

    @Override
    public boolean isTimestamp() {
        checkNotNull(inner);
        return FFI.DType_is_timestamp(inner);
    }

    @Override
    public TimeUnit getTimeUnit() {
        checkNotNull(inner);
        return TimeUnit.from(FFI.DType_time_unit(inner));
    }

    @Override
    public Optional<String> getTimeZone() {
        checkNotNull(inner);
        try (Memory memory = new Memory(MAX_TIMEZONE_LEN)) {
            var lenRef = new IntByReference();
            FFI.DType_time_zone(inner, memory, lenRef);

            if (lenRef.getValue() == 0) {
                return Optional.empty();
            }

            var data = memory.getByteArray(0, lenRef.getValue());
            var zone = new String(data, StandardCharsets.UTF_8);
            return Optional.of(zone);
        }
    }

    @Override
    public void close() {
        checkNotNull(inner);

        FFI.DType_free(inner);
        this.inner = null;
    }
}
