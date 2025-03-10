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

import static com.google.common.base.Preconditions.checkNotNull;

import com.jakewharton.nopen.annotation.Open;
import com.sun.jna.Memory;
import com.sun.jna.ptr.IntByReference;
import dev.vortex.api.Array;
import dev.vortex.api.DType;
import dev.vortex.jni.FFI;
import java.nio.charset.StandardCharsets;

/**
 * Core Vortex array type that all logical arrays inherit from.
 */
@Open
public class NativeArray extends BaseWrapped<FFI.FFIArray> implements Array {
    // Assume no strings larger than 1MiB.
    private static final int MAX_STRING_LEN = 1_024 * 1_024;

    private final boolean isDate;
    private final boolean isTimestamp;

    public NativeArray(FFI.FFIArray inner) {
        super(inner);
        var dtype = FFI.FFIArray_dtype(inner);
        this.isDate = FFI.DType_is_date(dtype);
        this.isTimestamp = FFI.DType_is_timestamp(dtype);
    }

    @Override
    public void close() {
        checkNotNull(inner, "inner");

        // Free all resources
        FFI.FFIArray_free(inner);

        inner = null;
    }

    /**
     * Get the length of the array.
     */
    @Override
    public long getLen() {
        return FFI.FFIArray_len(inner);
    }

    @Override
    public DType getDataType() {
        checkNotNull(inner, "inner");

        var dtype = FFI.FFIArray_dtype(inner);
        return new NativeDType(dtype);
    }

    @Override
    public Array getField(int index) {
        checkNotNull(inner, "inner");

        return new NativeArray(FFI.FFIArray_get_field(inner, index));
    }

    @Override
    public Array slice(int start, int stop) {
        checkNotNull(inner, "inner");
        return new NativeArray(FFI.FFIArray_slice(inner, start, stop));
    }

    @Override
    public boolean getNull(int index) {
        // check validity of the array
        return false;
    }

    @Override
    public byte getByte(int index) {
        checkNotNull(inner, "inner");
        return FFI.FFIArray_get_i8(inner, index);
    }

    @Override
    public short getShort(int index) {
        checkNotNull(inner, "inner");
        return FFI.FFIArray_get_i16(inner, index);
    }

    @Override
    public int getInt(int index) {
        checkNotNull(inner, "inner");
        if (isDate || isTimestamp) {
            return FFI.FFIArray_get_storage_i32(inner, index);
        } else {
            return FFI.FFIArray_get_i32(inner, index);
        }
    }

    @Override
    public long getLong(int index) {
        checkNotNull(inner, "inner");
        if (isDate || isTimestamp) {
            return FFI.FFIArray_get_storage_i64(inner, index);
        } else {
            return FFI.FFIArray_get_i64(inner, index);
        }
    }

    @Override
    public boolean getBool(int index) {
        return false;
    }

    @Override
    public float getFloat(int index) {
        checkNotNull(inner, "inner");
        return FFI.FFIArray_get_f32(inner, index);
    }

    @Override
    public double getDouble(int index) {
        checkNotNull(inner, "inner");
        return FFI.FFIArray_get_f64(inner, index);
    }

    @Override
    public String getUTF8(int index) {
        try (Memory memory = new Memory(MAX_STRING_LEN)) {
            var lenRef = new IntByReference();
            FFI.FFIArray_get_utf8(inner, index, memory, lenRef);
            var written = memory.getByteArray(0, lenRef.getValue());
            return new String(written, StandardCharsets.UTF_8);
        }
    }

    @Override
    public byte[] getBinary(int index) {
        try (Memory memory = new Memory(MAX_STRING_LEN)) {
            var lenRef = new IntByReference();
            FFI.FFIArray_get_utf8(inner, index, memory, lenRef);
            return memory.getByteArray(0, lenRef.getValue());
        }
    }
}
