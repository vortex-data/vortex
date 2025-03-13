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
package dev.vortex.jni;

import com.sun.jna.*;
import com.sun.jna.ptr.IntByReference;

/**
 * Bindings from the {@code vortex-ffi} C ABI to Java using JNA.
 */
public final class FFI {
    static {
        Native.register("vortex_ffi");
    }

    // Array interactions
    public static native long FFIArray_len(FFIArray array);

    public static native FFIDType FFIArray_dtype(FFIArray array);

    public static native void FFIArray_free(FFIArray array);

    public static native FFIArray FFIArray_slice(FFIArray array, int start, int stop);

    public static native boolean FFIArray_is_null(FFIArray array, int index);

    public static native int FFIArray_null_count(FFIArray array);

    public static native FFIArray FFIArray_get_field(FFIArray array, int index);

    public static native byte FFIArray_get_u8(FFIArray array, int index);

    public static native short FFIArray_get_u16(FFIArray array, int index);

    public static native int FFIArray_get_u32(FFIArray array, int index);

    public static native long FFIArray_get_u64(FFIArray array, int index);

    public static native byte FFIArray_get_i8(FFIArray array, int index);

    public static native short FFIArray_get_i16(FFIArray array, int index);

    public static native int FFIArray_get_i32(FFIArray array, int index);

    public static native long FFIArray_get_i64(FFIArray array, int index);

    // Extension type accessors
    public static native byte FFIArray_get_storage_u8(FFIArray array, int index);

    public static native short FFIArray_get_storage_u16(FFIArray array, int index);

    public static native int FFIArray_get_storage_u32(FFIArray array, int index);

    public static native long FFIArray_get_storage_u64(FFIArray array, int index);

    public static native byte FFIArray_get_storage_i8(FFIArray array, int index);

    public static native short FFIArray_get_storage_i16(FFIArray array, int index);

    public static native int FFIArray_get_storage_i32(FFIArray array, int index);

    public static native long FFIArray_get_storage_i64(FFIArray array, int index);

    // TODO(aduffy): better f16?
    public static native short FFIArray_get_f16(FFIArray array, int index);

    public static native float FFIArray_get_f32(FFIArray array, int index);

    public static native double FFIArray_get_f64(FFIArray array, int index);

    public static native void FFIArray_get_utf8(FFIArray array, int index, Pointer dst, IntByReference len);

    public static native void FFIArray_get_binary(FFIArray array, int index, Pointer dst, IntByReference len);

    // DType interactions.
    public static native byte DType_get(FFIDType dtype);

    public static native boolean DType_nullable(FFIDType dtype);

    public static native FFIDType DType_new(byte variant, boolean nullable);

    public static native FFIDType DType_new_list(FFIDType elementType, boolean nullable);

    public static native FFIDType DType_new_struct(Pointer names, Pointer types, int len, boolean nullable);

    public static native int DType_field_count(FFIDType dtype);

    public static native void DType_field_name(FFIDType dtype, int index, Pointer name, IntByReference len);

    public static native FFIDType DType_field_dtype(FFIDType dtype, int index);

    public static native FFIDType DType_element_type(FFIDType dtype);

    public static native boolean DType_is_time(FFIDType dtype);

    public static native boolean DType_is_date(FFIDType dtype);

    public static native boolean DType_is_timestamp(FFIDType dtype);

    public static native byte DType_time_unit(FFIDType dtype);

    public static native void DType_time_zone(FFIDType dtype, Pointer zone, IntByReference len);

    public static native void DType_free(FFIDType dtype);

    // File interactions
    public static native FFIFile File_open(FileOpenOptions options);

    public static native FFIDType File_dtype(FFIFile file);

    public static native void File_free(FFIFile file);

    public static native FFIArrayStream File_scan(FFIFile file, FileScanOptions options);

    // ArrayStream interaction
    public static native FFIDType FFIArrayStream_dtype(FFIArrayStream stream);

    public static native boolean FFIArrayStream_next(FFIArrayStream stream);

    public static native FFIArray FFIArrayStream_current(FFIArrayStream stream);

    public static native void FFIArrayStream_free(FFIArrayStream stream);

    @Structure.FieldOrder({"path", "property_keys", "property_vals", "property_count"})
    public static final class FileOpenOptions extends Structure {
        public String path;
        public Pointer property_keys;
        public Pointer property_vals;
        public int property_count;

        public FileOpenOptions(String path, StringArray property_keys, StringArray property_vals, int property_count) {
            this.path = path;
            this.property_keys = property_keys;
            this.property_vals = property_vals;
            this.property_count = property_count;
        }
    }

    @Structure.FieldOrder({"projection", "projection_len"})
    public static final class FileScanOptions extends Structure {
        public Pointer projection;
        public int projection_len;

        public FileScanOptions(StringArray projection, int length) {
            this.projection = projection;
            this.projection_len = length;
        }
    }

    /**
     * Opaque pointer to an {@code FFIFile} from the Vortex FFI.
     */
    public static final class FFIFile extends PointerType {}

    /**
     * Opaque pointer to an {@code FFIArray} from the Vortex FFI.
     */
    public static final class FFIArray extends PointerType {}

    /**
     * Representation of the {@code DType} structure from the Vortex FFI.
     */
    public static final class FFIDType extends PointerType {}

    /**
     * Opaque pointer to an {@code FFIArrayStream} from the Vortex FFI.
     */
    public static final class FFIArrayStream extends PointerType {}
}
