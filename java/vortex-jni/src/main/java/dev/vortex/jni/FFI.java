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

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.PointerType;

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

    // DType interactions.
    public static native FFIDType FFIDType_new(char variant, boolean nullable);

    public static native FFIDType FFIDType_new_list(FFIDType elementType, boolean nullable);

    public static native FFIDType FFIDType_new_struct(Pointer names, Pointer types, int len, boolean nullable);

    // File interactions
    public static native FFIFile FFIFile_open(String path);

    public static native void FFIFile_free(FFIFile file);

    public static native FFIArrayStream FFIFile_scan(FFIFile file);

    // ArrayStream interaction
    public static native boolean FFIArrayStream_next(FFIArrayStream stream);

    public static native FFIArray FFIArrayStream_current(FFIArrayStream stream);

    public static native void FFIArrayStream_free(FFIArrayStream stream);

    /**
     * Opaque pointer to an {@code FFIFile} from the Vortex FFI.
     */
    public static final class FFIFile extends PointerType {}

    /**
     * Opaque pointer to an {@code FFIArray} from the Vortex FFI.
     */
    public static final class FFIArray extends PointerType {}

    /**
     * Opaque pointer to an {@code FFIDType} from the Vortex FFI.
     */
    public static final class FFIDType extends PointerType {}

    /**
     * Opaque pointer to an {@code FFIArrayStream} from the Vortex FFI.
     */
    public static final class FFIArrayStream extends PointerType {}
}
