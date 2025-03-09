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

import dev.vortex.api.ArrayStream;
import dev.vortex.api.DType;
import dev.vortex.api.File;
import dev.vortex.api.ScanOptions;
import dev.vortex.jni.FFI;

public final class NativeFile extends BaseWrapped<FFI.FFIFile> implements File {
    private NativeFile(FFI.FFIFile inner) {
        super(inner);
    }

    /**
     * Open a file at the provided path on the filesystem.
     */
    public static NativeFile open(String path) {
        return new NativeFile(FFI.File_open(path));
    }

    @Override
    public DType getDType() {
        return new NativeDType(FFI.File_dtype(inner));
    }

    /**
     * Create a new ScanBuilder with all of the relevant settings.
     */
    @Override
    public ArrayStream newScan(ScanOptions options) {
        var scan = FFI.File_scan(inner);
        return new NativeArrayStream(scan);
    }

    @Override
    public void close() {
        FFI.File_free(inner);
    }
}
