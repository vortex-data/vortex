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

import com.sun.jna.StringArray;
import dev.vortex.api.ArrayStream;
import dev.vortex.api.DType;
import dev.vortex.api.File;
import dev.vortex.api.ScanOptions;
import dev.vortex.jni.FFI;
import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

final class NativeFile extends BaseWrapped<FFI.FFIFile> implements File {
    private NativeFile(FFI.FFIFile inner) {
        super(inner);
    }

    /**
     * Open a handle to a local vortex File stored at the given path.
     */
    public static NativeFile open(String path) {
        return open(Paths.get(path));
    }

    /**
     * Open a handle to a local vortex File stored at the given path.
     */
    public static NativeFile open(Path path) {
        return open(path.toUri(), Map.of());
    }

    /**
     * Open a file at the provided URI, with configuration supplied.
     */
    public static NativeFile open(URI uri, Map<String, String> properties) {
        try (StringArray keys = new StringArray(properties.keySet().toArray(new String[0]));
                StringArray values = new StringArray(properties.values().toArray(new String[0]))) {
            FFI.FileOpenOptions options = new FFI.FileOpenOptions(uri.toString(), keys, values, properties.size());
            return new NativeFile(FFI.File_open(options));
        }
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
        String[] columns = options.columns().toArray(new String[0]);
        try (StringArray columnsPtr = new StringArray(columns)) {
            var scan = FFI.File_scan(inner, new FFI.FileScanOptions(columnsPtr, columns.length));
            return new NativeArrayStream(scan);
        }
    }

    @Override
    public void close() {
        FFI.File_free(inner);
    }
}
