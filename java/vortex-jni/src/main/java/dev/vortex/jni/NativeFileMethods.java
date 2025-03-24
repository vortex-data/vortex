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

import java.util.List;
import java.util.Map;

public final class NativeFileMethods {
    static {
        NativeLoader.loadJni();
    }

    private NativeFileMethods() {}

    /**
     * Open a file using the native library with the provided URI and options.
     *
     * @param uri     The URI of the file to open. e.g. "file://path/to/file".
     * @param options A map of options to provide for opening the file.
     * @return A native pointer to the opened file. This will be 0 if the open call failed.
     */
    public static native long open(String uri, Map<String, String> options);

    /**
     * Get the data type of the file associated with the given pointer.
     *
     * @param pointer The native pointer to a file. Must be a value returned by {@link #open(String, Map)}.
     * @return Native pointer to the DType of the file. This pointer is owned by the file and should not be freed.
     */
    public static native long dtype(long pointer);

    /**
     * Close the file associated with the given pointer.
     *
     * @param pointer The native pointer to a file. Must be a value returned by {@link #open(String, Map)}.
     */
    public static native void close(long pointer);

    public static native long scan(long pointer, List<String> columns, byte[] predicateProto);
}
