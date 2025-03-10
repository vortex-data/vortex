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

import dev.vortex.api.Array;
import dev.vortex.api.ArrayStream;
import dev.vortex.jni.FFI;

public final class NativeArrayStream extends BaseWrapped<FFI.FFIArrayStream> implements ArrayStream {

    public NativeArrayStream(FFI.FFIArrayStream inner) {
        super(inner);
    }

    @Override
    public Array getCurrent() {
        checkNotNull(inner, "inner");
        var array = FFI.FFIArrayStream_current(inner);
        return new NativeArray(array);
    }

    @Override
    public boolean next() {
        checkNotNull(inner, "inner");
        return FFI.FFIArrayStream_next(inner);
    }

    @Override
    public void close() {
        checkNotNull(inner, "inner");
        FFI.FFIArrayStream_free(inner);
        inner = null;
    }
}
