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

public final class NativeDTypeMethods {
    static {
        NativeLoader.loadJni();
    }

    private NativeDTypeMethods() {}

    public static native void free(long pointer);

    public static native byte getVariant(long pointer);

    public static native boolean isNullable(long pointer);

    public static native List<String> getFieldNames(long pointer);

    // Returns a list of DType pointers.
    public static native List<Long> getFieldTypes(long pointer);

    public static native long getElementType(long pointer);

    public static native boolean isDate(long pointer);

    public static native boolean isTime(long pointer);

    public static native boolean isTimestamp(long pointer);

    public static native byte getTimeUnit(long pointer);

    public static native String getTimeZone(long pointer);
}
