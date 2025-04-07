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

public final class NativeLogging {
    static {
        NativeLoader.loadJni();
    }

    private NativeLogging() {}

    public static final int ERROR = 0;
    public static final int WARN = 1;
    public static final int INFO = 2;
    public static final int DEBUG = 3;
    public static final int TRACE = 4;

    /**
     * Initialize logging to the desired level. Must be one of:
     * <ul>
     *  <li>{@link #ERROR}</li>
     *  <li>{@link #WARN}</li>
     *  <li>{@link #INFO}</li>
     *  <li>{@link #DEBUG}</li>
     *  <li>{@link #TRACE}</li>
     * </ul>
     */
    public static native void initLogging(int level);
}
