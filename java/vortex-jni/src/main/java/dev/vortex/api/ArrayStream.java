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
package dev.vortex.api;

public interface ArrayStream extends AutoCloseable {
    Array getCurrent();

    DType getDataType();

    /**
     * Fetch the next element of the stream.
     * <p>
     * The value will be available via {@link #getCurrent()}. If the stream is finished, this will return false.
     * <p>
     * It is an error to call this method if a previous invocation returned false.
     */
    boolean next();

    @Override
    void close();
}
