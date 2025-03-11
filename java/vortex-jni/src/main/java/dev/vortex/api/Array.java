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

public interface Array extends AutoCloseable {
    long getLen();

    DType getDataType();

    Array getField(int index);

    Array slice(int start, int stop);

    boolean getNull(int index);

    int getNullCount();

    byte getByte(int index);

    short getShort(int index);

    int getInt(int index);

    long getLong(int index);

    boolean getBool(int index);

    float getFloat(int index);

    double getDouble(int index);

    String getUTF8(int index);

    byte[] getBinary(int index);

    @Override
    void close();
}
