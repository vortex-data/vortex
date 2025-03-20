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
package dev.vortex.api.expressions.proto;

import com.google.common.base.Preconditions;

final class Casts {
    private Casts() {}

    static byte toByte(int value) {
        Preconditions.checkArgument(value <= Byte.MAX_VALUE && value >= Byte.MIN_VALUE);
        return (byte) value;
    }

    static byte toUnsignedByte(int value) {
        Preconditions.checkArgument(Integer.compareUnsigned(value, 255) <= 0);
        return (byte) value;
    }

    static short toShort(int value) {
        Preconditions.checkArgument(value <= Short.MAX_VALUE && value >= Short.MIN_VALUE);
        return (short) value;
    }

    static short toUnsignedShort(int value) {
        Preconditions.checkArgument(Integer.compareUnsigned(value, 65_535) <= 0);
        return (short) value;
    }
}
