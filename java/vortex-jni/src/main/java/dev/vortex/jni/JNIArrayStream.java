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

import com.google.common.base.Preconditions;
import dev.vortex.api.Array;
import dev.vortex.api.ArrayStream;
import dev.vortex.api.DType;
import java.util.Optional;
import java.util.OptionalLong;

public final class JNIArrayStream implements ArrayStream {
    private OptionalLong pointer;
    private Optional<Array> next;

    public JNIArrayStream(long pointer) {
        Preconditions.checkArgument(pointer > 0, "Invalid pointer address: " + pointer);
        this.pointer = OptionalLong.of(pointer);
        advance();
    }

    @Override
    public boolean hasNext() {
        return next.isPresent();
    }

    @Override
    public Array next() {
        Array array = this.next.get();
        advance();
        return array;
    }

    @Override
    public DType getDataType() {
        return new JNIDType(NativeArrayStreamMethods.getDType(pointer.getAsLong()), false);
    }

    @Override
    public void close() {
        NativeArrayStreamMethods.free(pointer.getAsLong());
        pointer = OptionalLong.empty();
        next = Optional.empty();
    }

    private void advance() {
        long next = NativeArrayStreamMethods.take(pointer.getAsLong());
        if (next <= 0) {
            this.next = Optional.empty();
        } else {
            this.next = Optional.of(new JNIArray(next));
        }
    }
}
