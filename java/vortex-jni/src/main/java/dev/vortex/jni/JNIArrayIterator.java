// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import com.google.common.base.Preconditions;
import dev.vortex.api.Array;
import dev.vortex.api.ArrayIterator;
import dev.vortex.api.DType;
import java.util.Optional;
import java.util.OptionalLong;

public final class JNIArrayIterator implements ArrayIterator {
    private OptionalLong pointer;
    private Optional<Array> next;

    public JNIArrayIterator(long pointer) {
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
        return new JNIDType(NativeArrayIteratorMethods.getDType(pointer.getAsLong()), false);
    }

    @Override
    public void close() {
        if (pointer.isEmpty()) {
            return;
        }

        NativeArrayIteratorMethods.free(pointer.getAsLong());
        pointer = OptionalLong.empty();
        next = Optional.empty();
    }

    private void advance() {
        long next = NativeArrayIteratorMethods.take(pointer.getAsLong());
        if (next <= 0) {
            this.next = Optional.empty();
        } else {
            this.next = Optional.of(new JNIArray(next));
        }
    }
}
