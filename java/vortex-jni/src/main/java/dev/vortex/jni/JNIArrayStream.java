package dev.vortex.jni;

import dev.vortex.api.Array;
import dev.vortex.api.ArrayStream;
import dev.vortex.api.DType;

import java.util.OptionalLong;

public final class JNIArrayStream implements ArrayStream {
    private OptionalLong pointer;

    public JNIArrayStream(long pointer) {
        this.pointer = OptionalLong.of(pointer);
    }

    @Override
    public Array getCurrent() {
        return new JNIArray(NativeArrayStreamMethods.take(pointer.getAsLong()));
    }

    @Override
    public DType getDataType() {
        return new JNIDType(NativeArrayStreamMethods.getDType(pointer.getAsLong()), false);
    }

    @Override
    public boolean next() {
        return NativeArrayStreamMethods.hasNext(pointer.getAsLong());
    }

    @Override
    public void close() {
        NativeArrayStreamMethods.free(pointer.getAsLong());
        pointer = OptionalLong.empty();
    }
}
