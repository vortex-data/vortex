// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import com.google.common.base.Preconditions;
import dev.vortex.api.ArrayIterator;
import dev.vortex.api.DType;
import dev.vortex.api.File;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.proto.Expressions;
import java.util.OptionalLong;

public final class JNIFile implements File {
    private OptionalLong pointer;

    public JNIFile(long pointer) {
        Preconditions.checkArgument(pointer > 0, "Invalid pointer address: " + pointer);
        this.pointer = OptionalLong.of(pointer);
    }

    @Override
    public DType getDType() {
        return new JNIDType(NativeFileMethods.dtype(pointer.getAsLong()));
    }

    @Override
    public long rowCount() {
        return NativeFileMethods.rowCount(pointer.getAsLong());
    }

    @Override
    public ArrayIterator newScan(ScanOptions options) {
        byte[] predicateProto = null;

        if (options.predicate().isPresent()) {
            predicateProto = Expressions.serialize(options.predicate().get()).toByteArray();
        }

        long[] rowIndices = options.rowIndices().orElse(null);
        long[] rowRange = options.rowRange().orElse(null);

        return new JNIArrayIterator(
                NativeFileMethods.scan(pointer.getAsLong(), options.columns(), predicateProto, rowRange, rowIndices));
    }

    @Override
    public void close() {
        if (pointer.isEmpty()) {
            return;
        }
        NativeFileMethods.close(pointer.getAsLong());
        pointer = OptionalLong.empty();
    }
}
