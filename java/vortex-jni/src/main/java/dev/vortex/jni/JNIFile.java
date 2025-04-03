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
import dev.vortex.api.ArrayStream;
import dev.vortex.api.DType;
import dev.vortex.api.File;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.expressions.proto.ExpressionProtoSerializer;
import java.util.OptionalLong;

public final class JNIFile implements File {
    private OptionalLong pointer;

    public JNIFile(long pointer) {
        Preconditions.checkArgument(pointer > 0, "Invalid pointer address: " + pointer);
        this.pointer = OptionalLong.of(pointer);
    }

    @Override
    public DType getDType() {
        return new JNIDType(NativeFileMethods.dtype(pointer.getAsLong()), false);
    }

    @Override
    public ArrayStream newScan(ScanOptions options) {
        byte[] predicateProto = null;

        if (options.predicate().isPresent()) {
            predicateProto = ExpressionProtoSerializer.serialize(
                            options.predicate().get())
                    .toByteArray();
        }

        long[] rowIndices = options.rowIndices().orElse(null);

        return new JNIArrayStream(
                NativeFileMethods.scan(pointer.getAsLong(), options.columns(), predicateProto, rowIndices));
    }

    @Override
    public void close() {
        NativeFileMethods.close(pointer.getAsLong());
        pointer = OptionalLong.empty();
    }
}
