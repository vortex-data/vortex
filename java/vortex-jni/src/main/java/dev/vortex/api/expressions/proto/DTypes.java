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

import dev.vortex.proto.DTypeProtos;

final class DTypes {
    private DTypes() {}

    static DTypeProtos.DType nullType() {
        return DTypeProtos.DType.newBuilder()
                .setNull(DTypeProtos.Null.newBuilder().build())
                .build();
    }

    static DTypeProtos.DType bool(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setBool(DTypeProtos.Bool.newBuilder().setNullable(nullable).build())
                .build();
    }

    static DTypeProtos.DType int8(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setPrimitive(DTypeProtos.Primitive.newBuilder()
                        .setType(DTypeProtos.PType.I8)
                        .setNullable(nullable)
                        .build())
                .build();
    }

    static DTypeProtos.DType int16(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setPrimitive(DTypeProtos.Primitive.newBuilder()
                        .setType(DTypeProtos.PType.I16)
                        .setNullable(nullable)
                        .build())
                .build();
    }

    static DTypeProtos.DType int32(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setPrimitive(DTypeProtos.Primitive.newBuilder()
                        .setType(DTypeProtos.PType.I32)
                        .setNullable(nullable)
                        .build())
                .build();
    }

    static DTypeProtos.DType int64(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setPrimitive(DTypeProtos.Primitive.newBuilder()
                        .setType(DTypeProtos.PType.I64)
                        .setNullable(nullable)
                        .build())
                .build();
    }

    static DTypeProtos.DType float32(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setPrimitive(DTypeProtos.Primitive.newBuilder()
                        .setType(DTypeProtos.PType.F32)
                        .setNullable(nullable)
                        .build())
                .build();
    }

    static DTypeProtos.DType float64(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setPrimitive(DTypeProtos.Primitive.newBuilder()
                        .setType(DTypeProtos.PType.F64)
                        .setNullable(nullable)
                        .build())
                .build();
    }

    static DTypeProtos.DType string(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setUtf8(DTypeProtos.Utf8.newBuilder().setNullable(nullable).build())
                .build();
    }

    static DTypeProtos.DType binary(boolean nullable) {
        return DTypeProtos.DType.newBuilder()
                .setBinary(DTypeProtos.Binary.newBuilder().setNullable(nullable).build())
                .build();
    }
}
