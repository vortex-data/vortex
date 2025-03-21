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

import com.google.protobuf.ByteString;
import com.google.protobuf.NullValue;
import dev.vortex.proto.ScalarProtos;

final class Scalars {
    private Scalars() {}

    static ScalarProtos.Scalar nullNull() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.nullType())
                .build();
    }

    static ScalarProtos.Scalar bool(boolean value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setBoolValue(value)
                        .build())
                .setDtype(DTypes.bool(false))
                .build();
    }

    static ScalarProtos.Scalar nullBool() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.bool(true))
                .build();
    }

    static ScalarProtos.Scalar int8(byte value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setInt8Value(value)
                        .build())
                .setDtype(DTypes.int8(false))
                .build();
    }

    static ScalarProtos.Scalar nullInt8() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.int8(true))
                .build();
    }

    static ScalarProtos.Scalar int16(short value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setInt16Value(value)
                        .build())
                .setDtype(DTypes.int16(false))
                .build();
    }

    static ScalarProtos.Scalar nullInt16() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.int16(true))
                .build();
    }

    static ScalarProtos.Scalar int32(int value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setInt32Value(value)
                        .build())
                .setDtype(DTypes.int32(false))
                .build();
    }

    static ScalarProtos.Scalar nullInt32() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.int32(true))
                .build();
    }

    static ScalarProtos.Scalar int64(long value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setInt64Value(value)
                        .build())
                .setDtype(DTypes.int64(false))
                .build();
    }

    static ScalarProtos.Scalar nullInt64() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.int64(true))
                .build();
    }

    static ScalarProtos.Scalar float32(float value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(
                        ScalarProtos.ScalarValue.newBuilder().setF32Value(value).build())
                .setDtype(DTypes.float32(false))
                .build();
    }

    static ScalarProtos.Scalar nullFloat32() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.float32(true))
                .build();
    }

    static ScalarProtos.Scalar float64(double value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(
                        ScalarProtos.ScalarValue.newBuilder().setF64Value(value).build())
                .setDtype(DTypes.float64(false))
                .build();
    }

    static ScalarProtos.Scalar nullFloat64() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.float64(true))
                .build();
    }

    static ScalarProtos.Scalar string(String value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setStringValue(value)
                        .build())
                .setDtype(DTypes.string(false))
                .build();
    }

    static ScalarProtos.Scalar nullString() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.string(true))
                .build();
    }

    static ScalarProtos.Scalar bytes(byte[] value) {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setBytesValue(ByteString.copyFrom(value))
                        .build())
                .setDtype(DTypes.binary(false))
                .build();
    }

    static ScalarProtos.Scalar nullBytes() {
        return ScalarProtos.Scalar.newBuilder()
                .setValue(ScalarProtos.ScalarValue.newBuilder()
                        .setNullValue(NullValue.NULL_VALUE)
                        .build())
                .setDtype(DTypes.binary(true))
                .build();
    }
}
