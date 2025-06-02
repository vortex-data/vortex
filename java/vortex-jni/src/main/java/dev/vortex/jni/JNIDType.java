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
import com.google.common.collect.Lists;
import dev.vortex.api.DType;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;

public final class JNIDType implements DType {
    private final boolean shouldFree;
    private OptionalLong pointer;

    public JNIDType(long pointer, boolean shouldFree) {
        Preconditions.checkArgument(pointer > 0, "Invalid pointer address: " + pointer);
        this.pointer = OptionalLong.of(pointer);
        this.shouldFree = shouldFree;
    }

    @Override
    public Variant getVariant() {
        return Variant.from(NativeDTypeMethods.getVariant(pointer.getAsLong()));
    }

    @Override
    public boolean isNullable() {
        return NativeDTypeMethods.isNullable(pointer.getAsLong());
    }

    @Override
    public List<String> getFieldNames() {
        return NativeDTypeMethods.getFieldNames(pointer.getAsLong());
    }

    @Override
    public List<DType> getFieldTypes() {
        return Lists.transform(
                NativeDTypeMethods.getFieldTypes(pointer.getAsLong()), typePtr -> new JNIDType(typePtr, false));
    }

    @Override
    public DType getElementType() {
        // How to propagate the Borrow checker rules to Java side.
        return new JNIDType(NativeDTypeMethods.getElementType(pointer.getAsLong()), false);
    }

    @Override
    public boolean isDate() {
        return NativeDTypeMethods.isDate(pointer.getAsLong());
    }

    @Override
    public boolean isTime() {
        return NativeDTypeMethods.isTime(pointer.getAsLong());
    }

    @Override
    public boolean isTimestamp() {
        return NativeDTypeMethods.isTimestamp(pointer.getAsLong());
    }

    @Override
    public TimeUnit getTimeUnit() {
        return TimeUnit.from(NativeDTypeMethods.getTimeUnit(pointer.getAsLong()));
    }

    @Override
    public Optional<String> getTimeZone() {
        return Optional.ofNullable(NativeDTypeMethods.getTimeZone(pointer.getAsLong()));
    }

    @Override
    public boolean isDecimal() {
        return NativeDTypeMethods.isDecimal(pointer.getAsLong());
    }

    @Override
    public int getPrecision() {
        return NativeDTypeMethods.getDecimalPrecision(pointer.getAsLong());
    }

    @Override
    public byte getScale() {
        return NativeDTypeMethods.getDecimalScale(pointer.getAsLong());
    }

    @Override
    public void close() {
        if (shouldFree) {
            NativeDTypeMethods.free(pointer.getAsLong());
            pointer = OptionalLong.empty();
        }
    }
}
