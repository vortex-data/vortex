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
package dev.vortex.spark.read;

import dev.vortex.api.Array;
import dev.vortex.spark.SparkTypes;
import org.apache.spark.sql.types.Decimal;
import org.apache.spark.sql.vectorized.ColumnVector;
import org.apache.spark.sql.vectorized.ColumnarArray;
import org.apache.spark.sql.vectorized.ColumnarMap;
import org.apache.spark.unsafe.types.UTF8String;

public final class VortexColumnVector extends ColumnVector {
    private final Array array;

    public VortexColumnVector(Array array) {
        super(SparkTypes.toDataType(array.getDataType()));
        this.array = array;
    }

    @Override
    public void close() {
        try {
            array.close();
        } catch (Exception e) {
            throw new RuntimeException("Failed to close Vortex Array", e);
        }
    }

    @Override
    public boolean hasNull() {
        return array.getDataType().isNullable();
    }

    @Override
    public int numNulls() {
        // TODO(aduffy): Ad FFI function for mask.
        return 0;
    }

    @Override
    public boolean isNullAt(int rowId) {
        // TODO(aduffy): Validity check
        return array.getNull(rowId);
    }

    @Override
    public boolean getBoolean(int rowId) {
        // TODO(aduffy): perform unsafe access to the rows
        return array.getBool(rowId);
    }

    @Override
    public byte getByte(int rowId) {
        // TODO(aduffy): implement using the binary return values here instead.
        return array.getByte(rowId);
    }

    @Override
    public short getShort(int rowId) {
        return array.getShort(rowId);
    }

    @Override
    public int getInt(int rowId) {
        return array.getInt(rowId);
    }

    @Override
    public long getLong(int rowId) {
        return array.getLong(rowId);
    }

    @Override
    public float getFloat(int rowId) {
        return array.getFloat(rowId);
    }

    @Override
    public double getDouble(int rowId) {
        return array.getDouble(rowId);
    }

    @Override
    public ColumnarArray getArray(int rowId) {
        // TODO(aduffy): figure out array FFI support
        throw new UnsupportedOperationException("TODO: implement getArray");
    }

    @Override
    public ColumnarMap getMap(int ordinal) {
        // TODO(aduffy): figure out struct FFI support
        throw new UnsupportedOperationException("TODO: implement getMap");
    }

    @Override
    public Decimal getDecimal(int rowId, int precision, int scale) {
        throw new UnsupportedOperationException("Vortex does not support DECIMAL types");
    }

    @Override
    public UTF8String getUTF8String(int rowId) {
        return UTF8String.fromString(array.getUTF8(rowId));
    }

    @Override
    public byte[] getBinary(int rowId) {
        return array.getBinary(rowId);
    }

    @Override
    public ColumnVector getChild(int ordinal) {
        throw new UnsupportedOperationException("TODO(aduffy): implement getChild");
    }
}
