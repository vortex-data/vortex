// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import java.math.BigDecimal;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;

public interface Array extends AutoCloseable {
    long getLen();

    long nbytes();

    /**
     * Export to an ArrowVector. The data will now be owned by the VectorSchemaRoot after this operation.
     */
    VectorSchemaRoot exportToArrow(BufferAllocator allocator, VectorSchemaRoot reuse);

    DType getDataType();

    Array getField(int index);

    Array slice(int start, int stop);

    boolean getNull(int index);

    int getNullCount();

    byte getByte(int index);

    short getShort(int index);

    int getInt(int index);

    long getLong(int index);

    boolean getBool(int index);

    float getFloat(int index);

    double getDouble(int index);

    BigDecimal getBigDecimal(int index);

    String getUTF8(int index);

    void getUTF8_ptr_len(int index, long[] ptr, int[] len);

    byte[] getBinary(int index);

    @Override
    void close();
}
