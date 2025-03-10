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

import static com.google.common.base.Preconditions.checkArgument;

import dev.vortex.api.Array;
import dev.vortex.api.DType;
import org.apache.spark.sql.vectorized.ColumnarBatch;

public final class VortexColumnarBatch extends ColumnarBatch {
    private VortexColumnarBatch(VortexColumnVector[] columns, int numRows) {
        super(columns, numRows);
    }

    public static VortexColumnarBatch of(Array array) {
        var dataType = array.getDataType();
        checkArgument(
                dataType.getVariant() == DType.Variant.STRUCT,
                "VortexColumnarBatch can only be built from STRUCT type array");

        var columns = new VortexColumnVector[array.getDataType().getFieldNames().size()];
        for (int i = 0; i < columns.length; i++) {
            var field = array.getField(i);
            columns[i] = new VortexColumnVector(field);
        }

        // NOTE: casting from long -> int may fail.
        var len = array.getLen();
        checkArgument(len <= Integer.MAX_VALUE, "array len overflows Integer.MAX_VALUE");
        return new VortexColumnarBatch(columns, (int) len);
    }
}
