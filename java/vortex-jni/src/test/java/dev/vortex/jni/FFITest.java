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

import org.junit.jupiter.api.Test;

import java.nio.file.Paths;

import static dev.vortex.jni.FFI.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

public final class FFITest {
    @Test
    public void testScan() {
        var path = Paths.get(".")
                .toAbsolutePath()
                .getParent()
                .getParent()
                .getParent()
                .resolve("bench-vortex/data/tpch/1/vortex_compressed/lineitem.vortex")
                .toAbsolutePath()
                .toString();
        var file = FFIFile_open(path);
        var stream = FFIFile_scan(file);

        long batchCount = 0;
        long rowCount = 0;
        while (FFIArrayStream_next(stream)) {
            var batch = FFIArrayStream_current(stream);
            var len = FFIArray_len(batch);
            rowCount += len;
            FFIArray_free(batch);

            batchCount += 1;
        }

        // Close the resources
        FFIArrayStream_free(stream);
        FFIFile_free(file);

        assertEquals(6001215, rowCount);
        assertEquals(58, batchCount);
    }
}
