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

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.api.DType;
import dev.vortex.api.ScanOptions;
import dev.vortex.impl.NativeFile;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.Test;

public final class FFITest {
    private static final Path LINEITEM = Paths.get(".")
            .toAbsolutePath()
            .getParent()
            .getParent()
            .getParent()
            .resolve("bench-vortex/data/tpch/1/vortex_compressed/lineitem.vortex")
            .toAbsolutePath();

    @Test
    public void testDType() {
        // Provide a simple test for DType checking.
        try (NativeFile lineitem = NativeFile.open(LINEITEM.toString())) {
            try (DType dtype = lineitem.getDType()) {
                System.out.println("dtype: " + dtype);
            }
        }
    }

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

        long rowCount = 0;
        try (var file = NativeFile.open(path);
                var scan = file.newScan(ScanOptions.of())) {

            while (scan.next()) {
                try (var array = scan.getCurrent()) {
                    rowCount += array.getLen();
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Failed closing resources", e);
        }

        assertEquals(6001215L, rowCount);
    }
}
