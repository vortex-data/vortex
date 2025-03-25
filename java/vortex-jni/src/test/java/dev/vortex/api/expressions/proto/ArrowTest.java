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

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.api.*;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.junit.jupiter.api.Test;

public final class ArrowTest {
    @Test
    public void testReadAsArrow() {
        try (RootAllocator rootAllocator = new RootAllocator(1024 * 1024 * 1024)) {
            // See if we can read as Arrow instead.
            try (File file =
                    Files.open("/Volumes/Code/vortex/bench-vortex/data/tpch/1/vortex_compressed/region.vortex")) {
                try (ArrayStream chunkStream = file.newScan(ScanOptions.of())) {
                    while (chunkStream.hasNext()) {
                        Array chunk = chunkStream.next();
                        try (VectorSchemaRoot arrow = chunk.exportToArrow(rootAllocator)) {
                            assertEquals(5, arrow.getRowCount());
                        }
                    }
                }
            }
        }
    }
}
