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
package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;

final class VortexScanTest {
    private static final Path BENCH_PATH = Paths.get("/Volumes/Code/vortex/bench-vortex/data/tpch/1/vortex_compressed");

    @Test
    public void testSparkRead() {
        SparkSession spark =
                SparkSession.builder().appName("test").master("local").getOrCreate();

        var filePath = BENCH_PATH.resolve("part.vortex").toAbsolutePath().toString();

        System.out.println("Loading table from " + filePath);

        var parts = spark.read().format("vortex").load(filePath);
        assertEquals(200_000L, parts.count());
    }
}
