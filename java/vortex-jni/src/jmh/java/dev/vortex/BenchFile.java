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
package dev.vortex;

import com.jakewharton.nopen.annotation.Open;
import dev.vortex.api.File;
import dev.vortex.api.Files;
import java.net.URI;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

@BenchmarkMode(value = Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Thread)
@Open
public class BenchFile {
    static final URI FILE =
            URI.create("s3a://vortex-iceberg-dev/warehouse/db/trips/data/202409-citibike-tripdata_2.vortex");
    static final String AWS_ACCESS_KEY = System.getenv("AWS_ACCESS_KEY");
    static final String AWS_SECRET_KEY = System.getenv("AWS_SECRET_KEY");
    static final Map<String, String> PROPS = Map.of(
            "aws_access_key_id", AWS_ACCESS_KEY,
            "aws_secret_access_key", AWS_SECRET_KEY);

    File opened;

    @Benchmark
    public void open() {
        opened = Files.open(FILE, PROPS);
    }

    @TearDown(Level.Invocation)
    public void tearDown() {
        if (opened != null) {
            opened.close();
        }
    }
}
