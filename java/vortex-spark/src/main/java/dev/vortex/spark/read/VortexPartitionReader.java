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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import dev.vortex.api.Array;
import dev.vortex.api.ArrayStream;
import dev.vortex.api.File;
import dev.vortex.api.ScanOptions;
import dev.vortex.impl.NativeFile;
import dev.vortex.spark.VortexFilePartition;
import java.io.IOException;
import java.util.Objects;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A {@link PartitionReader} that reads columnar batches out of a Vortex file into
 * Vortex memory format.
 */
final class VortexPartitionReader implements PartitionReader<ColumnarBatch> {
    private final VortexFilePartition partition;

    private File file;
    private ArrayStream arrayStream;

    private Array current;

    VortexPartitionReader(VortexFilePartition partition) {
        this.partition = partition;
        initNativeResources();
    }

    @Override
    public boolean next() {
        checkState(arrayStream != null, "arrayStream");

        if (!arrayStream.next()) {
            return false;
        }

        current = arrayStream.getCurrent();
        return true;
    }

    @Override
    public ColumnarBatch get() {
        return VortexColumnarBatch.of(checkNotNull(current, "current"));
    }

    /**
     * Initialize the Vortex File and ArrayStream resources.
     */
    void initNativeResources() {
        file = NativeFile.open(partition.getPath());
        arrayStream = file.newScan(ScanOptions.of());
    }

    @Override
    public void close() throws IOException {
        checkState(Objects.nonNull(file), "File was closed");
        checkState(Objects.nonNull(arrayStream), "ArrayStream was closed");

        current.close();
        current = null;

        arrayStream.close();
        arrayStream = null;

        file.close();
        file = null;
    }
}
