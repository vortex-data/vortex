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

import dev.vortex.spark.VortexFilePartition;
import java.io.Serializable;
import org.apache.spark.sql.catalyst.InternalRow;
import org.apache.spark.sql.connector.read.InputPartition;
import org.apache.spark.sql.connector.read.PartitionReader;
import org.apache.spark.sql.connector.read.PartitionReaderFactory;
import org.apache.spark.sql.vectorized.ColumnarBatch;

/**
 * A {@link PartitionReaderFactory} for Vortex file partitions.
 */
enum ReaderFactory implements PartitionReaderFactory, Serializable {
    INSTANCE;

    private static final boolean SUPPORTS_COLUMNAR_READS = true;

    @Override
    public PartitionReader<InternalRow> createReader(InputPartition partition) {
        throw new UnsupportedOperationException("ReaderFactory only supports columnar reads");
    }

    @Override
    public PartitionReader<ColumnarBatch> createColumnarReader(InputPartition partition) {
        return new VortexPartitionReader((VortexFilePartition) partition);
    }

    @Override
    public boolean supportColumnarReads(InputPartition partition) {
        return SUPPORTS_COLUMNAR_READS;
    }
}
