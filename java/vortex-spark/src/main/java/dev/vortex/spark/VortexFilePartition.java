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

import com.google.common.collect.ImmutableList;
import java.io.Serializable;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.read.InputPartition;

/**
 * An {@link InputPartition} for reading a whole Vortex file.
 */
public final class VortexFilePartition implements InputPartition, Serializable {
    private final String path;
    private final ImmutableList<Column> columns;

    public VortexFilePartition(String path, ImmutableList<Column> columns) {
        this.path = path;
        this.columns = columns;
    }

    public String getPath() {
        return path;
    }

    public ImmutableList<Column> getColumns() {
        return columns;
    }
}
