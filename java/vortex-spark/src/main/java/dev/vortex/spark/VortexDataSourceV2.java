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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import dev.vortex.api.File;
import dev.vortex.impl.Files;
import dev.vortex.spark.read.VortexTable;
import java.util.Map;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Spark V2 data source for reading Vortex files.
 * <p>
 * This class is automatically registered so it can be discovered by the Spark runtime. The current way to
 * use it is to do {@link org.apache.spark.sql.SparkSession#read} and specify the format as "vortex".
 */
public final class VortexDataSourceV2 implements TableProvider, DataSourceRegister {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String PATH_KEY = "path";
    private static final String PATHS_KEY = "paths";

    public VortexDataSourceV2() {}

    @Override
    public StructType inferSchema(CaseInsensitiveStringMap options) {
        // Look at the last file in the listing and dump its schema.
        // TODO(aduffy): support schema evolution/merging?
        var pathToInfer = Iterables.getLast(getPaths(options));

        try (File file = Files.open(pathToInfer)) {
            var columns = SparkTypes.toColumns(file.getDType());
            return CatalogV2Util.v2ColumnsToStructType(columns);
        }
    }

    @Override
    public Table getTable(StructType schema, Transform[] _partitioning, Map<String, String> properties) {
        var uncased = new CaseInsensitiveStringMap(properties);

        var paths = getPaths(uncased);
        var columns = ImmutableList.<Column>builder()
                .add(CatalogV2Util.structTypeToV2Columns(schema))
                .build();
        return new VortexTable(paths, columns);
    }

    @Override
    public String shortName() {
        return "vortex";
    }

    private static ImmutableList<String> getPaths(CaseInsensitiveStringMap uncased) {
        if (uncased.containsKey(PATH_KEY)) {
            return ImmutableList.of(uncased.get(PATH_KEY));
        } else if (uncased.containsKey(PATHS_KEY)) {
            return decodePathsSafe(uncased.get(PATHS_KEY));
        } else {
            throw new IllegalArgumentException("Missing required option: \"path\" or \"paths\"");
        }
    }

    private static ImmutableList<String> decodePathsSafe(String pathsJson) {
        try {
            return ImmutableList.copyOf(MAPPER.readValue(pathsJson, String[].class));
        } catch (Exception e) {
            throw new IllegalArgumentException("Failed to decode \"paths\" option", e);
        }
    }
}
