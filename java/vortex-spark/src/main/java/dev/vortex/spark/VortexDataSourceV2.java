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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableList;
import java.util.Map;
import java.util.Optional;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.*;
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

    // Only create the table one time.
    private Optional<Table> cachedTable;

    public VortexDataSourceV2() {
        // Create a new one with options.
    }

    @Override
    public StructType inferSchema(CaseInsensitiveStringMap options) {
        ImmutableList.Builder<String> pathsBuilder = ImmutableList.builder();
        if (options.containsKey(PATH_KEY)) {
            pathsBuilder.add(options.get(PATH_KEY));
        } else if (options.containsKey(PATHS_KEY)) {
            try {
                pathsBuilder.add(MAPPER.readValue(options.get(PATHS_KEY), String[].class));
            } catch (JsonProcessingException e) {
                throw new RuntimeException("Failed to deserialize option \"paths\" as String[]", e);
            }
        }
        ImmutableList<String> paths = pathsBuilder.build();

        return StructType$.MODULE$.apply(ImmutableList.of(
                StructField.apply("simple_field", StringType$.MODULE$, true, new MetadataBuilder().build())));
    }

    // Infer schema using one of the files instead.
    // We can load one of the files, open a handle, and see about caching the schema names here.

    @Override
    public Table getTable(StructType schema, Transform[] _partitioning, Map<String, String> _properties) {
        return new VortexTable("my_new_table", schema);
    }

    @Override
    public String shortName() {
        return "vortex";
    }
}
