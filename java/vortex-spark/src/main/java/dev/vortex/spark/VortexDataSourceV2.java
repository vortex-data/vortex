// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import dev.vortex.api.File;
import dev.vortex.api.Files;
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

    /**
     * Creates a new instance of the Vortex data source.
     * <p>
     * This no-argument constructor is required for Spark to instantiate the data source
     * through reflection.
     */
    public VortexDataSourceV2() {}

    /**
     * Infers the schema of the Vortex files specified in the options.
     * <p>
     * This method examines the last file in the provided paths to determine the schema.
     * Currently, schema evolution and merging across multiple files is not supported.
     *
     * @param options the data source options containing file paths
     * @return the inferred Spark SQL schema
     * @throws RuntimeException if required path options are missing
     * @throws RuntimeException if there's an error reading the file or converting the schema
     */
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

    /**
     * Creates a Vortex table instance with the given schema and properties.
     * <p>
     * This method creates a VortexTable that can be used to read data from the specified
     * Vortex files. The partitioning parameter is currently ignored.
     *
     * @param schema the table schema
     * @param _partitioning table partitioning transforms (currently ignored)
     * @param properties the table properties containing file paths and other options
     * @return a VortexTable instance for reading the data
     * @throws RuntimeException if required path properties are missing
     */
    @Override
    public Table getTable(StructType schema, Transform[] _partitioning, Map<String, String> properties) {
        var uncased = new CaseInsensitiveStringMap(properties);

        var paths = getPaths(uncased);
        var columns = ImmutableList.<Column>builder()
                .add(CatalogV2Util.structTypeToV2Columns(schema))
                .build();
        return new VortexTable(paths, columns);
    }

    /**
     * Returns the short name identifier for this data source.
     * <p>
     * This name is used by Spark when registering the data source and can be used
     * in SQL queries and DataFrame read operations to specify this format.
     *
     * @return the short name "vortex"
     */
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
