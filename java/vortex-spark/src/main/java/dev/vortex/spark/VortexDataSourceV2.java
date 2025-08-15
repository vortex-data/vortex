// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import dev.vortex.api.File;
import dev.vortex.api.Files;
import dev.vortex.spark.read.VortexTable;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.spark.sql.connector.catalog.CatalogV2Util;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * Spark V2 data source for reading and writing Vortex files.
 * <p>
 * This class is automatically registered so it can be discovered by the Spark runtime.
 * For reading: {@link org.apache.spark.sql.SparkSession#read} and specify the format as "vortex".
 * For writing: {@link org.apache.spark.sql.Dataset#write} and specify the format as "vortex".
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
        // For write operations, the path might not exist yet
        // In that case, return an empty schema to signal Spark to use the DataFrame's schema
        var paths = getPaths(options);
        var pathToInfer = Iterables.getLast(paths);

        Path path = java.nio.file.Paths.get(pathToInfer);

        System.err.println("DEBUG: inferSchema called with path: " + pathToInfer);

        // Check if the path exists
        if (!java.nio.file.Files.exists(path)) {
            // For write operations, return empty schema (Spark will use DataFrame's schema)
            // We can't return null as that causes issues in getTable
            System.err.println("DEBUG: Path does not exist, returning empty schema");
            return new StructType();
        }

        System.err.println("DEBUG: Path exists, isDirectory=" + java.nio.file.Files.isDirectory(path));

        // If it's a directory, look for Vortex files inside
        if (java.nio.file.Files.isDirectory(path)) {
            try (var stream = java.nio.file.Files.list(path)) {
                // Find the first .vortex file in the directory
                var vortexFile =
                        stream.filter(p -> p.toString().endsWith(".vortex")).findFirst();

                if (vortexFile.isPresent()) {
                    pathToInfer = vortexFile.get().toString();
                    System.err.println("DEBUG: Found vortex file for schema: " + pathToInfer);
                } else {
                    // No vortex files found, return empty schema
                    System.err.println("DEBUG: No .vortex files found in directory");
                    return new StructType();
                }
            } catch (Exception e) {
                System.err.println("DEBUG: Exception listing directory: " + e.getMessage());
                return new StructType();
            }
        }

        try (File file = Files.open(pathToInfer)) {
            var columns = SparkTypes.toColumns(file.getDType());
            return CatalogV2Util.v2ColumnsToStructType(columns);
        }
    }

    /**
     * Creates a Vortex table instance with the given schema and properties.
     * <p>
     * This method creates a VortexWritableTable that can be used to both read from and write to
     * Vortex files. The partitioning parameter is currently ignored.
     *
     * @param schema        the table schema
     * @param _partitioning table partitioning transforms (currently ignored)
     * @param properties    the table properties containing file paths and other options
     * @return a VortexTable instance for reading and writing data
     * @throws RuntimeException if required path properties are missing
     */
    @Override
    public Table getTable(StructType schema, Transform[] _partitioning, Map<String, String> properties) {
        var uncased = new CaseInsensitiveStringMap(properties);

        var paths = getPaths(uncased);

        // Convert schema to columns
        ImmutableList<Column> columns;
        if (schema != null && schema.fields().length > 0) {
            columns = ImmutableList.<Column>builder()
                    .add(CatalogV2Util.structTypeToV2Columns(schema))
                    .build();
        } else {
            // For write operations where the path doesn't exist, inferSchema returns empty
            // But Spark still needs a valid schema for write operations
            // The actual write schema will come from LogicalWriteInfo in newWriteBuilder
            columns = ImmutableList.of();
        }

        // Support both read and write operations
        String outputPath = uncased.get(PATH_KEY);
        if (outputPath != null) {
            // This is a write operation - pass the schema along
            return new VortexTable(paths, columns, outputPath, uncased, schema);
        } else {
            return new VortexTable(paths, columns);
        }
    }

    /**
     * Indicates whether this data source supports external metadata (schemas).
     * <p>
     * Returns true to indicate that this data source accepts external schemas,
     * which is necessary for write operations where the DataFrame provides the schema.
     *
     * @return true to accept external schemas
     */
    @Override
    public boolean supportsExternalMetadata() {
        return true;
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
            String path = uncased.get(PATH_KEY);
            return expandPathToFiles(path);
        } else if (uncased.containsKey(PATHS_KEY)) {
            return decodePathsSafe(uncased.get(PATHS_KEY));
        } else {
            throw new IllegalArgumentException("Missing required option: \"path\" or \"paths\"");
        }
    }

    /**
     * Expands a path to individual Vortex files.
     * If the path is a directory, returns all .vortex files in the directory.
     * If the path is a file, returns the file itself.
     */
    private static ImmutableList<String> expandPathToFiles(String pathStr) {
        Path path = java.nio.file.Paths.get(pathStr);

        if (!java.nio.file.Files.exists(path)) {
            // For write operations, the path might not exist yet
            return ImmutableList.of(pathStr);
        }

        if (java.nio.file.Files.isDirectory(path)) {
            try (var stream = java.nio.file.Files.list(path)) {
                List<String> vortexFiles = stream.filter(p -> p.toString().endsWith(".vortex"))
                        .map(Path::toString)
                        .sorted() // Sort for consistent ordering
                        .collect(Collectors.toList());

                if (vortexFiles.isEmpty()) {
                    // No vortex files found, return the directory (for write operations)
                    return ImmutableList.of(pathStr);
                }

                return ImmutableList.copyOf(vortexFiles);
            } catch (Exception e) {
                // Fall back to the original path
                return ImmutableList.of(pathStr);
            }
        } else {
            // Single file
            return ImmutableList.of(pathStr);
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
