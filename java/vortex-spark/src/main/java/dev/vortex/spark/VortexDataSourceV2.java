// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import dev.vortex.api.DataSource;
import dev.vortex.jni.NativeFiles;
import dev.vortex.spark.config.HadoopUtils;
import dev.vortex.spark.read.PartitionPathUtils;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableProvider;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.sources.DataSourceRegister;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import scala.Option;

/**
 * Spark V2 data source for reading and writing Vortex files.
 *
 * <p>This class is automatically registered so it can be discovered by the Spark runtime. For reading:
 * {@link org.apache.spark.sql.SparkSession#read} and specify the format as "vortex". For writing:
 * {@link org.apache.spark.sql.Dataset#write} and specify the format as "vortex".
 */
public final class VortexDataSourceV2 implements TableProvider, DataSourceRegister {
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String PATH_KEY = "path";
    private static final String PATHS_KEY = "paths";

    private final Option<SparkSession> sparkSession;

    /**
     * Creates a new instance of the Vortex data source.
     *
     * <p>This no-argument constructor is required for Spark to instantiate the data source through reflection.
     */
    public VortexDataSourceV2() {
        this.sparkSession = SparkSession.getActiveSession();
    }

    /**
     * Infers the schema of the Vortex files specified in the options.
     *
     * <p>This method examines the last file in the provided paths to determine the schema. Currently, schema evolution
     * and merging across multiple files is not supported.
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

        // If path is not found, we report empty schema.
        // This will be replaced with whatever the DataFrame schema is
        if (paths.isEmpty()) {
            return new StructType();
        }

        var formatOptions = buildDataSourceOptions(options.asCaseSensitiveMap());

        var pathToInfer = Objects.requireNonNull(Iterables.getLast(paths));
        // If the path is a directory, scan the directory for a file and use that file
        if (!pathToInfer.endsWith(".vortex")) {
            Optional<String> firstFile =
                    NativeFiles.listFiles(VortexSparkSession.get(formatOptions), pathToInfer, formatOptions).stream()
                            .findFirst();

            if (firstFile.isEmpty()) {
                throw new RuntimeException(String.format("UNABLE_TO_INFER_SCHEMA format: %s", shortName()));
            } else {
                pathToInfer = firstFile.get();
            }
        }

        StructType dataSchema;
        {
            DataSource ds = DataSource.open(VortexSparkSession.get(formatOptions), pathToInfer, formatOptions);
            var arrowSchema = ds.arrowSchema(dev.vortex.arrow.ArrowAllocation.rootAllocator());
            StructField[] fields = arrowSchema.getFields().stream()
                    .map(f -> new StructField(
                            f.getName(), ArrowUtils.fromArrowField(f), f.isNullable(), Metadata.empty()))
                    .toArray(StructField[]::new);
            dataSchema = new StructType(fields);
        }

        // Discover partition columns from Hive-style directory paths and append them.
        Map<String, String> partitionValues = PartitionPathUtils.parsePartitionValues(pathToInfer);
        if (!partitionValues.isEmpty()) {
            Set<String> dataColumnNames = Stream.of(dataSchema.fieldNames()).collect(Collectors.toSet());
            for (Map.Entry<String, String> entry : partitionValues.entrySet()) {
                if (!dataColumnNames.contains(entry.getKey())) {
                    DataType type = PartitionPathUtils.inferPartitionColumnType(entry.getValue());
                    dataSchema = dataSchema.add(entry.getKey(), type, true);
                }
            }
        }

        return dataSchema;
    }

    /**
     * Infers partition transforms by inspecting Hive-style {@code key=value} segments in the first listed file path.
     *
     * <p>Spark calls this before {@link #getTable(StructType, Transform[], Map)} when the caller did not provide
     * explicit partitioning. Returning identity transforms here lets downstream components (notably
     * {@link dev.vortex.spark.read.VortexScanBuilder}) tell which schema columns are encoded in the directory layout
     * rather than stored inside the Vortex files, which matters for predicate pushdown.
     */
    @Override
    public Transform[] inferPartitioning(CaseInsensitiveStringMap options) {
        var paths = getPaths(options);
        if (paths.isEmpty()) {
            return new Transform[0];
        }
        var formatOptions = buildDataSourceOptions(options.asCaseSensitiveMap());
        String pathToInfer = Objects.requireNonNull(Iterables.getLast(paths));
        if (!pathToInfer.endsWith(".vortex")) {
            Optional<String> firstFile =
                    NativeFiles.listFiles(VortexSparkSession.get(formatOptions), pathToInfer, formatOptions).stream()
                            .findFirst();
            if (firstFile.isEmpty()) {
                return new Transform[0];
            }
            pathToInfer = firstFile.get();
        }
        Map<String, String> partitionValues = PartitionPathUtils.parsePartitionValues(pathToInfer);
        if (partitionValues.isEmpty()) {
            return new Transform[0];
        }
        return partitionValues.keySet().stream().map(Expressions::identity).toArray(Transform[]::new);
    }

    /**
     * Creates a Vortex table instance with the given schema and properties.
     *
     * <p>This method creates a VortexWritableTable that can be used to both read from and write to Vortex files. The
     * partitioning parameter is currently ignored.
     *
     * @param schema the table schema
     * @param partitioning table partitioning transforms
     * @param properties the table properties containing file paths and other options
     * @return a VortexTable instance for reading and writing data
     * @throws RuntimeException if required path properties are missing
     */
    @Override
    public Table getTable(StructType schema, Transform[] partitioning, Map<String, String> properties) {
        var uncased = new CaseInsensitiveStringMap(properties);
        ImmutableList<String> paths = getPaths(uncased);
        return new VortexTable(paths, schema, buildDataSourceOptions(properties), partitioning);
    }

    /**
     * Indicates whether this data source supports external metadata (schemas).
     *
     * <p>Returns true to indicate that this data source accepts external schemas, which is necessary for write
     * operations where the DataFrame provides the schema.
     *
     * @return true to accept external schemas
     */
    @Override
    public boolean supportsExternalMetadata() {
        return true;
    }

    /**
     * Returns the short name identifier for this data source.
     *
     * <p>This name is used by Spark when registering the data source and can be used in SQL queries and DataFrame read
     * operations to specify this format.
     *
     * @return the short name "vortex"
     */
    @Override
    public String shortName() {
        return "vortex";
    }

    private Map<String, String> buildDataSourceOptions(Map<String, String> properties) {
        var hadoopConf = sparkSession.get().sessionState().newHadoopConf();

        var options = ImmutableMap.<String, String>builder();
        options.putAll(properties);

        // Forward any S3-relevant properties from hadoopConf to the reader config.
        options.putAll(HadoopUtils.s3PropertiesFromHadoopConf(hadoopConf));
        // Forward any Azure-relevant properties from hadoopConf to the reader config.
        options.putAll(HadoopUtils.azurePropertiesFromHadoopConf(hadoopConf));

        return options.build();
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
