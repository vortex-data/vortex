// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connector.write.LogicalWriteInfo;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Write options the writers cannot use fail on the driver, before Spark launches a task.
 *
 * <p>Spark's V1 overwrite deletes the old files before any {@code FileFormat} hook sees the options, so this cannot
 * keep them, but it does stop the job from starting and reports the option directly rather than as a task failure.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexWriteOptionValidationTest {
    private static final StructType SCHEMA = new StructType().add("id", DataTypes.LongType);

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexWriteOptionValidationTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    static Stream<Arguments> unusableOptions() {
        return Stream.of(
                Arguments.of("batch.size", "lots"),
                Arguments.of("vortex.session.provider", "dev.vortex.spark.NoSuchProvider"),
                Arguments.of("vortex.workerThreads", "-1"));
    }

    @ParameterizedTest
    @MethodSource("unusableOptions")
    void anUnusableOptionFailsAV1WriteOnTheDriver(String option, String value) {
        String output = tempDir.resolve("v1-" + option + "-" + value).toString();
        // A first write fixes the JVM's worker pool, after which a later value used to go unchecked.
        spark.range(10).write().format("vortex").save(output);

        assertNamesOption(option, assertThrows(IllegalArgumentException.class, () -> spark.range(10)
                .write()
                .format("vortex")
                .option(option, value)
                .mode(SaveMode.Overwrite)
                .save(output)));
    }

    @ParameterizedTest
    @MethodSource("unusableOptions")
    void anUnusableOptionFailsWhenAV2WriteIsPlanned(String option, String value) {
        // DataFrameWriter sends file sources through the V1 format, so the V2 table is driven directly.
        CaseInsensitiveStringMap tableOptions = new CaseInsensitiveStringMap(Map.of("path", tempDir.toString()));
        VortexTable table = (VortexTable) new VortexDataSourceV2().getTable(tableOptions, SCHEMA);

        assertNamesOption(
                option,
                assertThrows(IllegalArgumentException.class, () -> table.newWriteBuilder(writeInfo(option, value))));
    }

    @Test
    void anOutOfRangeBatchSizeStillFallsBackToTheDefault() {
        String output = tempDir.resolve("out-of-range-batch").toString();
        spark.range(10).write().format("vortex").option("batch.size", "100000").save(output);

        assertEquals(10, spark.read().format("vortex").load(output).count());
    }

    private static void assertNamesOption(String option, IllegalArgumentException failure) {
        assertTrue(
                failure.getMessage().toLowerCase(Locale.ROOT).contains(option.toLowerCase(Locale.ROOT)),
                failure::getMessage);
    }

    private static LogicalWriteInfo writeInfo(String option, String value) {
        return new LogicalWriteInfo() {
            @Override
            public String queryId() {
                return "query";
            }

            @Override
            public StructType schema() {
                return SCHEMA;
            }

            @Override
            public CaseInsensitiveStringMap options() {
                return new CaseInsensitiveStringMap(Map.of(option, value));
            }
        };
    }
}
