// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Only files that end with {@code .vortex} belong to a Vortex dataset.
 *
 * <p>Spark's file index keeps {@code _metadata} and {@code _common_metadata}, and it keeps every extension it does not
 * recognise, so the connector is the only thing that can decide what belongs to the dataset. It skips every other
 * file, as the connector before the file-source rewrite did. These tests hold the paths that see the listing — schema
 * inference, scan statistics, {@code COUNT(*)} pushdown, and the V1 and V2 scans — to the same answer.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexFileExtensionTest {
    private static final String USE_V1_SOURCE_LIST = "spark.sql.sources.useV1SourceList";

    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexFileExtensionTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.adaptive.enabled", "false")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    @Test
    void schemaInferenceRejectsADirectoryOfNonVortexFiles() throws IOException {
        Path output = tempDir.resolve("renamed");
        write(output, 20);
        renameEveryVortexFile(output, ".dat");

        Exception failure = assertThrows(
                Exception.class,
                () -> spark.read().format("vortex").load(output.toString()).schema());

        assertTrue(rootMessage(failure).contains(".vortex"), rootMessage(failure));
    }

    @ParameterizedTest(name = "useV1SourceList={0}, aggregatePushdown={1}")
    @CsvSource({"'',true", "'',false", "vortex,true", "vortex,false"})
    void aStrayNonVortexFileIsSkipped(String useV1SourceList, boolean aggregatePushdown) throws IOException {
        Path output = tempDir.resolve("stray-" + useV1SourceList + "-" + aggregatePushdown);
        write(output, 20);
        // Spark's listing hides names that begin with `_` or `.`, and keeps everything else.
        Files.write(output.resolve("stray.dat"), new byte[] {1, 2, 3});

        String previous = spark.conf().get(USE_V1_SOURCE_LIST);
        spark.conf().set(USE_V1_SOURCE_LIST, useV1SourceList);
        try {
            Dataset<Row> data = spark.read()
                    .format("vortex")
                    .option("vortex.aggregatePushdown", aggregatePushdown)
                    .load(output.toString());

            assertEquals(20, data.count());
            assertEquals(20, data.collectAsList().size());
            assertEquals(10, data.filter("id >= 10").count());
        } finally {
            spark.conf().set(USE_V1_SOURCE_LIST, previous);
        }
    }

    @Test
    void aDirectPathToANonVortexFileIsRejected() throws IOException {
        Path file = tempDir.resolve("bare.dat");
        Files.write(file, new byte[] {1, 2, 3});

        Exception failure = assertThrows(
                Exception.class,
                () -> spark.read().format("vortex").load(file.toString()).count());

        assertTrue(rootMessage(failure).contains(".vortex"), rootMessage(failure));
    }

    @Test
    void anUppercaseExtensionIsStillAVortexFile() throws IOException {
        Path output = tempDir.resolve("uppercase");
        write(output, 15);
        renameEveryVortexFile(output, ".VORTEX");

        assertEquals(15, spark.read().format("vortex").load(output.toString()).count());
    }

    private void write(Path output, int rows) {
        spark.range(0, rows)
                .selectExpr("cast(id as int) as id", "concat('value_', cast(id as string)) as value")
                .repartition(2)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());
    }

    private static void renameEveryVortexFile(Path directory, String extension) throws IOException {
        try (Stream<Path> files = Files.list(directory)) {
            for (Path file : files.toList()) {
                String name = file.getFileName().toString();
                if (name.endsWith(".vortex")) {
                    Files.move(file, file.resolveSibling(name.replace(".vortex", extension)));
                }
            }
        }
    }

    private static String rootMessage(Throwable failure) {
        StringBuilder messages = new StringBuilder();
        for (Throwable cause = failure; cause != null; cause = cause.getCause()) {
            messages.append(cause.getMessage()).append('\n');
        }
        return messages.toString();
    }
}
