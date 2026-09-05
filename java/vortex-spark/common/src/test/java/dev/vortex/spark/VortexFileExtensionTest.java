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

/**
 * Every file in a Vortex dataset must end with {@code .vortex}.
 *
 * <p>Spark's file index keeps {@code _metadata} and {@code _common_metadata}, and it keeps every extension it does not
 * recognise, so the connector is the only thing that can decide what belongs to the dataset. These tests hold the three
 * paths that see the listing — schema inference, scan statistics, and the scan itself — to the same answer.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexFileExtensionTest {
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

    @Test
    void aStrayNonVortexFileFailsTheScan() throws IOException {
        Path output = tempDir.resolve("stray");
        write(output, 20);
        // Spark's listing hides names that begin with `_` or `.`, and keeps everything else.
        Files.write(output.resolve("stray.dat"), new byte[] {1, 2, 3});

        // Inference still finds a Vortex file, so the stray one survives until the scan reaches it.
        Dataset<Row> data = spark.read().format("vortex").load(output.toString());

        Exception failure = assertThrows(Exception.class, data::count);
        assertTrue(rootMessage(failure).contains("stray.dat"), rootMessage(failure));
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
