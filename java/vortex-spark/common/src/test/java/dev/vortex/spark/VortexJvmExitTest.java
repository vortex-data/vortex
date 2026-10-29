// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

/** A JVM that wrote Vortex files must still exit once {@code main} returns. */
@DisabledOnOs(value = OS.WINDOWS, disabledReason = "native workers keep the jni crate's non-daemon attachment there")
public final class VortexJvmExitTest {

    @TempDir
    Path tempDir;

    @Test
    public void jvmExitsAfterWrites() throws Exception {
        Path log = tempDir.resolve("child.log");
        Process child = new ProcessBuilder(List.of(
                        Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                        "-Xmx512m",
                        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
                        "--add-opens=java.base/java.nio=ALL-UNNAMED",
                        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
                        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
                        "-cp",
                        System.getProperty("java.class.path"),
                        VortexJvmExitTest.class.getName(),
                        tempDir.resolve("table").toUri().toString()))
                .redirectErrorStream(true)
                .redirectOutput(log.toFile())
                .start();
        boolean exited = child.waitFor(120, TimeUnit.SECONDS);
        if (!exited) {
            child.destroyForcibly().waitFor();
        }
        assertTrue(exited, "the JVM stayed alive after main returned; see " + log);
        assertEquals(0, child.exitValue(), "the child failed; see " + log);
    }

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .getOrCreate();
        spark.range(100_000)
                .selectExpr("id", "cast(id as string) as s")
                .write()
                .format("vortex")
                .save(args[0]);
        if (spark.read().format("vortex").load(args[0]).count() != 100_000) {
            throw new IllegalStateException("wrong row count");
        }
        spark.stop();
    }
}
