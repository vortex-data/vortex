// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.bench;

import java.nio.file.Path;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;

/** Local Spark session and dataset writer shared by the benchmarks. */
final class BenchmarkSparkSession {
    private BenchmarkSparkSession() {}

    static SparkSession create(String appName) {
        return SparkSession.builder()
                .appName(appName)
                .master("local[4]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.ui.enabled", "false")
                .config("spark.sql.shuffle.partitions", "4")
                .getOrCreate();
    }

    /**
     * Writes {@code rows} rows spread over {@code fileCount} Vortex files. Each row carries an integer id, a string,
     * and a double so scans touch several encodings.
     */
    static void writeDataset(SparkSession spark, Path output, long rows, int fileCount) {
        spark.range(0, rows)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id % 1000 as string)) as value",
                        "cast(id as double) / 7.0 as measure")
                .repartition(fileCount)
                .write()
                .format("vortex")
                .mode(SaveMode.Overwrite)
                .save(output.toString());
    }
}
