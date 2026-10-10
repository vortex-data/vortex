// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Column pruning and resolution of user-specified read schemas.
 *
 * <p>A query over partition columns alone, and a count Spark declines to push down, both leave the read data schema
 * empty. The scan must still push an empty projection, or it pulls every column off storage to answer them.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexProjectionTest {
    private SparkSession spark;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() {
        spark = SparkSession.builder()
                .appName("VortexProjectionTest")
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

    @AfterEach
    void restoreActiveSession() {
        SparkSession.setActiveSession(spark);
    }

    @ParameterizedTest
    @CsvSource({"v2,false", "v2,true", "v1,false", "v1,true", "v1_rows,false", "v1_rows,true"})
    void explicitSchemaRespectsCaseSensitivity(String mode, boolean caseSensitive) {
        SparkSession session = newActiveSession();
        session.conf().set("spark.sql.sources.useV1SourceList", mode.startsWith("v1") ? "vortex" : "");
        session.conf().set("spark.sql.codegen.wholeStage", !mode.equals("v1_rows"));
        session.conf().set("spark.sql.caseSensitive", caseSensitive);
        Path output = tempDir.resolve("schema_case_" + mode + "_" + caseSensitive);
        session.range(1, 3).coalesce(1).write().format("vortex").save(output.toString());

        Dataset<Row> data = session.read()
                .format("vortex")
                .schema(new StructType().add("ID", DataTypes.LongType, true))
                .load(output.toString());
        List<Row> expected = caseSensitive
                ? List.of(RowFactory.create((Object) null), RowFactory.create((Object) null))
                : List.of(RowFactory.create(1L), RowFactory.create(2L));
        assertEquals(expected, data.orderBy("ID").collectAsList());
        assertEquals(
                caseSensitive ? List.of() : List.of(RowFactory.create(2L)),
                data.where("ID > 1").collectAsList());
    }

    @Test
    void explicitSchemaRejectsAmbiguousColumnNames() {
        SparkSession session = newActiveSession();
        session.conf().set("spark.sql.sources.useV1SourceList", "");
        session.conf().set("spark.sql.caseSensitive", true);
        Path output = tempDir.resolve("ambiguous_schema");
        session.range(1)
                .selectExpr("id", "id + 1 AS ID")
                .coalesce(1)
                .write()
                .format("vortex")
                .save(output.toString());
        session.conf().set("spark.sql.caseSensitive", false);
        Dataset<Row> data = session.read()
                .format("vortex")
                .schema(new StructType().add("id", DataTypes.LongType, true))
                .load(output.toString());

        Throwable failure = assertThrows(Exception.class, data::collectAsList);
        while (failure.getCause() != null) {
            failure = failure.getCause();
        }
        assertTrue(failure.getMessage().contains("Ambiguous columns id and ID"), failure.getMessage());
    }

    @Test
    void selectingOnlyAPartitionColumnReadsNoDataColumn() {
        Dataset<Row> data = writeAndRead("partition_only", true).select("group");

        List<Row> rows = data.collectAsList();

        assertEquals(30, rows.size());
        assertEquals(15, rows.stream().filter(row -> row.getInt(0) == 0).count());
        assertEquals(15, rows.stream().filter(row -> row.getInt(0) == 1).count());
        // No data column is named anywhere in the scan, so nothing but the partition value was read.
        assertFalse(plan(data).contains("value"), plan(data));
    }

    @Test
    void countWithAggregatePushdownOffStillCountsEveryRow() {
        Path output = tempDir.resolve("count_no_pushdown");
        write(output, false);
        Dataset<Row> data = spark.read()
                .format("vortex")
                .option("vortex.aggregatePushdown", "false")
                .load(output.toString());

        Dataset<Row> count = data.selectExpr("count(*) AS count");

        assertEquals(30L, count.first().getLong(0));
        assertTrue(plan(count).contains("PushedAggregation: []"), plan(count));
    }

    @Test
    void constantProjectionStillReportsEveryRow() {
        Dataset<Row> data = writeAndRead("constant", false);

        assertEquals(30, data.selectExpr("1 AS one").collectAsList().size());
    }

    private String plan(Dataset<Row> data) {
        return data.queryExecution().executedPlan().toString();
    }

    private SparkSession newActiveSession() {
        SparkSession session = spark.newSession();
        // Spark 3.5's FileDataSourceV2 resolves its session through SparkSession.active.
        SparkSession.setActiveSession(session);
        return session;
    }

    private Dataset<Row> writeAndRead(String name, boolean partitioned) {
        Path output = tempDir.resolve(name);
        write(output, partitioned);
        return spark.read().format("vortex").load(output.toString());
    }

    private void write(Path output, boolean partitioned) {
        Dataset<Row> data = spark.range(0, 30)
                .selectExpr(
                        "cast(id as int) as id",
                        "concat('value_', cast(id as string)) as value",
                        "cast(id % 2 as int) as group")
                .repartition(3);
        if (partitioned) {
            data.write()
                    .format("vortex")
                    .partitionBy("group")
                    .mode(SaveMode.Overwrite)
                    .save(output.toString());
        } else {
            data.write().format("vortex").mode(SaveMode.Overwrite).save(output.toString());
        }
    }
}
