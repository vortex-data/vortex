// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.apache.spark.sql.AnalysisException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration tests for Spark SQL access to Vortex: managed tables created without a {@code LOCATION} clause, and
 * direct file queries through {@link VortexCatalog}.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSqlTest {

    private SparkSession spark;
    private Path warehouseDir;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() throws IOException {
        warehouseDir = Files.createTempDirectory("vortex-warehouse");
        spark = SparkSession.builder()
                .appName("VortexSqlTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.warehouse.dir", warehouseDir.toUri().toString())
                .config("spark.sql.catalog.vortex", VortexCatalog.class.getName())
                .config("spark.ui.enabled", "false")
                .getOrCreate();
    }

    @AfterAll
    public void tearDown() {
        if (spark != null) {
            spark.stop();
        }
    }

    /**
     * Spark 3.5's built-in session catalog cannot read tables backed by a DataSource-V2-only provider: its
     * {@code FindDataSourceTable} rule falls back to the V1 {@code DataSource} path, which rejects the provider with
     * "vortex is not a valid Spark SQL Data Source". Spark 4 resolves such tables through the provider directly. On
     * Spark 3.5 the {@link VortexSessionCatalog} extension provides the same support — see
     * {@link VortexSessionCatalogTest}, which runs these scenarios on both versions.
     */
    private void assumeSupportsSqlTables() {
        assumeTrue(
                spark.version().startsWith("4."),
                "CREATE TABLE ... USING vortex requires Spark 4 or the VortexSessionCatalog extension");
    }

    @Test
    @DisplayName("Managed table lifecycle: CREATE, SELECT while empty, INSERT, INSERT OVERWRITE, DROP")
    public void testManagedTableLifecycle() {
        assumeSupportsSqlTables();
        spark.sql("CREATE TABLE managed_students (id INT, name STRING, age INT) USING vortex");

        assertEquals(0, spark.sql("SELECT * FROM managed_students").count(), "New managed table should be empty");

        spark.sql("INSERT INTO managed_students VALUES (1, 'Alice', 20), (2, 'Bob', 21)");
        List<Row> rows = spark.sql("SELECT name FROM managed_students WHERE age > 20 ORDER BY name")
                .collectAsList();
        assertEquals(1, rows.size());
        assertEquals("Bob", rows.get(0).getString(0));

        spark.sql("INSERT OVERWRITE managed_students VALUES (3, 'Carol', 22)");
        assertEquals(1, spark.sql("SELECT * FROM managed_students").count(), "Overwrite should replace all rows");

        Path tableDir = warehouseDir.resolve("managed_students");
        assertTrue(Files.exists(tableDir), "Managed table data should live under the warehouse dir");
        spark.sql("DROP TABLE managed_students");
        assertFalse(Files.exists(tableDir), "Dropping a managed table should remove its data");
    }

    @Test
    @DisplayName("CREATE TABLE AS SELECT without a LOCATION clause")
    public void testCreateManagedTableAsSelect() {
        assumeSupportsSqlTables();
        spark.sql("CREATE TABLE ctas_source (id INT, name STRING) USING vortex");
        spark.sql("INSERT INTO ctas_source VALUES (1, 'Alice'), (2, 'Bob')");

        spark.sql("CREATE TABLE ctas_target USING vortex AS SELECT * FROM ctas_source WHERE id > 1");
        List<Row> rows = spark.sql("SELECT name FROM ctas_target").collectAsList();
        assertEquals(1, rows.size());
        assertEquals("Bob", rows.get(0).getString(0));

        spark.sql("DROP TABLE ctas_target");
        spark.sql("DROP TABLE ctas_source");
    }

    @Test
    @DisplayName("Reading the vortex format without a path option still fails")
    public void testReadWithoutPathStillThrows() {
        assertThrows(
                IllegalArgumentException.class,
                () -> spark.read().format("vortex").load(),
                "A read with no path should not silently return an empty DataFrame");
    }

    @Test
    @DisplayName("Direct file query through the vortex catalog")
    public void testDirectPathQuery() {
        Path dataDir = tempDir.resolve("direct_query");
        writeTestData(dataDir);

        List<Row> rows = spark.sql(String.format("SELECT name FROM vortex.`%s` WHERE age > 30 ORDER BY name", dataDir))
                .collectAsList();
        assertEquals(2, rows.size());
        assertEquals("Alice", rows.get(0).getString(0));
        assertEquals("Carol", rows.get(1).getString(0));
    }

    @Test
    @DisplayName("INSERT INTO a direct path through the vortex catalog")
    public void testDirectPathInsert() {
        Path dataDir = tempDir.resolve("direct_insert");
        writeTestData(dataDir);

        spark.sql(String.format("INSERT INTO vortex.`%s` VALUES (4, 'Dave', 50)", dataDir));
        assertEquals(
                4,
                spark.sql(String.format("SELECT * FROM vortex.`%s`", dataDir)).count());
    }

    @Test
    @DisplayName("Direct queries of missing paths and non-path names fail as table-not-found")
    public void testDirectPathNotFound() {
        assertThrows(AnalysisException.class, () -> spark.sql(
                        String.format("SELECT * FROM vortex.`%s`", tempDir.resolve("no_such_dir")))
                .collect());
        assertThrows(AnalysisException.class, () -> spark.sql("SELECT * FROM vortex.not_a_path")
                .collect());
    }

    private void writeTestData(Path dataDir) {
        StructType schema = DataTypes.createStructType(new StructField[] {
            DataTypes.createStructField("id", DataTypes.IntegerType, false),
            DataTypes.createStructField("name", DataTypes.StringType, false),
            DataTypes.createStructField("age", DataTypes.IntegerType, false)
        });
        Dataset<Row> df = spark.createDataFrame(
                Arrays.asList(
                        RowFactory.create(1, "Alice", 34),
                        RowFactory.create(2, "Bob", 27),
                        RowFactory.create(3, "Carol", 45)),
                schema);
        df.write().format("vortex").mode(SaveMode.Overwrite).save(dataDir.toString());
    }
}
