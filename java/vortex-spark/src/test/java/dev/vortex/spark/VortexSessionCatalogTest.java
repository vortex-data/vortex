// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.io.TempDir;

/**
 * Integration tests for {@link VortexSessionCatalog}, the session catalog extension that makes {@code CREATE TABLE ...
 * USING vortex} work on Spark 3.5 as well as Spark 4.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexSessionCatalogTest {

    private SparkSession spark;
    private Path warehouseDir;

    @TempDir
    Path tempDir;

    @BeforeAll
    public void setUp() throws IOException {
        warehouseDir = Files.createTempDirectory("vortex-warehouse");
        spark = SparkSession.builder()
                .appName("VortexSessionCatalogTest")
                .master("local[2]")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.sql.warehouse.dir", warehouseDir.toUri().toString())
                .config("spark.sql.catalog.spark_catalog", VortexSessionCatalog.class.getName())
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
    @DisplayName("Managed table lifecycle: CREATE, SELECT while empty, INSERT, INSERT OVERWRITE, DROP")
    public void testManagedTableLifecycle() {
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
    @DisplayName("External table with a LOCATION clause")
    public void testExternalTable() {
        Path location = tempDir.resolve("ext_students");
        spark.sql(
                String.format("CREATE TABLE ext_students (id INT, name STRING) USING vortex LOCATION '%s'", location));
        spark.sql("INSERT INTO ext_students VALUES (1, 'Alice')");
        assertEquals(1, spark.sql("SELECT * FROM ext_students").count());
        spark.sql("DROP TABLE ext_students");
    }

    @Test
    @DisplayName("Tables of other providers pass through the extension untouched")
    public void testOtherProviderPassthrough() {
        spark.sql("CREATE TABLE pq_table (id INT) USING parquet");
        spark.sql("INSERT INTO pq_table VALUES (7)");
        assertEquals(
                7, spark.sql("SELECT * FROM pq_table").collectAsList().get(0).getInt(0));
        spark.sql("DROP TABLE pq_table");
    }
}
