/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex.trino;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.trino.Session;
import io.trino.testing.*;

import java.math.BigDecimal;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * End-to-end tests for the Vortex table function.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public final class VortexTableFunctionTest {

    private static final String TEST_FILE_PATH =
            VortexTableFunctionTest.class.getResource("/minimal.vortex").getPath();

    private QueryRunner queryRunner;

    @BeforeAll
    public void setup() {
        queryRunner = new StandaloneQueryRunner(testSession());
        queryRunner.installPlugin(new VortexPlugin());
        queryRunner.createCatalog("vortex", "vortex", Map.of());
    }

    @AfterAll
    public void teardown() {
        if (queryRunner != null) {
            queryRunner.close();
        }
    }

    private static Session testSession() {
        return Session.builder(TestingSession.testSession())
                .setCatalog("vortex")
                .setSchema("system")
                .build();
    }

    @Test
    public void testReadVortexTableFunction() {
        String query = String.format(
                "SELECT * FROM TABLE(vortex.system.read_vortex(uri => '%s'))", "file://" + TEST_FILE_PATH);

        MaterializedResult result = queryRunner.execute(query);

        assertEquals(10, result.getRowCount(), "Expected 10 rows");
        assertEquals(3, result.getTypes().size(), "Expected 3 columns");
    }

    @Test
    public void testReadVortexWithProjection() {
        String query = String.format(
                "SELECT \"Name\", \"State\" FROM TABLE(vortex.system.read_vortex(uri => '%s'))",
                "file://" + TEST_FILE_PATH);

        MaterializedResult result = queryRunner.execute(query);

        assertEquals(10, result.getRowCount(), "Expected 10 rows");
        assertEquals(2, result.getTypes().size(), "Expected 2 columns");
    }

    @Test
    public void testReadVortexWithFilter() {
        String query = String.format(
                "SELECT * FROM TABLE(vortex.system.read_vortex(uri => '%s')) WHERE \"State\" = 'VA'",
                "file://" + TEST_FILE_PATH);

        MaterializedResult result = queryRunner.execute(query);

        assertEquals(1, result.getRowCount(), "Expected 1 row for VA state");

        Object name = result.getMaterializedRows().get(0).getField(0);
        assertEquals("John", name);
    }

    @Test
    public void testReadVortexColumnValues() {
        String query = String.format(
                "SELECT \"Name\", \"Salary\", \"State\" FROM TABLE(vortex.system.read_vortex(uri => '%s')) ORDER BY \"Salary\"",
                "file://" + TEST_FILE_PATH);

        MaterializedResult result = queryRunner.execute(query);

        assertEquals(10, result.getRowCount(), "Expected 10 rows");

        // Verify first row (Alice, 10.00, CA)
        var firstRow = result.getMaterializedRows().get(0);
        assertEquals("Alice", firstRow.getField(0));
        assertEquals(new BigDecimal("10.00"), firstRow.getField(1));
        assertEquals("CA", firstRow.getField(2));

        // Verify last row (John, 100.00, VA)
        var lastRow = result.getMaterializedRows().get(9);
        assertEquals("John", lastRow.getField(0));
        assertEquals(new BigDecimal("100.00"), lastRow.getField(1));
        assertEquals("VA", lastRow.getField(2));
    }
}
