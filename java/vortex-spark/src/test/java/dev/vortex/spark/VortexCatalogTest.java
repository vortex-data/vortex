// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.TableChange;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VortexCatalog}, the path-based catalog behind {@code SELECT * FROM vortex.`/path`}.
 *
 * <p>Characterizes which identifiers the catalog accepts as paths and which it rejects, plus the DDL surface it
 * deliberately does not implement. End-to-end querying through a real session is covered by {@code VortexSqlTest};
 * these tests pin the identifier contract without starting Spark.
 */
final class VortexCatalogTest {

    private static final StructType SCHEMA =
            new StructType(new StructField[] {new StructField("id", DataTypes.IntegerType, false, Metadata.empty())});

    private static VortexCatalog catalog() {
        VortexCatalog catalog = new VortexCatalog();
        catalog.initialize("vortex", new CaseInsensitiveStringMap(Map.of()));
        return catalog;
    }

    @Test
    @DisplayName("Takes its name from the session config that registered it")
    void nameComesFromInitialize() {
        VortexCatalog catalog = new VortexCatalog();
        catalog.initialize("my_vortex", new CaseInsensitiveStringMap(Map.of()));

        assertEquals("my_vortex", catalog.name());
    }

    @Test
    @DisplayName("Defaults to the name \"vortex\" before initialize is called")
    void nameDefaultsToVortex() {
        assertEquals("vortex", new VortexCatalog().name());
    }

    @Test
    @DisplayName("Lists no tables: the catalog holds no state, tables are addressed by path")
    void listTablesIsAlwaysEmpty() {
        assertArrayEquals(new Identifier[0], catalog().listTables(new String[0]));
        assertArrayEquals(new Identifier[0], catalog().listTables(new String[] {"any", "namespace"}));
    }

    @Test
    @DisplayName("An identifier without a path separator is not a table")
    void identifierWithoutSlashIsRejected() {
        assertThrows(NoSuchTableException.class, () -> catalog().loadTable(Identifier.of(new String[0], "not_a_path")));
    }

    @Test
    @DisplayName("A namespaced identifier is not a table, even when the name looks like a path")
    void namespacedIdentifierIsRejected() {
        assertThrows(NoSuchTableException.class, () -> catalog()
                .loadTable(Identifier.of(new String[] {"db"}, "/data/a.vortex")));
    }

    @Test
    @DisplayName("A path-shaped identifier that cannot be read surfaces as table not found")
    void unreadablePathIsReportedAsMissingTable() {
        // Schema inference throws for a path with no Vortex files; the catalog translates that into
        // NoSuchTableException so SQL users see "table not found" rather than an internal error.
        assertThrows(NoSuchTableException.class, () -> catalog()
                .loadTable(Identifier.of(new String[0], "/nonexistent/vortex/path")));
    }

    @Test
    @DisplayName("Dropping a table never deletes data, it reports that nothing was dropped")
    void dropTableReturnsFalse() {
        assertFalse(catalog().dropTable(Identifier.of(new String[0], "/data/a.vortex")));
    }

    @Test
    @DisplayName("CREATE TABLE is unsupported: write to the path instead")
    void createTableIsUnsupported() {
        assertThrows(UnsupportedOperationException.class, () -> catalog()
                .createTable(Identifier.of(new String[0], "/data/a.vortex"), SCHEMA, new Transform[0], Map.of()));
    }

    @Test
    @DisplayName("ALTER TABLE is unsupported: there is no metadata to alter")
    void alterTableIsUnsupported() {
        assertThrows(UnsupportedOperationException.class, () -> catalog()
                .alterTable(Identifier.of(new String[0], "/data/a.vortex"), new TableChange[0]));
    }

    @Test
    @DisplayName("RENAME TABLE is unsupported: there is no metadata to rename")
    void renameTableIsUnsupported() {
        assertThrows(UnsupportedOperationException.class, () -> catalog()
                .renameTable(
                        Identifier.of(new String[0], "/data/a.vortex"),
                        Identifier.of(new String[0], "/data/b.vortex")));
    }
}
