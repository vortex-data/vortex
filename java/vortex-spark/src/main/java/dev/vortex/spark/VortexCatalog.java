// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import java.util.Map;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCatalog;
import org.apache.spark.sql.connector.catalog.TableChange;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;

/**
 * A path-based Spark catalog for querying Vortex files directly from SQL.
 *
 * <p>Spark only supports {@code SELECT * FROM format.`path`} syntax for built-in file formats, so this catalog provides
 * the equivalent for Vortex. Register it under the name {@code vortex}:
 *
 * <pre>spark.sql.catalog.vortex=dev.vortex.spark.VortexCatalog</pre>
 *
 * <p>then query a Vortex file, or a directory of Vortex files, directly by path:
 *
 * <pre>SELECT * FROM vortex.`/path/to/data`;</pre>
 *
 * <p>The table identifier must look like a path — contain a {@code /} — and resolves to the same table a
 * {@code spark.read.format("vortex")} load of that path would produce, so reads, writes ({@code INSERT INTO
 * vortex.`/path/to/data`}), and pushdown all behave identically. The catalog holds no state and supports no DDL.
 */
public final class VortexCatalog implements TableCatalog {
    private static final String PATH_KEY = "path";

    private String name = "vortex";

    /**
     * Creates a new catalog instance.
     *
     * <p>This no-argument constructor is required for Spark to instantiate the catalog through reflection from the
     * {@code spark.sql.catalog.<name>} configuration.
     */
    public VortexCatalog() {}

    @Override
    public void initialize(String name, CaseInsensitiveStringMap options) {
        this.name = name;
    }

    @Override
    public String name() {
        return name;
    }

    /**
     * Returns no identifiers: this catalog holds no state, tables are addressed by path.
     *
     * @param namespace the namespace to list, ignored
     * @return an empty array
     */
    @Override
    public Identifier[] listTables(String[] namespace) {
        return new Identifier[0];
    }

    /**
     * Loads the Vortex file or directory of Vortex files at the path given by the identifier name.
     *
     * @param ident identifier whose name is a filesystem path or URL, e.g. {@code vortex.`/path/to/data`}
     * @return a table backed by the Vortex files at the path
     * @throws NoSuchTableException if the identifier does not look like a path, or the path cannot be read
     */
    @SuppressWarnings("deprecation")
    @Override
    public Table loadTable(Identifier ident) throws NoSuchTableException {
        String path = ident.name();
        if (ident.namespace().length != 0 || !path.contains("/")) {
            throw new NoSuchTableException(ident);
        }
        var options = new CaseInsensitiveStringMap(Map.of(PATH_KEY, path));
        var provider = new VortexDataSourceV2();
        StructType schema;
        Transform[] partitioning;
        try {
            schema = provider.inferSchema(options);
            partitioning = provider.inferPartitioning(options);
        } catch (RuntimeException e) {
            // Missing or unreadable paths surface as "table not found" to SQL users.
            throw new NoSuchTableException(ident);
        }
        return provider.getTable(schema, partitioning, Map.of(PATH_KEY, path));
    }

    /** Unsupported: tables are addressed by path, create them by writing data with the {@code vortex} format. */
    @Override
    public Table createTable(
            Identifier ident, StructType schema, Transform[] partitions, Map<String, String> properties) {
        throw new UnsupportedOperationException(
                "VortexCatalog does not support CREATE TABLE, write data to the path instead");
    }

    /** Unsupported: this catalog holds no table metadata to alter. */
    @Override
    public Table alterTable(Identifier ident, TableChange... changes) {
        throw new UnsupportedOperationException("VortexCatalog does not support ALTER TABLE");
    }

    /** Unsupported: this catalog never drops data, returns false. */
    @Override
    public boolean dropTable(Identifier ident) {
        return false;
    }

    /** Unsupported: this catalog holds no table metadata to rename. */
    @Override
    public void renameTable(Identifier oldIdent, Identifier newIdent) {
        throw new UnsupportedOperationException("VortexCatalog does not support RENAME TABLE");
    }
}
