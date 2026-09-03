// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import java.util.Map;
import org.apache.spark.sql.catalyst.analysis.NoSuchNamespaceException;
import org.apache.spark.sql.catalyst.analysis.NoSuchTableException;
import org.apache.spark.sql.catalyst.analysis.TableAlreadyExistsException;
import org.apache.spark.sql.connector.catalog.DelegatingCatalogExtension;
import org.apache.spark.sql.connector.catalog.Identifier;
import org.apache.spark.sql.connector.catalog.Table;
import org.apache.spark.sql.connector.catalog.TableCatalog;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.StructType;

/**
 * A session catalog extension that resolves {@code USING vortex} tables through the Vortex DataSource V2 connector.
 *
 * <p>Spark 3.5's built-in session catalog resolves the tables it stores through the V1 {@code DataSource} path, which
 * rejects DataSource-V2-only connectors like Vortex, so {@code CREATE TABLE ... USING vortex} tables cannot be read
 * back. (Spark 4 resolves them through the V2 provider directly and needs none of this.) Registering this extension as
 * the session catalog fixes that on Spark 3.5:
 *
 * <pre>spark.sql.catalog.spark_catalog=dev.vortex.spark.VortexSessionCatalog</pre>
 *
 * <p>All operations are delegated to the built-in session catalog — table metadata lives wherever it normally would,
 * including the Hive metastore — but any table whose provider is {@code vortex} is loaded as a Vortex DataSource V2
 * table, backed by the files at the table's location. Tables of every other provider are untouched.
 */
public final class VortexSessionCatalog extends DelegatingCatalogExtension {

    /**
     * Creates a new session catalog extension.
     *
     * <p>This no-argument constructor is required for Spark to instantiate the catalog through reflection from the
     * {@code spark.sql.catalog.spark_catalog} configuration.
     */
    public VortexSessionCatalog() {}

    @Override
    public Table loadTable(Identifier ident) throws NoSuchTableException {
        return asVortexTableIfVortex(super.loadTable(ident));
    }

    /**
     * Creates the table in the delegate session catalog, then returns it resolved through the Vortex connector when its
     * provider is {@code vortex}.
     *
     * <p>Spark does not route table creation through this overload — {@link DelegatingCatalogExtension} sends the
     * {@code Column[]} overload straight to the delegate, and that is the one both supported Spark versions call. It is
     * kept because the delegate may return {@code null} on the normal path, which {@link #asVortexTableIfVortex} now
     * tolerates; {@code CREATE TABLE ... AS SELECT} gets its Vortex table from {@link #loadTable} instead.
     */
    @SuppressWarnings("deprecation")
    @Override
    public Table createTable(
            Identifier ident, StructType schema, Transform[] partitions, Map<String, String> properties)
            throws TableAlreadyExistsException, NoSuchNamespaceException {
        return asVortexTableIfVortex(super.createTable(ident, schema, partitions, properties));
    }

    /**
     * Rebuilds a session-catalog table as a Vortex DataSource V2 table when its provider is {@code vortex} and it has a
     * location; returns every other table unchanged, and {@code null} for a {@code null} input. The schema and
     * partitioning stored in the catalog are used as-is, no file needs to be opened.
     */
    @SuppressWarnings("deprecation")
    private static Table asVortexTableIfVortex(Table table) {
        if (table == null) {
            // V2SessionCatalog.createTable returns null on its normal path.
            return null;
        }
        Map<String, String> properties = table.properties();
        VortexDataSourceV2 provider = new VortexDataSourceV2();
        if (!provider.shortName().equalsIgnoreCase(properties.get(TableCatalog.PROP_PROVIDER))) {
            return table;
        }
        String location = properties.get(TableCatalog.PROP_LOCATION);
        if (location == null) {
            return table;
        }
        return provider.getTable(table.schema(), table.partitioning(), Map.of("path", location));
    }
}
