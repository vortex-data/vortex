// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.Map;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.Test;

final class VortexScanTest {
    @Test
    void exposesPredicateColumnsEvenWhenProjectionIsEmpty() {
        Column id = Column.create("id", DataTypes.IntegerType);
        Predicate predicate = new Predicate(
                "=", new Expression[] {Expressions.column("id"), new LiteralValue<>(7, DataTypes.IntegerType)});
        VortexScan scan =
                new VortexScan(List.of("data.vortex"), List.of(id), List.of(), new Predicate[] {predicate}, Map.of());

        assertEquals(0, scan.readSchema().size());
        assertArrayEquals(new String[] {"id"}, scan.tableSchema().fieldNames());
        assertEquals(DataTypes.IntegerType, scan.tableSchema().apply("id").dataType());
        assertArrayEquals(new Predicate[] {predicate}, scan.pushedPredicates());
    }

    @Test
    void predicateArraysCannotReplaceTheScansPredicates() {
        Predicate predicate = new Predicate("IS_NULL", new Expression[] {Expressions.column("id")});
        Predicate[] predicates = {predicate};
        VortexScan scan = new VortexScan(List.of(), List.of(), List.of(), predicates, Map.of());

        predicates[0] = null;
        scan.pushedPredicates()[0] = null;

        assertArrayEquals(new Predicate[] {predicate}, scan.pushedPredicates());
        assertEquals(0, new VortexScan(List.of(), List.of(), List.of(), null, Map.of()).pushedPredicates().length);
    }
}
