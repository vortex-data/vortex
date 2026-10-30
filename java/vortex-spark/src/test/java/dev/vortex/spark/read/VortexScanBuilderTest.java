// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Map;
import org.apache.spark.sql.connector.catalog.Column;
import org.apache.spark.sql.connector.expressions.Expression;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.LiteralValue;
import org.apache.spark.sql.connector.expressions.NamedReference;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.connector.expressions.filter.Predicate;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.MapType;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.unsafe.types.UTF8String;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VortexScanBuilder}, focusing on how {@link VortexScanBuilder#pushPredicates(Predicate[])}
 * splits predicates between Vortex pushdown and Spark post-scan evaluation.
 *
 * <p>Characterizes the partition-column exclusion rule (partition columns live in directory paths, not in the Vortex
 * files, so predicates referencing them must stay with Spark), the handling of unsupported operators, and the defensive
 * copy returned by {@link VortexScanBuilder#pushedPredicates()}.
 */
final class VortexScanBuilderTest {

    private static VortexScanBuilder builderWithColumns(Transform[] partitionTransforms, Column... columns) {
        VortexScanBuilder builder = new VortexScanBuilder(Map.of(), partitionTransforms);
        builder.addPath("/tmp/data.vortex");
        for (Column column : columns) {
            builder.addColumn(column);
        }
        return builder;
    }

    private static Predicate equality(String column, int value) {
        return new Predicate("=", new Expression[] {ref(column), new LiteralValue<>(value, DataTypes.IntegerType)});
    }

    private static NamedReference ref(String name) {
        return Expressions.column(name);
    }

    // --- pushPredicates: data columns ---

    @Test
    @DisplayName("Predicate on a data column is pushed and removed from post-scan predicates")
    void dataColumnPredicateIsPushed() {
        VortexScanBuilder builder = builderWithColumns(
                new Transform[0],
                Column.create("id", DataTypes.IntegerType),
                Column.create("name", DataTypes.StringType));

        Predicate predicate = equality("id", 42);
        Predicate[] postScan = builder.pushPredicates(new Predicate[] {predicate});

        assertEquals(0, postScan.length);
        assertArrayEquals(new Predicate[] {predicate}, builder.pushedPredicates());
    }

    @Test
    @DisplayName("Predicate on a column missing from the read schema is left to Spark")
    void unknownColumnPredicateIsNotPushed() {
        VortexScanBuilder builder = builderWithColumns(new Transform[0], Column.create("id", DataTypes.IntegerType));

        Predicate predicate = equality("missing", 1);
        Predicate[] postScan = builder.pushPredicates(new Predicate[] {predicate});

        assertArrayEquals(new Predicate[] {predicate}, postScan);
        assertEquals(0, builder.pushedPredicates().length);
    }

    @Test
    @DisplayName("Predicate with an unsupported operator is left to Spark")
    void unsupportedOperatorIsNotPushed() {
        VortexScanBuilder builder = builderWithColumns(new Transform[0], Column.create("name", DataTypes.StringType));

        // LIKE with arbitrary user pattern is not a translatable V2 predicate name.
        Predicate predicate = new Predicate(
                "LIKE",
                new Expression[] {ref("name"), new LiteralValue<>(UTF8String.fromString("%a%"), DataTypes.StringType)});
        Predicate[] postScan = builder.pushPredicates(new Predicate[] {predicate});

        assertArrayEquals(new Predicate[] {predicate}, postScan);
        assertEquals(0, builder.pushedPredicates().length);
    }

    // --- pushPredicates: partition columns ---

    @Test
    @DisplayName("Predicate on a partition column is left to Spark")
    void partitionColumnPredicateIsNotPushed() {
        Transform[] transforms = new Transform[] {Expressions.identity("year")};
        VortexScanBuilder builder = builderWithColumns(
                transforms, Column.create("id", DataTypes.IntegerType), Column.create("year", DataTypes.IntegerType));

        Predicate predicate = equality("year", 2024);
        Predicate[] postScan = builder.pushPredicates(new Predicate[] {predicate});

        assertArrayEquals(new Predicate[] {predicate}, postScan);
        assertEquals(0, builder.pushedPredicates().length);
    }

    @Test
    @DisplayName("Mixed predicates split into pushed (data) and post-scan (partition)")
    void mixedPredicatesAreSplit() {
        Transform[] transforms = new Transform[] {Expressions.identity("year")};
        VortexScanBuilder builder = builderWithColumns(
                transforms, Column.create("id", DataTypes.IntegerType), Column.create("year", DataTypes.IntegerType));

        Predicate onData = equality("id", 7);
        Predicate onPartition = equality("year", 2024);
        Predicate[] postScan = builder.pushPredicates(new Predicate[] {onData, onPartition});

        assertArrayEquals(new Predicate[] {onPartition}, postScan);
        assertArrayEquals(new Predicate[] {onData}, builder.pushedPredicates());
    }

    @Test
    @DisplayName("AND spanning a data column and a partition column is left to Spark as a whole")
    void conjunctionSpanningPartitionColumnIsNotPushed() {
        Transform[] transforms = new Transform[] {Expressions.identity("year")};
        VortexScanBuilder builder = builderWithColumns(
                transforms, Column.create("id", DataTypes.IntegerType), Column.create("year", DataTypes.IntegerType));

        Predicate and =
                new org.apache.spark.sql.connector.expressions.filter.And(equality("id", 7), equality("year", 2024));
        Predicate[] postScan = builder.pushPredicates(new Predicate[] {and});

        assertArrayEquals(new Predicate[] {and}, postScan);
        assertEquals(0, builder.pushedPredicates().length);
    }

    // --- pushedPredicates ---

    @Test
    @DisplayName("pushedPredicates is empty before any pushPredicates call")
    void pushedPredicatesEmptyByDefault() {
        VortexScanBuilder builder = builderWithColumns(new Transform[0], Column.create("id", DataTypes.IntegerType));
        assertEquals(0, builder.pushedPredicates().length);
    }

    @Test
    @DisplayName("pushedPredicates returns a defensive copy")
    void pushedPredicatesReturnsCopy() {
        VortexScanBuilder builder = builderWithColumns(new Transform[0], Column.create("id", DataTypes.IntegerType));
        Predicate predicate = equality("id", 1);
        builder.pushPredicates(new Predicate[] {predicate});

        Predicate[] first = builder.pushedPredicates();

        // Mutating one returned array must not leak into subsequent calls.
        first[0] = null;
        assertArrayEquals(new Predicate[] {predicate}, builder.pushedPredicates());
    }

    // --- pruneColumns ---

    private static final StructType ABC = new StructType()
            .add("a", DataTypes.LongType)
            .add("b", DataTypes.LongType)
            .add("c", DataTypes.LongType);

    @Test
    @DisplayName("Struct columns are pruned to the required fields, with columns and fields in table order")
    void pruneColumnsPrunesStructFields() {
        VortexScanBuilder builder = builderWithColumns(
                new Transform[0],
                Column.create("id", DataTypes.IntegerType),
                Column.create("s", ABC),
                Column.create("name", DataTypes.StringType));

        builder.pruneColumns(new StructType()
                .add("s", new StructType().add("c", DataTypes.LongType).add("a", DataTypes.LongType))
                .add("id", DataTypes.IntegerType));

        assertEquals(
                new StructType()
                        .add("id", DataTypes.IntegerType)
                        .add("s", new StructType().add("a", DataTypes.LongType).add("c", DataTypes.LongType)),
                builder.build().readSchema());
    }

    @Test
    @DisplayName("Structs inside structs are pruned at every level")
    void pruneColumnsPrunesNestedStructs() {
        StructType outer = new StructType().add("p", DataTypes.StringType).add("inner", ABC);
        VortexScanBuilder builder = builderWithColumns(new Transform[0], Column.create("outer", outer));

        builder.pruneColumns(new StructType()
                .add("outer", new StructType().add("inner", new StructType().add("b", DataTypes.LongType))));

        assertEquals(
                new StructType()
                        .add("outer", new StructType().add("inner", new StructType().add("b", DataTypes.LongType))),
                builder.build().readSchema());
    }

    @Test
    @DisplayName("Arrays and maps of structs keep their full table type")
    void pruneColumnsKeepsArrayAndMapTypes() {
        ArrayType arr = DataTypes.createArrayType(ABC);
        MapType map = DataTypes.createMapType(DataTypes.StringType, ABC);
        VortexScanBuilder builder =
                builderWithColumns(new Transform[0], Column.create("arr", arr), Column.create("m", map));

        StructType c = new StructType().add("c", DataTypes.LongType);
        builder.pruneColumns(new StructType()
                .add("arr", DataTypes.createArrayType(c))
                .add("m", DataTypes.createMapType(DataTypes.StringType, c)));

        assertEquals(
                new StructType().add("arr", arr).add("m", map), builder.build().readSchema());
    }

    // --- build ---

    @Test
    @DisplayName("build without paths fails fast")
    void buildWithoutPathsThrows() {
        VortexScanBuilder builder = new VortexScanBuilder(Map.of());
        builder.addColumn(Column.create("id", DataTypes.IntegerType));
        assertEquals(
                "paths cannot be empty",
                assertThrows(IllegalStateException.class, builder::build).getMessage());
    }
}
