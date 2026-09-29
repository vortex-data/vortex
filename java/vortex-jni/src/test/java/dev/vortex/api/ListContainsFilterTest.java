// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * End-to-end coverage for {@link Expression#listContains(Expression, Expression)} as a scan filter.
 *
 * <p>Constructing a list-contains expression proves nothing on its own: the element type is only checked when the
 * expression is bound to a schema, so a list literal built from the wrong element type builds fine and fails later,
 * inside a scan. These tests therefore write a small file and read it back through the filter, asserting on the rows
 * that survive.
 *
 * <p>The {@code maybe} column is nullable and null on every odd id, and the sets below hold a null element, to pin the
 * two null rules apart: under SQL's, which {@link Expression#in} and {@link Expression#notIn} use, a null element makes
 * every non-match null, so {@code NOT IN} keeps nothing; under Vortex's it is simply never a match.
 *
 * <p>The large-set case is the reason the binding exists. A caller without it has to expand {@code IN} into a chain of
 * equality comparisons, which is why callers cap the set size and fall back to a bounding range; a single
 * {@code list_contains} node carries the whole set, and the stats rewrite falsifies it for a zone only when every
 * element misses that zone's bounds.
 */
public final class ListContainsFilterTest {
    private static final int ROW_COUNT = 6;

    @TempDir
    static Path tempDir;

    private static Session session;
    private static String filePath;

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @BeforeAll
    static void writeFile() throws IOException {
        session = Session.create();
        filePath =
                tempDir.resolve("list_contains.vortex").toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = new Schema(List.of(
                Field.notNullable("id", new ArrowType.Int(32, true)),
                Field.notNullable("name", new ArrowType.Utf8()),
                Field.nullable("maybe", new ArrowType.Int(32, true))));

        try (VortexWriter writer = VortexWriter.builder(session, filePath, schema, allocator)
                        .build();
                VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            IntVector id = (IntVector) root.getVector("id");
            VarCharVector name = (VarCharVector) root.getVector("name");
            IntVector maybe = (IntVector) root.getVector("maybe");
            id.allocateNew(ROW_COUNT);
            name.allocateNew(ROW_COUNT);
            maybe.allocateNew(ROW_COUNT);
            for (int i = 0; i < ROW_COUNT; i++) {
                id.setSafe(i, i + 1);
                name.setSafe(i, ("row-" + (i + 1)).getBytes(UTF_8));
                // Every other row is null; the rest carry their id.
                if (i % 2 == 0) {
                    maybe.setNull(i);
                } else {
                    maybe.setSafe(i, i + 1);
                }
            }
            root.setRowCount(ROW_COUNT);

            try (ArrowArray array = ArrowArray.allocateNew(allocator);
                    ArrowSchema arrowSchema = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, root, null, array, arrowSchema);
                writer.writeBatch(array.memoryAddress(), arrowSchema.memoryAddress());
            }
            writer.finish();
        }
    }

    @Test
    public void inKeepsOnlyTheMatchingRows() {
        Expression set = Expression.literalList(Expression.literal(2), Expression.literal(4), Expression.literal(99));
        assertEquals(List.of(2, 4), scanIds(Expression.in(Expression.column("id"), set)));
    }

    @Test
    public void notInKeepsTheComplement() {
        Expression set = Expression.literalList(Expression.literal(2), Expression.literal(4), Expression.literal(99));
        assertEquals(List.of(1, 3, 5, 6), scanIds(Expression.notIn(Expression.column("id"), set)));
    }

    @Test
    public void listContainsTakesTheListFirst() {
        // in(value, list) is listContains(list, value) with the operands the other way round; both spellings have to
        // agree, and passing the column as the list would be a type error rather than a silently different filter.
        Expression set = Expression.literalList(Expression.literal(3));
        assertEquals(List.of(3), scanIds(Expression.listContains(set, Expression.column("id"))));
    }

    @Test
    public void inWithANullElementKeepsOnlyMatches() {
        // maybe = [null, 2, null, 4, null, 6]; the set holds 2 and a null.
        Expression set = Expression.literalList(Expression.literal(2), Expression.nullLiteral(Expression.DType.I32));
        assertEquals(List.of(2), scanIds(Expression.in(Expression.column("maybe"), set)));
    }

    @Test
    public void notInWithANullElementKeepsNothing() {
        // SQL: x NOT IN (2, NULL) is false for 2 and null for everything else, so no row survives.
        Expression set = Expression.literalList(Expression.literal(2), Expression.nullLiteral(Expression.DType.I32));
        assertEquals(List.of(), scanIds(Expression.notIn(Expression.column("maybe"), set)));
        // Without the null element NOT IN keeps the non-null non-matches.
        assertEquals(
                List.of(4, 6),
                scanIds(Expression.notIn(Expression.column("maybe"), Expression.literalList(Expression.literal(2)))));
    }

    @Test
    public void vortexNullRulesTreatANullElementAsNoMatch() {
        // listContains without SQL semantics: the null element never matches, so a non-match stays false and its
        // negation keeps the non-null non-matches.
        Expression set = Expression.literalList(Expression.literal(2), Expression.nullLiteral(Expression.DType.I32));
        assertEquals(List.of(2), scanIds(Expression.listContains(set, Expression.column("maybe"))));
        assertEquals(List.of(4, 6), scanIds(Expression.not(Expression.listContains(set, Expression.column("maybe")))));
    }

    @Test
    public void aStringSetFiltersOnUtf8() {
        Expression set = Expression.literalList(Expression.literal("row-1"), Expression.literal("row-6"));
        assertEquals(List.of(1, 6), scanIds(Expression.in(Expression.column("name"), set)));
    }

    @Test
    public void aSetLargerThanAnyOrChainCapStillPushesDown() {
        // The whole point of the binding: 1000 literals stay one expression node. Only 5 is present in the file.
        Expression[] elements = new Expression[1000];
        for (int i = 0; i < elements.length; i++) {
            elements[i] = Expression.literal(i == 0 ? 5 : ROW_COUNT + i);
        }
        assertEquals(List.of(5), scanIds(Expression.in(Expression.column("id"), Expression.literalList(elements))));
    }

    @Test
    public void anEmptySetMatchesNothing() {
        Expression empty = Expression.literalEmptyList(Expression.DType.I32);
        assertTrue(scanIds(Expression.in(Expression.column("id"), empty)).isEmpty());
    }

    @Test
    public void aNullSetMatchesNothing() {
        // A null list yields null rather than false, which filters the row out just the same.
        Expression nullList = Expression.nullLiteralList(Expression.DType.I32);
        assertTrue(scanIds(Expression.in(Expression.column("id"), nullList)).isEmpty());
    }

    /** Reads the {@code id} column of every row that survives {@code filter}, in file order. */
    private static List<Integer> scanIds(Expression filter) {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        DataSource dataSource = DataSource.open(session, filePath);
        Scan scan = dataSource.scan(
                ScanOptions.builder().filter(filter).ordered(true).build());

        List<Integer> ids = new ArrayList<>();
        while (scan.hasNext()) {
            Partition partition = scan.next();
            try (ArrowReader reader = partition.scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot root = reader.getVectorSchemaRoot();
                    IntVector id = (IntVector) root.getVector("id");
                    for (int i = 0; i < root.getRowCount(); i++) {
                        ids.add(id.get(i));
                    }
                }
            } catch (IOException e) {
                throw new AssertionError("failed reading partition", e);
            }
        }
        return ids;
    }
}
