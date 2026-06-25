// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.bench;

import static java.nio.charset.StandardCharsets.UTF_8;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ViewVarCharVector;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Shared synthetic table used by both {@link VortexJniReadBenchmark} and {@link VortexJniBatchDiagnostic} so they
 * measure and inspect the exact same data shape: six columns (2× int64, 2× float64, 2× Utf8View) over {@link #ROWS}
 * rows, with a deterministic fixed seed.
 *
 * <p>{@code id} is sequential, {@code cat} is a periodic low-cardinality column kept non-null so a {@code cat='alpha'}
 * filter has selectivity exactly {@code 1/|CATS|}, and {@code tag} is high-cardinality with a 10% null rate to exercise
 * a validity buffer.
 */
final class BenchData {

    static final int ROWS = 2_000_000;
    static final String[] CATS = {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"
    };

    private BenchData() {}

    static Schema schema() {
        return new Schema(List.of(
                Field.notNullable("id", new ArrowType.Int(64, true)),
                Field.notNullable("x", new ArrowType.Int(64, true)),
                Field.notNullable("y", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                Field.notNullable("z", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                Field.nullable("cat", ArrowType.Utf8View.INSTANCE),
                Field.nullable("tag", ArrowType.Utf8View.INSTANCE)));
    }

    static void writeTable(Session session, BufferAllocator allocator, String uri, int chunk) throws Exception {
        Schema schema = schema();
        Random rnd = new Random(42);
        try (VortexWriter writer = VortexWriter.create(session, uri, schema, new HashMap<>(), allocator);
                VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            BigIntVector id = (BigIntVector) root.getVector("id");
            BigIntVector x = (BigIntVector) root.getVector("x");
            Float8Vector y = (Float8Vector) root.getVector("y");
            Float8Vector z = (Float8Vector) root.getVector("z");
            ViewVarCharVector cat = (ViewVarCharVector) root.getVector("cat");
            ViewVarCharVector tag = (ViewVarCharVector) root.getVector("tag");

            long written = 0;
            while (written < ROWS) {
                int batch = (int) Math.min(chunk, ROWS - written);
                for (FieldVector v : root.getFieldVectors()) {
                    v.reset();
                }
                for (int i = 0; i < batch; i++) {
                    long r = written + i;
                    id.setSafe(i, r);
                    x.setSafe(i, rnd.nextInt(1_000_000));
                    y.setSafe(i, rnd.nextDouble());
                    z.setSafe(i, rnd.nextDouble());
                    // cat stays non-null and deterministic so filter selectivity is exactly 1/|CATS|.
                    cat.setSafe(i, CATS[(int) (r % CATS.length)].getBytes(UTF_8));
                    // tag carries nulls (every 10th row) and high-cardinality values to exercise a validity buffer.
                    if (r % 10 == 0) {
                        tag.setNull(i);
                    } else {
                        tag.setSafe(i, Long.toString(r).getBytes(UTF_8));
                    }
                }
                root.setRowCount(batch);
                try (ArrowArray arr = ArrowArray.allocateNew(allocator);
                        ArrowSchema sch = ArrowSchema.allocateNew(allocator)) {
                    Data.exportVectorSchemaRoot(allocator, root, null, arr, sch);
                    writer.writeBatch(arr.memoryAddress(), sch.memoryAddress());
                }
                written += batch;
            }
        }
    }
}
