// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.bench;

import static java.nio.charset.StandardCharsets.UTF_8;

import dev.vortex.api.DataSource;
import dev.vortex.api.Expression;
import dev.vortex.api.Partition;
import dev.vortex.api.Scan;
import dev.vortex.api.ScanOptions;
import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.jni.NativeLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.BigIntVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.Float8Vector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ViewVarCharVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.FloatingPointPrecision;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * Measures read throughput through the vortex-jni boundary (JNI + the Arrow C Data Interface).
 *
 * <p>Every invocation scans the full {@link #ROWS}-row table, so {@code @OperationsPerInvocation(ROWS)} makes JMH
 * report <b>input rows scanned per second</b> directly (rather than scans/s that the reader must convert).
 *
 * <p>To isolate native pushdown from JVM-side consumption savings, the projection and filter lanes each have a
 * <i>control</i> that does the SAME Java-side work but WITHOUT the native pushdown:
 *
 * <ul>
 *   <li>{@code projection} (native projection of id,y) vs {@code projectionControl} (full scan, consume only id,y) —
 *       the speedup that remains is attributable to native projection, not to touching fewer Java vectors.
 *   <li>{@code selectiveFilter} (native filter cat='alpha') vs {@code filterControl} (full scan, filter in Java) —
 *       the speedup that remains is attributable to native filter pushdown, not to summing fewer rows.
 * </ul>
 *
 * <p>Rows are consumed column-at-a-time (numeric sums, view lengths, null counts) rather than into per-row Java
 * objects, so the numbers reflect format/boundary cost rather than JVM allocation. {@code ScanOptions} exposes no
 * read-batch knob, and Vortex coalesces to ~64K-row read batches regardless of the writer's chunk size (see
 * {@link #main}), so boundary cost is amortized over large batches by construction.
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@OperationsPerInvocation(VortexJniReadBenchmark.ROWS)
@Warmup(iterations = 3, time = 2)
@Measurement(iterations = 5, time = 2)
@Fork(
        value = 1,
        jvmArgsAppend = {
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
        })
@Threads(1)
@State(Scope.Benchmark)
public class VortexJniReadBenchmark {

    static final int ROWS = 2_000_000;
    static final int WRITE_CHUNK = 65536;
    static final String[] CATS = {
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa"
    };
    static final byte[] ALPHA = "alpha".getBytes(UTF_8);
    static final long EXPECTED_ALPHA_MATCHES = ROWS / CATS.length;

    BufferAllocator allocator;
    Session session;
    DataSource dataSource;
    Path file;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        NativeLoader.loadJni();
        allocator = new RootAllocator(Long.MAX_VALUE);
        session = Session.create();
        file = Files.createTempFile("vortex-jni-bench-", ".vortex");
        Files.deleteIfExists(file);
        String uri = file.toAbsolutePath().toUri().toString();
        writeTable(session, allocator, uri, WRITE_CHUNK);
        dataSource = DataSource.open(session, uri);
        validate();
    }

    @TearDown(Level.Trial)
    public void teardown() throws Exception {
        // Intentionally does not close the allocator: DataSource/Scan native resources are released by VortexCleaner
        // at GC time, which races an explicit allocator.close() and trips leak detection. The JMH fork exits after the
        // trial and reclaims everything; we only remove the temp file.
        dataSource = null;
        if (file != null) {
            Files.deleteIfExists(file);
        }
    }

    /** Fail the trial loudly if the generated file or pushdown semantics are wrong — fast garbage must not be cited. */
    private void validate() throws Exception {
        if (!(dataSource.rowCount() instanceof DataSource.RowCount.Exact exact) || exact.value() != ROWS) {
            throw new IllegalStateException("expected exactly " + ROWS + " rows, got " + dataSource.rowCount());
        }
        // Native filter must return exactly ROWS/|CATS| rows for cat='alpha'.
        Expression filter =
                Expression.binary(Expression.BinaryOp.EQ, Expression.column("cat"), Expression.literal(CATS[0]));
        long matched = 0;
        Scan scan = dataSource.scan(ScanOptions.builder().filter(filter).build());
        while (scan.hasNext()) {
            try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    matched += reader.getVectorSchemaRoot().getRowCount();
                }
            }
        }
        if (matched != EXPECTED_ALPHA_MATCHES) {
            throw new IllegalStateException("filter cat='alpha' returned " + matched + ", expected " + EXPECTED_ALPHA_MATCHES);
        }
        // Native projection must yield exactly [id, y].
        Expression projection = Expression.select(new String[] {"id", "y"}, Expression.root());
        Scan pscan = dataSource.scan(ScanOptions.builder().projection(projection).build());
        if (pscan.hasNext()) {
            try (ArrowReader reader = pscan.next().scanArrow(allocator)) {
                if (reader.loadNextBatch()) {
                    List<String> names = reader.getVectorSchemaRoot().getSchema().getFields().stream()
                            .map(Field::getName)
                            .toList();
                    if (!names.equals(List.of("id", "y"))) {
                        throw new IllegalStateException("projection schema expected [id, y], got " + names);
                    }
                }
            }
        }
    }

    private static Schema schema() {
        return new Schema(List.of(
                Field.notNullable("id", new ArrowType.Int(64, true)),
                Field.notNullable("x", new ArrowType.Int(64, true)),
                Field.notNullable("y", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                Field.notNullable("z", new ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)),
                Field.nullable("cat", ArrowType.Utf8View.INSTANCE),
                Field.nullable("tag", ArrowType.Utf8View.INSTANCE)));
    }

    private static void writeTable(Session session, BufferAllocator allocator, String uri, int chunk) throws Exception {
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

    /** Full scan consuming ALL six columns at the buffer level. */
    @Benchmark
    public void fullScan(Blackhole bh) throws Exception {
        long sumId = 0;
        long sumX = 0;
        double sumY = 0;
        double sumZ = 0;
        long catLen = 0;
        long tagLenOrNulls = 0;
        Scan scan = dataSource.scan(ScanOptions.of());
        while (scan.hasNext()) {
            try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot r = reader.getVectorSchemaRoot();
                    int rows = r.getRowCount();
                    BigIntVector id = (BigIntVector) r.getVector("id");
                    BigIntVector x = (BigIntVector) r.getVector("x");
                    Float8Vector y = (Float8Vector) r.getVector("y");
                    Float8Vector z = (Float8Vector) r.getVector("z");
                    VarCharVector cat = (VarCharVector) r.getVector("cat");
                    VarCharVector tag = (VarCharVector) r.getVector("tag");
                    for (int i = 0; i < rows; i++) {
                        sumId += id.get(i);
                        sumX += x.get(i);
                        sumY += y.get(i);
                        sumZ += z.get(i);
                        catLen += cat.getValueLength(i);
                        if (tag.isNull(i)) {
                            tagLenOrNulls++;
                        } else {
                            tagLenOrNulls += tag.getValueLength(i);
                        }
                    }
                }
            }
        }
        bh.consume(sumId);
        bh.consume(sumX);
        bh.consume(sumY);
        bh.consume(sumZ);
        bh.consume(catLen);
        bh.consume(tagLenOrNulls);
    }

    /** Native projection pushdown: only id,y cross the boundary. */
    @Benchmark
    public void projection(Blackhole bh) throws Exception {
        Expression projection = Expression.select(new String[] {"id", "y"}, Expression.root());
        bh.consume(consumeIdY(dataSource.scan(ScanOptions.builder().projection(projection).build())));
    }

    /** Control for {@link #projection}: full scan, but consume only id,y in Java (no native projection). */
    @Benchmark
    public void projectionControl(Blackhole bh) throws Exception {
        bh.consume(consumeIdY(dataSource.scan(ScanOptions.of())));
    }

    private double consumeIdY(Scan scan) throws Exception {
        long sumId = 0;
        double sumY = 0;
        while (scan.hasNext()) {
            try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot r = reader.getVectorSchemaRoot();
                    int rows = r.getRowCount();
                    BigIntVector id = (BigIntVector) r.getVector("id");
                    Float8Vector y = (Float8Vector) r.getVector("y");
                    for (int i = 0; i < rows; i++) {
                        sumId += id.get(i);
                        sumY += y.get(i);
                    }
                }
            }
        }
        return sumId + sumY;
    }

    /** Native filter pushdown: only matching rows cross the boundary. */
    @Benchmark
    public void selectiveFilter(Blackhole bh) throws Exception {
        Expression filter =
                Expression.binary(Expression.BinaryOp.EQ, Expression.column("cat"), Expression.literal(CATS[0]));
        long matched = 0;
        long sumId = 0;
        Scan scan = dataSource.scan(ScanOptions.builder().filter(filter).build());
        while (scan.hasNext()) {
            try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot r = reader.getVectorSchemaRoot();
                    int rows = r.getRowCount();
                    BigIntVector id = (BigIntVector) r.getVector("id");
                    for (int i = 0; i < rows; i++) {
                        sumId += id.get(i);
                        matched++;
                    }
                }
            }
        }
        bh.consume(matched);
        bh.consume(sumId);
    }

    /** Control for {@link #selectiveFilter}: full scan, evaluate cat='alpha' in Java (no native filter). */
    @Benchmark
    public void filterControl(Blackhole bh) throws Exception {
        long matched = 0;
        long sumId = 0;
        Scan scan = dataSource.scan(ScanOptions.of());
        while (scan.hasNext()) {
            try (ArrowReader reader = scan.next().scanArrow(allocator)) {
                while (reader.loadNextBatch()) {
                    VectorSchemaRoot r = reader.getVectorSchemaRoot();
                    int rows = r.getRowCount();
                    BigIntVector id = (BigIntVector) r.getVector("id");
                    VarCharVector cat = (VarCharVector) r.getVector("cat");
                    for (int i = 0; i < rows; i++) {
                        if (!cat.isNull(i) && Arrays.equals(cat.get(i), ALPHA)) {
                            sumId += id.get(i);
                            matched++;
                        }
                    }
                }
            }
        }
        bh.consume(matched);
        bh.consume(sumId);
    }

    /**
     * Diagnostic (not a benchmark): prints the distribution of read batch row counts for a few writer chunk sizes, to
     * show that Vortex coalesces to a stable read-batch granularity independent of how the file was written.
     */
    public static void main(String[] args) throws Exception {
        NativeLoader.loadJni();
        for (int chunk : new int[] {8192, 65536, 131072}) {
            BufferAllocator alloc = new RootAllocator(Long.MAX_VALUE);
            Session sess = Session.create();
            Path f = Files.createTempFile("vortex-jni-diag-" + chunk + "-", ".vortex");
            Files.deleteIfExists(f);
            String uri = f.toAbsolutePath().toUri().toString();
            writeTable(sess, alloc, uri, chunk);
            DataSource ds = DataSource.open(sess, uri);
            long batches = 0;
            long rowsSeen = 0;
            long minRows = Long.MAX_VALUE;
            long maxRows = 0;
            Scan scan = ds.scan(ScanOptions.of());
            while (scan.hasNext()) {
                Partition partition = scan.next();
                try (ArrowReader reader = partition.scanArrow(alloc)) {
                    while (reader.loadNextBatch()) {
                        int rows = reader.getVectorSchemaRoot().getRowCount();
                        batches++;
                        rowsSeen += rows;
                        minRows = Math.min(minRows, rows);
                        maxRows = Math.max(maxRows, rows);
                    }
                }
            }
            System.out.printf(
                    "writeChunkRows=%d -> %d read batches over %d rows (min=%d, max=%d, avg=%d)%n",
                    chunk, batches, rowsSeen, minRows, maxRows, batches == 0 ? 0 : rowsSeen / batches);
            Files.deleteIfExists(f);
        }
    }
}
