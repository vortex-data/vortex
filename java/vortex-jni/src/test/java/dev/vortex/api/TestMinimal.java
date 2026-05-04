// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.arrow.ArrowAllocation;
import dev.vortex.jni.NativeLoader;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;
import org.apache.arrow.c.ArrowArray;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.DecimalVector;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VarCharVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.ViewVarCharVector;
import org.apache.arrow.vector.ipc.ArrowReader;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public final class TestMinimal {
    static final class Person {
        public String name;
        public BigDecimal salary;
        public String state;

        public Person(String name, BigDecimal salary, String state) {
            this.name = name;
            this.salary = salary;
            this.state = state;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Person person)) return false;
            return Objects.equals(name, person.name)
                    && Objects.equals(salary, person.salary)
                    && Objects.equals(state, person.state);
        }

        @Override
        public int hashCode() {
            return Objects.hash(name, salary, state);
        }

        @Override
        public String toString() {
            return "Person{name='" + name + "', salary=" + salary + ", state='" + state + "'}";
        }
    }

    @TempDir
    static Path tempDir;

    static String writePath;

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @BeforeAll
    static void setup() throws IOException {
        Path outputPath = tempDir.resolve("minimal.vortex");
        writePath = outputPath.toAbsolutePath().toUri().toString();

        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Schema schema = new Schema(List.of(
                Field.notNullable("Name", ArrowType.Utf8View.INSTANCE),
                Field.notNullable("Salary", ArrowType.Decimal.createDecimal(9, 2, 128)),
                Field.nullable("State", ArrowType.Utf8View.INSTANCE)));
        Session session = Session.create();
        try (VortexWriter writer = VortexWriter.create(session, writePath, schema, new HashMap<>(), allocator);
                VectorSchemaRoot root = VectorSchemaRoot.create(schema, allocator)) {
            ViewVarCharVector nameVec = (ViewVarCharVector) root.getVector("Name");
            DecimalVector salaryVec = (DecimalVector) root.getVector("Salary");
            ViewVarCharVector stateVec = (ViewVarCharVector) root.getVector("State");

            nameVec.allocateNew(10);
            salaryVec.allocateNew(10);
            stateVec.allocateNew(10);

            for (int i = 0; i < 10; i++) {
                var person = MINIMAL_DATA.get(i);
                nameVec.setSafe(i, person.name.getBytes(UTF_8));
                salaryVec.setSafe(i, 1_000 * (i + 1));
                stateVec.setSafe(i, person.state.getBytes(UTF_8));
            }

            root.setRowCount(10);

            try (ArrowArray arrowArray = ArrowArray.allocateNew(allocator);
                    ArrowSchema arrowSchemaFfi = ArrowSchema.allocateNew(allocator)) {
                Data.exportVectorSchemaRoot(allocator, root, null, arrowArray, arrowSchemaFfi);
                writer.writeBatch(arrowArray.memoryAddress(), arrowSchemaFfi.memoryAddress());
            }
        }
    }

    private static final List<Person> MINIMAL_DATA = List.of(
            new Person("Alice", BigDecimal.valueOf(1000L, 2), "CA"),
            new Person("Bob", BigDecimal.valueOf(2000L, 2), "NY"),
            new Person("Carol", BigDecimal.valueOf(3000L, 2), "TX"),
            new Person("Dan", BigDecimal.valueOf(4000L, 2), "CA"),
            new Person("Edward", BigDecimal.valueOf(5000L, 2), "NY"),
            new Person("Frida", BigDecimal.valueOf(6000L, 2), "TX"),
            new Person("George", BigDecimal.valueOf(7000L, 2), "CA"),
            new Person("Henry", BigDecimal.valueOf(8000L, 2), "NY"),
            new Person("Ida", BigDecimal.valueOf(9000L, 2), "TX"),
            new Person("John", BigDecimal.valueOf(10_000L, 2), "VA"));

    @Test
    public void testFullScan() throws Exception {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource ds = DataSource.open(session, writePath);

        assertEquals(new DataSource.RowCount.Exact(10L), ds.rowCount());

        var schema = ds.arrowSchema(allocator);
        assertEquals(
                List.of("Name", "Salary", "State"),
                schema.getFields().stream().map(Field::getName).toList());

        List<Person> people = readAll(ds, ScanOptions.of(), allocator, TestMinimal::readFullBatch);
        assertEquals(MINIMAL_DATA, people);
    }

    @Test
    public void testProjectedScan() throws Exception {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource ds = DataSource.open(session, writePath);
        Expression projection = Expression.select(new String[] {"Name", "State"}, Expression.root());

        ScanOptions options = ScanOptions.builder().projection(projection).build();

        List<Person> people = readAll(ds, options, allocator, batch -> {
            List<Person> results = new ArrayList<>();
            VarCharVector names = (VarCharVector) batch.getVector("Name");
            VarCharVector states = (VarCharVector) batch.getVector("State");
            for (int i = 0; i < batch.getRowCount(); i++) {
                String name = names.isNull(i) ? null : new String(names.get(i), UTF_8);
                String state = states.isNull(i) ? null : new String(states.get(i), UTF_8);
                results.add(new Person(name, null, state));
            }
            return results;
        });
        assertEquals(MINIMAL_DATA.size(), people.size());
        for (int i = 0; i < MINIMAL_DATA.size(); i++) {
            assertEquals(MINIMAL_DATA.get(i).name, people.get(i).name);
            assertEquals(MINIMAL_DATA.get(i).state, people.get(i).state);
        }
    }

    @Test
    public void testProjectedScanWithFilter() throws Exception {
        BufferAllocator allocator = ArrowAllocation.rootAllocator();
        Session session = Session.create();
        DataSource ds = DataSource.open(session, writePath);
        Expression filter =
                Expression.binary(Expression.BinaryOp.EQ, Expression.column("State"), Expression.literal("VA"));

        ScanOptions options = ScanOptions.builder().filter(filter).build();
        List<Person> people = readAll(ds, options, allocator, TestMinimal::readFullBatch);
        assertEquals(List.of(new Person("John", BigDecimal.valueOf(10_000L, 2), "VA")), people);
    }

    private interface BatchReader {
        List<Person> read(VectorSchemaRoot root);
    }

    private static List<Person> readAll(
            DataSource ds, ScanOptions options, BufferAllocator allocator, BatchReader reader) throws Exception {
        List<Person> result = new ArrayList<>();
        Scan scan = ds.scan(options);
        while (scan.hasNext()) {
            Partition partition = scan.next();
            try (ArrowReader arrowReader = partition.scanArrow(allocator)) {
                while (arrowReader.loadNextBatch()) {
                    result.addAll(reader.read(arrowReader.getVectorSchemaRoot()));
                }
            }
        }
        return result;
    }

    private static List<Person> readFullBatch(VectorSchemaRoot root) {
        List<Person> result = new ArrayList<>();
        VarCharVector names = (VarCharVector) root.getVector("Name");
        FieldVector salaries = root.getVector("Salary");
        VarCharVector states = (VarCharVector) root.getVector("State");

        for (int i = 0; i < root.getRowCount(); i++) {
            String name = names.isNull(i) ? null : new String(names.get(i), UTF_8);
            String state = states.isNull(i) ? null : new String(states.get(i), UTF_8);
            BigDecimal salary = null;
            if (!salaries.isNull(i)) {
                Object v = salaries.getObject(i);
                if (v instanceof BigDecimal) {
                    salary = (BigDecimal) v;
                } else {
                    salary = new BigDecimal(v.toString());
                }
            }
            result.add(new Person(name, salary, state));
        }
        return result;
    }
}
