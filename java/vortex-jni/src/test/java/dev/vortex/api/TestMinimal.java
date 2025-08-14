// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;

import dev.vortex.api.expressions.Binary;
import dev.vortex.api.expressions.GetItem;
import dev.vortex.api.expressions.Literal;
import dev.vortex.api.expressions.Root;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.junit.jupiter.api.Test;

public final class TestMinimal {
    // POJO representing the person data type.
    static final class Person {
        public String name;
        public BigDecimal salary;
        public String state;

        public Person(String name, BigDecimal salary, String state) {
            this.name = name;
            this.salary = salary;
            this.state = state;
        }

        public Person(Map<String, Object> map) {
            this.name = (String) map.get("Name");
            this.salary = (BigDecimal) map.get("Salary");
            this.state = (String) map.get("State");
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Person)) return false;
            Person person = (Person) o;
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
            return "Person{" + "name='" + name + '\'' + ", salary=" + salary + ", state='" + state + '\'' + '}';
        }
    }

    private static final String MINIMAL_PATH =
            TestMinimal.class.getResource("/minimal.vortex").getPath();

    // data representing the complete `minimal` test table:
    /// =======================
    /// Name | Salary  | State
    /// =======================
    /// Alice   1000    CA
    /// Bob     2000    NY
    /// Carol   3000    TX
    /// Dan     4000    CA
    /// Edward  5000    NY
    /// Frida   6000    TX
    /// George  7000    CA
    /// Henry   8000    NY
    /// Ida     9000    TX
    /// John    10000   VA
    /// =======================
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

    private static final List<Person> PROJECTED_DATA = List.of(
            new Person("Alice", null, "CA"),
            new Person("Bob", null, "NY"),
            new Person("Carol", null, "TX"),
            new Person("Dan", null, "CA"),
            new Person("Edward", null, "NY"),
            new Person("Frida", null, "TX"),
            new Person("George", null, "CA"),
            new Person("Henry", null, "NY"),
            new Person("Ida", null, "TX"),
            new Person("John", null, "VA"));

    @Test
    public void testFullScan() {
        try (var file = Files.open(MINIMAL_PATH);
                var fullScan = file.newScan(ScanOptions.of())) {
            assertEquals(10, file.rowCount());

            var dtype = fullScan.getDataType();
            assertEquals(DType.Variant.STRUCT, dtype.getVariant());
            assertEquals(dtype.getFieldNames(), List.of("Name", "Salary", "State"));

            // Perform a full scan, check the result.
            var people = readToList(fullScan, new TestCase() {
                @Override
                public Array[] open(Array batch) {
                    return new Array[] {
                        batch.getField(0), batch.getField(1), batch.getField(2),
                    };
                }

                @Override
                public Person readRow(Array[] fields, int idx) {
                    return new Person(fields[0].getUTF8(idx), fields[1].getBigDecimal(idx), fields[2].getUTF8(idx));
                }
            });

            assertEquals(MINIMAL_DATA, people);
        }
    }

    @Test
    public void testProjectedScan() {
        var projectOptions = ScanOptions.builder().addColumns("Name", "State").build();

        try (var file = Files.open(MINIMAL_PATH);
                var projectedScan = file.newScan(projectOptions)) {
            // Do stuff.
            var dtype = projectedScan.getDataType();
            assertEquals(DType.Variant.STRUCT, dtype.getVariant());
            assertEquals(dtype.getFieldNames(), List.of("Name", "State"));

            var people = readToList(projectedScan, new TestCase() {
                @Override
                public Array[] open(Array batch) {
                    return new Array[] {
                        batch.getField(0), batch.getField(1),
                    };
                }

                @Override
                public Person readRow(Array[] fields, int idx) {
                    return new Person(fields[0].getUTF8(idx), null, fields[1].getUTF8(idx));
                }
            });

            assertEquals(PROJECTED_DATA, people);
        }
    }

    @Test
    public void testProjectedScanWithFilter() {
        var filterOptions = ScanOptions.builder()
                .addColumns("State", "Salary", "Name")
                .predicate(Binary.eq(GetItem.of(Root.INSTANCE, "State"), Literal.string("VA")))
                .build();

        try (var file = Files.open(MINIMAL_PATH);
                var filteredScan = file.newScan(filterOptions)) {
            var dtype = filteredScan.getDataType();
            assertEquals(DType.Variant.STRUCT, dtype.getVariant());
            assertEquals(dtype.getFieldNames(), List.of("State", "Salary", "Name"));

            var people = readToList(filteredScan, new TestCase() {
                @Override
                public Array[] open(Array batch) {
                    return new Array[] {
                        batch.getField(0), // state
                        batch.getField(1), // salary
                        batch.getField(2), // name
                    };
                }

                @Override
                public Person readRow(Array[] fields, int idx) {
                    var state = fields[0].getUTF8(idx);
                    var salary = fields[1].getBigDecimal(idx);
                    var name = fields[2].getUTF8(idx);
                    return new Person(name, salary, state);
                }
            });

            assertEquals(List.of(new Person("John", BigDecimal.valueOf(10_000L, 2), "VA")), people);
        }
    }

    interface TestCase {
        Array[] open(Array batch);

        Person readRow(Array[] fields, int idx);
    }

    private List<Person> readToList(ArrayIterator scan, TestCase testCase) {
        List<Person> people = new ArrayList<>();
        while (scan.hasNext()) {
            var batch = scan.next();
            Array[] fields = testCase.open(batch);

            for (int batchIdx = 0; batchIdx < batch.getLen(); batchIdx++) {
                people.add(testCase.readRow(fields, batchIdx));
            }
        }
        return people;
    }
}
