// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.spark.write.PartitionedVortexDataWriter.ResolvedTransform;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.List;
import org.apache.spark.sql.connector.expressions.Expressions;
import org.apache.spark.sql.connector.expressions.Transform;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link PartitionedVortexDataWriter#resolveTransforms}, which turns Spark partition transforms into the
 * directory keys and column indices a writer uses.
 *
 * <p>Resolution happens eagerly on the driver because Spark's {@code Transform} objects are Scala case classes that are
 * not Java-serializable, so only the resolved form may cross to the executors; the serialization round trip below is
 * what that eager step exists for.
 */
final class PartitionedVortexDataWriterTest {

    private static final StructType SCHEMA = new StructType(new StructField[] {
        new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
        new StructField("name", DataTypes.StringType, true, Metadata.empty()),
        new StructField("event_date", DataTypes.DateType, true, Metadata.empty()),
        new StructField("event_ts", DataTypes.TimestampType, true, Metadata.empty())
    });

    private static ResolvedTransform resolveOne(Transform transform) {
        ResolvedTransform[] resolved =
                PartitionedVortexDataWriter.resolveTransforms(new Transform[] {transform}, SCHEMA);
        assertEquals(1, resolved.length);
        return resolved[0];
    }

    @Test
    @DisplayName("No transforms resolve to no partitioning")
    void emptyInput() {
        assertEquals(0, PartitionedVortexDataWriter.resolveTransforms(new Transform[0], SCHEMA).length);
    }

    @Test
    @DisplayName("An identity transform keeps the column name as its directory key")
    void identity() {
        ResolvedTransform resolved = resolveOne(Expressions.identity("name"));

        assertEquals("name", resolved.directoryKey());
        assertEquals("identity", resolved.transformName());
        assertEquals(1, resolved.columnIndices().get(0));
        assertEquals(DataTypes.StringType, resolved.columnTypes().get(0));
        assertEquals(-1, resolved.bucketCount(), "only bucket transforms carry a bucket count");
    }

    @Test
    @DisplayName("Temporal transforms suffix the directory key with the unit they truncate to")
    void temporalDirectoryKeys() {
        assertEquals(
                "event_date_year", resolveOne(Expressions.years("event_date")).directoryKey());
        assertEquals(
                "event_date_month", resolveOne(Expressions.months("event_date")).directoryKey());
        assertEquals(
                "event_date_day", resolveOne(Expressions.days("event_date")).directoryKey());
        assertEquals("event_ts_hour", resolveOne(Expressions.hours("event_ts")).directoryKey());
    }

    @Test
    @DisplayName("Temporal transforms reject a non-temporal column, naming the transform and the type")
    void temporalTransformsRejectNonTemporalColumns() {
        IllegalArgumentException thrown =
                assertThrows(IllegalArgumentException.class, () -> resolveOne(Expressions.years("id")));

        assertTrue(thrown.getMessage().contains("years"), thrown.getMessage());
        assertTrue(thrown.getMessage().contains("IntegerType"), thrown.getMessage());
    }

    @Test
    @DisplayName("The hours transform additionally rejects a date column, which has no hour to truncate to")
    void hoursRejectsDateColumn() {
        IllegalArgumentException thrown =
                assertThrows(IllegalArgumentException.class, () -> resolveOne(Expressions.hours("event_date")));

        assertTrue(thrown.getMessage().contains("hours"), thrown.getMessage());
    }

    @Test
    @DisplayName("A bucket transform carries its bucket count and joins its columns with underscores")
    void bucket() {
        ResolvedTransform single = resolveOne(Expressions.bucket(8, "id"));
        assertEquals("id_bucket", single.directoryKey());
        assertEquals(8, single.bucketCount());

        ResolvedTransform multi = resolveOne(Expressions.bucket(4, "id", "name"));
        assertEquals("id_name_bucket", multi.directoryKey());
        assertEquals(4, multi.bucketCount());
        assertEquals(0, multi.columnIndices().get(0));
        assertEquals(1, multi.columnIndices().get(1));
        assertEquals(List.of(DataTypes.IntegerType, DataTypes.StringType), multi.columnTypes());
    }

    @Test
    @DisplayName("A bucket transform without a numBuckets argument is rejected")
    void bucketWithoutCount() {
        Transform noCount = Expressions.apply("bucket", Expressions.column("id"));

        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class, () -> resolveOne(noCount));

        assertTrue(thrown.getMessage().contains("numBuckets"), thrown.getMessage());
    }

    @Test
    @DisplayName("An unsupported transform name is rejected")
    void unsupportedTransform() {
        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> resolveOne(Expressions.apply("truncate", Expressions.column("name"))));

        assertTrue(thrown.getMessage().contains("truncate"), thrown.getMessage());
    }

    @Test
    @DisplayName("A transform that references no column is rejected")
    void transformWithoutReferences() {
        IllegalArgumentException thrown =
                assertThrows(IllegalArgumentException.class, () -> resolveOne(Expressions.apply("identity")));

        assertTrue(thrown.getMessage().contains("no column references"), thrown.getMessage());
    }

    @Test
    @DisplayName("A transform on a column outside the schema is rejected")
    void transformOnUnknownColumn() {
        assertThrows(IllegalArgumentException.class, () -> resolveOne(Expressions.identity("missing")));
    }

    @Test
    @DisplayName("Resolved transforms survive Java serialization, which is why they are resolved eagerly")
    void resolvedTransformsAreSerializable() throws IOException, ClassNotFoundException {
        ResolvedTransform original = resolveOne(Expressions.bucket(4, "id", "name"));

        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(original);
        }
        ResolvedTransform restored;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            restored = (ResolvedTransform) in.readObject();
        }

        assertEquals(original.directoryKey(), restored.directoryKey());
        assertEquals(original.transformName(), restored.transformName());
        assertEquals(original.bucketCount(), restored.bucketCount());
        assertEquals(original.columnTypes(), restored.columnTypes());
        assertEquals(original.columnIndices().asList(), restored.columnIndices().asList());
    }
}
