// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import org.apache.spark.sql.util.CaseInsensitiveStringMap;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link VortexOptions}.
 *
 * <p>Two properties matter most: options resolve case-insensitively, because Spark lower-cases the keys of the map it
 * hands to a table; and instances survive Java serialization, because they cross the boundary to the executors inside
 * {@code VortexFilePartition} and the reader/writer factories.
 */
final class VortexOptionsTest {

    @Test
    @DisplayName("Worker threads default to 4 and are read case-insensitively")
    void workerThreadsCaseInsensitive() {
        assertEquals(4, VortexOptions.empty().workerThreads());
        assertEquals(16, VortexOptions.of(Map.of("vortex.workerThreads", "16")).workerThreads());
        // The spelling that actually arrives, after Spark has lower-cased the keys.
        assertEquals(16, VortexOptions.of(Map.of("vortex.workerthreads", "16")).workerThreads());
        assertEquals(16, VortexOptions.of(Map.of("VORTEX.WORKERTHREADS", "16")).workerThreads());
    }

    @Test
    @DisplayName("Worker threads accept zero but reject a negative count")
    void workerThreadsRejectNegative() {
        assertEquals(0, VortexOptions.of(Map.of("vortex.workerThreads", "0")).workerThreads());

        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> VortexOptions.of(Map.of("vortex.workerThreads", "-1")).workerThreads());
        assertTrue(thrown.getMessage().contains("vortex.workerThreads"), thrown.getMessage());
    }

    @Test
    @DisplayName("A non-numeric worker thread count is rejected naming the option and the value")
    void workerThreadsRejectNonNumeric() {
        IllegalArgumentException thrown = assertThrows(
                IllegalArgumentException.class,
                () -> VortexOptions.of(Map.of("vortex.workerThreads", "eight")).workerThreads());

        assertTrue(thrown.getMessage().contains("vortex.workerThreads"), thrown.getMessage());
        assertTrue(thrown.getMessage().contains("eight"), thrown.getMessage());
    }

    @Test
    @DisplayName("Write batch size defaults to 2048 and honours the legacy key")
    void writeBatchSize() {
        assertEquals(2048, VortexOptions.empty().writeBatchSize());
        assertEquals(
                4096,
                VortexOptions.of(Map.of("vortex.write.batch.size", "4096")).writeBatchSize());
        assertEquals(4096, VortexOptions.of(Map.of("batch.size", "4096")).writeBatchSize());
        assertEquals(
                4096,
                VortexOptions.of(Map.of("VORTEX.WRITE.BATCH.SIZE", "4096")).writeBatchSize());
    }

    @Test
    @DisplayName("The documented write batch size key wins over the legacy one")
    void writeBatchSizePrefersCurrentKey() {
        VortexOptions options = VortexOptions.of(Map.of("vortex.write.batch.size", "4096", "batch.size", "1024"));

        assertEquals(4096, options.writeBatchSize());
    }

    @Test
    @DisplayName("An out-of-range write batch size falls back to the default and is reported")
    void writeBatchSizeOutOfRange() {
        VortexOptions tooSmall = VortexOptions.of(Map.of("vortex.write.batch.size", "0"));
        VortexOptions tooLarge = VortexOptions.of(Map.of("vortex.write.batch.size", "65537"));

        assertEquals(2048, tooSmall.writeBatchSize());
        assertEquals(
                Optional.of(new VortexOptions.RejectedOption("vortex.write.batch.size", 0)),
                tooSmall.rejectedWriteBatchSize());
        assertEquals(2048, tooLarge.writeBatchSize());
        assertEquals(
                Optional.of(new VortexOptions.RejectedOption("vortex.write.batch.size", 65537)),
                tooLarge.rejectedWriteBatchSize());

        VortexOptions inRange = VortexOptions.of(Map.of("vortex.write.batch.size", "4096"));
        assertEquals(Optional.empty(), inRange.rejectedWriteBatchSize());

        // An out-of-range legacy value is reported under the key the user actually set.
        VortexOptions legacy = VortexOptions.of(Map.of("batch.size", "999999"));
        assertEquals(2048, legacy.writeBatchSize());
        assertEquals(
                Optional.of(new VortexOptions.RejectedOption("batch.size", 999999)), legacy.rejectedWriteBatchSize());
    }

    @Test
    @DisplayName("Session provider is empty when unset or blank")
    void sessionProvider() {
        assertEquals(Optional.empty(), VortexOptions.empty().sessionProvider());
        assertEquals(
                Optional.empty(),
                VortexOptions.of(Map.of("vortex.session.provider", "")).sessionProvider());
        assertEquals(
                Optional.of("com.example.Provider"),
                VortexOptions.of(Map.of("vortex.session.provider", "com.example.Provider"))
                        .sessionProvider());
        assertEquals(
                Optional.of("com.example.Provider"),
                VortexOptions.of(Map.of("vortex.session.PROVIDER", "com.example.Provider"))
                        .sessionProvider());
    }

    @Test
    @DisplayName("An override replaces the option it means to, whatever the original spelling")
    void overrideReplacesDifferentlySpelledKey() {
        VortexOptions table = VortexOptions.of(Map.of("vortex.workerThreads", "4"));

        VortexOptions scan = table.withOverrides(Map.of("vortex.workerthreads", "16"));

        assertEquals(16, scan.workerThreads());
        // The stale spelling must be gone, or a case-sensitive reader would still see the old value.
        assertEquals(1, scan.asMap().size());
    }

    @Test
    @DisplayName("Overrides keep unrelated options and leave the receiver untouched")
    void overridesAreAdditiveAndNonMutating() {
        Map<String, String> initial = new LinkedHashMap<>();
        initial.put("aws_region", "us-east-1");
        initial.put("vortex.workerThreads", "4");
        VortexOptions table = VortexOptions.of(initial);

        VortexOptions scan = table.withOverrides(Map.of("vortex.workerThreads", "8"));

        assertEquals("us-east-1", scan.asMap().get("aws_region"));
        assertEquals(8, scan.workerThreads());
        assertEquals(4, table.workerThreads());
    }

    @Test
    @DisplayName("Overriding with nothing returns the same instance")
    void emptyOverridesReturnSameInstance() {
        VortexOptions options = VortexOptions.of(Map.of("vortex.workerThreads", "4"));

        assertSame(options, options.withOverrides(Map.of()));
    }

    @Test
    @DisplayName("Survives Java serialization, including the transient case-insensitive view")
    void roundTripsThroughSerialization() throws IOException, ClassNotFoundException {
        VortexOptions original = VortexOptions.of(Map.of("vortex.workerThreads", "16", "aws_region", "us-east-1"));
        // Resolve first, so the transient case-insensitive view is populated: serializing a fresh
        // instance would pass even if that field were not transient.
        assertEquals(16, original.workerThreads());

        VortexOptions restored = roundTrip(original);

        assertEquals(original, restored);
        // Resolved after deserialization, so the transient lookup map must have been rebuilt.
        assertEquals(16, restored.workerThreads());
        assertEquals("us-east-1", restored.asMap().get("aws_region"));
    }

    @Test
    @DisplayName("Equality and hashing follow the underlying options")
    void equalityFollowsOptions() {
        VortexOptions one = VortexOptions.of(Map.of("vortex.workerThreads", "4"));
        VortexOptions same = VortexOptions.of(Map.of("vortex.workerThreads", "4"));
        VortexOptions other = VortexOptions.of(Map.of("vortex.workerThreads", "8"));

        assertEquals(one, same);
        assertEquals(one.hashCode(), same.hashCode());
        assertNotEquals(one, other);
        assertEquals(VortexOptions.empty(), VortexOptions.of(Map.of()));
    }

    @Test
    @DisplayName("The raw map is what the native bindings receive, unchanged")
    void asMapPreservesOriginalSpelling() {
        VortexOptions options = VortexOptions.of(Map.of("aws_region", "us-east-1"));

        assertEquals(Map.of("aws_region", "us-east-1"), options.asMap());
    }

    @Test
    @DisplayName("A valid current key wins even when the legacy key holds garbage")
    void currentKeyShortCircuitsLegacyKey() {
        VortexOptions options =
                VortexOptions.of(Map.of("vortex.write.batch.size", "4096", "batch.size", "not-a-number"));

        assertEquals(4096, options.writeBatchSize());
        assertEquals(Optional.empty(), options.rejectedWriteBatchSize());
    }

    @Test
    @DisplayName("Surrounding whitespace in a value is tolerated")
    void trimsValues() {
        assertEquals(
                16, VortexOptions.of(Map.of("vortex.workerThreads", " 16 ")).workerThreads());
        assertEquals(
                4096,
                VortexOptions.of(Map.of("vortex.write.batch.size", " 4096 ")).writeBatchSize());
    }

    @Test
    @DisplayName("The map handed to the native bindings cannot be mutated")
    void asMapIsImmutable() {
        Map<String, String> exposed =
                VortexOptions.of(Map.of("aws_region", "us-east-1")).asMap();

        assertThrows(UnsupportedOperationException.class, () -> exposed.put("aws_region", "eu-west-1"));
    }

    @Test
    @DisplayName("Wrapping copies the map, so later changes to the caller's map are not observed")
    void ofCopiesTheSuppliedMap() {
        Map<String, String> mutable = new LinkedHashMap<>();
        mutable.put("vortex.workerThreads", "4");
        VortexOptions options = VortexOptions.of(mutable);

        mutable.put("vortex.workerThreads", "16");

        assertEquals(4, options.workerThreads());
    }

    @Test
    @DisplayName("An override supplied through Spark's own map displaces the original spelling")
    void overrideThroughSparkMapDisplacesOriginal() {
        // This is the shape production sees: Spark hands the table a CaseInsensitiveStringMap, whose
        // keys are already lower-cased, to override the table-level options.
        VortexOptions table = VortexOptions.of(Map.of("vortex.workerThreads", "4"));

        VortexOptions scan = table.withOverrides(new CaseInsensitiveStringMap(Map.of("vortex.workerThreads", "16")));

        assertEquals(16, scan.workerThreads());
        assertEquals(1, scan.asMap().size(), scan.asMap().toString());
    }

    @Test
    @DisplayName("Options are rejected rather than silently treated as empty")
    void ofRejectsNull() {
        assertThrows(NullPointerException.class, () -> VortexOptions.of(null));
    }

    private static VortexOptions roundTrip(VortexOptions options) throws IOException, ClassNotFoundException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(options);
        }
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            return (VortexOptions) in.readObject();
        }
    }
}
