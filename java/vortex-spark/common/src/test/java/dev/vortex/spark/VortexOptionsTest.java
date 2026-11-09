// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

final class VortexOptionsTest {
    @ParameterizedTest
    @ValueSource(
            strings = {
                "vortex.readConcurrency",
                "vortex.readconcurrency",
                "VORTEX.READCONCURRENCY",
                "Vortex.ReadConcurrency"
            })
    void anyCasingFindsTheValue(String key) {
        VortexOptions options = VortexOptions.of(Map.of("vortex.readConcurrency", "4"));

        assertEquals(Optional.of("4"), options.get(key));
        assertEquals(4, options.getInt(key, 1));
    }

    @Test
    void anUnsetOptionFallsBack() {
        VortexOptions options = VortexOptions.empty();

        assertEquals(Optional.empty(), options.get("vortex.absent"));
        assertEquals("fallback", options.get("vortex.absent", "fallback"));
        assertEquals(7, options.getInt("vortex.absent", 7));
        assertTrue(options.getBoolean("vortex.absent", true));
    }

    @Test
    void booleansAreReadWithoutRegardToCase() {
        VortexOptions options = VortexOptions.of(Map.of("on", "TRUE", "off", " False "));

        assertTrue(options.getBoolean("on", false));
        assertFalse(options.getBoolean("off", true));
    }

    @Test
    void aValueOfTheWrongShapeNamesItsOption() {
        VortexOptions options = VortexOptions.of(Map.of("vortex.count", "many", "vortex.flag", "yes"));

        assertTrue(assertThrows(IllegalArgumentException.class, () -> options.getInt("vortex.count", 1))
                .getMessage()
                .contains("vortex.count"));
        assertTrue(assertThrows(IllegalArgumentException.class, () -> options.getBoolean("vortex.flag", true))
                .getMessage()
                .contains("vortex.flag"));
    }

    @Test
    void hadoopKeysKeepTheCasingTheyWereGivenIn() {
        Map<String, String> given = Map.of("fs.s3a.Endpoint", "https://storage.example");

        // Hadoop configuration keys are case-sensitive, so the map that feeds a Configuration must not be
        // lower-cased along with the vortex.* lookups.
        assertEquals(given, VortexOptions.of(given).asCaseSensitiveMap());
    }

    @Test
    void optionsSurviveSerialization() throws IOException, ClassNotFoundException {
        VortexOptions options = VortexOptions.of(Map.of("vortex.workerThreads", "2"));

        VortexOptions shipped = roundTrip(options);

        assertEquals(2, shipped.getInt("vortex.workerthreads", 1));
        assertEquals(options, shipped);
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
