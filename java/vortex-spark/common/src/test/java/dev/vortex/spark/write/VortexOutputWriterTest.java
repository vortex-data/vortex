// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.write;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.spark.VortexOptions;
import java.util.Map;
import org.junit.jupiter.api.Test;

final class VortexOutputWriterTest {
    private static final int DEFAULT = 2048;

    @Test
    void anUnsetBatchSizeFallsBackToTheDefault() {
        assertEquals(DEFAULT, VortexOutputWriter.configuredBatchSize(VortexOptions.empty()));
    }

    @Test
    void theGenericBatchSizeIsHonoured() {
        assertEquals(4096, VortexOutputWriter.configuredBatchSize(options(Map.of("batch.size", "4096"))));
    }

    @Test
    void theVortexBatchSizeOverridesTheGenericOne() {
        VortexOptions options = options(Map.of("batch.size", "4096", "vortex.write.batch.size", "512"));

        // A job that sets one batch size across formats can still say something different for Vortex.
        assertEquals(512, VortexOutputWriter.configuredBatchSize(options));
    }

    @Test
    void theOverrideStandsEvenWhenTheGenericValueIsUnusable() {
        VortexOptions options = options(Map.of("batch.size", "lots", "vortex.write.batch.size", "512"));

        assertEquals(512, VortexOutputWriter.configuredBatchSize(options));
    }

    @Test
    void eitherNameIsFoundUnderAnyCasing() {
        assertEquals(64, VortexOutputWriter.configuredBatchSize(options(Map.of("BATCH.SIZE", "64"))));
        assertEquals(64, VortexOutputWriter.configuredBatchSize(options(Map.of("Vortex.Write.Batch.Size", "64"))));
    }

    @Test
    void aBatchSizeOfTheWrongShapeNamesItsOption() {
        VortexOptions options = options(Map.of("vortex.write.batch.size", "lots"));

        assertTrue(assertThrows(IllegalArgumentException.class, () -> VortexOutputWriter.configuredBatchSize(options))
                .getMessage()
                .contains("vortex.write.batch.size"));
    }

    private static VortexOptions options(Map<String, String> values) {
        return VortexOptions.of(values);
    }
}
