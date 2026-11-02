// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.spark.VortexOptions;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.junit.jupiter.api.Test;

final class VortexIoTest {
    @Test
    void readConcurrencyDefaultsToNativeDefault() {
        VortexIo io = VortexIo.create(VortexOptions.empty(), new Configuration());

        assertEquals(0, io.readConcurrency());
    }

    @Test
    void readConcurrencyIsParsedAndValidated() {
        Configuration conf = new Configuration();

        assertEquals(
                8,
                VortexIo.create(options(VortexIo.READ_CONCURRENCY_OPTION, "8"), conf)
                        .readConcurrency());
        assertThrows(
                IllegalArgumentException.class,
                () -> VortexIo.create(options(VortexIo.READ_CONCURRENCY_OPTION, "-1"), conf));
        assertThrows(
                IllegalArgumentException.class,
                () -> VortexIo.create(options(VortexIo.READ_CONCURRENCY_OPTION, "many"), conf));
    }

    @Test
    void readConcurrencyIsFoundUnderAnyKeyCase() {
        Configuration conf = new Configuration();

        assertEquals(
                6, VortexIo.create(options("VORTEX.READCONCURRENCY", "6"), conf).readConcurrency());
        assertEquals(
                6, VortexIo.create(options("vortex.readconcurrency", "6"), conf).readConcurrency());
    }

    @Test
    void settingsAndConfigurationSurviveSerialization() throws IOException, ClassNotFoundException {
        Configuration conf = new Configuration();
        conf.set("fs.s3a.endpoint", "https://storage.example");
        VortexIo io = VortexIo.create(options(VortexIo.READ_CONCURRENCY_OPTION, "4"), conf);

        VortexIo shipped = roundTrip(io);

        assertEquals(4, shipped.readConcurrency());
        assertEquals("https://storage.example", shipped.hadoopConf().get("fs.s3a.endpoint"));
    }

    private static VortexOptions options(String key, String value) {
        return VortexOptions.of(Map.of(key, value));
    }

    private static VortexIo roundTrip(VortexIo io) throws IOException, ClassNotFoundException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(io);
        }
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            return (VortexIo) in.readObject();
        }
    }
}
