// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import dev.vortex.io.NativeReadable;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.hadoop.conf.Configuration;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class HadoopReadableTest {
    private static final int SIZE = 512 * 1024;

    private final Configuration conf = new Configuration();

    @TempDir
    Path tempDir;

    @Test
    void readsEveryRangeOfTheFile() throws IOException {
        byte[] content = content(SIZE);
        Path file = write("data.vortex", content);

        try (NativeReadable readable = HadoopReadable.open(conf, file.toString())) {
            assertEquals(SIZE, readable.length());
            assertEquals(file.toString(), readable.name());

            for (int[] range : new int[][] {{0, SIZE}, {0, 1}, {SIZE - 1, 1}, {1234, 65536}, {SIZE / 2, SIZE / 2}}) {
                ByteBuffer buffer = ByteBuffer.allocateDirect(range[1]);
                readable.readFully(range[0], buffer);
                assertArrayEquals(
                        slice(content, range[0], range[1]), drain(buffer), "range " + range[0] + "+" + range[1]);
            }
        }
    }

    @Test
    void suppliedLengthIsTrustedOverAnExtraStat() throws IOException {
        Path file = write("data.vortex", content(64));

        try (NativeReadable readable = HadoopReadable.open(conf, file.toString(), 64)) {
            assertEquals(64, readable.length());
        }
    }

    @Test
    void concurrentReadsEachGetTheirOwnStream() throws Exception {
        byte[] content = content(SIZE);
        Path file = write("data.vortex", content);
        int readers = 8;
        int chunk = SIZE / readers;

        try (NativeReadable readable = HadoopReadable.open(conf, file.toString())) {
            ExecutorService pool = Executors.newFixedThreadPool(readers);
            try {
                List<Callable<byte[]>> reads = new ArrayList<>();
                for (int i = 0; i < readers; i++) {
                    int offset = i * chunk;
                    reads.add(() -> {
                        byte[] read = new byte[chunk];
                        // Read the same range repeatedly so streams keep being returned to and taken
                        // from the pool while other threads are doing the same.
                        for (int round = 0; round < 20; round++) {
                            ByteBuffer buffer = ByteBuffer.allocateDirect(chunk);
                            readable.readFully(offset, buffer);
                            read = drain(buffer);
                        }
                        return read;
                    });
                }

                List<Future<byte[]>> results = pool.invokeAll(reads);
                for (int i = 0; i < readers; i++) {
                    assertArrayEquals(
                            slice(content, i * chunk, chunk), results.get(i).get(), "reader " + i);
                }
            } finally {
                pool.shutdownNow();
            }
        }
    }

    @Test
    void readsPastTheEndFail() throws IOException {
        Path file = write("data.vortex", content(1024));

        try (NativeReadable readable = HadoopReadable.open(conf, file.toString())) {
            assertThrows(EOFException.class, () -> readable.readFully(512, ByteBuffer.allocateDirect(1024)));
            assertThrows(EOFException.class, () -> readable.readFully(1024, ByteBuffer.allocateDirect(1)));
        }
    }

    @Test
    void readingAfterCloseFails() throws IOException {
        Path file = write("data.vortex", content(1024));

        NativeReadable readable = HadoopReadable.open(conf, file.toString());
        readable.readFully(0, ByteBuffer.allocateDirect(16));
        readable.close();

        assertThrows(IllegalStateException.class, () -> readable.readFully(0, ByteBuffer.allocateDirect(16)));
    }

    @Test
    void writtenBytesAreReadBackThroughTheHadoopBridge() throws IOException {
        byte[] content = content(4096);
        Path file = tempDir.resolve("written.vortex");

        try (HadoopWritable writable = HadoopWritable.create(conf, file.toString())) {
            writable.write(content, 0, 1024);
            writable.flush();
            writable.write(content, 1024, content.length - 1024);
        }

        assertArrayEquals(content, Files.readAllBytes(file));

        try (NativeReadable readable = HadoopReadable.open(conf, file.toString())) {
            ByteBuffer buffer = ByteBuffer.allocateDirect(content.length);
            readable.readFully(0, buffer);
            assertArrayEquals(content, drain(buffer));
        }
    }

    @Test
    void createReplacesAnExistingFile() throws IOException {
        Path file = tempDir.resolve("replaced.vortex");
        Files.write(file, new byte[] {1, 2, 3, 4, 5, 6, 7, 8});

        try (HadoopWritable writable = HadoopWritable.create(conf, file.toString())) {
            writable.write(new byte[] {9}, 0, 1);
        }

        assertArrayEquals(new byte[] {9}, Files.readAllBytes(file));
    }

    private Path write(String name, byte[] content) throws IOException {
        Path file = tempDir.resolve(name);
        Files.write(file, content);
        return file;
    }

    private static byte[] content(int size) {
        byte[] content = new byte[size];
        for (int i = 0; i < size; i++) {
            content[i] = (byte) (i % 251);
        }
        return content;
    }

    private static byte[] slice(byte[] content, int offset, int length) {
        byte[] expected = new byte[length];
        System.arraycopy(content, offset, expected, 0, length);
        return expected;
    }

    private static byte[] drain(ByteBuffer buffer) {
        byte[] read = new byte[buffer.position()];
        buffer.flip();
        buffer.get(read);
        return read;
    }
}
