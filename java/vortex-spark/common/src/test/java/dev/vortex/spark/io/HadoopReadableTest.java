// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.io.NativeReadable;
import java.io.EOFException;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FutureDataInputStreamBuilder;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.fs.impl.FutureDataInputStreamBuilderImpl;
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
            assertEquals(NativeReadable.DEFAULT_COALESCE_DISTANCE, readable.coalesceDistance());
            assertEquals(NativeReadable.DEFAULT_COALESCE_MAX_SIZE, readable.coalesceMaxSize());
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
    void everyStreamUsesTheResolvedLengthAndRandomReadPolicy() throws IOException {
        byte[] content = content(64);
        Path file = write("builder.vortex", content);
        for (long suppliedLength : new long[] {-1, content.length}) {
            try (TrackingFileSystem fs = trackingFileSystem();
                    HadoopReadable readable =
                            HadoopReadable.open(fs.getConf(), trackedPath(fs, file), suppliedLength)) {
                assertEquals(suppliedLength < 0 ? 1 : 0, fs.stats);
                // Open one stream directly and let a pooled read open a second.
                try (var first = readable.openStream()) {
                    ByteBuffer buffer = ByteBuffer.allocateDirect(content.length);
                    readable.readFully(0, buffer);
                    assertArrayEquals(content, drain(buffer));
                }
                assertEquals(2, fs.openOptions.size());
                for (Configuration options : fs.openOptions) {
                    assertEquals("64", options.get("fs.option.openfile.length"));
                    assertEquals("random", options.get("fs.option.openfile.read.policy"));
                }
            }
        }
    }

    @Test
    void failedAsynchronousOpensPreserveTheirOriginalExceptions() throws IOException {
        Path file = write("failed-open.vortex", content(64));
        IOException missing = new FileNotFoundException("missing object");
        for (Throwable failure : new Throwable[] {
            missing,
            new UncheckedIOException(missing),
            new IllegalArgumentException("invalid option"),
            new AssertionError("open failed")
        }) {
            try (TrackingFileSystem fs = trackingFileSystem();
                    HadoopReadable readable = HadoopReadable.open(fs.getConf(), trackedPath(fs, file), 64)) {
                fs.nextOpen = CompletableFuture.failedFuture(failure);
                Throwable expected = failure instanceof UncheckedIOException ? missing : failure;
                assertSame(
                        expected,
                        assertThrows(expected.getClass(), () -> readable.readFully(0, ByteBuffer.allocateDirect(1))));
            }
        }
    }

    @Test
    void interruptedOpenPreservesInterruptAndClosesALateStream() throws IOException {
        Path file = write("interrupted-open.vortex", content(64));
        try (TrackingFileSystem fs = trackingFileSystem();
                HadoopReadable readable = HadoopReadable.open(fs.getConf(), trackedPath(fs, file), 64)) {
            CompletableFuture<FSDataInputStream> pending = new CompletableFuture<>();
            fs.nextOpen = pending;
            Thread.currentThread().interrupt();
            try {
                InterruptedIOException failure = assertThrows(
                        InterruptedIOException.class, () -> readable.readFully(0, ByteBuffer.allocateDirect(1)));
                assertTrue(Thread.currentThread().isInterrupted());
                assertTrue(failure.getCause() instanceof InterruptedException);
            } finally {
                Thread.interrupted();
            }
            assertFalse(pending.isCancelled());
            FSDataInputStream lateStream = fs.open(new org.apache.hadoop.fs.Path(trackedPath(fs, file)));
            pending.complete(lateStream);
            assertThrows(IOException.class, lateStream::read);
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

    private static TrackingFileSystem trackingFileSystem() throws IOException {
        Configuration trackingConf = new Configuration();
        trackingConf.setClass("fs.tracking.impl", TrackingFileSystem.class, FileSystem.class);
        return (TrackingFileSystem) FileSystem.get(URI.create("tracking://" + UUID.randomUUID()), trackingConf);
    }

    private static String trackedPath(TrackingFileSystem fs, Path file) {
        return new org.apache.hadoop.fs.Path("tracking", fs.getUri().getAuthority(), file.toString()).toString();
    }

    /** Hadoop instantiates this filesystem through the test configuration. */
    public static final class TrackingFileSystem extends RawLocalFileSystem {
        private URI uri;
        private int stats;
        private final List<Configuration> openOptions = new ArrayList<>();
        private CompletableFuture<FSDataInputStream> nextOpen;

        @Override
        public void initialize(URI name, Configuration configuration) throws IOException {
            uri = name;
            super.initialize(name, configuration);
        }

        @Override
        public URI getUri() {
            return uri != null ? uri : super.getUri();
        }

        @Override
        public FileStatus getFileStatus(org.apache.hadoop.fs.Path path) throws IOException {
            stats++;
            return super.getFileStatus(path);
        }

        @Override
        public FutureDataInputStreamBuilder openFile(org.apache.hadoop.fs.Path path) {
            return new FutureDataInputStreamBuilderImpl(this, path) {
                @Override
                public CompletableFuture<FSDataInputStream> build() throws IOException {
                    assertTrue(getMandatoryKeys().isEmpty(), "Open options must remain optional for other filesystems");
                    openOptions.add(new Configuration(getOptions()));
                    return nextOpen != null
                            ? nextOpen
                            : CompletableFuture.completedFuture(TrackingFileSystem.this.open(path, getBufferSize()));
                }
            };
        }
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
