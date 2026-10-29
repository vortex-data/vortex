// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.api.Session;
import dev.vortex.api.VortexWriter;
import dev.vortex.jni.NativeLoader;
import dev.vortex.jni.NativeRuntime;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.types.pojo.ArrowType;
import org.apache.arrow.vector.types.pojo.Field;
import org.apache.arrow.vector.types.pojo.Schema;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;

/** Native workers that call back into Java attach to the JVM; they must not keep it alive or leak once retired. */
@DisabledOnOs(value = OS.WINDOWS, disabledReason = "native workers keep the jni crate's non-daemon attachment there")
final class NativeWorkerAttachmentTest {
    private static final int WORKERS = 4;

    @BeforeAll
    static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @AfterEach
    void stopWorkers() {
        NativeRuntime.setWorkerThreads(0);
    }

    /** Blocks the first write until every writer's first write has started, so each runs on its own worker. */
    private static final class Sink implements NativeWritable {
        private final CountDownLatch started;
        private final CountDownLatch release;
        private volatile Thread thread;

        Sink(CountDownLatch started, CountDownLatch release) {
            this.started = started;
            this.release = release;
        }

        @Override
        public void write(byte[] buffer, int offset, int length) {
            if (thread == null) {
                thread = Thread.currentThread();
                started.countDown();
                try {
                    assertTrue(release.await(30, TimeUnit.SECONDS));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }

        @Override
        public void flush() {}

        @Override
        public void close() {}
    }

    @Test
    void retiredWorkersDetachFromTheJvm() throws Exception {
        Schema schema = new Schema(List.of(Field.notNullable("id", new ArrowType.Int(64, true))));
        Session session = Session.create();
        NativeRuntime.setWorkerThreads(WORKERS);
        CountDownLatch started = new CountDownLatch(WORKERS);
        CountDownLatch release = new CountDownLatch(1);
        List<Sink> sinks = new ArrayList<>();
        try (BufferAllocator allocator = new RootAllocator()) {
            List<VortexWriter> writers = new ArrayList<>();
            for (int i = 0; i < WORKERS; i++) {
                Sink sink = new Sink(started, release);
                sinks.add(sink);
                writers.add(
                        VortexWriter.builder(session, sink, schema, allocator).build());
            }
            boolean allStarted = started.await(30, TimeUnit.SECONDS);
            release.countDown();
            assertTrue(allStarted, "every writer should reach its first upcall on a worker");
            for (VortexWriter writer : writers) {
                writer.close();
            }
        }
        for (Sink sink : sinks) {
            assertTrue(sink.thread.isDaemon(), "a worker attached as a non-daemon thread keeps the JVM alive");
        }

        NativeRuntime.setWorkerThreads(0);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        long alive;
        do {
            Thread.sleep(100);
            alive = sinks.stream()
                    .map(sink -> sink.thread)
                    .filter(Thread::isAlive)
                    .count();
        } while (alive > 0 && System.nanoTime() < deadline);
        assertEquals(0, alive, "retired workers must detach from the JVM");
    }
}
