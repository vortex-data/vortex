// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import dev.vortex.io.PooledReadable;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.StreamCapabilities;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.util.functional.FutureIO;

/** A {@link dev.vortex.io.NativeReadable} that serves Vortex's reads from a Hadoop {@link FileSystem}. */
public final class HadoopReadable extends PooledReadable {
    private final Path path;
    private final FileSystem fs;

    /** Opens a readable over {@code path}, stating the file for its length. */
    public static HadoopReadable open(Configuration conf, String path) {
        return open(conf, path, -1);
    }

    /**
     * Opens a readable over {@code path}. A non-negative {@code length} is taken as authoritative — the driver already
     * stats every file when it plans the scan, so executors need not stat them again.
     */
    public static HadoopReadable open(Configuration conf, String path, long length) {
        Path file = new Path(path);
        try {
            FileSystem fs = file.getFileSystem(conf);
            long resolved = length >= 0 ? length : fs.getFileStatus(file).getLen();
            return new HadoopReadable(path, file, fs, resolved);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to open " + path, e);
        }
    }

    private HadoopReadable(String name, Path path, FileSystem fs, long length) {
        super(name, length);
        this.path = path;
        this.fs = fs;
    }

    @Override
    protected PositionalStream openStream() throws IOException {
        // Given the length, S3A on Hadoop 3.3.5+ skips the metadata HEAD per stream; older Hadoop
        // (Spark 3.5) ignores these keys. String keys because 3.3.4 lacks the OpenFileOptions constants.
        CompletableFuture<FSDataInputStream> opened = fs.openFile(path)
                .opt("fs.option.openfile.length", Long.toString(length()))
                .opt("fs.option.openfile.read.policy", "random")
                .build();
        try {
            return new HadoopStream(opened.get());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            InterruptedIOException failure = new InterruptedIOException("Interrupted while opening " + path);
            failure.initCause(e);
            // Cancelling a future need not stop the underlying open. Close any stream it produces later.
            var unused = opened.thenAccept(IOUtils::closeStream);
            throw failure;
        } catch (ExecutionException e) {
            return FutureIO.raiseInnerCause(e);
        }
    }

    private static final class HadoopStream implements PositionalStream {
        private final FSDataInputStream stream;
        // Hadoop mandates this capability probe: streams may implement ByteBufferPositionedReadable
        // yet throw UnsupportedOperationException when their inner stream cannot serve it.
        private final boolean byteBufferPositionedRead;

        private HadoopStream(FSDataInputStream stream) {
            this.stream = stream;
            this.byteBufferPositionedRead = stream.hasCapability(StreamCapabilities.PREADBYTEBUFFER);
        }

        @Override
        public void readFully(long position, ByteBuffer buffer, ScratchBytes scratch) throws IOException {
            if (byteBufferPositionedRead) {
                // HDFS and friends fill Vortex's own memory, with no staging array in between.
                stream.readFully(position, buffer);
                return;
            }

            int length = buffer.remaining();
            byte[] target = scratch.bytes(length);
            stream.readFully(position, target, 0, length);
            buffer.put(target, 0, length);
        }

        @Override
        public void close() throws IOException {
            stream.close();
        }
    }
}
