// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import dev.vortex.io.NativeWritable;
import java.io.IOException;
import java.io.UncheckedIOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/**
 * A {@link NativeWritable} that streams a Vortex file into a Hadoop {@link FileSystem}.
 *
 * <p>Vortex writes and flushes but never closes; the file is finalized when the owner closes this.
 */
public final class HadoopWritable implements NativeWritable {
    private final FSDataOutputStream stream;

    /** Creates (or overwrites) the file at {@code path}, along with any missing parent directories. */
    public static HadoopWritable create(Configuration conf, String path) {
        Path file = new Path(path);
        try {
            FileSystem fs = file.getFileSystem(conf);
            return new HadoopWritable(fs.create(file, true));
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to create " + path, e);
        }
    }

    private HadoopWritable(FSDataOutputStream stream) {
        this.stream = stream;
    }

    @Override
    public void write(byte[] buffer, int offset, int length) throws IOException {
        stream.write(buffer, offset, length);
    }

    @Override
    public void flush() throws IOException {
        stream.flush();
    }

    @Override
    public void close() throws IOException {
        stream.close();
    }
}
