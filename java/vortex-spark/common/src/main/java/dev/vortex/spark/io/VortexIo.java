// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import dev.vortex.io.NativeReadable;
import dev.vortex.spark.VortexOptions;
import java.io.Serializable;
import java.util.Locale;
import java.util.Objects;
import org.apache.hadoop.conf.Configuration;

/**
 * The Hadoop configuration this Spark job reaches storage with, and the read settings that go with it.
 *
 * <p>File contents are read and written through Hadoop streams, so the connector sees the same schemes and credential
 * providers as Spark's file index and commit protocol. Reads open a {@link HadoopReadable}; writes go through
 * {@link HadoopWritable} on the task path Spark's commit protocol assigns.
 *
 * <p>Built on the driver and shipped to executors, so the configuration travels with it.
 */
public final class VortexIo implements Serializable {
    private static final long serialVersionUID = 1L;

    /**
     * Option bounding how many concurrent read upcalls the native reader issues against one file. Each in-flight upcall
     * leases one pooled Hadoop input stream. Zero keeps the native default.
     */
    public static final String READ_CONCURRENCY_OPTION = "vortex.readConcurrency";

    private static final int DEFAULT_READ_CONCURRENCY = 0;

    private final SerializableHadoopConf conf;
    private final int readConcurrency;

    /** Captures {@code hadoopConf} and the read settings found in the format options. */
    public static VortexIo create(VortexOptions options, Configuration hadoopConf) {
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(hadoopConf, "hadoopConf");
        return new VortexIo(new SerializableHadoopConf(hadoopConf), parseReadConcurrency(options));
    }

    /** Hadoop I/O over a default configuration, for callers that have no Spark session to draw one from. */
    public static VortexIo defaults() {
        return new VortexIo(new SerializableHadoopConf(new Configuration()), DEFAULT_READ_CONCURRENCY);
    }

    private VortexIo(SerializableHadoopConf conf, int readConcurrency) {
        this.conf = conf;
        this.readConcurrency = readConcurrency;
    }

    private static int parseReadConcurrency(VortexOptions options) {
        int parsed = options.getInt(READ_CONCURRENCY_OPTION, DEFAULT_READ_CONCURRENCY);
        if (parsed < 0) {
            throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "%s must be >= 0, got %d", READ_CONCURRENCY_OPTION, parsed));
        }
        return parsed;
    }

    public Configuration hadoopConf() {
        return conf.value();
    }

    /** Bound on concurrent read upcalls per file; zero keeps the native default. */
    public int readConcurrency() {
        return readConcurrency;
    }

    /**
     * Opens a byte source for {@code file}, stating it only if the listing that produced it did not report a size. The
     * caller owns the result and must close it once the scan built on it is done.
     *
     * <p>Every read path in the connector goes through here, so this is where the {@code *.vortex} requirement is
     * enforced.
     *
     * @throws IllegalArgumentException if the path does not carry {@link VortexFile#EXTENSION}
     */
    public NativeReadable openReadable(VortexFile file) {
        VortexFile.requireVortexExtension(file.path());
        return HadoopReadable.open(hadoopConf(), file.path(), file.length());
    }
}
