// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import dev.vortex.io.NativeReadable;
import dev.vortex.spark.VortexOptions;
import java.io.Serializable;
import java.util.Locale;
import java.util.Objects;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

/**
 * The Hadoop configuration this Spark job reaches storage with, and the read settings that go with it.
 *
 * <p>File contents are read and written through Hadoop streams, so the connector sees the same schemes and credential
 * providers as Spark's file index and commit protocol. Reads open a {@link HadoopReadable}; writes go through
 * {@link HadoopWritable} on the task path Spark's commit protocol assigns.
 *
 * <p>Built on the driver and shipped to executors, so the configuration travels with it, or is broadcast to them.
 */
public final class VortexIo implements Serializable {
    private static final long serialVersionUID = 2L;

    /**
     * Option bounding how many concurrent read upcalls the native reader issues against one file. Each in-flight upcall
     * leases one pooled Hadoop input stream. Zero keeps the native default.
     */
    public static final String READ_CONCURRENCY_OPTION = "vortex.readConcurrency";

    private static final int DEFAULT_READ_CONCURRENCY = 0;

    /**
     * Maximum gap between merged reads, in bytes. Defaults to 1 MiB; zero merges only adjacent or overlapping reads.
     */
    public static final String COALESCE_DISTANCE_OPTION = "vortex.read.coalesce.distance";

    /**
     * Maximum span of a merged read, in bytes. Must be positive; defaults to 16 MiB. Individual reads may be larger.
     */
    public static final String COALESCE_MAX_SIZE_OPTION = "vortex.read.coalesce.maxSize";

    /**
     * Option bounding how many splits of a file the native scan reads at once, in all. Each split in flight holds the
     * decoded chunks of every column it overlaps, and every task in an executor runs its own scan, which by default
     * keeps several splits in flight per core. Lowering this reduces memory held by concurrent split decoding and Arrow
     * conversion. A scan with both a filter and a limit may also retain one trimmed array per split while preparing
     * output. This does not cap total native memory or I/O read-ahead. Zero keeps the native default.
     */
    public static final String SPLIT_CONCURRENCY_OPTION = "vortex.splitConcurrency";

    private static final int DEFAULT_SPLIT_CONCURRENCY = 0;

    /** The configuration this instance carries itself, or null when it reads {@link #broadcastConf}. */
    private final SerializableHadoopConf conf;

    /** The configuration broadcast to every executor once, or null when this instance carries {@link #conf}. */
    private final Broadcast<SerializableHadoopConf> broadcastConf;

    private final ReadSettings settings;

    private record ReadSettings(int readConcurrency, int splitConcurrency, int coalesceDistance, int coalesceMaxSize)
            implements Serializable {
        static ReadSettings parse(VortexOptions options) {
            Objects.requireNonNull(options, "options");
            return new ReadSettings(
                    nonNegative(options, READ_CONCURRENCY_OPTION, DEFAULT_READ_CONCURRENCY),
                    nonNegative(options, SPLIT_CONCURRENCY_OPTION, DEFAULT_SPLIT_CONCURRENCY),
                    nonNegative(options, COALESCE_DISTANCE_OPTION, NativeReadable.DEFAULT_COALESCE_DISTANCE),
                    positive(options, COALESCE_MAX_SIZE_OPTION, NativeReadable.DEFAULT_COALESCE_MAX_SIZE));
        }
    }

    /** Captures {@code hadoopConf} and the read settings found in the format options. */
    public static VortexIo create(VortexOptions options, Configuration hadoopConf) {
        ReadSettings settings = ReadSettings.parse(options);
        Objects.requireNonNull(hadoopConf, "hadoopConf");
        return new VortexIo(new SerializableHadoopConf(hadoopConf), null, settings);
    }

    /**
     * Like {@link #create}, but broadcasts {@code hadoopConf} through {@code context}. Each task that deserializes the
     * result then shares the executor's copy, instead of reading the whole configuration back from its own bytes.
     */
    public static VortexIo broadcast(VortexOptions options, Configuration hadoopConf, SparkContext context) {
        ReadSettings settings = ReadSettings.parse(options);
        Objects.requireNonNull(hadoopConf, "hadoopConf");
        ClassTag<SerializableHadoopConf> tag = ClassTag$.MODULE$.apply(SerializableHadoopConf.class);
        return new VortexIo(null, context.broadcast(new SerializableHadoopConf(hadoopConf), tag), settings);
    }

    /** Hadoop I/O over a default configuration, for callers that have no Spark session to draw one from. */
    public static VortexIo defaults() {
        return create(VortexOptions.empty(), new Configuration());
    }

    private VortexIo(
            SerializableHadoopConf conf, Broadcast<SerializableHadoopConf> broadcastConf, ReadSettings settings) {
        this.conf = conf;
        this.broadcastConf = broadcastConf;
        this.settings = settings;
    }

    private static int nonNegative(VortexOptions options, String key, int defaultValue) {
        int parsed = options.getInt(key, defaultValue);
        if (parsed < 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "%s must be >= 0, got %d", key, parsed));
        }
        return parsed;
    }

    private static int positive(VortexOptions options, String key, int defaultValue) {
        int parsed = options.getInt(key, defaultValue);
        if (parsed <= 0) {
            throw new IllegalArgumentException(String.format(Locale.ROOT, "%s must be > 0, got %d", key, parsed));
        }
        return parsed;
    }

    public Configuration hadoopConf() {
        return conf != null ? conf.value() : broadcastConf.value().value();
    }

    /** Bound on concurrent read upcalls per file; zero keeps the native default. */
    public int readConcurrency() {
        return settings.readConcurrency();
    }

    /** The most splits of a file the native scan reads at once; zero keeps the native default. */
    public int splitConcurrency() {
        return settings.splitConcurrency();
    }

    /**
     * Opens a byte source for {@code file}, stating it only if the listing that produced it did not report a size. The
     * caller owns the result and must close it once the scan built on it is done.
     *
     * <p>The readers skip listed files without {@link VortexFile#EXTENSION} before they get here. The check stops any
     * other path from opening such a file as Vortex data.
     *
     * @throws IllegalArgumentException if the path does not carry {@link VortexFile#EXTENSION}
     */
    public NativeReadable openReadable(VortexFile file) {
        VortexFile.requireVortexExtension(file.path());
        return HadoopReadable.open(
                hadoopConf(), file.path(), file.length(), settings.coalesceDistance(), settings.coalesceMaxSize());
    }
}
