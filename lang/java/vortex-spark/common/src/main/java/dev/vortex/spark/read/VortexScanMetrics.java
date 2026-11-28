// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.io.IOException;
import java.util.concurrent.TimeUnit;
import org.apache.spark.sql.connector.metric.CustomMetric;
import org.apache.spark.sql.connector.metric.CustomSumMetric;
import org.apache.spark.sql.connector.metric.CustomTaskMetric;
import org.apache.spark.sql.connector.read.PartitionReader;

/**
 * The metrics a Vortex scan reports in the Spark UI, summed over the tasks of the scan.
 *
 * <p>Spark creates each metric on the driver from its class name, so every metric is a public class with a public
 * constructor that takes no arguments.
 */
public final class VortexScanMetrics {
    static final String FILES_OPENED = "vortexFilesOpened";
    static final String SPLITS_SKIPPED = "vortexSplitsSkipped";
    static final String BATCHES = "vortexBatches";
    static final String OPEN_TIME = "vortexOpenTimeMs";
    static final String SCAN_TIME = "vortexScanTimeMs";

    private VortexScanMetrics() {}

    /** The metrics every Vortex scan reports. */
    public static CustomMetric[] supported() {
        return new CustomMetric[] {new FilesOpened(), new SplitsSkipped(), new Batches(), new OpenTime(), new ScanTime()
        };
    }

    /**
     * The metrics of one task, which every file reader of the task adds to.
     *
     * <p>Spark's file partition reader does not pass on the metrics of the readers of its files, so the task reports
     * these through {@link #reporting}. V1 readers count into a task of their own, which nothing reports.
     */
    public static final class Task {
        private long filesOpened;
        private long splitsSkipped;
        private long batches;
        private long openNanos;
        private long scanNanos;

        void fileOpened() {
            filesOpened++;
        }

        void splitSkipped() {
            splitsSkipped++;
        }

        void batchRead() {
            batches++;
        }

        void addOpenNanos(long nanos) {
            openNanos += nanos;
        }

        void addScanNanos(long nanos) {
            scanNanos += nanos;
        }

        CustomTaskMetric[] values() {
            return new CustomTaskMetric[] {
                metric(FILES_OPENED, filesOpened),
                metric(SPLITS_SKIPPED, splitsSkipped),
                metric(BATCHES, batches),
                metric(OPEN_TIME, TimeUnit.NANOSECONDS.toMillis(openNanos)),
                metric(SCAN_TIME, TimeUnit.NANOSECONDS.toMillis(scanNanos)),
            };
        }
    }

    /** {@code reader}, reporting the metrics of {@code task}. */
    static <T> PartitionReader<T> reporting(PartitionReader<T> reader, Task task) {
        return new PartitionReader<>() {
            @Override
            public boolean next() throws IOException {
                return reader.next();
            }

            @Override
            public T get() {
                return reader.get();
            }

            @Override
            public CustomTaskMetric[] currentMetricsValues() {
                return task.values();
            }

            @Override
            public void close() throws IOException {
                reader.close();
            }
        };
    }

    private static CustomTaskMetric metric(String name, long value) {
        return new CustomTaskMetric() {
            @Override
            public String name() {
                return name;
            }

            @Override
            public long value() {
                return value;
            }
        };
    }

    /** Files, or splits of files, the scan opened. */
    public static final class FilesOpened extends CustomSumMetric {
        @Override
        public String name() {
            return FILES_OPENED;
        }

        @Override
        public String description() {
            return "Vortex files opened";
        }
    }

    /** Opened splits that the scan proved to hold no rows to read, before it decoded any data. */
    public static final class SplitsSkipped extends CustomSumMetric {
        @Override
        public String name() {
            return SPLITS_SKIPPED;
        }

        @Override
        public String description() {
            return "Vortex splits with no rows to read";
        }
    }

    /** Arrow batches the native scan returned. */
    public static final class Batches extends CustomSumMetric {
        @Override
        public String name() {
            return BATCHES;
        }

        @Override
        public String description() {
            return "Vortex batches read";
        }
    }

    /** Time spent opening files, reading their footers, and planning their native scans. */
    public static final class OpenTime extends CustomSumMetric {
        @Override
        public String name() {
            return OPEN_TIME;
        }

        @Override
        public String description() {
            return "Vortex file open time (ms)";
        }
    }

    /** Time spent in the native scan: reading, decoding, filtering, and exporting batches. */
    public static final class ScanTime extends CustomSumMetric {
        @Override
        public String name() {
            return SCAN_TIME;
        }

        @Override
        public String description() {
            return "Vortex native scan time (ms)";
        }
    }
}
