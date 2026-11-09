// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.io.NativeReadable;
import dev.vortex.jni.NativeDataSource;
import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import org.apache.arrow.c.ArrowSchema;
import org.apache.arrow.c.Data;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * A set of Vortex files opened through a {@link Session}. Data sources are cheap to open (only the first file is read
 * eagerly, to determine the schema) and can be scanned multiple times.
 *
 * <p>Call {@link #close()} to release this handle and its cached file resources promptly. Scans and partitions already
 * created from it remain valid and retain the resources they still need. {@link VortexCleaner} releases an unclosed
 * handle when it becomes unreachable.
 */
public final class DataSource implements AutoCloseable {
    private final Session session;
    private final long pointer;
    private final Cleaner.Cleanable cleanable;
    private boolean closed;

    private DataSource(Session session, long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid data source pointer");
        this.session = Objects.requireNonNull(session, "session");
        this.pointer = pointer;
        cleanable = VortexCleaner.register(this, () -> NativeDataSource.free(pointer));
    }

    /** Open a single URI. */
    public static DataSource open(Session session, String uri) {
        return open(session, uri, Collections.emptyMap());
    }

    /**
     * Open one or more URIs or globs. When a glob is used, the first match is opened eagerly; subsequent matches are
     * opened lazily on scan.
     *
     * @param session open session
     * @param uri single URI or glob
     * @param properties object-store credentials / options
     */
    public static DataSource open(Session session, String uri, Map<String, String> properties) {
        return open(session, List.of(uri), properties);
    }

    /**
     * Open one or more URIs or globs. When a glob is used, the first match is opened eagerly; subsequent matches are
     * opened lazily on scan.
     *
     * @param session open session
     * @param uris URIs or globs to scan
     * @param properties object-store credentials / options
     */
    public static DataSource open(Session session, List<String> uris, Map<String, String> properties) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(uris, "uris");
        Preconditions.checkArgument(!uris.isEmpty(), "at least one uri is required");
        String[] uriArray = uris.toArray(String[]::new);
        Preconditions.checkArgument(
                Arrays.stream(uriArray).allMatch(Objects::nonNull), "uris must not contain null values");
        long sessionPointer = session.nativePointer();
        long pointer = NativeDataSource.open(sessionPointer, uriArray, properties);
        return new DataSource(session, pointer);
    }

    /** Open a single file through a caller-provided byte source. See {@link #open(Session, List, int)}. */
    public static DataSource open(Session session, NativeReadable readable) {
        return open(session, List.of(readable), 0);
    }

    /** Open files through caller-provided byte sources with the default read concurrency. */
    public static DataSource open(Session session, List<NativeReadable> readables) {
        return open(session, readables, 0);
    }

    /**
     * Open one or more files through caller-provided byte sources instead of native storage clients. Every read the
     * scan performs becomes an upcall into the corresponding {@link NativeReadable}, so this is the integration point
     * for external I/O abstractions (for example Iceberg's {@code FileIO}). Each file is identified by
     * {@link NativeReadable#name()}, which must be unique within {@code readables}.
     *
     * <p>The readables must remain open until this data source is closed and every derived scan, partition, and Arrow
     * reader has finished reading. Native code never closes them.
     *
     * @param session open session
     * @param readables byte sources
     * @param readConcurrency maximum in-flight {@code readFully} calls across all files of this data source; {@code 0}
     *     selects the default. Each in-flight read occupies one native thread and typically one stream on its readable.
     */
    public static DataSource open(Session session, List<NativeReadable> readables, int readConcurrency) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(readables, "readables");
        Preconditions.checkArgument(!readables.isEmpty(), "at least one readable is required");
        Preconditions.checkArgument(readConcurrency >= 0, "readConcurrency must not be negative");
        Object[] readableArray = readables.toArray();
        String[] names = new String[readableArray.length];
        long[] lengths = new long[readableArray.length];
        for (int i = 0; i < readableArray.length; i++) {
            Preconditions.checkArgument(readableArray[i] != null, "readables must not contain null values");
            NativeReadable readable = readables.get(i);
            names[i] = readable.name();
            Preconditions.checkArgument(names[i] != null, "readable at index %s returned a null name", i);
            lengths[i] = readable.length();
            Preconditions.checkArgument(lengths[i] >= 0, "readable for %s reported negative length", names[i]);
        }
        long pointer =
                NativeDataSource.openFiles(session.nativePointer(), readableArray, names, lengths, readConcurrency);
        return new DataSource(session, pointer);
    }

    /** Arrow schema of the data source (and of scans produced from it). */
    public synchronized Schema arrowSchema(BufferAllocator allocator) {
        Preconditions.checkState(!closed, "data source is closed");
        try (ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
            NativeDataSource.arrowSchema(session.nativePointer(), pointer, schema.memoryAddress());
            return Data.importSchema(allocator, schema, null);
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    /**
     * Row count along with the precision of that estimate. Mirrors the Rust {@code Precision<u64>} returned by
     * {@code DataSource::row_count}: {@link RowCount.Unknown} when no estimate is available, {@link RowCount.Estimate}
     * for an inexact hint, {@link RowCount.Exact} when the count is authoritative.
     */
    public synchronized RowCount rowCount() {
        Preconditions.checkState(!closed, "data source is closed");
        try {
            long[] out = new long[2];
            NativeDataSource.rowCount(pointer, out);
            return switch ((int) out[1]) {
                case 1 -> new RowCount.Estimate(out[0]);
                case 2 -> new RowCount.Exact(out[0]);
                default -> RowCount.Unknown.INSTANCE;
            };
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    /** Precision-aware row count. See {@link #rowCount()}. */
    public sealed interface RowCount {
        /** Returns the row count as a long, or {@code OptionalLong.empty()} when unknown. */
        OptionalLong asOptional();

        /** Row count is not known. */
        final class Unknown implements RowCount {
            public static final Unknown INSTANCE = new Unknown();

            private Unknown() {}

            @Override
            public OptionalLong asOptional() {
                return OptionalLong.empty();
            }
        }

        /** Estimated row count; the actual value may differ. */
        record Estimate(long value) implements RowCount {
            @Override
            public OptionalLong asOptional() {
                return OptionalLong.of(value);
            }
        }

        /** Exact row count. */
        record Exact(long value) implements RowCount {
            @Override
            public OptionalLong asOptional() {
                return OptionalLong.of(value);
            }
        }
    }

    /**
     * Sum of the on-storage byte sizes of all files included in this data source along with the precision of that
     * estimate. Mirrors the Rust {@code Option<Precision<u64>>} returned by {@code DataSource::byte_size}:
     * {@link ByteSize.Unknown} when no estimate is available (for example when the filesystem listing did not return
     * sizes), {@link ByteSize.Estimate} for an inexact hint (some files contribute extrapolated sizes), and
     * {@link ByteSize.Exact} when every file has a known size.
     */
    public synchronized ByteSize byteSize() {
        Preconditions.checkState(!closed, "data source is closed");
        try {
            long[] out = new long[2];
            NativeDataSource.byteSize(pointer, out);
            return switch ((int) out[1]) {
                case 1 -> new ByteSize.Estimate(out[0]);
                case 2 -> new ByteSize.Exact(out[0]);
                default -> ByteSize.Unknown.INSTANCE;
            };
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    /** Precision-aware byte size. See {@link #byteSize()}. */
    public sealed interface ByteSize {
        /** Returns the byte size as a long, or {@code OptionalLong.empty()} when unknown. */
        OptionalLong asOptional();

        /** Byte size is not known. */
        final class Unknown implements ByteSize {
            public static final Unknown INSTANCE = new Unknown();

            private Unknown() {}

            @Override
            public OptionalLong asOptional() {
                return OptionalLong.empty();
            }
        }

        /** Estimated byte size; the actual value may differ. */
        record Estimate(long value) implements ByteSize {
            @Override
            public OptionalLong asOptional() {
                return OptionalLong.of(value);
            }
        }

        /** Exact byte size. */
        record Exact(long value) implements ByteSize {
            @Override
            public OptionalLong asOptional() {
                return OptionalLong.of(value);
            }
        }
    }

    /** Submit a scan. */
    public synchronized Scan scan(ScanOptions options) {
        Preconditions.checkState(!closed, "data source is closed");
        try {
            Objects.requireNonNull(options, "options");

            long projectionPtr =
                    options.projection().map(Expression::nativePointer).orElse(0L);
            long filterPtr = options.filter().map(Expression::nativePointer).orElse(0L);
            long begin = options.rowRangeBegin().orElse(0L);
            long end = options.rowRangeEnd().orElse(0L);
            ScanOptions.SelectionMode selectionMode = options.selectionMode();
            long[] selectionIndices = selectionIndices(options);
            byte[] selectionRoaringBitmap = selectionRoaringBitmap(options);
            long limit = options.limit().orElse(0L);
            boolean ordered = options.ordered();
            int splitConcurrency = options.splitConcurrency().orElse(0);

            long scanPtr = dev.vortex.jni.NativeScan.create(
                    pointer,
                    projectionPtr,
                    filterPtr,
                    begin,
                    end,
                    selectionIndices,
                    selectionRoaringBitmap,
                    selectionMode.code(),
                    limit,
                    ordered,
                    splitConcurrency);
            return Scan.fromPointer(session, scanPtr);
        } finally {
            Reference.reachabilityFence(this);
        }
    }

    /** Releases this handle once; existing scans, partitions, and batches remain valid. */
    @Override
    public synchronized void close() {
        if (!closed) {
            closed = true;
            cleanable.clean();
        }
    }

    private static long[] selectionIndices(ScanOptions options) {
        return switch (options.selectionMode()) {
            case INCLUDE, EXCLUDE -> options.selectionIndices().orElse(null);
            default -> null;
        };
    }

    private static byte[] selectionRoaringBitmap(ScanOptions options) {
        return switch (options.selectionMode()) {
            case INCLUDE_ROARING, EXCLUDE_ROARING ->
                options.selectionRoaringBitmap()
                        .orElseThrow(() -> new IllegalArgumentException(
                                "selection roaring bitmap is required for roaring selection modes"));
            default -> null;
        };
    }
}
