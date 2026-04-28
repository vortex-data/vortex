// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeDataSource;
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
 * <p>Native resources are released automatically via {@link VortexCleaner} when the data source becomes unreachable.
 */
public final class DataSource {
    private final Session session;
    private final long pointer;

    private DataSource(Session session, long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid data source pointer");
        this.session = Objects.requireNonNull(session, "session");
        this.pointer = pointer;
        VortexCleaner.register(this, () -> NativeDataSource.free(pointer));
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

    /** Arrow schema of the data source (and of scans produced from it). */
    public Schema arrowSchema(BufferAllocator allocator) {
        try (ArrowSchema schema = ArrowSchema.allocateNew(allocator)) {
            NativeDataSource.arrowSchema(pointer, schema.memoryAddress());
            return Data.importSchema(allocator, schema, null);
        }
    }

    /**
     * Row count along with the precision of that estimate. Mirrors the Rust {@code Option<Precision<u64>>} returned by
     * {@code DataSource::row_count}: {@link RowCount.Unknown} when no estimate is available, {@link RowCount.Estimate}
     * for an inexact hint, {@link RowCount.Exact} when the count is authoritative.
     */
    public RowCount rowCount() {
        long[] out = new long[2];
        NativeDataSource.rowCount(pointer, out);
        return switch ((int) out[1]) {
            case 1 -> new RowCount.Estimate(out[0]);
            case 2 -> new RowCount.Exact(out[0]);
            default -> RowCount.Unknown.INSTANCE;
        };
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

    /** Submit a scan. */
    public Scan scan(ScanOptions options) {
        Objects.requireNonNull(options, "options");

        long projectionPtr = options.projection().map(Expression::nativePointer).orElse(0L);
        long filterPtr = options.filter().map(Expression::nativePointer).orElse(0L);
        long begin = options.rowRangeBegin().orElse(0L);
        long end = options.rowRangeEnd().orElse(0L);
        long[] selectionIndices = options.selectionIndices().orElse(null);
        byte selectionMode = options.selectionMode().code();
        long limit = options.limit().orElse(0L);
        boolean ordered = options.ordered();

        long scanPtr = dev.vortex.jni.NativeScan.create(
                pointer, projectionPtr, filterPtr, begin, end, selectionIndices, selectionMode, limit, ordered);
        return Scan.fromPointer(session, scanPtr);
    }
}
