// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import com.google.common.base.Preconditions;
import dev.vortex.api.Session;
import dev.vortex.api.VortexColumnStatistics;
import dev.vortex.io.NativeReadable;
import java.lang.ref.Reference;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Static utilities for discovering, deleting, and inspecting the metadata and statistics of Vortex files on an object
 * store. The caller supplies a {@link Session}; its runtime handle is forwarded to the underlying object store.
 */
public final class NativeFiles {
    static {
        NativeLoader.loadJni();
    }

    private NativeFiles() {}

    /** List all Vortex files reachable under {@code uri}. */
    public static List<String> listFiles(Session session, String uri, Map<String, String> options) {
        return listFiles(session.nativePointer(), uri, options);
    }

    /** Delete files at the given URIs. Silently tolerates an empty list. */
    public static void delete(Session session, String[] uris, Map<String, String> options) {
        delete(session.nativePointer(), uris, options);
    }

    /**
     * Read the user-defined metadata segments written into the Vortex file at {@code uri}, as opaque bytes keyed by the
     * names the {@code VortexWriter} builder was given. Returns an empty map for a file that carries no metadata.
     *
     * <p>This opens the file on its own. Metadata lives in dedicated segments that opening a file does not read by
     * default, so a segment that falls outside the file-tail read costs an extra round trip.
     */
    public static Map<String, byte[]> readMetadata(Session session, String uri, Map<String, String> options) {
        return readMetadata(session.nativePointer(), uri, options);
    }

    /**
     * Read the user-defined metadata segments of a Vortex file through a caller-provided {@link NativeReadable},
     * instead of a native storage client. See {@link #readMetadata(Session, String, Map)}.
     *
     * <p>The readable must stay open for the duration of the call; native code never closes it.
     *
     * <p>{@link NativeReadable#name()} keys the file in the session's footer cache, exactly as it does when the
     * readable is scanned, so reading metadata and then scanning the same file share one footer read.
     */
    public static Map<String, byte[]> readMetadata(Session session, NativeReadable readable) {
        Objects.requireNonNull(readable, "readable");
        String name = readable.name();
        Preconditions.checkArgument(name != null && !name.isEmpty(), "readable reported an empty name");
        long length = readable.length();
        Preconditions.checkArgument(length >= 0, "readable for %s reported negative length", name);
        return readMetadataFromReadable(session.nativePointer(), readable, name, length);
    }

    /**
     * Read the per-column statistics stored in the footer of the Vortex file at {@code uri}, in Arrow schema order.
     * Statistics the file does not store are empty in the returned values.
     *
     * <p>This opens the file on its own, but reads only the footer and no column data.
     */
    public static List<VortexColumnStatistics> readStatistics(
            Session session, String uri, Map<String, String> options) {
        return List.of(readStatistics(session.nativePointer(), uri, options));
    }

    /**
     * Read the per-column statistics of a Vortex file through a caller-provided {@link NativeReadable}, instead of a
     * native storage client. See {@link #readStatistics(Session, String, Map)}.
     *
     * <p>The readable must stay open for the duration of the call; native code never closes it.
     *
     * <p>{@link NativeReadable#name()} keys the file in the session's footer cache, exactly as it does when the
     * readable is scanned, so reading statistics and then scanning the same file share one footer read.
     */
    public static List<VortexColumnStatistics> readStatistics(Session session, NativeReadable readable) {
        Objects.requireNonNull(readable, "readable");
        String name = readable.name();
        Preconditions.checkArgument(name != null && !name.isEmpty(), "readable reported an empty name");
        long length = readable.length();
        Preconditions.checkArgument(length >= 0, "readable for %s reported negative length", name);
        return List.of(readStatisticsFromReadable(session.nativePointer(), readable, name, length));
    }

    /**
     * The natural row splits of the Vortex file behind {@code readable}, as their boundaries: split {@code i} holds
     * rows {@code [boundaries[i], boundaries[i + 1])}. The splits use the default split policy, following the file's
     * chunk boundaries and subdividing large chunks.
     *
     * <p>The readable must stay open for the duration of the call; native code never closes it.
     *
     * <p>{@link NativeReadable#name()} keys the file in the session's footer cache, exactly as it does when the
     * readable is scanned.
     */
    public static long[] splits(Session session, NativeReadable readable) {
        return splitsFromReadable(session, readable, null);
    }

    /**
     * {@link #splits(Session, NativeReadable)} for a scan of the top-level {@code columns}: the splits follow the
     * chunks of those columns only. This is much cheaper than the splits of every column for a scan of a few columns of
     * a wide file. Columns that the file does not hold are ignored, and when it holds none of them, the splits follow
     * every column.
     */
    public static long[] splits(Session session, NativeReadable readable, List<String> columns) {
        Objects.requireNonNull(columns, "columns");
        return splitsFromReadable(session, readable, columns.toArray(String[]::new));
    }

    private static long[] splitsFromReadable(Session session, NativeReadable readable, String[] columns) {
        Objects.requireNonNull(session, "session");
        Objects.requireNonNull(readable, "readable");
        String name = readable.name();
        Preconditions.checkArgument(name != null && !name.isEmpty(), "readable reported an empty name");
        long length = readable.length();
        Preconditions.checkArgument(length >= 0, "readable for %s reported negative length", name);
        try {
            return splitsFromReadable(session.nativePointer(), readable, name, length, columns);
        } finally {
            Reference.reachabilityFence(session);
        }
    }

    private static native List<String> listFiles(long sessionPointer, String uri, Map<String, String> options);

    private static native void delete(long sessionPointer, String[] uris, Map<String, String> options);

    private static native Map<String, byte[]> readMetadata(
            long sessionPointer, String uri, Map<String, String> options);

    private static native Map<String, byte[]> readMetadataFromReadable(
            long sessionPointer, Object readable, String name, long length);

    private static native VortexColumnStatistics[] readStatistics(
            long sessionPointer, String uri, Map<String, String> options);

    private static native VortexColumnStatistics[] readStatisticsFromReadable(
            long sessionPointer, Object readable, String name, long length);

    private static native long[] splitsFromReadable(
            long sessionPointer, Object readable, String name, long length, String[] columns);
}
