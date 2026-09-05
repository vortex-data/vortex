// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.io;

import java.io.Serializable;
import java.util.Locale;

/** A Vortex file and, when a listing already reported it, its size on storage. */
public final class VortexFile implements Serializable {
    private static final long serialVersionUID = 1L;

    /** The extension every Vortex data file must carry. */
    public static final String EXTENSION = ".vortex";

    /** Size of a file nothing has stat'ed yet. Whoever opens it pays for the stat, if it needs one at all. */
    public static final long UNKNOWN_LENGTH = -1;

    private final String path;
    private final long length;

    /**
     * @param path the path as the caller spelled it, not as Hadoop qualifies it
     * @param length size in bytes, or {@link #UNKNOWN_LENGTH} for a path named directly rather than listed
     */
    public VortexFile(String path, long length) {
        this.path = path;
        this.length = length;
    }

    public String path() {
        return path;
    }

    public long length() {
        return length;
    }

    /** A file whose size is not known yet. */
    public static VortexFile unsized(String path) {
        return new VortexFile(path, UNKNOWN_LENGTH);
    }

    /**
     * Returns whether {@code path} names a Vortex data file.
     *
     * <p>Spark's file index hides names that begin with {@code _} or {@code .}, but it keeps {@code _metadata} and
     * {@code _common_metadata}, and it keeps every other extension. The connector reads only files that carry
     * {@link #EXTENSION}, so this is the one test that decides what belongs to a Vortex dataset.
     */
    public static boolean hasVortexExtension(String path) {
        return path.toLowerCase(Locale.ROOT).endsWith(EXTENSION);
    }

    /**
     * Fails unless {@code path} names a Vortex data file.
     *
     * @throws IllegalArgumentException if the path does not carry {@link #EXTENSION}
     */
    public static void requireVortexExtension(String path) {
        if (!hasVortexExtension(path)) {
            throw new IllegalArgumentException(String.format(
                    Locale.ROOT,
                    "%s is not a Vortex file: every file in a Vortex dataset must end with %s. "
                            + "Remove the file, or restrict the listing with the pathGlobFilter option.",
                    path,
                    EXTENSION));
        }
    }
}
