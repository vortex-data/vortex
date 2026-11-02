// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import org.roaringbitmap.longlong.Roaring64NavigableMap;

/**
 * A pushed-down {@code TABLESAMPLE} without replacement: each row, or with {@code blocks} each chunk of rows, is kept
 * when a pseudo-random number drawn for it falls in {@code [lowerBound, upperBound)}.
 *
 * <p>The number is a hash of the seed, the file's path, and the row's index in its file, so a row's fate does not
 * depend on how Spark splits the file, and the same seed selects the same rows on every run. Readers hand the kept rows
 * to the native scan as a row selection, so chunks without a kept row are never decoded.
 */
public record VortexSample(double lowerBound, double upperBound, long seed, boolean blocks) implements Serializable {
    private static final long serialVersionUID = 1L;

    /** Separates the draws for chunks from the draws for rows, so the two samples are independent. */
    private static final long BLOCK_STREAM = 0x5a17c6e3b9d04f21L;

    /** The rows of {@code [begin, end)} of the file at {@code path} that the sample keeps. */
    Roaring64NavigableMap select(String path, long begin, long end) {
        long fileSeed = mix(seed ^ mix(hash(path)));
        Roaring64NavigableMap rows = new Roaring64NavigableMap();
        for (long row = begin; row < end; row++) {
            // The top 53 bits of the hash, as a double in [0, 1).
            double draw = (mix(fileSeed + row) >>> 11) * 0x1.0p-53;
            if (draw >= lowerBound && draw < upperBound) {
                rows.addLong(row);
            }
        }
        return rows;
    }

    /**
     * The rows of {@code [begin, end)} in the whole chunks the sample keeps, of the file at {@code path} whose natural
     * splits have {@code boundaries}. A chunk's fate hangs on its index alone, so a block sample reads whole chunks
     * however Spark splits the file.
     */
    Roaring64NavigableMap selectBlocks(String path, long[] boundaries, long begin, long end) {
        long fileSeed = mix(seed ^ mix(hash(path)) ^ BLOCK_STREAM);
        Roaring64NavigableMap rows = new Roaring64NavigableMap();
        for (int block = 0; block + 1 < boundaries.length; block++) {
            double draw = (mix(fileSeed + block) >>> 11) * 0x1.0p-53;
            long first = Math.max(boundaries[block], begin);
            long last = Math.min(boundaries[block + 1], end);
            if (draw >= lowerBound && draw < upperBound && first < last) {
                rows.addRange(first, last);
            }
        }
        return rows;
    }

    /** {@code rows} in the portable format the native scan reads. */
    static byte[] portable(Roaring64NavigableMap rows) {
        try (ByteArrayOutputStream bytes = new ByteArrayOutputStream();
                DataOutputStream output = new DataOutputStream(bytes)) {
            rows.serializePortable(output);
            output.flush();
            return bytes.toByteArray();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** A hash of the path's bytes that, unlike {@link String#hashCode}, spreads over all 64 bits. */
    private static long hash(String path) {
        long hash = 0xcbf29ce484222325L;
        for (byte b : path.getBytes(StandardCharsets.UTF_8)) {
            hash = (hash ^ (b & 0xff)) * 0x100000001b3L;
        }
        return hash;
    }

    /** The SplitMix64 finalizer, which turns consecutive inputs into independent-looking outputs. */
    private static long mix(long value) {
        long z = value + 0x9e3779b97f4a7c15L;
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        return z ^ (z >>> 31);
    }
}
