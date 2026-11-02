// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.io;

import java.io.Closeable;
import java.io.IOException;

/** Helpers for closing groups of resources. */
public final class Closeables {
    private Closeables() {}

    /**
     * Closes every item, always attempting all of them; the first failure is thrown once everything has been visited,
     * with any further failures suppressed.
     */
    public static void closeAll(Iterable<? extends Closeable> closeables) throws IOException {
        IOException failure = null;
        for (Closeable closeable : closeables) {
            try {
                closeable.close();
            } catch (IOException e) {
                if (failure == null) {
                    failure = e;
                } else {
                    failure.addSuppressed(e);
                }
            }
        }
        if (failure != null) {
            throw failure;
        }
    }
}
