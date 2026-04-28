// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import dev.vortex.api.Session;
import java.util.List;
import java.util.Map;

/**
 * Static utilities for discovering and deleting Vortex files on an object store. The caller supplies a {@link Session};
 * its runtime handle is forwarded to the underlying object store.
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

    private static native List<String> listFiles(long sessionPointer, String uri, Map<String, String> options);

    private static native void delete(long sessionPointer, String[] uris, Map<String, String> options);
}
