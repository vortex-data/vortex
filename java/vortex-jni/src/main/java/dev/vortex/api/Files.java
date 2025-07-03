// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.jni.JNIFile;
import dev.vortex.jni.NativeFileMethods;
import java.net.URI;
import java.nio.file.Paths;
import java.util.Map;

public final class Files {

    private Files() {}

    public static File open(String path) {
        if (path.startsWith("/")) {
            return open(Paths.get(path).toUri(), Map.of());
        }
        return open(URI.create(path), Map.of());
    }

    public static File open(URI uri, Map<String, String> properties) {
        long ptr = NativeFileMethods.open(uri.toString(), properties);
        Preconditions.checkArgument(ptr > 0, "Failed to open file: %s", uri);
        return new JNIFile(ptr);
    }
}
