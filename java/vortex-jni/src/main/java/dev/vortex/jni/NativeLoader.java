// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import com.google.common.io.ByteStreams;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;

/** Loads the native vortex-jni shared library from the classpath. */
public final class NativeLoader {
    private static boolean loaded = false;

    private NativeLoader() {}

    /** Load the native library into the current JVM. Thread-safe and idempotent. */
    public static synchronized void loadJni() {
        if (loaded) {
            return;
        }

        String osName = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String osArch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        String libName = "libvortex_jni";

        String osShortName;
        String libExt;
        if (osName.contains("win")) {
            osShortName = "win";
            libExt = ".dll";
            libName += libExt;
        } else if (osName.contains("mac")) {
            osShortName = "darwin";
            libExt = ".dylib";
            libName += libExt;
        } else if (osName.contains("nix") || osName.contains("nux")) {
            osShortName = "linux";
            libExt = ".so";
            libName += libExt;
        } else {
            throw new UnsupportedOperationException("Unsupported OS: " + osName);
        }

        String libPath = "/native/" + osShortName + "-" + osArch + "/" + libName;
        try (InputStream in = NativeLoader.class.getResourceAsStream(libPath)) {
            if (in == null) {
                throw new FileNotFoundException("Library not found: " + libPath);
            }
            File tempFile = File.createTempFile("libvortex_jni", libExt);
            tempFile.deleteOnExit();

            try (OutputStream out = new FileOutputStream(tempFile)) {
                ByteStreams.copy(in, out);
            }
            libName = tempFile.getAbsolutePath();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load library: " + e.getMessage(), e);
        }

        System.load(libName);
        loaded = true;
    }
}
