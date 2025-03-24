/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.jni;

import com.google.common.io.ByteStreams;
import java.io.*;
import java.util.Locale;

public final class NativeLoader {
    private static boolean loaded = false;

    private NativeLoader() {}

    public static synchronized void loadJni() {
        if (loaded) {
            return;
        }

        // Load the native library
        String osName = System.getProperty("os.name").toLowerCase(Locale.ROOT);
        String osArch = System.getProperty("os.arch").toLowerCase(Locale.ROOT);
        String libName = "libvortex_jni";

        String osShortName;
        if (osName.contains("win")) {
            osShortName = "win";
            libName += ".dll";
        } else if (osName.contains("mac")) {
            osShortName = "darwin";
            libName += ".dylib";
        } else if (osName.contains("nix") || osName.contains("nux")) {
            osShortName = "linux";
            libName += ".so";
        } else {
            throw new UnsupportedOperationException("Unsupported OS: " + osName);
        }

        // Extract the library from classpath
        // This assumes the library is in the same package as this class
        String libPath = "/native/" + osShortName + "-" + osArch + "/" + libName;
        try (InputStream in = NativeLoader.class.getResourceAsStream(libPath)) {
            if (in == null) {
                throw new FileNotFoundException("Library not found: " + libPath);
            }
            File tempFile = File.createTempFile("libvortex_jni", ".dylib");
            tempFile.deleteOnExit();

            try (OutputStream out = new FileOutputStream(tempFile)) {
                ByteStreams.copy(in, out);
            }
            libName = tempFile.getAbsolutePath();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load library: " + e.getMessage(), e);
        }

        // Load the library
        System.load(libName);
        loaded = true;
    }
}
