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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

public final class NativeLoader {
    private static final String CACHE_FILE_NAME = "libvortex_jni.so";
    private static final AtomicBoolean LOADED = new AtomicBoolean();

    private NativeLoader() {}

    public static synchronized void loadJni() {
        if (LOADED.get()) {
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
        // We unpack the library to a temp file, before putting it into a stable location.
        String libPath = "/native/" + osShortName + "-" + osArch + "/" + libName;
        try (InputStream in = NativeLoader.class.getResourceAsStream(libPath)) {
            if (in == null) {
                throw new FileNotFoundException("Library not found: " + libPath);
            }
            File tempFile = File.createTempFile("libvortex_jni", ".dylib");
            tempFile.deleteOnExit();

            // Copy the data to temp file
            Files.copy(in, tempFile.toPath());

            // Atomically move the file into the cache directory.
            // NOTE: this is only atomic when the tmpdir and the target are on the same file system.
            Path cacheDir = Files.createDirectories(cacheDir());
            Path outPath = cacheDir.resolve(CACHE_FILE_NAME);
            Files.move(tempFile.toPath(), outPath);

            libName = outPath.toAbsolutePath().toString();
        } catch (IOException e) {
            throw new RuntimeException("Failed to load library: " + e.getMessage(), e);
        }

        // Load the library
        System.load(libName);
        LOADED.set(true);
    }

    private static Path cacheDir() {
        // If the user home dir is detectable, create the target dir inside of it.
        // Otherwise we fallback to Java temp dir.
        // If there is no tmpdir defined, we fallback to the working directory that the JVM was launched from.
        String rootDir = Stream.of(System.getProperty("user.home"), System.getProperty("java.io.tmpdir"), ".")
                .filter(Objects::nonNull)
                .findFirst()
                .get();

        return Paths.get(rootDir, ".vortex", "jni");
    }
}
