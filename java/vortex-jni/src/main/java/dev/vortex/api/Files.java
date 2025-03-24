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
