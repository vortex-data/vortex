// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import java.util.Map;

/**
 * Native JNI methods for Vortex file writing.
 */
public final class NativeWriterMethods {
    
    static {
        NativeLoader.loadJni();
    }
    
    private NativeWriterMethods() {}
    
    /**
     * Creates a new native Vortex writer.
     *
     * @param filePath the path where the Vortex file will be written
     * @param schemaJson the Arrow schema in JSON format
     * @param options additional writer options
     * @return a native pointer to the writer, or 0 on failure
     */
    public static native long create(String filePath, String schemaJson, Map<String, String> options);
    
    /**
     * Writes a batch of Arrow data to the Vortex file.
     *
     * @param writerPtr the native writer pointer
     * @param arrowData the Arrow IPC format data
     * @return true if successful, false otherwise
     */
    public static native boolean writeBatch(long writerPtr, byte[] arrowData);
    
    /**
     * Closes the native writer and finalizes the file.
     *
     * @param writerPtr the native writer pointer
     * @return true if successful, false otherwise
     */
    public static native boolean close(long writerPtr);
}