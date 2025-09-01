// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

public final class NativeArrayIteratorMethods {
    static {
        NativeLoader.loadJni();
    }

    private NativeArrayIteratorMethods() {}

    /**
     * Free all resources associated with the stream behind the pointer.
     */
    public static native void free(long pointer);

    /**
     * Returns a pointer to the next element in the stream, or -1 if there are no more elements.
     * <p>
     * An exception is thrown if the stream is closed, either via free or due to a previous call
     * to this method returning -1.
     */
    public static native long take(long pointer);

    /**
     * Get a pointer to the DType of the elements of the stream.
     */
    public static native long getDType(long pointer);
}
