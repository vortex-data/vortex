package dev.vortex.jni;

public final class NativeArrayStreamMethods {
    private NativeArrayStreamMethods() {}

    public static native void free(long pointer);

    public static native long take(long pointer);

    public static native long getDType(long pointer);

    public static native boolean hasNext(long pointer);
}
