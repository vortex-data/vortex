package dev.vortex.api;

import dev.vortex.jni.FFI;

/**
 * Base class for all objects that are wrappers around some native {@code T} exposed by FFI.
 * <p>
 * Each wrapped type has a close implementation that will free the native resource.
 */
public abstract class BaseWrapped<T> implements AutoCloseable {
    protected final T inner;

    protected BaseWrapped(T inner) {
        this.inner = inner;
    }

    @Override
    public abstract void close();
}
