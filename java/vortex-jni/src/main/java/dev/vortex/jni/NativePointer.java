// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.jni;

import com.google.common.base.Preconditions;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A native pointer cell that can be safely taken at most once.
 *
 * <p>Wraps an {@link AtomicLong} that holds a non-zero native address until it is {@link #take() taken}, after which
 * the cell is empty. {@link #read()} fails fast with {@link IllegalStateException} once the pointer has been taken,
 * turning use-after-free into a Java exception instead of a JVM crash.
 */
public final class NativePointer {
    private final AtomicLong pointer;

    private NativePointer(long pointer) {
        this.pointer = new AtomicLong(pointer);
    }

    /** Wrap a freshly-allocated, non-zero native pointer. */
    public static NativePointer of(long pointer) {
        Preconditions.checkArgument(pointer != 0, "pointer must be non-zero");
        return new NativePointer(pointer);
    }

    /**
     * Destructively read the pointer value, replacing it with a null reference.
     *
     * @throws IllegalStateException if the pointer has already been taken or freed.
     */
    public long take() {
        long ref = pointer.getAndSet(0L);
        Preconditions.checkState(ref != 0, "pointer already taken or freed");
        return ref;
    }

    /**
     * Destructively read the pointer value, returning {@code 0} if it was already taken. Intended for cleanup callbacks
     * where ownership may legitimately have been transferred elsewhere first.
     */
    public long tryTake() {
        return pointer.getAndSet(0L);
    }

    /**
     * Read the pointer address.
     *
     * @throws IllegalStateException if the pointer has been freed previously.
     */
    public long read() {
        long ref = pointer.get();
        Preconditions.checkState(ref != 0, "cannot perform read() of freed pointer");
        return ref;
    }
}
