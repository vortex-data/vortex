// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import dev.vortex.VortexCleaner;
import dev.vortex.jni.NativeExpression;
import java.lang.ref.Reference;

/**
 * A comparison value that can change while a scan runs, read by {@link Expression#dynamic}.
 *
 * <p>A value starts unset. {@link #set(Expression)} replaces it, and every comparison over it sees the new value from
 * its next evaluation on, in any thread. Expressions share the value rather than copying it, so they stay valid after
 * this handle becomes unreachable. The native value is released automatically via {@link VortexCleaner}.
 */
public final class DynamicValue {
    private final long pointer;

    private DynamicValue(long pointer) {
        Preconditions.checkArgument(pointer != 0, "invalid dynamic value pointer");
        this.pointer = pointer;
        VortexCleaner.register(this, () -> NativeExpression.dynamicValueFree(pointer));
    }

    /**
     * An unset value whose type is that of the literal {@code typePrototype}, typically a typed null such as
     * {@code Expression.nullLiteral(DType.I64)}.
     */
    public static DynamicValue create(Expression typePrototype) {
        // The Cleaner of an unreachable expression frees its native pointer, so keep the input alive for the call.
        try {
            return new DynamicValue(NativeExpression.dynamicValueCreate(typePrototype.nativePointer()));
        } finally {
            Reference.reachabilityFence(typePrototype);
        }
    }

    /** Sets the value to the literal {@code value}, cast to this value's type. */
    public void set(Expression value) {
        try {
            NativeExpression.dynamicValueSet(pointer, value.nativePointer());
        } finally {
            Reference.reachabilityFence(value);
            Reference.reachabilityFence(this);
        }
    }

    long nativePointer() {
        return pointer;
    }
}
