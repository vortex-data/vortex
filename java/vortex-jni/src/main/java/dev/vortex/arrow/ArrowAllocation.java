// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.arrow;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

/**
 * Utility class for managing Apache Arrow memory allocation.
 *
 * <p>This class provides a global shared root allocator for Arrow memory operations used throughout the Vortex JNI
 * layer. The allocator is configured with the maximum available heap size to allow for efficient memory management.
 */
public final class ArrowAllocation {
    private static final RootAllocator ROOT_ALLOCATOR = new RootAllocator(Long.MAX_VALUE);

    private ArrowAllocation() {}

    /**
     * Returns the shared root allocator instance for Apache Arrow operations.
     *
     * <p>This allocator is shared across all Arrow operations in the JVM and is configured to use the maximum available
     * memory. It should be used as the parent allocator for all Arrow memory operations within the Vortex system.
     *
     * @return the shared {@link BufferAllocator} instance
     */
    public static BufferAllocator rootAllocator() {
        return ROOT_ALLOCATOR;
    }
}
