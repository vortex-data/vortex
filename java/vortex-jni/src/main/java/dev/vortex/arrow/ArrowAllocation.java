package dev.vortex.arrow;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;

public final class ArrowAllocation {
    private static final RootAllocator ROOT_ALLOCATOR = new RootAllocator(Long.MAX_VALUE);

    private ArrowAllocation() {
    }

    public static BufferAllocator rootAllocator() {
        return ROOT_ALLOCATOR;
    }
}
