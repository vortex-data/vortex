// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-scoped buffer allocation.

use std::any::Any;

pub use vortex_buffer::BufferAllocator;
pub use vortex_buffer::BufferAllocatorRef;
pub use vortex_buffer::StaticBufferAllocator;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::VortexSession;

/// Session-scoped memory configuration for Vortex arrays.
#[derive(Clone, Debug)]
pub struct MemorySession {
    allocator: BufferAllocatorRef,
}

impl MemorySession {
    /// Creates a new memory configuration using the provided allocator.
    pub fn new(allocator: BufferAllocatorRef) -> Self {
        Self { allocator }
    }

    /// Returns the configured allocator.
    pub fn allocator(&self) -> BufferAllocatorRef {
        self.allocator.clone()
    }

    /// Updates the configured allocator.
    pub fn set_allocator(&mut self, allocator: BufferAllocatorRef) {
        self.allocator = allocator;
    }
}

impl Default for MemorySession {
    fn default() -> Self {
        Self::new(BufferAllocatorRef::statically_allocated())
    }
}

impl SessionVar for MemorySession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Extension methods for session-scoped buffer allocation.
pub trait MemorySessionExt: SessionExt {
    /// Returns the memory configuration.
    fn memory(&self) -> SessionGuard<'_, MemorySession> {
        self.get::<MemorySession>()
    }

    /// Returns the configured buffer allocator.
    fn allocator(&self) -> BufferAllocatorRef {
        self.memory().allocator()
    }

    /// Configures the session allocator and returns the session.
    fn with_allocator(self, allocator: BufferAllocatorRef) -> VortexSession {
        let session = self.session();
        session.get_mut::<MemorySession>().set_allocator(allocator);
        session
    }
}

impl<S: SessionExt> MemorySessionExt for S {}

#[cfg(test)]
pub(crate) mod test_allocator {
    use std::alloc::Layout;
    use std::ptr::NonNull;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use allocator_api2::alloc::AllocError;
    use allocator_api2::alloc::Allocator;
    use allocator_api2::alloc::Global;

    use super::BufferAllocatorRef;

    #[derive(Debug)]
    struct CountingAllocator(Arc<AtomicUsize>);

    unsafe impl Allocator for CountingAllocator {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Global.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
            // SAFETY: the allocation came from Global with this layout.
            unsafe { Global.deallocate(ptr, layout) }
        }
    }

    pub(crate) fn counting_allocator() -> (BufferAllocatorRef, Arc<AtomicUsize>) {
        let allocations = Arc::new(AtomicUsize::new(0));
        (
            BufferAllocatorRef::new(CountingAllocator(Arc::clone(&allocations))),
            allocations,
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BufferAllocatorRef;

    use super::MemorySession;

    #[test]
    fn memory_session_replaces_allocator() {
        let allocator = BufferAllocatorRef::statically_allocated();
        let mut session = MemorySession::default();
        session.set_allocator(allocator);
        let buffer = session.allocator().copy_from([1u32, 2, 3]);
        assert_eq!(buffer.as_slice(), [1, 2, 3]);
    }
}
