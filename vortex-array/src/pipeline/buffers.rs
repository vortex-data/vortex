// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::task::{Poll, ready};

use vortex_buffer::Buffer;
use vortex_error::{VortexResult, vortex_panic};

use crate::pipeline::KernelContext;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufferId(usize);

impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferId {
    /// Creates a new `BufferId` with a unique identifier.
    pub fn new() -> Self {
        BufferId(NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

impl Deref for BufferId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Wrapper for managed buffer access with automatic caching
#[derive(Debug, Clone)]
pub enum BufferHandle<T> {
    Pending(BufferId),
    Ready(Buffer<T>),
}

impl<T> Hash for BufferHandle<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            BufferHandle::Pending(id) => id.hash(state),
            BufferHandle::Ready(buffer) => buffer.as_ptr().hash(state),
        }
    }
}

impl<T> PartialEq for BufferHandle<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BufferHandle::Pending(id1), BufferHandle::Pending(id2)) => id1 == id2,
            (BufferHandle::Ready(buf1), BufferHandle::Ready(buf2)) => {
                buf1.as_ptr() == buf2.as_ptr()
            }
            _ => false,
        }
    }
}

impl<T> BufferHandle<T> {
    pub fn new(buffer: Buffer<T>) -> Self {
        BufferHandle::Ready(buffer)
    }

    pub fn new_pending(id: BufferId) -> Self {
        BufferHandle::Pending(id)
    }

    pub fn into_typed<S>(self) -> BufferHandle<S> {
        match self {
            BufferHandle::Ready(buffer) => {
                BufferHandle::Ready(Buffer::<S>::from_byte_buffer(buffer.into_byte_buffer()))
            }
            BufferHandle::Pending(id) => BufferHandle::Pending(id),
        }
    }

    pub fn buffer(self) -> Option<Buffer<T>> {
        match self {
            BufferHandle::Ready(buffer) => Some(buffer),
            BufferHandle::Pending(_) => None,
        }
    }

    pub fn get_or_load(&mut self, ctx: &dyn KernelContext) -> Poll<VortexResult<&Buffer<T>>> {
        if let BufferHandle::Ready(buffer) = self {
            return Poll::Ready(Ok(buffer));
        }

        let buffer_id = match self {
            BufferHandle::Pending(id) => *id,
            BufferHandle::Ready(_) => unreachable!("BufferHandle should not be ready here"),
        };
        let buffer = ready!(ctx.buffer(buffer_id))?;
        *self = BufferHandle::Ready(Buffer::<T>::from_byte_buffer(buffer));

        if let BufferHandle::Ready(buffer) = self {
            Poll::Ready(Ok(buffer))
        } else {
            vortex_panic!("BufferHandle should be ready after loading");
        }
    }
}
