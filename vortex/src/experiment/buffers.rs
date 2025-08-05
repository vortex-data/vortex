// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::encodings::EvaluationContext;
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::task::{Poll, ready};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_utils::aliases::hash_map::HashMap;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufferId(usize);

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

    pub fn get_or_load(&mut self, ctx: &dyn EvaluationContext) -> Poll<VortexResult<&Buffer<T>>> {
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

impl EvaluationContext for HashMap<BufferId, ByteBuffer> {
    fn buffer(&self, buffer_id: BufferId) -> Poll<VortexResult<ByteBuffer>> {
        match self.get(&buffer_id) {
            Some(buffer) => Poll::Ready(Ok(buffer.clone())),
            None => Poll::Pending,
        }
    }
}
