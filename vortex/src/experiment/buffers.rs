// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::encodings::EvaluationContext;
use std::ops::Deref;
use std::sync::atomic::AtomicUsize;
use std::task::{Poll, ready};
use vortex_buffer::{Buffer, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult};
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
pub struct BufferHandle<T> {
    buffer_id: BufferId,
    buffer: Option<Buffer<T>>,
}

impl<T> BufferHandle<T> {
    pub fn new(buffer_id: BufferId) -> Self {
        Self {
            buffer_id,
            buffer: None,
        }
    }

    pub fn get_or_load(&mut self, ctx: &dyn EvaluationContext) -> Poll<VortexResult<&Buffer<T>>> {
        if self.buffer.is_none() {
            let byte_buffer = ready!(ctx.buffer(self.buffer_id))?;
            self.buffer = Some(Buffer::<T>::from_byte_buffer(byte_buffer));
        }
        Poll::Ready(Ok(self.buffer.as_ref().vortex_expect("infallible")))
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
