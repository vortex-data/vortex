// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

/// A destination for I/O reads that can be finalized into a [`BufferHandle`].
pub trait WriteTarget: Send + 'static {
    /// Returns the buffer as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [u8];

    /// Returns the length of the buffer in bytes.
    fn len(&self) -> usize;

    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Finalize the target into a buffer handle.
    fn into_handle(self: Box<Self>) -> VortexResult<BufferHandle>;
}

impl WriteTarget for ByteBufferMut {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        self.as_mut()
    }

    fn len(&self) -> usize {
        ByteBufferMut::len(self)
    }

    fn into_handle(self: Box<Self>) -> VortexResult<BufferHandle> {
        Ok(BufferHandle::new_host(self.freeze()))
    }
}
