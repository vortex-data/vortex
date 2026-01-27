// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;

/// A destination memory region for writes.
pub enum WriteRegion<'a> {
    /// A standard host slice that can be written by the CPU.
    HostSlice(&'a mut [u8]),
    /// A registered host memory region suitable for RDMA writes.
    Registered(RegisteredRegion<'a>),
    /// A device memory region suitable for GPU-direct or other device DMA.
    Device(DeviceRegion<'a>),
}

/// A registered host memory region suitable for RDMA writes.
pub struct RegisteredRegion<'a> {
    pub ptr: *mut u8,
    pub len: usize,
    pub lkey: u32,
    pub rkey: u32,
    pub(crate) _lifetime: PhantomData<&'a mut [u8]>,
}

/// A device memory region suitable for device DMA.
pub struct DeviceRegion<'a> {
    pub ptr: *mut u8,
    pub len: usize,
    pub(crate) _lifetime: PhantomData<&'a mut [u8]>,
}

/// A destination for I/O reads that can be finalized into a [`BufferHandle`].
pub trait WriteDestination: Send + 'static {
    /// Returns the length of the buffer in bytes.
    fn len(&self) -> usize;

    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the writable region for this target.
    fn region(&mut self) -> WriteRegion<'_>;

    /// Finalize the target into a buffer handle.
    fn into_handle(self: Box<Self>) -> BoxFuture<'static, VortexResult<BufferHandle>>;
}

impl WriteDestination for ByteBufferMut {
    fn len(&self) -> usize {
        ByteBufferMut::len(self)
    }

    fn region(&mut self) -> WriteRegion<'_> {
        WriteRegion::HostSlice(self.as_mut())
    }

    fn into_handle(self: Box<Self>) -> BoxFuture<'static, VortexResult<BufferHandle>> {
        async move { Ok(BufferHandle::new_host(self.freeze())) }.boxed()
    }
}
