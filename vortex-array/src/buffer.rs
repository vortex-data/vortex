// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::vortex_bail;
use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::dyn_traits::DynEq;
use vortex_utils::dyn_traits::DynHash;

/// A buffer can be either on the CPU or on an attached device (e.g. GPU).
/// The Device implementation will come later.
#[derive(Debug, Clone)]
pub enum BufferHandle {
    /// On the host/cpu.
    Host(ByteBuffer),
    /// On the device.
    Device(Arc<dyn DeviceBuffer>),
}

impl BufferHandle {
    pub fn len(&self) -> usize {
        match self {
            BufferHandle::Host(b) => b.len(),
            BufferHandle::Device(d) => d.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            BufferHandle::Host(b) => b.is_empty(),
            BufferHandle::Device(d) => d.is_empty(),
        }
    }
}

/// A buffer that is stored on a device (e.g. GPU).
pub trait DeviceBuffer: 'static + Send + Sync + Debug + DynEq + DynHash {
    /// Downcast to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Returns the length of the buffer in bytes.
    fn len(&self) -> usize;

    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to copy the device buffer to a host ByteBuffer.
    fn to_host(self: Arc<Self>) -> VortexResult<ByteBuffer>;
}

impl Hash for dyn DeviceBuffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn DeviceBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}
impl Eq for dyn DeviceBuffer {}

impl BufferHandle {
    /// Fetches the cpu buffer and fails otherwise.
    pub fn bytes(&self) -> ByteBuffer {
        match self {
            BufferHandle::Host(b) => b.clone(),
            BufferHandle::Device(b) => b
                .clone()
                .to_host()
                .vortex_expect("failed to move device buffer to host buffer"),
        }
    }

    /// Fetches the cpu buffer and fails otherwise.
    pub fn into_bytes(self) -> ByteBuffer {
        match self {
            BufferHandle::Host(b) => b,
            BufferHandle::Device(_) => todo!(),
        }
    }

    /// Attempts to convert this handle into a CPU ByteBuffer.
    /// Returns an error if the buffer is on a device.
    pub fn try_to_bytes(self) -> VortexResult<ByteBuffer> {
        match self {
            BufferHandle::Host(b) => Ok(b),
            BufferHandle::Device(_) => vortex_bail!("cannot move device_buffer to buffer"),
        }
    }
}
