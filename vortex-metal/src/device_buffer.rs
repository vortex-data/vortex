// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

/// A [`DeviceBuffer`] wrapping a Metal GPU allocation.
///
/// Like the host `BufferHandle` variant, all slicing/referencing works in terms of byte units.
/// On Apple Silicon, Metal uses unified memory, so the buffer contents are directly accessible
/// from both CPU and GPU.
#[derive(Clone)]
pub struct MetalDeviceBuffer {
    /// The underlying Metal buffer
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Offset in bytes from the start of the allocation
    offset: usize,
    /// Length in bytes
    len: usize,
    /// Minimum required alignment of the buffer
    alignment: Alignment,
}

impl MetalDeviceBuffer {
    /// Creates a new Metal device buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The Metal buffer
    /// * `alignment` - The alignment of the buffer
    pub fn new(buffer: Retained<ProtocolObject<dyn MTLBuffer>>, alignment: Alignment) -> Self {
        let len = buffer.length();
        Self {
            buffer,
            offset: 0,
            len,
            alignment,
        }
    }

    /// Returns a reference to the underlying Metal buffer.
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Returns the offset in bytes from the start of the allocation.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns a pointer to the buffer contents at the current offset.
    ///
    /// On Apple Silicon with shared memory, this pointer is directly accessible from the CPU.
    ///
    /// # Safety
    ///
    /// The caller must ensure proper synchronization between CPU and GPU access.
    pub fn contents_ptr(&self) -> *mut std::ffi::c_void {
        // SAFETY: contents() returns a valid pointer for the buffer's lifetime
        let base_ptr = self.buffer.contents().as_ptr();
        // SAFETY: Adding offset within buffer bounds
        unsafe { base_ptr.add(self.offset) }
    }

    /// Wraps this buffer into a `BufferHandle`.
    pub fn into_buffer_handle(self) -> BufferHandle {
        BufferHandle::new_device(Arc::new(self))
    }
}

/// Extension trait for getting a Metal buffer from a [`BufferHandle`].
pub trait MetalBufferExt {
    /// Returns a reference to the Metal device buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a Metal buffer.
    fn metal_buffer(&self) -> VortexResult<&MetalDeviceBuffer>;
}

impl MetalBufferExt for BufferHandle {
    fn metal_buffer(&self) -> VortexResult<&MetalDeviceBuffer> {
        let device_buffer = self
            .as_device_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on device"))?;

        device_buffer
            .as_any()
            .downcast_ref::<MetalDeviceBuffer>()
            .ok_or_else(|| vortex_err!("expected MetalDeviceBuffer, was {device_buffer:?}"))
    }
}

impl Debug for MetalDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalDeviceBuffer")
            .field("buffer_ptr", &self.buffer.contents())
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("alignment", &self.alignment)
            .finish()
    }
}

impl Hash for MetalDeviceBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash based on the buffer pointer, offset, and length
        (self.buffer.contents().as_ptr() as usize).hash(state);
        self.offset.hash(state);
        self.len.hash(state);
    }
}

impl PartialEq for MetalDeviceBuffer {
    fn eq(&self, other: &Self) -> bool {
        // Equal if they point to the same extent of GPU memory
        std::ptr::eq(
            self.buffer.contents().as_ptr(),
            other.buffer.contents().as_ptr(),
        ) && self.offset == other.offset
            && self.len == other.len
    }
}

impl Eq for MetalDeviceBuffer {}

impl DeviceBuffer for MetalDeviceBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Synchronous copy of Metal device to host memory.
    ///
    /// On Apple Silicon with unified memory, this is essentially a memcpy
    /// since the buffer is already accessible from the CPU.
    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        // On Apple Silicon, Metal buffers with shared storage mode are directly
        // accessible from the CPU. We just need to copy the data.
        let mut host_buffer = ByteBufferMut::with_capacity_aligned(self.len, alignment);

        let src_ptr = self.contents_ptr();

        // SAFETY: We're copying from a valid Metal buffer to our host buffer.
        // The Metal buffer contents are valid for the buffer's lifetime.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptr.cast::<u8>(),
                host_buffer.spare_capacity_mut().as_mut_ptr().cast(),
                self.len,
            );
            host_buffer.set_len(self.len);
        }

        Ok(host_buffer.freeze().into_byte_buffer())
    }

    /// Copies a device buffer to host memory asynchronously.
    ///
    /// On Apple Silicon with unified memory, this completes immediately since
    /// the data is already accessible. For discrete GPUs, this would schedule
    /// a blit command.
    fn copy_to_host(
        &self,
        alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        // For unified memory on Apple Silicon, we can just do a synchronous copy
        // wrapped in an async block.
        let buffer = self.copy_to_host_sync(alignment)?;
        Ok(Box::pin(async move { Ok(buffer) }))
    }

    /// Slices the Metal device buffer to a subrange.
    ///
    /// **IMPORTANT**: this is a byte range, not elements range.
    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        assert!(
            range.end <= self.len,
            "Slice range end {} exceeds allocation size {}",
            range.end,
            self.len
        );

        let new_offset = self.offset + range.start;
        let new_len = range.end - range.start;

        Arc::new(MetalDeviceBuffer {
            buffer: self.buffer.clone(),
            offset: new_offset,
            len: new_len,
            alignment: self.alignment,
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        let effective_ptr = self.buffer.contents().as_ptr() as usize + self.offset;
        if effective_ptr.is_multiple_of(*alignment) {
            Ok(Arc::new(MetalDeviceBuffer {
                buffer: self.buffer.clone(),
                offset: self.offset,
                len: self.len,
                alignment,
            }))
        } else {
            // Metal buffers are typically aligned to at least 16 bytes.
            // If we need higher alignment, we would need to allocate a new buffer.
            Err(vortex_err!(
                "Cannot align MetalDeviceBuffer to {} (current offset: {})",
                alignment,
                self.offset
            ))
        }
    }
}

// Implement Send + Sync for MetalDeviceBuffer
// SAFETY: Metal buffers can be shared across threads on Apple platforms.
// The underlying Metal runtime handles synchronization.
unsafe impl Send for MetalDeviceBuffer {}
unsafe impl Sync for MetalDeviceBuffer {}
